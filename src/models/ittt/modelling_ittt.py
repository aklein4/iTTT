import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from transformers.modeling_utils import PreTrainedModel

from models.ittt.configuration_ittt import ItttConfig
from models.reference_llama.modelling_llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.torch_utils import simple_rms_norm


@torch.compile(fullgraph=True, mode="reduce-overhead")
def newtonschulz(
    x: torch.FloatTensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.FloatTensor:
    # https://kellerjordan.github.io/posts/muon/
    
    assert x.ndim == 3, "x must be a 2D and batch"
    bs = x.shape[0]

    a, b, c = (3.4445, -4.7750, 2.0315)

    y = x / (
        x.reshape(bs, -1).norm(dim=-1)[:, None, None] + eps
    )

    if x.shape[-2] > x.shape[-1]:
        y = y.transpose(-2, -1)

    for _ in range(steps):
        m = y @ y.transpose(-2, -1)
        n = b * m + c * m @ m
        y = a * y + n @ y

    if x.shape[-2] > x.shape[-1]:
        y = y.transpose(-2, -1)

    return y


class ItttFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, z, mod):
        ctx.save_for_backward(x)
        ctx.mod = mod
        return z.clone()
    

    @staticmethod
    def backward(ctx, grad):

        x, = ctx.saved_tensors
        mod: ItttLinear = ctx.mod

        og_grad = grad.clone()
        x_dtype = x.dtype
        
        do_loss = mod.momentum is not None

        with torch.set_grad_enabled(do_loss):
                
            x_leaf = x.float()
            g = grad.float()

            x_leaf = x_leaf.detach().requires_grad_(do_loss)
            g = g.detach().requires_grad_(False)
        
            x = simple_rms_norm(
                x_leaf - x_leaf.mean(-2, keepdim=True),
                eps=mod.eps
            ) # [b, s, i]
            g = simple_rms_norm(g, eps=mod.eps)  # [b, s, r]

            # [b, r, i]
            this_update = g.transpose(-2, -1) @ x

            # TODO: this scale is a little weird on the first chunk
            mod.update = (
                this_update / math.sqrt(x.shape[-2]) # approx 1 std
            )

            if do_loss:

                w = mod.momentum_step(this_update)

                pred_g = torch.einsum("boi,bji->bjo", w, x)
                pred_g = simple_rms_norm(pred_g, eps=mod.eps)

                mod.prev_loss = F.mse_loss(pred_g, g) / mod.num_ittt
                
                weighted_loss = mod.aux_weight * mod.prev_loss
                x_grad = torch.autograd.grad(weighted_loss, x_leaf)[0]
            
                x_grad = x_grad.to(x_dtype)

            else:
                x_grad = None

        return x_grad, og_grad, None

        
class ItttLinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        base_lr: float,
        momentum_beta: float,
        aux_weight: float,
        eps: float = 1e-7,
        momentum_dtype=torch.bfloat16,
        state_dtype=torch.float32,
    ):
        super().__init__()

        # save config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank

        self.base_lr = base_lr
        self.momentum_beta = momentum_beta
        self.aux_weight = aux_weight

        self.eps = eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.momentum_dtype = momentum_dtype
        self.state_dtype = state_dtype

        # save linear
        self.weight = linear.weight
        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_parameter("bias", None)
        
        # ittt params
        self.log_lr = nn.Parameter(
            torch.zeros(rank, self.in_features)
        )
        self.out_proj = nn.Parameter(
            torch.randn(self.out_features, rank) / math.sqrt(self.rank)
        )

        self.num_ittt = None

        # ephemeral state
        self.state = None
        self.momentum = None
        self.update = None
        self.prev_loss = None

        self.svd_init()

    
    @torch.no_grad()
    def svd_init(self):

        u, s, v = torch.linalg.svd(self.weight, full_matrices=False)

        self.out_proj.copy_(
            u[:, :self.rank] *
            s[None, :self.rank]
        )
    

    def get_lr(self):
        return self.base_lr * torch.exp(self.log_lr * self.scalar_scaler)


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        if self.state is not None:

            s = self.get_lr()[None] * self.state

        else:
            s = torch.zeros_like(self.log_lr)[None]

        z = torch.einsum("boi,bji->bjo", s, x)
        z = ItttFunction.apply(x, z, self)

        y_lora = F.linear(z, self.out_proj)
        y_base = F.linear(x, self.weight, self.bias)

        y = y_base + y_lora

        return y

    
    @torch.no_grad()
    def reset_state(self):
        self.state = None
        self.momentum = None
        self.update = None
        self.prev_loss = None


    def get_prev_loss(self):
        return self.prev_loss


    def momentum_step(self, x):
        return (
            self.momentum_beta * self.momentum.to(x.dtype) +
            (1 - self.momentum_beta) * x
        )

    
    @torch.no_grad()
    def update_state(self):
        if self.update is None:
            return
                
        if self.momentum is None:
            self.momentum = (
                (1 - self.momentum_beta) * self.update.detach().to(self.momentum_dtype)
            )
        else:
            self.momentum = (
                self.momentum_beta * self.momentum +
                (1 - self.momentum_beta) * self.update.detach().to(self.momentum_dtype)
            )
        self.update = None

        # we don't worry about adam-like biased momentum because newton-schulz normalizes anyway
        delta = -newtonschulz(
            self.momentum,
            eps=self.eps
        ).to(self.state_dtype)

        if self.state is None:
            self.state = delta
        else:
            self.state += delta

        self.prev_loss = None


class ItttModel(PreTrainedModel):

    config: ItttConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True


    def __init__(self, config: ItttConfig):
        super().__init__(config)

        self.llama = LlamaForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float32,
            attn_implementation=config._attn_implementation
        )

        self.start_layer = config.start_layer
        self.rank = config.rank
        self.eps = self.llama.config.rms_norm_eps

        for layer in self.llama.model.layers[self.start_layer:]:
            layer: LlamaDecoderLayer

            layer.self_attn.q_proj = ItttLinear(
                layer.self_attn.q_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                aux_weight=config.aux_weight,
                eps=self.eps
            )
            layer.self_attn.o_proj = ItttLinear(
                layer.self_attn.o_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                aux_weight=config.aux_weight,
                eps=self.eps
            )
            layer.mlp.down_proj = ItttLinear(
                layer.mlp.down_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                aux_weight=config.aux_weight,
                eps=self.eps
            )

        self.num_ittt = 0
        for m in self.modules():
            if isinstance(m, ItttLinear):
                self.num_ittt += 1
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.num_ittt = self.num_ittt
        
        self.post_init()

    
    def _init_weights(self, module):
        # We don't want to re-initialize the weights, so we override this method to do nothing.
        return


    @torch.no_grad()
    def reset_state(self):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.reset_state()
                

    @torch.no_grad()
    def update_state(self):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.update_state()
    

    def get_prev_loss(self):

        loss = None

        for m in self.modules():
            if isinstance(m, ItttLinear):

                l = m.get_prev_loss()
                if l is not None:
                    if loss is None:
                        loss = l
                    else:
                        loss = loss + l

        return loss

    def forward(self, *args, **kwargs):
        return self.llama(*args, **kwargs)
        