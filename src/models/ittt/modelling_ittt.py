import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from transformers.modeling_utils import PreTrainedModel

from models.ittt.configuration_ittt import ItttConfig
from models.reference_llama.modelling_llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.torch_utils import simple_rms_norm
import utils.constants as constants


# @torch.compile(fullgraph=True, mode="reduce-overhead")
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

        tp = torch.float32 if constants.DEVICE == "cpu" else torch.bfloat16

        if mod.state is not None:
            with torch.enable_grad():

                s = mod.state.clone().detach().to(tp).requires_grad_(True)
                w = mod.get_lr()[None].detach().to(tp) * s

                z_for_grad = torch.einsum("boi,bji->bjo", w, x.to(tp))            

                s_grad = torch.autograd.grad(
                    z_for_grad,
                    s,
                    grad_outputs=grad.detach().to(tp),
                )[0]

                if mod.state_grad is None:
                    mod.state_grad = s_grad.detach().to(mod.state_dtype)
                else:
                    mod.state_grad += s_grad.detach().to(mod.state_dtype)

        do_fo = mod.last_state_grad is not None
        
        if do_fo:
            x_leaf = torch.chunk(x, 2, dim=0)[0]
            g = torch.chunk(grad, 2, dim=0)[1]

        else:
            x_leaf = x
            g = grad

        with torch.set_grad_enabled(do_fo):

            x_leaf = x_leaf.clone().detach().float().requires_grad_(do_fo)
            g = g.clone().detach().float().requires_grad_(False)

            x = (x_leaf - x_leaf.mean(dim=-2, keepdim=True))
            x = F.normalize(x, dim=-2, eps=mod.eps) * math.sqrt(x.shape[-2])  # [b, s, i]

            g = F.normalize(g, dim=-2, eps=mod.eps) * math.sqrt(x.shape[-2])  # [b, s, r]

            x = x.to(tp)
            g = g.to(tp)

            # [b, r, i]
            update = -( # negative for gradient DESCENT
                g.transpose(-2, -1) @ x
            ) 
            
            update = update / (
                math.sqrt(x.shape[-2]) # approx 1 std
                * math.sqrt(mod.in_features) # kaiming scaling
            )
            # update = newtonschulz(update)

            if do_fo:

                if mod.state_grad is not None:
                    curr_grad = torch.chunk(mod.state_grad, 2, dim=0)[1]
                    grad_for_s = (mod.last_state_grad - curr_grad).detach().to(tp)
                else:
                    grad_for_s = mod.last_state_grad.detach().to(tp)

                x_grad = torch.autograd.grad(
                    update,
                    x_leaf,
                    grad_outputs=grad_for_s,
                )[0]

                x_grad = torch.cat([x_grad, torch.zeros_like(x_grad)], dim=0).detach().to(x_dtype)
            
                update = torch.cat([update, update], dim=0)

            else:
                x_grad = None

            mod.update = update.detach()

        return x_grad, og_grad, None

        
class ItttLinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        base_lr: float,
        momentum_beta: float,
        eps: float = 1e-7,
        state_dtype=torch.float32,
    ):
        super().__init__()

        # save config
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank

        self.base_lr = base_lr
        self.momentum_beta = momentum_beta

        self.eps = eps
        self.scalar_scaler = math.sqrt(self.in_features)

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

        # ephemeral state
        self.state = None
        self.update = None
        self.state_grad = None
        self.last_state_grad = None

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
    def relative_grad_error(self):
        if self.state_grad is None or self.last_state_grad is None:
            return None
        
        err = (self.state_grad.chunk(2, dim=0)[1] - self.last_state_grad).norm()
        denom = self.last_state_grad.norm() + self.eps

        return (err / denom).item()


    @torch.no_grad()
    def finalize_state(self):
        self.last_state_grad = self.state_grad
        self.state = None
        self.update = None
        self.state_grad = None        

    
    @torch.no_grad()
    def reset_state(self):
        self.state = None
        self.update = None
        self.state_grad = None
        self.last_state_grad = None

    
    @torch.no_grad()
    def update_state(self):
        if self.update is None:
            return
                
        # we don't worry about adam-like biased momentum because newton-schulz normalizes anyway
        delta = self.update.to(self.state_dtype)
        self.update = None

        if self.state is None:
            self.state = delta
        else:
            self.state += delta


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


    def __init__(self, config):
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
                eps=self.eps
            )
            layer.self_attn.o_proj = ItttLinear(
                layer.self_attn.o_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                eps=self.eps
            )
            layer.mlp.down_proj = ItttLinear(
                layer.mlp.down_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                eps=self.eps
            )

        self.post_init()

    
    def _init_weights(self, module):
        # We don't want to re-initialize the weights, so we override this method to do nothing.
        return


    @torch.no_grad()
    def relative_grad_error(self):
        errors = []

        for module in self.modules():
            if isinstance(module, ItttLinear):
                
                err = module.relative_grad_error()
                if err is not None:
                    errors.append(err)

        if len(errors) == 0:
            return None

        return sum(errors) / len(errors)


    @torch.no_grad()
    def finalize_state(self):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.finalize_state()


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
                

    def forward(self, *args, **kwargs):
        return self.llama(*args, **kwargs)
        