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
    def forward(ctx, x, y_lora, mod):
        ctx.save_for_backward(x)
        ctx.mod = mod
        return y_lora.clone()
    

    @staticmethod
    def backward(ctx, grad):
        og_grad = grad.clone()

        x, = ctx.saved_tensors
        mod = ctx.mod

        with torch.enable_grad():

            z_leaf = torch.zeros(
                x.shape[0], x.shape[1], mod.rank,
                device=x.device,
                dtype=mod.momentum_dtype,
                requires_grad=True
            )
            ema_proj = mod.ema_out_proj.clone().detach().to(mod.momentum_dtype).requires_grad_(False)

            pred_y_lora = F.linear(z_leaf, ema_proj)

            g = torch.autograd.grad(
                pred_y_lora,
                z_leaf,
                grad.to(mod.momentum_dtype),
            )[0].detach()

        x = x.to(mod.momentum_dtype)
        g = g.to(mod.momentum_dtype)

        x = F.linear(x, mod.ema_in_proj.to(mod.momentum_dtype))

        x = simple_rms_norm(x, eps=mod.eps) # [b, s, i]
        g = F.normalize(g, dim=-2, eps=mod.eps) * math.sqrt(x.shape[-2])  # [b, s, r]

        # [b, r, i]
        update = (
            g.transpose(-2, -1) @ x
        ) / math.sqrt(x.shape[-2]) # approx 1 std

        if mod.momentum is None:
            mod.momentum = (
                (1 - mod.momentum_beta) * update
            )
        else:
            mod.momentum = (
                mod.momentum_beta * mod.momentum +
                (1 - mod.momentum_beta) * update
            )

        return None, og_grad, None

        
class ItttLinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        base_lr: float,
        momentum_beta: float,
        ema_beta: float,
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
        self.ema_beta = ema_beta

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
            torch.zeros(rank, self.rank)
        )

        self.in_proj = nn.Parameter(
            torch.randn(rank, self.in_features) / math.sqrt(self.in_features)
        )
        self.register_buffer(
            "ema_in_proj",
            self.in_proj.data.clone(),
            persistent=True
        )
        self.ema_in_proj: nn.Buffer

        self.out_proj = nn.Parameter(
            torch.randn(self.out_features, rank) / math.sqrt(self.rank)
        )
        self.register_buffer(
            "ema_out_proj",
            self.out_proj.data.clone(),
            persistent=True
        )
        self.ema_out_proj: nn.Buffer

        # ephemeral state
        self.state = None
        self.momentum = None

        self.svd_init()

    
    @torch.no_grad()
    def svd_init(self):

        u, s, v = torch.linalg.svd(self.weight, full_matrices=False)

        self.in_proj.copy_(
            v[:self.rank, :] *
            torch.sqrt(s[:self.rank, None])
        )
        self.ema_in_proj.copy_(self.in_proj.data.clone())

        self.out_proj.copy_(
            u[:, :self.rank] *
            torch.sqrt(s[None, :self.rank])
        )
        self.ema_out_proj.copy_(self.out_proj.data.clone())


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        if self.state is not None:

            lr = (
                self.base_lr *
                torch.exp(self.log_lr * self.scalar_scaler)
            )
            s = lr[None] * self.state

        else:
            s = torch.zeros_like(self.log_lr)[None]

        z = F.linear(x, self.in_proj)
        z = torch.einsum("boi,bji->bjo", s, z)

        y_lora = F.linear(z, self.out_proj)
        y_lora = ItttFunction.apply(x, y_lora, self)

        y_base = F.linear(x, self.weight, self.bias)

        y = y_base + y_lora

        return y


    @torch.no_grad()
    def ema_update(self):
        self.ema_in_proj.copy_(
            self.ema_beta * self.ema_in_proj +
            (1 - self.ema_beta) * self.in_proj.data
        )
        self.ema_out_proj.copy_(
            self.ema_beta * self.ema_out_proj +
            (1 - self.ema_beta) * self.out_proj.data
        )

    
    @torch.no_grad()
    def reset_state(self):
        self.state = None
        self.momentum = None

    
    @torch.no_grad()
    def update_state(self):
        if self.momentum is None:
            return
                
        # we don't worry about adam-like biased momentum because newton-schulz normalizes anyway
        delta = -newtonschulz(
            self.momentum,
            eps=self.eps
        ).to(self.state_dtype)

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

            # layer.self_attn.q_proj = ItttLinear(
            #     layer.self_attn.q_proj,
            #     rank=config.rank,
            #     base_lr=config.base_lr,
            #     momentum_beta=config.momentum_beta,
            #     ema_beta=config.ema_beta,
            #     eps=self.eps
            # )
            layer.self_attn.o_proj = ItttLinear(
                layer.self_attn.o_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                ema_beta=config.ema_beta,
                eps=self.eps
            )
            layer.mlp.down_proj = ItttLinear(
                layer.mlp.down_proj,
                rank=config.rank,
                base_lr=config.base_lr,
                momentum_beta=config.momentum_beta,
                ema_beta=config.ema_beta,
                eps=self.eps
            )

        self.post_init()

    
    def _init_weights(self, module):
        # We don't want to re-initialize the weights, so we override this method to do nothing.
        return


    @torch.no_grad()
    def ema_update(self):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                m.ema_update()

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
        