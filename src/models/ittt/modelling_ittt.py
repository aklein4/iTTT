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
    def forward(ctx, x, y, mod, prefix):
        ctx.save_for_backward(x)
        ctx.mod = mod
        ctx.prefix = prefix
        return y.clone()
    

    @staticmethod
    def backward(ctx, g):
        og_grad = g.clone()

        x, = ctx.saved_tensors
        mod = ctx.mod

        x = x.to(mod.momentum_dtype)
        g = g.to(mod.momentum_dtype)

        x = simple_rms_norm(x, eps=mod.eps) # [b, s, i]
        g = F.normalize(g, dim=-2, eps=mod.eps) * math.sqrt(x.shape[-2])  # [b, s, r]

        # [b, r, i]
        update = (
            g.transpose(-2, -1) @ x
        ) / math.sqrt(x.shape[-2]) # approx 1 std

        momentum_name = f"{ctx.prefix}_momentum"
        momentum = getattr(mod, momentum_name)

        if momentum is None:
            momentum = (
                (1 - mod.momentum_beta) * update
            )
        else:
            momentum = (
                mod.momentum_beta * momentum +
                (1 - mod.momentum_beta) * update
            )

        setattr(mod, momentum_name, momentum)

        return None, og_grad, None

        
class ItttLinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        base_lr: float,
        momentum_beta: float,
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
        self.register_buffer(
            "down_lr_scale", torch.ones(1), persistent=True
        )
        self.down_lr_scale: nn.Buffer
        self.down_log_lr = nn.Parameter(
            torch.zeros(rank, self.in_features)
        )
        self.down_base = nn.Parameter(
            torch.randn(rank, self.in_features) / math.sqrt(self.in_features)
        )

        self.register_buffer(
            "up_lr_scale", torch.ones(1), persistent=True
        )
        self.up_lr_scale: nn.Buffer
        self.up_log_lr = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )
        self.up_base = nn.Parameter(
            torch.randn(self.out_features, rank) / math.sqrt(rank)
        )

        # ephemeral state
        self.down_state = None
        self.down_momentum = None

        self.up_state = None
        self.up_momentum = None

        self.svd_init()

    
    @torch.no_grad()
    def svd_init(self):

        u, s, v = torch.linalg.svd(self.weight, full_matrices=False)
        t = torch.sqrt(s)

        self.down_base.copy_(
            v[:self.rank, :],
            t[:self.rank, None]
        )
        self.down_lr_scale.copy_(
            (self.down_base.norm() / math.sqrt(self.in_features * self.rank)) # around 1
            / (1 / math.sqrt(self.in_features))
        )

        self.up_base.copy_(
            u[:, :self.rank],
            t[None, :self.rank]
        )
        self.up_lr_scale.copy_(
            (self.up_base.norm() / math.sqrt(self.out_features * self.rank))
            / (1 / math.sqrt(self.out_features))
        )
    

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"
        assert (self.down_state is None) == (self.up_state is None), "both up and down state must be None or not None"

        if self.down_state is not None:

            s_down = (
                self.base_lr *
                self.down_lr_scale *
                torch.exp(self.down_log_lr * self.scalar_scaler)
            )[None] * self.down_state

            s_up = (
                self.base_lr *
                self.up_lr_scale *
                torch.exp(self.up_log_lr * self.scalar_scaler)
            )[None] * self.up_state

        else:
            s_down = torch.zeros_like(self.down_log_lr)[None]
            s_up = torch.zeros_like(self.up_log_lr)[NOne]

        z = torch.einsum("boi,bji->bjo", s_down, x)
        z = ItttFunction.apply(x, z, self, "down")

        y_lora = torch.einsum("boi,bji->bjo", s_up, z)
        y_lora = ItttFunction.apply(z, y_lora, self, "up")

        y_base = F.linear(x, self.weight, self.bias)

        y = y_base + y_lora

        return y

    
    @torch.no_grad()
    def reset_state(self):
        self.down_state = None
        self.down_momentum = None

        self.up_state = None
        self.up_momentum = None

    
    @torch.no_grad()
    def update_state(self):
        assert (self.down_state is None) == (self.up_state is None), "both up and down state must be None or not None"

        if self.down_momentum is None:
            return
                
        # we don't worry about adam-like biased momentum because newton-schulz normalizes anyway
        down_delta = -newtonschulz(
            self.down_momentum,
            eps=self.eps
        ).to(self.state_dtype)

        if self.down_state is None:
            self.down_state = down_delta
        else:
            self.down_state += down_delta

        up_delta = -newtonschulz(
            self.up_momentum,
            eps=self.eps
        ).to(self.state_dtype)

        if self.up_state is None:
            self.up_state = up_delta
        else:
            self.up_state += up_delta


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
        