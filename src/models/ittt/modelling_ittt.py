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
    def forward(ctx, x, mod):
        ctx.save_for_backward(x)
        ctx.mod = mod
        return x.clone()
    

    @staticmethod
    def backward(ctx, grad):
        og_grad = grad.clone()

        x, = ctx.saved_tensors
        mod = ctx.mod
        
        x = simple_rms_norm(x, eps=mod.eps) # [b, s, i]
        g = simple_rms_norm(grad, eps=mod.eps) # [b, s, r]

        # [b, r, i]
        update = g.transpose(-2, -1) @ x

        mod.down_update = (
            update # scale here is not really important
        )

        return og_grad, None

        
class ItttLinear(nn.Module):

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        eps: float = 1e-7,
    ):
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.eps = eps

        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_parameter("bias", None)
        
        self.weight = linear.weight

        self.down_0 = nn.Parameter(
            torch.randn(rank, self.in_features) / math.sqrt(self.in_features)
        )
        self.down_log_lr = nn.Parameter(
            torch.zeros(rank, self.in_features)
        )

        self.down_state = None
        self.down_update = None

        self.up = nn.Parameter(
            torch.randn(self.out_features, rank) / math.sqrt(self.rank)
        )

        self.svd_init()

    
    @torch.no_grad()
    def svd_init(self):

        u, s, v = torch.linalg.svd(self.weight, full_matrices=False)
        s_sqrt = torch.sqrt(s[:self.rank])

        self.down_0.copy_(v[:self.rank] * s_sqrt[:, None])
        self.up.copy_(u[:, :self.rank] * s_sqrt[None, :])

        self.weight -= self.up @ self.down_0
    

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:
        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        if self.down_state is not None:

            s = (
                torch.exp(self.down_log_lr)[None] *
                self.down_state
            )
            s = newtonschulz(s, eps=self.eps)

            s = s + self.down_0[None]

        else:
            s = self.down_0[None]

        z = torch.einsum("boi,bji->bjo", s, x)
        z = ItttFunction.apply(z, self)

        y_lora = F.linear(z, self.up)
        y_base = F.linear(x, self.weight, self.bias)

        y = y_base + y_lora

        return y


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

            layer.self_attn.o_proj = ItttLinear(
                layer.self_attn.o_proj,
                rank=config.rank,
                eps=self.eps
            )
            layer.mlp.down_proj = ItttLinear(
                layer.mlp.down_proj,
                rank=config.rank,
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

                m.down_state = None
                m.down_update = None


    @torch.no_grad()
    def update_state(self, dtype=torch.float32):
        for m in self.modules():
            if isinstance(m, ItttLinear):
                
                if m.down_update is None:
                    continue

                if m.down_state is None:
                    m.down_state = m.down_update.to(dtype)
                else:
                    m.down_state += m.down_update.to(dtype)
                
                m.down_update = None


    def forward(self, *args, **kwargs):
        return self.llama(*args, **kwargs)
        