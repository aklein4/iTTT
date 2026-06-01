import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN

from models.ittt.configuration_ittt import ItttConfig
from models.reference_llama.modelling_llama import LlamaForCausalLM, LlamaDecoderLayer
from utils.training_utils import lm_loss
from tqdm import tqdm


@torch.compile(fullgraph=True, mode="reduce-overhead")
def newtonschulz(
    x: torch.FloatTensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.FloatTensor:
    # https://kellerjordan.github.io/posts/muon/
    
    a, b, c = (3.4445, -4.7750, 2.0315)

    y = x / (
        x.reshape(*x.shape[:-2], -1).norm(dim=-1)[:, None, None] + eps
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


class FastWeightFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        buffer: torch.FloatTensor,
    ) -> torch.FloatTensor:
        ctx.save_for_backward(x)
        ctx.dtype = buffer.dtype
        return y.clone()


    @staticmethod
    def backward(
        ctx,
        grad: torch.FloatTensor
    ) -> tuple[None, torch.FloatTensor, None]:

        x, = ctx.saved_tensors
        dtype: torch.dtype = ctx.dtype

        # [b, r, i]
        update = (
            grad.to(dtype).transpose(-2, -1) @
            x.to(dtype)
        )
    
        return None, grad, update

        
class FastWeight(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: ItttConfig,
    ):
        super().__init__()

        # save config
        self.in_features = in_features
        self.out_features = out_features

        self.base_lr = config.base_lr
        self.momentum_beta = config.momentum_beta

        self.eps = config.rms_norm_eps
        self.scalar_scaler = math.sqrt(self.in_features)

        self.momentum_dtype = getattr(torch, config.momentum_dtype)
        self.state_dtype = getattr(torch, config.state_dtype)
        
        # ittt params
        self.log_lr = nn.Parameter(
            torch.zeros(self.out_features, self.in_features)
        )

        # ephemeral state
        self.state: nn.Buffer
        self.momentum: nn.Buffer
        self.prev_whitened: nn.Buffer
                    

    def get_lr(self):
        return (
            self.base_lr *
            torch.exp(self.log_lr * self.scalar_scaler) /
            math.sqrt(self.in_features)
        )


    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:

        assert x.ndim == 3, "x must be 3D (batch, seq_len, dim)"

        s = self.get_lr()[None] * self.state.detach()

        y = torch.einsum("boi,bli->blo", s, x)
        y = FastWeightFunction.apply(x, y, self.momentum)

        return y


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):

        state = torch.zeros(
            bs, self.out_features, self.in_features,
            device=device, dtype=self.state_dtype,
        )
        momentum = torch.zeros_like(
            state, dtype=self.momentum_dtype
        )
        prev_whitened = torch.zeros_like(
            state, dtype=self.momentum_dtype
        )

        self.register_buffer("state", state, persistent=False)
        self.register_buffer("momentum", momentum, persistent=False)
        self.register_buffer("prev_whitened", prev_whitened, persistent=False)
        
        self.state.requires_grad_(False)

        self.momentum.requires_grad_(True)
        self.momentum.grad = torch.zeros_like(self.momentum)

        self.prev_whitened.requires_grad_(False)


    @torch.no_grad()
    def empty_state(self):

        self.state.zero_()

        self.momentum.zero_()
        self.momentum.grad.zero_()

        self.prev_whitened.zero_()

    
    @torch.no_grad()
    def update_state(self):
        
        update = self.momentum.grad

        new_momentum = torch.lerp(
            self.momentum,
            update,
            1 - self.momentum_beta
        )
        new_whitened = newtonschulz(
            new_momentum, eps=self.eps
        )

        # approximates newton_schulz as a linear function:
        # f(ax) = af(x), f(x+y) = f(x) + f(y)
        delta = (
            (new_whitened - self.prev_whitened * self.momentum_beta) /
            (1 - self.momentum_beta)
        )

        # scale delta to element-wise scale of 1
        delta = delta * math.sqrt(max(self.in_features, self.out_features))
        
        self.state.add_(-delta.to(self.state_dtype))
        
        self.momentum.copy_(new_momentum.detach())
        self.momentum.grad.zero_()

        self.prev_whitened.copy_(new_whitened.detach())


class FastWeightMLP(nn.Module):
    def __init__(
        self,
        config: ItttConfig,
    ):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fast_weight_size = config.fast_weight_size
        
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.gate_fast = nn.Linear(self.hidden_size, self.fast_weight_size, bias=False)
        self.up_fast = nn.Linear(self.hidden_size, self.fast_weight_size, bias=False)
        self.fast = FastWeight(
            self.fast_weight_size, self.fast_weight_size, config
        )
        self.down_fast = nn.Linear(self.fast_weight_size, self.hidden_size, bias=False)


    def forward(self, x):
    
        h_base = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        y_base = self.down_proj(h_base)

        h_fast = self.act_fn(self.gate_fast(x)) * self.up_fast(x)
        h_fast = self.fast(h_fast)
        y_fast = self.down_fast(h_fast)

        return y_base + y_fast


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

        for k, v in self.llama.config.to_dict().items():
            setattr(self.config, k, v)
        
        self.disable_fast_weights = config.disable_fast_weights
        if self.disable_fast_weights:
            return

        for layer in self.llama.model.layers:
            layer: LlamaDecoderLayer

            layer.mlp = FastWeightMLP(config)
            mlp = layer.mlp

            mlp.gate_fast.weight.data.copy_(
                mlp.gate_proj.weight.data[:mlp.fast_weight_size, :]
            )
            mlp.up_fast.weight.data.copy_(
                mlp.up_proj.weight.data[:mlp.fast_weight_size, :]
            )
            mlp.down_fast.weight.data.copy_(
                mlp.down_proj.weight.data[:, :mlp.fast_weight_size]
            )

        self.post_init()

    
    def _init_weights(self, module):
        # We don't want to re-initialize the weights, so we override this method to do nothing.
        return

    
    def forward(self, *args, **kwargs):
        return self.llama(*args, **kwargs)


    @torch.no_grad()
    def init_state(self, bs: int, device: torch.device):
        for m in self.modules():
            if isinstance(m, FastWeight):
                m.init_state(bs, device)


    @torch.no_grad()
    def empty_state(self):
        for m in self.modules():
            if isinstance(m, FastWeight):
                m.empty_state()
    

    @torch.no_grad()
    def update_state(self):

        to_update = []
        for name, mod in self.llama.model.layers[0].named_modules():
            if isinstance(mod, FastWeight):
                to_update.append(name)

        for name in to_update:
            self.update_state_named(name)


    @torch.no_grad()
    def update_state_named(self, name: str):
        # updates named module across all layers in parallel
        
        try:
            ref: FastWeight = self.llama.model.layers[0].get_submodule(name)
        except:
            ref: FastWeight = self.llama.model.layers[0]._orig_mod.get_submodule(name)

        updates = []
        momentums = []
        prev_whiteneds = []
        for layer in self.llama.model.layers:
            layer: LlamaDecoderLayer

            try:
                m: FastWeight = layer.get_submodule(name)
            except:
                m: FastWeight = layer._orig_mod.get_submodule(name)

            updates.append(m.momentum.grad)
            momentums.append(m.momentum)
            prev_whiteneds.append(m.prev_whitened)
        
        updates = torch.stack(updates, dim=1)
        momentums = torch.stack(momentums, dim=1)
        prev_whiteneds = torch.stack(prev_whiteneds, dim=1)

        new_momentums = torch.lerp(
            momentums,
            updates,
            1 - ref.momentum_beta
        )
        new_whiteneds = newtonschulz(
            new_momentums, eps=ref.eps
        )

        deltas = (
            (new_whiteneds - prev_whiteneds * ref.momentum_beta) /
            (1 - ref.momentum_beta)
        )

        deltas = deltas * math.sqrt(max(ref.in_features, ref.out_features))

        state_deltas = -deltas.to(ref.state_dtype)

        for i, layer in enumerate(self.llama.model.layers):
            layer: LlamaDecoderLayer

            try:
                m: FastWeight = layer.get_submodule(name)
            except:
                m: FastWeight = layer._orig_mod.get_submodule(name)

            m.state.add_(state_deltas[:, i].detach())

            m.momentum.copy_(new_momentums[:, i].detach())
            m.momentum.grad.zero_()

            m.prev_whitened.copy_(new_whiteneds[:, i].detach())


    def get_logits(self, *args, **kwargs):
        return self.compute_logits(*args, **kwargs)

    def compute_logits(
        self,
        input_ids: torch.LongTensor,
        output_ids: torch.LongTensor | None = None,
        chunk_size: int | None = None,
        cpu_logits: bool = False,
        verbose: bool = False,
        add_bos: bool = False,
    ):
        # TODO: fix this

        if output_ids is not None:
            input_ids = torch.cat([input_ids, output_ids], dim=-1)
        
        if chunk_size is None:
            chunk_size = self.config.chunk_size

        chunks = torch.split(input_ids, chunk_size, dim=-1)

        ac_kwargs = {
            "device_type": str(input_ids.device),
            "dtype": torch.bfloat16,
        }

        self.init_state(input_ids.shape[0], input_ids.device)

        all_logits = []

        # first chunk
        with torch.enable_grad():
            with torch.autocast(**ac_kwargs):

                logits = self(
                    chunks[0],
                    logits_to_keep=slice(0, -1)
                )[0]
    
                loss = lm_loss(
                    logits, chunks[0],
                    shift_logits=False,
                    ignore_index=self.config.pad_token_id,
                )

                if cpu_logits:
                    logits = logits.cpu()
                all_logits.append(logits.detach())

            loss.backward()

        # remaining chunks
        for i in tqdm(range(1, len(chunks)), desc="Processing Chunks", leave=False, disable=(not verbose)):
            
            first_chunk = chunks[i-1]
            second_chunk = chunks[i]
            
            if i > 1 and add_bos:
                first_chunk = torch.cat(
                    [
                    torch.full_like(first_chunk[:, :1], self.config.bos_token_id),
                    first_chunk
                    ],
                    dim=-1
                )

            all_chunk = torch.cat([first_chunk, second_chunk], dim=-1)

            self.update_state()

            with torch.enable_grad():
                with torch.autocast(**ac_kwargs):

                    logits = self(
                        all_chunk,
                        logits_to_keep=slice(first_chunk.shape[-1]-1, -1)
                    ).logits

                    loss = lm_loss(
                        all_chunk[:, first_chunk.shape[-1]:],
                        logits,
                        shift_logits=False,
                        shift_labels=False,
                        ignore_index=self.config.pad_token_id,
                    )

                    if cpu_logits:
                        logits = logits.cpu()
                    all_logits.append(logits.detach())

                loss.backward()

        self.zero_grad(True)
        self.empty_state()
            
        logits = torch.cat(all_logits, dim=1).detach()

        if output_ids is not None:
            logits = logits[:, -output_ids.shape[-1]:, :]
        
        return logits
