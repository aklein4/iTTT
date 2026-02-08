import torch
import torch.nn as nn
import torch.nn.functional as F


class _ScaleGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        if isinstance(ctx.scale, dict):
            return grad_output * ctx.scale["value"], None
        return grad_output * ctx.scale, None

def scale_gradient(x, scale):
    return _ScaleGradient.apply(x, scale)


class _PrintGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        return x
    
    @staticmethod
    def backward(ctx, grad_output):

        if ctx.name is not None:
            print(f"Gradient of {ctx.name}:")
        else:
            print("Gradient:")
        print(grad_output)
        
        return grad_output, None

def print_gradient(x, name=None):
    return _PrintGradient.apply(x, name)


def unsqueeze_to_batch(x, target):
    while x.dim() < target.dim():
        x = x[None]

    return x


def expand_to_batch(x, target):
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[None]
        num_unsqueeze += 1

    x = x.expand(
        *([target.shape[i] for i in range(num_unsqueeze)] + list(og_shape))
    )

    return x


def unsqueeze_to_channel(x, target):
    while x.dim() < target.dim():
        x = x[..., None]

    return x


def expand_to_channel(x, target):
    og_shape = x.shape

    num_unsqueeze = 0
    while x.dim() < target.dim():
        x = x[..., None]
        num_unsqueeze += 1

    x = x.expand(
        *(list(og_shape) + [target.shape[i] for i in range(num_unsqueeze)])
    )

    return x


def safe_copy_state(src, dst, strict=True):

    state = {
        k: v.clone().detach() for k, v in src.state_dict().items()
    }

    dst.load_state_dict(state, strict=strict)


def shift(
    x: torch.Tensor,
    n: int,
    dim: int,
    direction: str,
    narrow: bool,
):

    zero_shape = list(x.shape)
    zero_shape[dim] = n
    z = torch.zeros(*zero_shape, device=x.device, dtype=x.dtype)

    if direction == 'right':
        if narrow:
            x = torch.narrow(x, dim, 0, x.shape[dim] - n)
        
        l = [z, x]

    elif direction == 'left':
        if narrow:
            x = torch.narrow(x, dim, n, x.shape[dim] - n)
        
        l = [x, z]

    else:
        raise ValueError(f"Invalid direction: {direction}")
    
    return torch.cat(l, dim=dim)

