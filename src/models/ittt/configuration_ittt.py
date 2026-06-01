
from transformers.configuration_utils import PretrainedConfig


class ItttConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ItttModel`].


    Args:
        base_model (`str`, *optional*, defaults to `"HuggingFaceTB/SmolLM2-360M"`):
            The name or path of the base model to use for the iTTT initialization.
        fast_weight_size (`int`, *optional*, defaults to 256):
            The rank of the low-rank updates.
        base_lr (`float`, *optional*, defaults to 1e-3):
            The base learning rate for iTTT updates.
        momentum_beta (`float`, *optional*, defaults to 0.90):
            The beta parameter for momentum in iTTT updates.
        momentum_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            The data type for momentum buffers in iTTT updates.
        state_dtype (`str`, *optional*, defaults to `"float32"`):
            The data type for the fast weight states in iTTT updates.
        disable_fast_weights (`bool`, *optional*, defaults to `False`):
            Whether to disable the use of fast weights in the model.
    ```"""

    model_type = "ittt"


    def __init__(
        self,
        base_model: str="HuggingFaceTB/SmolLM2-360M",
        fast_weight_size: int=256,
        base_lr: float=1e-3,
        momentum_beta: float=0.90,
        momentum_dtype: str="bfloat16",
        state_dtype: str="float32",
        disable_fast_weights: bool=False,
        **kwargs,
    ):
        
        self.base_model = base_model

        self.fast_weight_size = fast_weight_size

        self.base_lr = base_lr
        self.momentum_beta = momentum_beta

        self.momentum_dtype = momentum_dtype
        self.state_dtype = state_dtype

        disable_fast_weights = disable_fast_weights

        super().__init__(**kwargs)
