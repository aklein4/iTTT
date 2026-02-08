
from transformers.configuration_utils import PretrainedConfig


class ItttConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ItttModel`].


    Args:
        base_model (`str`, *optional*, defaults to `"HuggingFaceTB/SmolLM2-360M"`):
            The name or path of the base model to use for the iTTT initialization.
        start_layer (`int`, *optional*, defaults to 0):
            The layer from which to start applying the iTTT updates.
        rank (`int`, *optional*, defaults to 256):
            The rank of the low-rank updates.
        base_lr (`float`, *optional*, defaults to 1e-3):
            The base learning rate for iTTT updates.
        momentum_beta (`float`, *optional*, defaults to 0.75):
            The beta parameter for momentum in iTTT updates.
    ```"""

    model_type = "ittt"


    def __init__(
        self,
        base_model: str="HuggingFaceTB/SmolLM2-360M",
        start_layer: int=0,
        rank: int=256,
        base_lr: float=1e-3,
        momentum_beta: float=0.75,
        **kwargs,
    ):
        
        self.base_model = base_model

        self.start_layer = start_layer
        self.rank = rank

        self.base_lr = base_lr
        self.momentum_beta = momentum_beta

        super().__init__(**kwargs)
