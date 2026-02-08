import torch

import hydra
import omegaconf

import datasets

import utils.constants as constants
from utils.import_utils import import_model, import_trainer
from utils.torch_utils import safe_copy_state


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: omegaconf.DictConfig):

    print("Loading model...")
    if config.model.pretrained is not None:
        
        model_config = import_model(config.model.config.type)(
            **config.model.config.kwargs
        )
        model = import_model(config.model.type)(model_config)

        state_model = import_model(config.model.type).from_pretrained(
            config.model.pretrained,
        )

        safe_copy_state(
            state_model, model, strict=False
        )

        for k in state_model.state_dict().keys():
            if k not in model.state_dict().keys():
                print(f"Warning: parameter {k} not used from checkpoint!")
        for k in model.state_dict().keys():
            if k not in state_model.state_dict().keys():
                print(f"Warning: parameter {k} not found in checkpoint!")
            
        model_config = model.config.to_dict()
        state_config = state_model.config.to_dict()
        for k in model_config.keys():
            if model_config[k] != state_config[k]:
                print(
                    f"Warning: config key {k} has different value in checkpoint ({state_config[k]}) and model ({model_config[k]})!"
                )

    else:
        
        model_config = import_model(config.model.config.type)(
            **config.model.config.kwargs
        )
        model = import_model(config.model.type)(model_config)

    model = model.to(constants.DEVICE, torch.float32)
    
    print("Loading dataset...")
    dataset = datasets.load_dataset(
        config.dataset.name,
        **config.dataset.kwargs,
    )

    print("Loading trainer...")
    trainer = import_trainer(config.trainer.type)(
        config,
        model,
        dataset,
    )

    print("Entering trainer...")
    trainer.train()


if __name__ == '__main__':
    main()
    