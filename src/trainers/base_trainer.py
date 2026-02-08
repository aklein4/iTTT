import torch

import os
from tqdm import tqdm

import wandb
import huggingface_hub as hf
from transformers import PreTrainedModel

from omegaconf import DictConfig, OmegaConf

from utils.import_utils import import_optimizer, import_collator
import utils.constants as constants
from utils.logging_utils import LogSection


class BaseTrainer:

    model: PreTrainedModel


    def __init__(
        self,
        config: DictConfig,
        model,
        dataset,
    ):
        self.config = config
        self.model = model
        self.dataset = dataset

        self.checkpoint_path = None
        self.repo_name = None
        if not self.config.debug:

            os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)

            # create the huggingface save repo
            full_name = f"{self.config.project}_{self.config.name}"
            
            self.checkpoint_path = os.path.join(
                constants.LOCAL_DATA_PATH,
                full_name
            )

            if self.config.trainer.push_to_hub:
                self.repo_name = f"{constants.HF_ID}/{full_name}"
                hf.create_repo(
                    self.repo_name, private=False, exist_ok=True, token=constants.HF_TOKEN
                )

            # create the wandb project
            wandb.init(
                project=self.config.project,
                name=self.config.name,
                notes=self.config.notes,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        self.post_init()

    
    def post_init(self):
        return


    def safe_log(self, **kwargs):
        if self.config.debug:
            return

        d = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.item()
            else:
                d[k] = v

        wandb.log(d)
        

    @torch.no_grad()
    def save_checkpoint(
        self,
        step
    ):
        if self.config.debug:
            return

        with LogSection(f"step {step} checkpoint saving", newline=True):

            path = os.path.join(
                self.checkpoint_path,
                f"step_{step:012d}"
            )
            self.model.save_pretrained(
                path,
                push_to_hub=False,
            )

            if self.config.trainer.push_to_hub:
                hf.upload_folder(
                    repo_id=self.repo_name,
                    folder_path=path,
                    path_in_repo=f"step_{step:012d}",
                    token=constants.HF_TOKEN,
                )

        
    def manual_checkpoint(self, step):
        if step % self.config.trainer.manual_checkpoint_interval != 0:
            return False

        file = os.path.join(
            constants.LOCAL_DATA_PATH,
            "save_now.txt"
        )
        if not os.path.exists(file):
            with open(file, "w") as f:
                f.write("0")
            return False
        
        with open(file, "r") as f:
            check = f.read().strip() == "1"

        if check:
            with open(file, "w") as f:
                f.write("0")

        return check


    def train(self):

        # init model
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.train()
        if self.config.trainer.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # init optimizer
        optimizer = import_optimizer(self.config.optimizer.type)(
           self.model.parameters(),
            **self.config.optimizer.kwargs
        )

        # init data loader
        collator = import_collator(self.config.collator.type)(
            **self.config.collator.kwargs
        )
        collator.skip_steps = self.config.trainer.skip_steps
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=collator,
            shuffle=False,
            drop_last=True,
        )

        # init loop vars
        step = 0
        epoch = 0
        pbar = tqdm(desc=f"Training {self.config.project}/{self.config.name}")
        pbar.update(0)

        # train indefinitely
        while True:
            for batch in loader:

                # skip steps if needed
                if self.config.trainer.skip_steps is not None:
                    if step < self.config.trainer.skip_steps:
                        step += 1
                        pbar.update(1)
                        continue

                loss, aux = self.train_step(
                    step,
                    optimizer,
                    **batch
                )

                # update pbar
                pbar.update(1)
                pbar.set_postfix(
                    loss=loss.item() if isinstance(loss, torch.Tensor) else loss
                )

                # log to wandb
                triggered_save = aux.pop("save_checkpoint", False)
                self.safe_log(
                    loss=loss,
                    epoch=epoch,
                    **aux,
                    **optimizer.get_log_info()
                )

                # increment step before checkpointing, so save path is "after N steps of training"
                step += 1

                # save checkpoint
                if triggered_save or step % self.config.trainer.checkpoint_interval == 0 or self.manual_checkpoint(step):
                    self.save_checkpoint(step)

            epoch += 1


    def train_step(
        self,
        step,
        optimizer,
        **batch
    ):
        
        with torch.autocast(
            device_type=str(constants.DEVICE),
            dtype=getattr(torch, self.config.trainer.autocast_dtype),
        ):
            loss, aux = self.train_forward(
                step,
                **batch
            )
        
        loss.backward()

        if step == 0:
            self.debug_gradients()

        if self.config.trainer.grad_norm_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.trainer.grad_norm_clip
            )
            aux["grad_norm"] = grad_norm

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        return loss, aux


    def train_forward(
        self,
        step,
        **batch
    ):
        raise NotImplementedError("train_forward must be implemented in child class!")
    

    def debug_gradients(self):

        with open(os.path.join(constants.LOCAL_DATA_PATH, "gradients.txt"), "w") as f:

            f.write("\n === FINITE GRADIENT === \n\n")
            for n, p in self.model.named_parameters():
                if p.grad is not None and torch.all(p.grad.isfinite()):
                    f.write(f"{n}\n")

            f.write("\n === NO GRADIENT === \n\n")
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    f.write(f"{n}\n")

            f.write("\n === NONFINITE GRADIENT === \n\n")
            for n, p in self.model.named_parameters():
                if p.grad is not None and torch.any(~p.grad.isfinite()):
                    f.write(f"{n}\n")
