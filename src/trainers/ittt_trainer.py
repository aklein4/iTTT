import torch

from tqdm import tqdm

from trainers.base_trainer import BaseTrainer
from models.ittt.modelling_ittt import ItttModel
from utils.training_utils import lm_loss
import utils.constants as constants


class ItttTrainer(BaseTrainer):
    
    model: ItttModel

    half_loss: bool = False


    def loss(self, input_ids, logits):
        ignore_index = -100
        if self.model.llama.config.pad_token_id is not None:
            ignore_index = self.model.llama.config.pad_token_id

        if self.half_loss:
            input_ids = torch.chunk(input_ids, 2, dim=0)[1]
            logits = torch.chunk(logits, 2, dim=0)[1]

        return lm_loss(
            input_ids, logits,
            shift_logits=False,
            ignore_index=ignore_index,
        )


    def process_chunks(
        self,
        chunks,
        ac_kwargs,
        desc="Processing Chunks"
    ):
        
        # first chunk
        with torch.autocast(**ac_kwargs):

            logits = self.model(
                chunks[0],
                logits_to_keep=slice(0, -1)
            ).logits
            loss = self.loss(chunks[0], logits)

        loss.backward()

        aux = {
            "lm_loss/chunk_00": loss.item(),
        }
        total_loss = loss.item()

        # remaining chunks
        for i in tqdm(range(1, len(chunks)), desc=desc, leave=False):
            in_chunk = chunks[i-1]
            out_chunk = chunks[i]
            all_chunk = torch.cat([in_chunk, out_chunk], dim=-1)

            self.model.update_state()

            with torch.autocast(**ac_kwargs):

                logits = self.model(
                    all_chunk,
                    logits_to_keep=slice(in_chunk.shape[-1]-1, -1)
                ).logits
                loss = self.loss(
                    all_chunk[:, in_chunk.shape[-1]-1:],
                    logits
                )

            loss.backward()

            aux[f"lm_loss/chunk_{i:02d}"] = loss.item()
            total_loss += loss.item()
        
        return total_loss, aux


    def train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        input_ids: torch.LongTensor,
    ):
        
        # get the arguments
        chunks = torch.split(
            input_ids, self.config.trainer.chunk_size,
            dim=-1
        )
        ac_kwargs = {
            "device_type": str(constants.DEVICE),
            "dtype": getattr(torch, self.config.trainer.autocast_dtype),
        }

        # fill last_state_grad
        self.model.reset_state()
        # self.process_chunks(chunks, ac_kwargs, desc="Filling State Grads")
        total_loss, aux = self.process_chunks(chunks, ac_kwargs, desc="Filling State Grads")
        self.model.finalize_state()
        self.model.zero_grad()

        # process again with the actual gradients
        double_chunks = torch.split(
            torch.cat([input_ids, input_ids], dim=0),
            self.config.trainer.chunk_size,
            dim=-1
        )
        self.half_loss = True
        total_loss, aux = self.process_chunks(double_chunks, ac_kwargs)
        self.half_loss = False
        aux["relative_grad_error"] = self.model.relative_grad_error()
        self.model.reset_state()
        
        # regular optimization step
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

        # finalize outputs
        final_loss = total_loss / len(chunks)
        aux["num_atoms"] = input_ids.numel()

        decades = {}
        for key, value in aux.items():

            if "chunk_" in key:
                if key.endswith("00"):
                    continue

                decade = key.split("_")[-1][0]

                if decade not in decades:
                    decades[decade] = []
                decades[decade].append(value)

        for decade, values in decades.items():
            aux[f"grouped_lm_loss/decade_{decade}"] = sum(values) / len(values)

        return final_loss, aux
    