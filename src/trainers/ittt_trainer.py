import torch

from tqdm import tqdm

from trainers.base_trainer import BaseTrainer
from models.ittt.modelling_ittt import ItttModel
from utils.training_utils import lm_loss
import utils.constants as constants


class ItttTrainer(BaseTrainer):
    
    model: ItttModel


    def loss(self, input_ids, logits):
        ignore_index = -100
        if self.model.llama.config.pad_token_id is not None:
            ignore_index = self.model.llama.config.pad_token_id

        return lm_loss(
            input_ids, logits,
            shift_logits=False,
            ignore_index=ignore_index,
        )


    def train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        input_ids: torch.LongTensor,
    ):
        chunks = torch.split(
            input_ids, self.config.trainer.chunk_size,
            dim=-1
        )

        ac_kwargs = {
            "device_type": str(constants.DEVICE),
            "dtype": getattr(torch, self.config.trainer.autocast_dtype),
        }

        self.model.reset_state()

        # first chunk
        with torch.autocast(**ac_kwargs):

            logits = self.model(
                chunks[0],
                logits_to_keep=slice(0, -1)
            ).logits
            loss = self.loss(chunks[0], logits)

        loss.backward()
        self.model.update_state()

        aux = {
            "lm_loss/chunk_00": loss.item(),
        }
        total_loss = loss.item()

        # remaining chunks
        for i in tqdm(range(1, len(chunks)), desc="Processing Chunks", leave=False):
            in_chunk = chunks[i-1]
            out_chunk = chunks[i]
            all_chunk = torch.cat([in_chunk, out_chunk], dim=-1)

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
            self.model.update_state()

            aux[f"lm_loss/chunk_{i:02d}"] = loss.item()
            total_loss += loss.item()

        # do things again without updating
        with torch.autocast(**ac_kwargs):

            logits = self.model(
                chunks[0],
                logits_to_keep=slice(0, -1)
            ).logits
            loss = self.loss(chunks[0], logits)

        loss.backward()

        aux["encoded_loss/chunk_00"] = loss.item()
        encoded_total_loss = loss.item()

        # remaining chunks
        for i in tqdm(range(1, len(chunks)), desc="Processing Chunks Again", leave=False):
            in_chunk = chunks[i-1]
            out_chunk = chunks[i]
            all_chunk = torch.cat([in_chunk, out_chunk], dim=-1)

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

            aux[f"encoded_loss/chunk_{i:02d}"] = loss.item()
            encoded_total_loss += loss.item()
        
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

        # do this again just in case
        self.model.reset_state()

        # finalize outputs
        final_loss = total_loss / len(chunks)
        aux["num_atoms"] = input_ids.numel()

        decades = {}
        encoded_decades = {}
        for key, value in aux.items():

            if "lm_loss/chunk_" in key:
                if key.endswith("00"):
                    continue

                decade = key.split("_")[-1][0]

                if decade not in decades:
                    decades[decade] = []
                decades[decade].append(value)

            elif "encoded_loss/chunk_" in key:
                if key.endswith("00"):
                    continue

                decade = key.split("_")[-1][0]

                if decade not in encoded_decades:
                    encoded_decades[decade] = []
                encoded_decades[decade].append(value)

        for decade, values in decades.items():
            aux[f"grouped_lm_loss/decade_{decade}"] = sum(values) / len(values)
        for decade, values in encoded_decades.items():
            aux[f"grouped_encoded_loss/decade_{decade}"] = sum(values) / len(values)

        aux["encoded_total_loss"] = encoded_total_loss

        return final_loss, aux
    