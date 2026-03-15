import torch

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import datasets

from models.reference_llama.modelling_llama import LlamaForCausalLM
from models.ittt.modelling_ittt import ItttModel
from models.ittt.configuration_ittt import ItttConfig
from collators.tokenize import TokenizeCollator
import utils.constants as constants
from utils.training_utils import lm_loss


MODEL_URL = 'meta-llama/Llama-3.2-1B'
DATA_URL = "Geralt-Targaryen/books3"

ITTT_KWARGS = {
    "start_layer": 0,
    "rank": 1024,
    "base_lr": 1e-3,
    "momentum_beta": 0.90,  
}
CHUNK_SIZE = 1024

NUM_EXAMPLES = 1024
BS = 1

SEQUENCE_LENGTH = 1024 * 64

DO_LM = True
DO_SLIDING = True

def main():
    
    print("Loading data...")
    data = datasets.load_dataset(DATA_URL, split='train', streaming=True)
    collator = TokenizeCollator(MODEL_URL, SEQUENCE_LENGTH)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=BS,
        collate_fn=collator,
    )

    print("Loading model...")
    lm_model = LlamaForCausalLM.from_pretrained(
        MODEL_URL,
        dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to(constants.DEVICE)
    # lm_model.lm_head.weight = lm_model.lm_head.weight.cpu()

    print("Loading ItttModel...")
    itt_config = ItttConfig(
        base_model=MODEL_URL,
        _attn_implementation="flash_attention_2",
        **ITTT_KWARGS
    )
    ittt_model = ItttModel(itt_config).to(constants.DEVICE)
    # ittt_model.lm_head.weight = ittt_model.lm_head.weight.cpu()
    ittt_model.gradient_checkpointing_enable()

    print("Running models...")
    lm_losses = []
    ittt_losses = []
    sliding_losses = []
    for i, batch in tqdm(enumerate(loader), total=NUM_EXAMPLES // BS):

        pad_token_id = collator.tokenizer.pad_token_id

        input_ids = batch['input_ids'].to(constants.DEVICE)
        inputs_for_model = torch.where(
            input_ids == pad_token_id,
            torch.zeros_like(input_ids),
            input_ids,
        )

        logits = ittt_model.compute_logits(
            inputs_for_model, CHUNK_SIZE,
            labels=input_ids, ignore_index=pad_token_id,
            verbose=True,
        )
        loss = lm_loss(
            input_ids, logits,
            shift_logits=False,
            ignore_index=pad_token_id,
            reduction='none',
        )
        ittt_losses.append(loss.cpu())
        del logits

        if DO_SLIDING:
            logits = ittt_model.compute_logits(
                inputs_for_model, CHUNK_SIZE,
                labels=input_ids, ignore_index=pad_token_id,
                verbose=True, do_update=False,
            )
            loss = lm_loss(
                input_ids, logits,
                shift_logits=False,
                ignore_index=pad_token_id,
                reduction='none',
            )
            sliding_losses.append(loss.cpu())
            del logits

        if DO_LM:
            with torch.no_grad():
                with torch.autocast(str(constants.DEVICE), torch.bfloat16, enabled=False):

                    logits = lm_model.forward(
                        inputs_for_model,
                        logits_to_keep=slice(0, -1)
                    ).logits

                loss = lm_loss(
                    input_ids, logits,
                    shift_logits=False,
                    ignore_index=pad_token_id,
                    reduction='none',
                )
            lm_losses.append(loss.cpu())
            del logits

        if len(ittt_losses) >= NUM_EXAMPLES // BS:
            break
    
    if len(lm_losses) > 0:
        lm_losses = torch.cat(lm_losses, dim=0)
        torch.save(
            lm_losses,
            os.path.join(constants.LOCAL_DATA_PATH, "lm_losses.pt")
        )
    
    ittt_losses = torch.cat(ittt_losses, dim=0)
    torch.save(
        ittt_losses,
        os.path.join(constants.LOCAL_DATA_PATH, "ittt_losses.pt")
    )

    if len(sliding_losses) > 0:
        sliding_losses = torch.cat(sliding_losses, dim=0)
        torch.save(
            sliding_losses,
            os.path.join(constants.LOCAL_DATA_PATH, "sliding_losses.pt")
        )


def analyze_results():

    lm_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "lm_losses.pt")).float().numpy()
    ittt_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "ittt_losses.pt")).float().numpy()
    
    sliding_losses = torch.load(os.path.join(constants.LOCAL_DATA_PATH, "sliding_losses.pt"))
    if isinstance(sliding_losses, list):
        sliding_losses = torch.cat(sliding_losses, dim=0)
    sliding_losses = sliding_losses.float().numpy()

    df = pd.DataFrame({
        "lm_loss": lm_losses.mean(0),
        "ittt_loss": ittt_losses.mean(0),
        "sliding_loss": sliding_losses.mean(0),
    })

    for col in df.columns:

        x = np.arange(len(df[col]))
        y_running = df[col].rolling(window=250)
        
        plt.plot(x, y_running.mean(), label=col)
    
    plt.legend()
    plt.grid()
    plt.savefig("comparison_to_dense.png")


if __name__ == "__main__":
    main()
    analyze_results()
