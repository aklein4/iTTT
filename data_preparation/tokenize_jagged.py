
import numpy as np
from functools import partial

import datasets

from transformers import AutoTokenizer


TOKENIZER_URL = "HuggingFaceTB/SmolLM2-135M" # "TinyLlama/TinyLlama_v1.1"

DATASET = (
    "allenai/dolma3_longmino_mix-50B-1025",
)

BS = 512

SAVE_URL = "aklein4/longmino-50B-SmolLM2"


def tokenize_batch(
    example,
    tokenizer: AutoTokenizer=None,
):

    input_ids = tokenizer(
        example["text"],
        add_special_tokens=True,
        return_tensors="np",
        padding=False,
        truncation=False,
    ).input_ids.astype(np.uint16)

    print([len(x) for x in input_ids])
    exit()

    keep = input_ids[:, -1] != tokenizer.pad_token_id

    out = {
        "input_ids": [x for x in input_ids],
        "keep": [x for x in keep],
    }

    return out


def tokenize_dataset(
    data_url: tuple[str, str],
    tokenizer: AutoTokenizer,
):
    
    data = datasets.load_dataset(*data_url, streaming=False)

    data = data.map(
        partial(tokenize_batch, tokenizer=tokenizer),
        remove_columns=data.column_names,
        batched=True,
        batch_size=BS,
    )

    return data


def main():

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_URL)

    data = tokenize_dataset(DATASET, tokenizer)

    data = data.shuffle(seed=42)

    data.push_to_hub(
        SAVE_URL, 
        private=False,
    )


if __name__ == "__main__":
    main()
