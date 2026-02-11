
import numpy as np
from functools import partial

import datasets

from transformers import AutoTokenizer


TOKENIZER_URL = "HuggingFaceTB/SmolLM2-135M" # "TinyLlama/TinyLlama_v1.1"

DATASET = (
    "Geralt-Targaryen/books3",
)

BS = 512

SAVE_URL = "aklein4/books3-SmolLM2"


def tokenize_batch(
    example,
    tokenizer: AutoTokenizer=None,
):

    input_ids = tokenizer(
        example["text"],
        add_special_tokens=True,
        padding=False,
        truncation=False,
    ).input_ids

    lengths = [
        len(x) for x in input_ids
    ]
    input_ids = [
        np.array(x).astype(np.uint16) for x in input_ids
    ]

    out = {
        "input_ids": [x for x in input_ids],
        "num_tokens": [x for x in lengths],
    }

    return out


def tokenize_dataset(
    data_url: tuple[str, str],
    tokenizer: AutoTokenizer,
):
    
    data = datasets.load_dataset(*data_url, streaming=False)

    data = data.map(
        partial(tokenize_batch, tokenizer=tokenizer),
        remove_columns=["text"],
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
