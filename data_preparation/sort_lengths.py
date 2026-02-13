
import numpy as np
from functools import partial

import datasets


INPUT_URL = "aklein4/books3-SmolLM2"
OUTPUT_URL = "aklein4/books3-SmolLM2-sorted"


LENGTHS = [
    1024 * 1024,
    1024 * 512,
    1024 * 256,
    1024 * 128,
    1024 * 64,
    1024 * 32,
]


def truncate_batch(
    example,
    length: int=None,
):
    input_ids = [x[:length] for x in example["input_ids"]]
    num_tokens = [len(x) for x in input_ids]

    return {
        "input_ids": input_ids,
        "num_tokens": num_tokens,
    }


def longer_than(
    example,
    length: int=None,
):
    return [
        x >= length
        for x in example["num_tokens"]
    ]


def shorter_than(
    example,
    length: int=None,
):
    return [
        x < length
        for x in example["num_tokens"]
    ]


def main():

    data = datasets.load_dataset(INPUT_URL, streaming=False, split="train")

    for length in LENGTHS:

        subset = data.filter(
            partial(longer_than, length=length),
            batched=True,
            batch_size=32,
        )
        subset = subset.map(
            partial(truncate_batch, length=length),
            batched=True,
            batch_size=32,
        )

        subset.push_to_hub(
            OUTPUT_URL,
            config_name=f"length_{length}",
        )

        data = data.filter(
            partial(shorter_than, length=length),
            batched=True,
            batch_size=32,
        )


def length_histogram():

    data = datasets.load_dataset(INPUT_URL, streaming=False, split="train")

    lengths = np.array(data["num_tokens"])
    lengths = np.clip(lengths, 0, 1024 * 1024 * 1.1)

    import matplotlib.pyplot as plt

    plt.hist(lengths, bins=100)
    plt.grid()
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.title("Histogram of Lengths")
    plt.savefig("length_histogram.png")


if __name__ == "__main__":
    main()
    # length_histogram()
