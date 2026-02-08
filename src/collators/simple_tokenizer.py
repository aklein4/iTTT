
from transformers import AutoTokenizer

import utils.constants as constants


class SimpleTokenizer:

    def __init__(
        self,
        tokenizer_url: str,
        max_length: int,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.max_length = max_length


    def __call__(
        self,
        raw
    ):
        
        text = [x["text"] for x in raw]

        input_ids = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        input_ids = input_ids.long().to(constants.DEVICE)

        return {
            "input_ids": input_ids
        }
    