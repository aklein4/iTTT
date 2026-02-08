import torch

from transformers import AutoTokenizer

import utils.constants as constants


class PadMinimizer:

    def __init__(
        self,
        tokenizer_url: str,
        max_length: int,
        true_batch_size: int,
        replace_pad_token: int = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.max_length = max_length
        self.true_batch_size = true_batch_size
        self.replace_pad_token = replace_pad_token

        self.step = 0
        self.skip_steps = None


    def __call__(
        self,
        raw
    ):
        if self.skip_steps is not None and self.step < (self.skip_steps - 3):
            self.step += 1
            return {}

        text = [x["text"] for x in raw]

        input_ids = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        input_ids = input_ids.long().to(constants.DEVICE)

        # minimize padding by selecting samples with least padding tokens
        pad_counts = (input_ids == self.tokenizer.pad_token_id).sum(-1)
        order = torch.argsort(pad_counts, dim=0)

        input_ids = input_ids[order][:self.true_batch_size]

        if self.replace_pad_token is not None:
            input_ids = torch.where(
                input_ids == self.tokenizer.pad_token_id,
                torch.full_like(input_ids, self.replace_pad_token),
                input_ids
            )

        return {
            "input_ids": input_ids
        }
    