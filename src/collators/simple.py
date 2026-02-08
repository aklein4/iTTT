import torch

from utils.data_utils import load_any_array
import utils.constants as constants


class SimpleCollator:

    def __init__(
        self,
        expected_length: int,
    ):
        self.expected_length = expected_length


    def __call__(
        self,
        raw
    ):
        
        input_ids = [
            load_any_array(x["input_ids"]) for x in raw
        ]
        for ids in input_ids:
            assert len(ids) == self.expected_length, f"Expected length {self.expected_length}, got {len(ids)}"

        input_ids = torch.stack(input_ids, dim=0)
        input_ids = input_ids.long().to(constants.DEVICE)

        return {
            "input_ids": input_ids
        }
    