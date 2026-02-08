import torch

import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get the base path of src
BASE_PATH = os.path.dirname( # src
    os.path.dirname( # utils
        __file__ # utils.constants
    )
)

# local data path
LOCAL_DATA_PATH = os.path.join(BASE_PATH, "local_data")

# modules for classes
MODEL_MODULE = "models"
TRAINER_MODULE = "trainers"
COLLATOR_MODULE = "collators"
OPTIMIZER_MODULE = "optimizers"

# huggingface login id
HF_ID = "aklein4"

# token for huggingface
HF_TOKEN = os.getenv("HF_TOKEN")
