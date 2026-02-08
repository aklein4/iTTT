#!/bin/bash

: '
Setup a TPU VM to use the repo.
 - MUST RUN WITH dot (.) command

Arguments:
    $1: wandb token

Example:
    . setup_vm.sh <WANDB_TOKEN>
'

# install torch (already installed on lambda?)
# pip install torch==2.9.0 torchvision==0.24.0

# install extras
pip install -r requirements.txt

# login to wandb
python -m wandb login $1
