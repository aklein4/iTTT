A simple and highly configurable skeleton for training ML models using pytorch and transformers. Ideal for exotic ML research.

## Notes
 - remember to remove any identifying information (such as HF id) before anonymous submission

## Lambda Instance Setup Instructions

1. Create instance with version: `Lambda Stack 22.04`. This comes with:
```
torch==2.7.0
torchvision==0.22.0
flash-attn==2.7.4.post1
```

2. (Optional) install github and login to improve git in VS Code remote SSH
```
sudo apt install gh
gh auth login
```

3. Clone repo

`git clone https://github.com/aklein4/ml-skeleton.git`

4. Setup environment

`cd ~/ml-skeleton && . setup_vm.sh <WANDB_TOKEN>`

5. (Optional) Set your git config to enable contributions
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```