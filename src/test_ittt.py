import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from models.ittt.modelling_ittt import ItttModel


URL = "aklein4/iTTT_longattn"
SUBFOLDER = "step_000000001000"


def main():
    
    model = ItttModel.from_pretrained(
        URL,
        subfolder=SUBFOLDER,
    )

    w = model.llama.model.layers[24].mlp.down_proj

    plt.hist(
        (w.log_lr * w.scalar_scaler).flatten().cpu().detach().numpy(),
        bins=100,
    )
    plt.grid()
    plt.savefig("log_lr_hist.png")
    plt.clf()

    x = (w.log_lr * w.scalar_scaler).float().cuda()
    
    n = F.normalize(x, dim=-1)
    sims = n @ n.transpose(-1, -2)

    indices = []
    ind = 0
    while True:

        indices.append(ind)
        if len(indices) == n.shape[0]:
            break

        sims[:, ind] = -1e9
        ind = torch.argmax(sims[ind]).item()

    x = x[indices]

    plt.matshow(
        x.cpu().detach().numpy(),
    )
    plt.colorbar()
    plt.savefig("lr_mat.png")


if __name__ == '__main__':
    main()
