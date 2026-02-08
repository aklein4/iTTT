import torch

import wandb
import numpy as np


class LogSection:

    def __init__(self, name: str, newline: bool = False):
        """ A context manager for logging sections of code.
         - prints a starting message when entering
         - prints a finishing message when exiting

        Args:
            name (str): name of the section
        """
        self.name = name
        self.newline = newline
    
    
    def __enter__(self):
        if self.newline:
            print("")
        print(f"Starting {self.name}...", flush=True)
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            print(f"Finished {self.name}.", flush=True)
        else:
            raise exc_type(exc_value).with_traceback(traceback)
        

def wandb_image_histogram(x, quantile=1.0):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    assert len(x.shape) == 1

    n = x.shape[0]

    m = np.quantile(x, quantile)

    img = np.zeros((n//4, n))
    img += np.clip((x[None, :] / m), 0.0, 1.0)

    return wandb.Image(img, mode='L')
