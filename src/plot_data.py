
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = "new_loss.csv"

RUNS = [
    "longattn-seperated",
    "longattn",
    "longattn-aux",
]


def main():
    
    df = pd.read_csv(FILE)

    for run in RUNS:
        
        # dec_0 = df[f"{run} - grouped_lm_loss/decade_0"]
        # dec_6 = df[f"{run} - grouped_lm_loss/decade_6"]
        # x = dec_6 / dec_0
        x = df[f"{run} - loss"]

        x_running = x.rolling(window=100)
        plt.plot(x_running.mean(), label=run)

    plt.legend()
    plt.grid()
    plt.savefig("data.png")


if __name__ == "__main__":
    main()