
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = "loss.csv"

RUNS = [
    "longattn-seperated",
    "longattn-no-muon",
    # "longattn-aux-v2",
    "longattn-fo-grad"
]
COLORS = ["black", "red", "blue"]


def main():
    
    df = pd.read_csv(FILE)

    a = df[f"{RUNS[0]} - loss"][:500].rolling(window=200).mean()
    b = df[f"{RUNS[-1]} - loss"][:500].rolling(window=200).mean()
    plt.plot(b - a)
    plt.grid()
    plt.savefig("diff.png")
    return

    for i, run in enumerate(RUNS):
        
        # dec_0 = df[f"{run} - grouped_lm_loss/decade_0"]
        # dec_6 = df[f"{run} - grouped_lm_loss/decade_6"]
        # x = dec_6 / dec_0
        x = df[f"{run} - loss"][:800]

        x_running = x.rolling(window=200)
        plt.plot(x_running.mean(), label=run, color=COLORS[i])

    plt.legend()
    plt.grid()
    plt.savefig("data.png")


if __name__ == "__main__":
    main()