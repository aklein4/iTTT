
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHUNK_FILE = "local_data/lm_loss_chunks.csv"
DECADE_FILE = "local_data/lm_loss_decades.csv"
BASELINE_CHUNK_FILE = "local_data/baseline_chunks.csv"

RUN = "longattn-seperated"
BASELINE_RUN = "longattn-baseline"

CHUNK_SIZE = 512

NUM_CHUNKS = 64
NUM_DECADES = 7

MEAN_OVER = 100
ROLL_OVER = 250


def chunk_key(chunk, run=RUN):
    return f"{run} - lm_loss/chunk_{chunk:02d}"

def decade_key(decade, run=RUN):
    return f"{run} - grouped_lm_loss/decade_{decade:01d}"


def plot_chunks():
    
    df = pd.read_csv(CHUNK_FILE)
    baseline_df = pd.read_csv(BASELINE_CHUNK_FILE)

    x = []
    baseline = []
    drop_in = []
    trained = []
    for i in range(NUM_CHUNKS):

        x.append(i * CHUNK_SIZE)

        baseline.append(
            np.mean(baseline_df[chunk_key(i, BASELINE_RUN)].iloc[:MEAN_OVER]
        ))

        drop_in.append(
            np.mean(df[chunk_key(i)].iloc[:MEAN_OVER]
        ))
        
        trained.append(
            np.mean(df[chunk_key(i)].iloc[-MEAN_OVER:]
        ))

    plt.plot(x, baseline, label="Baseline", color="red", linewidth=2)
    plt.plot(x, drop_in, label="iTTT", color="tab:cyan", linewidth=2)
    plt.plot(x, trained, label="iTTT (finetuned)", color="blue", linewidth=2)

    plt.legend()
    plt.grid()

    plt.xlabel("Token Position")
    plt.ylabel("Loss (log perplexity)")
    plt.title("Loss by Token Position")

    plt.savefig("local_data/loss_by_token_position.png")
    plt.clf()


def plot_decades():
    
    df = pd.read_csv(DECADE_FILE)

    for i in range(NUM_DECADES):

        y = df[decade_key(i)].rolling(ROLL_OVER).mean()
        
        if i == 0:
            token_range = f"{CHUNK_SIZE:05d} - {(10 * CHUNK_SIZE)-1:05d}"
        else:
            token_range = f"{i * 10 * CHUNK_SIZE:05d} - {((i + 1) * 10 * CHUNK_SIZE)-1:05d}"
        
        color = plt.cm.viridis(0.9*(1 - i / (NUM_DECADES-1)))

        plt.plot(y, label=token_range, color=color, linewidth=2)

    plt.legend()
    plt.grid()

    plt.xlabel("Training Step")
    plt.ylabel("Loss (log perplexity)")
    plt.title("Loss Through Training at Token Position Ranges")

    plt.xlim(right=len(y)*1.6)

    plt.savefig("local_data/loss_by_token_decade.png")
    plt.clf()


if __name__ == "__main__":
    plot_chunks()
    plot_decades()
