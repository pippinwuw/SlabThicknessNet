"""Entry point: train then evaluate."""

import matplotlib.pyplot as plt
import numpy as np

from src.train import train
from src.evaluate import evaluate


def plot_history(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("SmoothL1 Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    val = history["val_loss"]
    ax2.plot(epochs, val, "b-", alpha=0.4)
    ax2.scatter(epochs, val, c="b", s=12)
    ax2.scatter([int(np.argmin(val)) + 1], [np.min(val)],
                c="r", s=80, marker="*", zorder=5,
                label=f"best (epoch {np.argmin(val) + 1})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val SmoothL1 Loss")
    ax2.set_title("Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.close()
    print("[plot] saved training_history.png")


if __name__ == "__main__":
    model, history = train()
    plot_history(history)
    evaluate()
