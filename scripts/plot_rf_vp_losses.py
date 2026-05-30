"""
scripts/plot_rf_vp_losses.py — Part 6.A combined training-loss curve.

Overlays the DDPM/VP score-matching loss and the Rectified Flow velocity loss
on the same axes (log scale) for direct visual comparison.

Usage:
    python scripts/plot_rf_vp_losses.py \
        --vp_losses runs/vp/train_losses.npy \
        --rf_losses runs/rectflow/train_losses.npy \
        --out plots/part6a_loss_curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_losses", type=str, default="runs/vp/train_losses.npy",
                   help="Path to the DDPM/VP train_losses.npy")
    p.add_argument("--rf_losses", type=str, default="runs/rectflow/train_losses.npy",
                   help="Path to the Rectified Flow train_losses.npy")
    p.add_argument("--out", type=str, default="plots/part6a_loss_curves.png")
    return p.parse_args()


def maybe_load(path: str) -> np.ndarray | None:
    f = Path(path)
    if not f.exists():
        print(f"[warn] missing loss file: {path} (skipping that curve)")
        return None
    arr = np.load(f)
    print(f"[ok] loaded {path}  ({len(arr)} epochs, final={arr[-1]:.4f})")
    return arr


def main():
    args = get_args()
    vp = maybe_load(args.vp_losses)
    rf = maybe_load(args.rf_losses)

    if vp is None and rf is None:
        raise SystemExit("No loss curves found. Train the models first.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    if vp is not None:
        plt.plot(np.arange(1, len(vp) + 1), vp,
                 marker="o", markersize=3, label="DDPM / VP score loss")
    if rf is not None:
        plt.plot(np.arange(1, len(rf) + 1), rf,
                 marker="s", markersize=3, label="Rectified Flow velocity loss")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (log scale)")
    plt.title("Part 6.A — DDPM vs. Rectified Flow training loss")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure -> {args.out}")


if __name__ == "__main__":
    main()
