"""
scripts/sample.py  —  Generate and compare samples (Parts 5C, 6B, 6D)
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


def save_grid(samples: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a (B,1,H,W) tensor in [-1,1] as an image grid."""
    grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=nrow)
    plt.figure(figsize=(nrow, max(1, samples.size(0) // nrow)))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="em",
                   choices=["em", "pc", "ddim", "rectflow", "all", "part6b_grids"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--vp_checkpoint", type=str, default=None)
    p.add_argument("--rf_checkpoint", type=str, default=None)
    p.add_argument("--reflow_checkpoint", type=str, default=None)
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--n_corrector", type=int, default=1)
    p.add_argument("--snr", type=float, default=0.16)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--out", type=str, default="samples.png")
    p.add_argument("--out_dir", type=str, default="plots/part6b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vp_model(checkpoint: str, beta_min: float, beta_max: float, T: int, device):
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return VPSDE(beta_min, beta_max, T), model


def load_rf_model(checkpoint: str, device) -> tuple[RectifiedFlow, UNet]:
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return RectifiedFlow(), model


def fixed_initial_noise(n: int, device) -> torch.Tensor:
    """Same z ~ N(0,I) used across methods; DDPM scales by sigma(1)."""
    return torch.randn(n, 1, 28, 28, device=device)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    shape = (args.n_samples, 1, 28, 28)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.method == "em":
        ckpt = args.checkpoint or args.vp_checkpoint
        sde, model = load_vp_model(ckpt, args.beta_min, args.beta_max, args.T, device)
        z = fixed_initial_noise(args.n_samples, device)
        samples = sde.euler_maruyama(model, shape, num_steps=args.num_steps,
                                     device=device, initial_noise=z)
        save_grid(samples, args.out, title=f"DDPM EM ({args.num_steps} steps)")

    elif args.method == "pc":
        ckpt = args.checkpoint or args.vp_checkpoint
        sde, model = load_vp_model(ckpt, args.beta_min, args.beta_max, args.T, device)
        samples = sde.predictor_corrector(model, shape, num_steps=args.num_steps,
                                          n_corrector=args.n_corrector, snr=args.snr,
                                          device=device)
        save_grid(samples, args.out,
                  title=f"DDPM PC ({args.num_steps} steps, {args.n_corrector} correctors)")

    elif args.method == "ddim":
        ckpt = args.checkpoint or args.vp_checkpoint
        sde, model = load_vp_model(ckpt, args.beta_min, args.beta_max, args.T, device)
        z = fixed_initial_noise(args.n_samples, device)
        samples = sde.ddim(model, shape, num_steps=args.num_steps,
                           device=device, initial_noise=z)
        save_grid(samples, args.out, title=f"DDIM ({args.num_steps} steps)")

    elif args.method == "rectflow":
        ckpt = args.checkpoint or args.rf_checkpoint
        flow, model = load_rf_model(ckpt, device)
        z = fixed_initial_noise(args.n_samples, device)
        samples = flow.euler_sample(model, shape, num_steps=args.num_steps,
                                    device=device, initial_noise=z)
        save_grid(samples, args.out, title=f"Rectified Flow ({args.num_steps} steps)")

    elif args.method == "part6b_grids":
        vp_ckpt = args.vp_checkpoint or args.checkpoint
        rf_ckpt = args.rf_checkpoint
        sde, vp_model = load_vp_model(vp_ckpt, args.beta_min, args.beta_max, args.T, device)
        flow, rf_model = load_rf_model(rf_ckpt, device)
        step_counts = [1, 5, 10, 50, 100, 200, 1000]
        os.makedirs(args.out_dir, exist_ok=True)
        z = fixed_initial_noise(args.n_samples, device)
        for steps in step_counts:
            save_grid(
                flow.euler_sample(rf_model, shape, num_steps=steps, device=device, initial_noise=z),
                f"{args.out_dir}/rectflow_{steps}.png",
                title=f"Flow Matching — {steps} steps",
            )
            save_grid(
                sde.ddim(vp_model, shape, num_steps=steps, device=device, initial_noise=z),
                f"{args.out_dir}/ddim_{steps}.png",
                title=f"DDIM — {steps} steps",
            )
            save_grid(
                sde.euler_maruyama(vp_model, shape, num_steps=steps, device=device, initial_noise=z),
                f"{args.out_dir}/em_{steps}.png",
                title=f"DDPM EM — {steps} steps",
            )
        print(f"Saved grids under {args.out_dir}/")

    elif args.method == "all":
        vp_ckpt = args.vp_checkpoint or args.checkpoint
        rf_ckpt = args.rf_checkpoint
        reflow_ckpt = args.reflow_checkpoint
        sde, vp_model = load_vp_model(vp_ckpt, args.beta_min, args.beta_max, args.T, device)
        flow, rf_model = load_rf_model(rf_ckpt, device)
        n = 8
        z = fixed_initial_noise(n, device)
        small_shape = (n, 1, 28, 28)
        rows = [
            ("DDPM EM (1000)", sde.euler_maruyama(vp_model, small_shape, 1000, device, z)),
            ("DDIM (100)", sde.ddim(vp_model, small_shape, 100, device, z)),
            ("Rect. Flow (100)", flow.euler_sample(rf_model, small_shape, 100, device, z)),
        ]
        if reflow_ckpt:
            _, reflow_model = load_rf_model(reflow_ckpt, device)
            rows.append(("Reflow (1)", flow.euler_sample(reflow_model, small_shape, 1, device, z)))
        fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2, len(rows) * 1.2))
        for i, (label, samples) in enumerate(rows):
            for j in range(n):
                img = (samples[j].clamp(-1, 1) * 0.5 + 0.5).squeeze().cpu().numpy()
                axes[i, j].imshow(img, cmap="gray")
                axes[i, j].axis("off")
                if j == 0:
                    axes[i, j].set_ylabel(label, rotation=0, labelpad=40, fontsize=9)
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        plt.close()
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
