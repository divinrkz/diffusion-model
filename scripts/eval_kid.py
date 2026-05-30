"""
scripts/eval_kid.py  —  Part 6B: KID evaluation table
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError("Install torch-fidelity: uv sync --extra fidelity")

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]
METHODS = ["rectflow", "ddim", "em"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint", type=str, required=True)
    p.add_argument("--rf_checkpoint", type=str, required=True)
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="plots/part6b_kid_table.md")
    return p.parse_args()


def save_samples_to_dir(samples: torch.Tensor, directory: str):
    os.makedirs(directory, exist_ok=True)
    samples = samples.clamp(-1, 1) * 0.5 + 0.5
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{i:05d}.png"))


def prepare_real_dir(root: str) -> str:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    real_dir = os.path.join(root, "_fidelity_real")
    os.makedirs(real_dir, exist_ok=True)
    for i in range(len(ds)):
        img, _ = ds[i]
        save_image(img * 0.5 + 0.5, os.path.join(real_dir, f"{i:05d}.png"))
    return real_dir


def compute_kid(generated_dir: str, real_dir: str) -> tuple[float, float]:
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(1000, len(os.listdir(generated_dir))),
        verbose=False,
    )
    return metrics["kernel_inception_distance_mean"], metrics["kernel_inception_distance_std"]


@torch.no_grad()
def generate_all(model_kind: str, model, sde_or_flow, n_samples, batch_size, num_steps, device):
    chunks = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        shape = (b, 1, 28, 28)
        if model_kind == "rectflow":
            out = sde_or_flow.euler_sample(model, shape, num_steps=num_steps, device=device)
        elif model_kind == "ddim":
            out = sde_or_flow.ddim(model, shape, num_steps=num_steps, device=device)
        elif model_kind == "em":
            out = sde_or_flow.euler_maruyama(model, shape, num_steps=num_steps, device=device)
        else:
            raise ValueError(model_kind)
        chunks.append(out.cpu())
        remaining -= b
    return torch.cat(chunks, dim=0)


def main():
    args = get_args()
    device = torch.device(args.device)

    vp_model = UNet(in_channels=1, base_channels=64).to(device)
    vp_model.load_state_dict(torch.load(args.vp_checkpoint, map_location=device))
    vp_model.eval()
    sde = VPSDE(args.beta_min, args.beta_max, args.T)

    rf_model = UNet(in_channels=1, base_channels=64).to(device)
    rf_model.load_state_dict(torch.load(args.rf_checkpoint, map_location=device))
    rf_model.eval()
    flow = RectifiedFlow()

    real_dir = prepare_real_dir("data")
    results: dict[tuple[str, int], tuple[float, float]] = {}
    cache_path = Path(args.out).with_suffix(".json")

    import json
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        for k, v in raw.items():
            method, steps = k.split("|")
            results[(method, int(steps))] = (v["mean"], v["std"])
        print(f"Resuming from cache ({len(results)} entries)")

    with tempfile.TemporaryDirectory() as tmp:
        for method in METHODS:
            backend = flow if method == "rectflow" else sde
            model = rf_model if method == "rectflow" else vp_model
            for steps in STEP_COUNTS:
                key = (method, steps)
                if key in results:
                    print(f"Skipping cached {method} steps={steps}")
                    continue
                print(f"Generating {args.n_samples} samples: {method}, steps={steps} ...")
                samples = generate_all(method, model, backend, args.n_samples,
                                       args.batch_size, steps, device)
                gen_dir = os.path.join(tmp, f"{method}_{steps}")
                save_samples_to_dir(samples, gen_dir)
                mean, std = compute_kid(gen_dir, real_dir)
                results[key] = (mean, std)
                print(f"  KID = {mean:.4f} ± {std:.4f}")
                cache_path.write_text(json.dumps(
                    {f"{m}|{s}": {"mean": v[0], "std": v[1]} for (m, s), v in results.items()}
                ))

    lines = [
        "| Steps | Flow Matching | DDIM | DDPM EM |",
        "|-------|---------------|------|---------|",
    ]
    for steps in STEP_COUNTS:
        rf = results.get(("rectflow", steps))
        dd = results.get(("ddim", steps))
        em = results.get(("em", steps))
        def fmt(x):
            if x is None:
                return "—"
            m, s = x
            tag = " **(baseline)**" if steps == 1000 and x == em else ""
            return f"{m:.3f} ± {s:.3f}{tag}"
        lines.append(f"| {steps} | {fmt(rf)} | {fmt(dd)} | {fmt(em)} |")

    table = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table + "\n")
    print("\n" + table)
    print(f"\nSaved table -> {args.out}")


if __name__ == "__main__":
    main()
