#!/usr/bin/env bash
# Part 6.B — sample grids + KID table
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then PY="uv run python"; else PY=".venv/bin/python"; fi

VP_CKPT="${VP_CKPT:-runs/vp/best.pt}"
RF_CKPT="${RF_CKPT:-runs/rectflow/best.pt}"

echo ">>> Generating 8x8 sample grids (same initial noise across methods)"
${PY} scripts/sample.py --method part6b_grids \
    --vp_checkpoint "${VP_CKPT}" --rf_checkpoint "${RF_CKPT}" \
    --n_samples 64 --seed 42 --out_dir plots/part6b

echo ">>> Computing KID table (1000 samples per cell — slow on CPU)"
${PY} scripts/eval_kid.py \
    --vp_checkpoint "${VP_CKPT}" --rf_checkpoint "${RF_CKPT}" \
    --n_samples 1000 --out plots/part6b_kid_table.md

echo "Done."
