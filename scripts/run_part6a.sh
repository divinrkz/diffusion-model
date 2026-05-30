#!/usr/bin/env bash
# scripts/run_part6a.sh — Part 6.A: train Rectified Flow + DDPM/VP baseline
# and plot both training-loss curves on the same (log-scale) axes.
#
# Usage:
#   bash scripts/run_part6a.sh
#   EPOCHS=50 BATCH=64 LR=1e-4 bash scripts/run_part6a.sh
#
# Both models use the SAME hyperparameters (Part 5 settings) so the loss
# curves are produced under matched training budgets.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- Hyperparameters (same for both models) ---
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-64}"
LR="${LR:-1e-4}"

VP_DIR="runs/vp"
RF_DIR="runs/rectflow"

# Prefer `uv run` if available, else fall back to the venv / system python.
if command -v uv >/dev/null 2>&1; then
    PY="uv run python"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

echo "=========================================================="
echo " Part 6.A  |  epochs=${EPOCHS}  batch=${BATCH}  lr=${LR}"
echo " Python launcher: ${PY}"
echo "=========================================================="

# --- 1) DDPM / VP score model (Part 5 baseline) ---
echo ">>> [1/3] Training DDPM/VP baseline -> ${VP_DIR}"
${PY} scripts/train_vp.py \
    --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" \
    --save_dir "${VP_DIR}"

# --- 2) Rectified Flow ---
echo ">>> [2/3] Training Rectified Flow -> ${RF_DIR}"
${PY} scripts/train_rectflow.py \
    --epochs "${EPOCHS}" --lr "${LR}" --batch_size "${BATCH}" \
    --save_dir "${RF_DIR}"

# --- 3) Combined loss curve ---
echo ">>> [3/3] Plotting combined loss curves"
${PY} scripts/plot_rf_vp_losses.py \
    --vp_losses "${VP_DIR}/train_losses.npy" \
    --rf_losses "${RF_DIR}/train_losses.npy" \
    --out plots/part6a_loss_curves.png

echo "Done. See plots/part6a_loss_curves.png"
