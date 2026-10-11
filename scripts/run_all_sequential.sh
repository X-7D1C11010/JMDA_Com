#!/usr/bin/env bash
# One-command experiment order:
#   1) supervised DA, 2) JMDA-Net UDA, 3) comparison-method UDA,
#   4) CoOp VLM baseline, 5) unified efficiency benchmark.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${1:-100}"
PYTHON="${PYTHON:-python}"
export PYTHON
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[1/5] Supervised domain adaptation"
bash "$HERE/scripts/run_server.sh" sda "$EPOCHS"

echo "[2/5] JMDA-Net unsupervised domain adaptation"
bash "$HERE/scripts/run_server.sh" jmda_uda "$EPOCHS"

echo "[3/5] Comparison-method unsupervised domain adaptation"
bash "$HERE/scripts/run_server.sh" uda "$EPOCHS"

echo "[4/5] CoOp VLM baseline (reported separately from DA methods)"
bash "$HERE/scripts/run_server.sh" vlm "$EPOCHS"

if [ "${RUN_EFFICIENCY:-1}" = "1" ]; then
  echo "[5/5] Unified efficiency benchmark"
  "$PYTHON" "$HERE/tools/benchmark_efficiency.py" \
    --methods all \
    --protocol benchmark \
    --benchmark_repeats "${EFFICIENCY_REPEATS:-3}" \
    --benchmark_train_batch_size "${EFFICIENCY_TRAIN_BATCH_SIZE:-8}" \
    --efficiency_output_dir "$HERE/efficiency_results/benchmark"
else
  echo "[5/5] Efficiency benchmark skipped (RUN_EFFICIENCY=0)"
fi

echo "All phases completed."
echo "Training records: $HERE/results/runs.csv"
echo "Raw logs: $HERE/server_logs"
echo "Efficiency results: $HERE/efficiency_results"
