#!/usr/bin/env bash
# One-command experiment order (original protocol per comparison method):
#   1) JMDA-Net UDA, 2) original UDA comparison methods,
#   3) MCFT supervised multimodal baseline, 4) unified efficiency benchmark.
# Existing JMDA-Net supervised results are intentionally not rerun here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${1:-100}"
PYTHON="${PYTHON:-python}"
export PYTHON
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[1/4] JMDA-Net unsupervised domain adaptation"
bash "$HERE/scripts/run_server.sh" jmda_uda "$EPOCHS"

echo "[2/4] Original-protocol UDA comparison methods"
bash "$HERE/scripts/run_server.sh" uda "$EPOCHS"

echo "[3/4] MCFT supervised multimodal baseline (reported separately)"
bash "$HERE/scripts/run_server.sh" supervised "$EPOCHS"

if [ "${RUN_EFFICIENCY:-1}" = "1" ]; then
  echo "[4/4] Unified efficiency benchmark"
  "$PYTHON" "$HERE/tools/benchmark_efficiency.py" \
    --methods all \
    --protocol benchmark \
    --benchmark_repeats "${EFFICIENCY_REPEATS:-3}" \
    --benchmark_train_batch_size "${EFFICIENCY_TRAIN_BATCH_SIZE:-8}" \
    --efficiency_output_dir "$HERE/efficiency_results/benchmark"
else
  echo "[4/4] Efficiency benchmark skipped (RUN_EFFICIENCY=0)"
fi

echo "All phases completed."
echo "Training records: $HERE/results/runs.csv"
echo "Raw logs: $HERE/server_logs"
echo "Efficiency results: $HERE/efficiency_results"
