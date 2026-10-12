#!/usr/bin/env bash
# Formal UDA sequence only:
#   1) JMDA-Net UDA (its internal iterations use seeds 42..)
#   2) comparison UDA methods for every requested seed
# CoOp is intentionally excluded because it is a supervised prompt-learning
# baseline and must be reported separately.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${1:-100}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JMDA_NUM_ITERATIONS="${JMDA_NUM_ITERATIONS:-5}"

read -r -a BASELINE_SEEDS <<< "${JMDA_BASELINE_SEEDS:-42 43 44 45 46}"
if [ "${#BASELINE_SEEDS[@]}" -eq 0 ]; then
  echo "JMDA_BASELINE_SEEDS must contain at least one seed."
  exit 1
fi

echo "[1/2] JMDA-Net strict UDA: epochs=$EPOCHS iterations=$JMDA_NUM_ITERATIONS"
bash "$HERE/scripts/run_server.sh" jmda_uda "$EPOCHS"

echo "[2/2] UDA comparison methods: seeds=${BASELINE_SEEDS[*]}"
for seed in "${BASELINE_SEEDS[@]}"; do
  echo ">>> UDA comparison seed=$seed"
  JMDA_SEED="$seed" bash "$HERE/scripts/run_server.sh" uda "$EPOCHS"
done

echo "All UDA phases completed. CoOp was not run."
echo "Use final_acc (not best_acc) from $HERE/results/runs.csv for UDA reporting."

