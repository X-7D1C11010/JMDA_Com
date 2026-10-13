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
export JMDA_SUITE_ID="${JMDA_SUITE_ID:-uda_$(date +%Y%m%d_%H%M%S)_$$}"

read -r -a BASELINE_SEEDS <<< "${JMDA_BASELINE_SEEDS:-42 43 44 45 46}"
if [ "${#BASELINE_SEEDS[@]}" -eq 0 ]; then
  echo "JMDA_BASELINE_SEEDS must contain at least one seed."
  exit 1
fi
if ! [[ "$JMDA_NUM_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JMDA_NUM_ITERATIONS must be a positive integer."
  exit 1
fi
if [ "$JMDA_NUM_ITERATIONS" -ne "${#BASELINE_SEEDS[@]}" ]; then
  echo "JMDA_NUM_ITERATIONS must equal the number of baseline seeds for a fair suite."
  exit 1
fi
FIRST_SEED="${BASELINE_SEEDS[0]}"
if ! [[ "$FIRST_SEED" =~ ^[0-9]+$ ]]; then
  echo "Baseline seeds must be non-negative integers."
  exit 1
fi
for index in "${!BASELINE_SEEDS[@]}"; do
  expected=$((FIRST_SEED + index))
  if [ "${BASELINE_SEEDS[$index]}" != "$expected" ]; then
    echo "JMDA uses consecutive seeds; expected seed $expected at position $index."
    exit 1
  fi
done

echo "[1/2] JMDA-Net strict UDA: epochs=$EPOCHS iterations=$JMDA_NUM_ITERATIONS"
JMDA_SEED="$FIRST_SEED" bash "$HERE/scripts/run_server.sh" jmda_uda "$EPOCHS"

echo "[2/2] UDA comparison methods: seeds=${BASELINE_SEEDS[*]}"
for seed in "${BASELINE_SEEDS[@]}"; do
  echo ">>> UDA comparison seed=$seed"
  JMDA_SEED="$seed" bash "$HERE/scripts/run_server.sh" uda "$EPOCHS"
done

"${PYTHON:-python}" "$HERE/tools/summarize_run_results.py" \
  --runs-csv "$HERE/results/runs.csv" \
  --suite-id "$JMDA_SUITE_ID" \
  --protocol UDA \
  --output-dir "$HERE/results/summaries"

echo "All UDA phases completed. CoOp was not run."
echo "Use final_acc (not best_acc) from $HERE/results/runs.csv for UDA reporting."

