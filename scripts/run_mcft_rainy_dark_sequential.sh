#!/usr/bin/env bash
# Focused MCFT weather experiment: rainy and dark only.
# Every seed is a fresh process/model; no checkpoint is inherited across runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EPOCHS="${1:-${MCFT_RAINY_DARK_EPOCHS:-100}}"
LR="${MCFT_RAINY_DARK_LR:-0.0002}"
read -r -a SEEDS <<< "${MCFT_RAINY_DARK_SEEDS:-42 43 44 45 46}"

if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive_epochs]" >&2
  exit 1
fi
if ! [[ "$LR" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
  echo "MCFT_RAINY_DARK_LR must be a non-negative number: $LR" >&2
  exit 1
fi
if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "MCFT_RAINY_DARK_SEEDS must contain at least one seed" >&2
  exit 1
fi

export JMDA_DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
export JMDA_TARGETS="rainy dark"
export SUPERVISED_MCFT_EPOCHS="$EPOCHS"
export SUPERVISED_MCFT_LR="$LR"

echo "MCFT focused run: targets=${JMDA_TARGETS}; epochs=$EPOCHS; lr=$LR; seeds=${SEEDS[*]}"
for seed in "${SEEDS[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "invalid seed in MCFT_RAINY_DARK_SEEDS: $seed" >&2
    exit 1
  fi
  echo ">>> MCFT rainy+dark seed=$seed (fresh model; independent run)"
  JMDA_SEED="$seed" bash "$HERE/scripts/run_server.sh" supervised "$EPOCHS"
done

echo "MCFT rainy+dark workflow completed."
