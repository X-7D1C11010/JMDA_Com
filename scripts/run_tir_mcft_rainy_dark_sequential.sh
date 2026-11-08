#!/usr/bin/env bash
# Strictly sequential combined experiment:
#   1) TIR on VDS2Raw, seeds 42..46
#   2) MCFT on rainy and dark weather only, seeds 42..46
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIS_INITIAL_SCALE="${1:-${VDS2RAW_TIR_AIS_INITIAL_SCALE:-1.50}}"
MCFT_EPOCHS="${2:-${MCFT_RAINY_DARK_EPOCHS:-100}}"
TIR_EPOCHS="${3:-${VDS2RAW_TIR_EPOCHS:-100}}"
PYTHON_BIN="${PYTHON:-python}"

if ! "$PYTHON_BIN" -c \
    'import math, sys; value = float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0.0 else 1)' \
    "$AIS_INITIAL_SCALE"; then
  echo "AIS gate must be finite and strictly positive: $AIS_INITIAL_SCALE" >&2
  exit 1
fi
for epoch_spec in "$MCFT_EPOCHS" "$TIR_EPOCHS"; do
  if ! [[ "$epoch_spec" =~ ^[1-9][0-9]*$ ]]; then
    echo "epochs must be positive integers: $epoch_spec" >&2
    exit 1
  fi
done

export VDS2RAW_METHOD_ORDER="tir"
export VDS2RAW_SEEDS="${COMBINED_TIR_SEEDS:-42 43 44 45 46}"
export VDS2RAW_TIR_AIS_INITIAL_SCALE="$AIS_INITIAL_SCALE"
export VDS2RAW_TIR_EPOCHS="$TIR_EPOCHS"

echo "[1/2] TIR: AIS initial gate=$AIS_INITIAL_SCALE; epochs=$TIR_EPOCHS; seeds=${VDS2RAW_SEEDS}"
bash "$HERE/scripts/run_vds2raw_methods.sh" formal

export MCFT_RAINY_DARK_EPOCHS="$MCFT_EPOCHS"
echo "[2/2] MCFT: rainy+dark only; epochs=$MCFT_EPOCHS; lr=${MCFT_RAINY_DARK_LR:-0.0002}; seeds=${MCFT_RAINY_DARK_SEEDS:-42 43 44 45 46}"
bash "$HERE/scripts/run_mcft_rainy_dark_sequential.sh" "$MCFT_EPOCHS"

echo "Sequential TIR + MCFT rainy/dark workflow completed."
