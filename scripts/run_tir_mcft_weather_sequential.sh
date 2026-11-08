#!/usr/bin/env bash
# Sequential formal/smoke workflow:
#   1) VDS2Raw RGB+NIR+AIS TIR
#   2) weather VIS+IR fully supervised MCFT
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in
  formal|smoke) ;;
  *) echo "usage: $0 {formal|smoke}" >&2; exit 1 ;;
esac

export VDS2RAW_DATA_ROOT="${VDS2RAW_DATA_ROOT:-/home/lixiang/lx/VDS2Raw/threeclass_ready}"
export VDS2RAW_METHOD_ORDER="tir"
export VDS2RAW_SEEDS="${COMBINED_TIR_SEEDS:-42 43 44 45 46}"

export JMDA_DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
read -r -a MCFT_SEEDS <<< "${COMBINED_MCFT_SEEDS:-42 43 44 45 46}"
MCFT_EPOCHS="${WEATHER_MCFT_EPOCHS:-100}"
if [ "$MODE" = "smoke" ]; then
  VDS2RAW_SEEDS="42"
  export VDS2RAW_SEEDS
  MCFT_SEEDS=(42)
  MCFT_EPOCHS=1
fi

echo "[1/2] TIR: VDS2Raw RGB+NIR+AIS; seeds=${VDS2RAW_SEEDS}"
bash "$HERE/scripts/run_vds2raw_methods.sh" "$MODE"

echo "[2/2] MCFT: weather VIS+IR; seeds=${MCFT_SEEDS[*]}; epochs=$MCFT_EPOCHS"
export SUPERVISED_MCFT_EPOCHS="$MCFT_EPOCHS"
for seed in "${MCFT_SEEDS[@]}"; do
  echo ">>> weather MCFT seed=$seed (fresh model; independent run)"
  JMDA_SEED="$seed" bash "$HERE/scripts/run_server.sh" supervised "$MCFT_EPOCHS"
done

echo "Sequential TIR + weather MCFT workflow completed."

