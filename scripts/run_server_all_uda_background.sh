#!/usr/bin/env bash
# Launch pinned VIS+IR baselines or a controlled experiment in one detached job.
# Usage: bash scripts/run_server_all_uda_background.sh [target] [seeds_csv] [variant] [methods_csv]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-backlight}"
SEEDS="${2:-42}"
if [[ ! "$SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Expected comma-separated nonnegative seeds, got: $SEEDS" >&2
  exit 2
fi
VARIANT="${3:-historical_best}"
case "$VARIANT" in
  historical_best|compare|tuned|diagnose|dann_capped|dann_ir_pretrain|coral_lr5|sampling|cdan_source_only|cdan_grl_cap|mdd_ramp8|mdd_ramp4|mcc_source_only|mcc_joint_zero|augmentation|ir_pretrain|ema_teacher) ;;
  *) echo "Unknown historical UDA variant: $VARIANT" >&2; exit 2 ;;
esac

export JMDA_TARGETS="$TARGET"
export JMDA_SEED="${SEEDS%%,*}"
export UDA_HISTORICAL_SEEDS="$SEEDS"
export JMDA_CLASS_NAMES_FILE="${JMDA_CLASS_NAMES_FILE:-$HERE/class.txt}"
export UDA_METHODS="DANN,DeepCORAL,CDAN,MDD,MCC"
if [ "$#" -ge 4 ]; then export UDA_METHODS="$4"; fi
export UDA_TRAINING_PROFILE=historical_best
export UDA_HISTORICAL_VARIANT="$VARIANT"
export JMDA_SUITE_ID="${JMDA_SUITE_ID:-uda_${VARIANT}_$(date +%Y%m%d_%H%M%S)_$$}"
export JMDA_GPU_SCHEDULER=1
export JMDA_GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-0,1}"
export JMDA_GPU_MAX_UTIL_PERCENT="${JMDA_GPU_MAX_UTIL_PERCENT:-10}"
export JMDA_GPU_STABLE_CHECKS="${JMDA_GPU_STABLE_CHECKS:-3}"
export JMDA_GPU_POLL_SECONDS="${JMDA_GPU_POLL_SECONDS:-30}"
export GPU_MIN_FREE_DANN_MIB="${GPU_MIN_FREE_DANN_MIB:-16000}"
export GPU_MIN_FREE_DEEPCORAL_MIB="${GPU_MIN_FREE_DEEPCORAL_MIB:-16000}"
export GPU_MIN_FREE_CDAN_MIB="${GPU_MIN_FREE_CDAN_MIB:-18000}"
export GPU_MIN_FREE_MDD_MIB="${GPU_MIN_FREE_MDD_MIB:-20000}"
export GPU_MIN_FREE_MCC_MIB="${GPU_MIN_FREE_MCC_MIB:-18000}"

echo "Sequential UDA suite: DANN -> DeepCORAL -> CDAN -> MDD -> MCC (when selected)"
echo "Target=$TARGET Seeds=$SEEDS Variant=$VARIANT Methods=$UDA_METHODS BatchSize=16 per domain"
echo "GPU candidates=$JMDA_GPU_CANDIDATES; utilization<=${JMDA_GPU_MAX_UTIL_PERCENT}%"
exec bash "$HERE/scripts/run_server_background.sh" uda
