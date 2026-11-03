#!/usr/bin/env bash
# One-seed validation of the log-driven VDS2Raw tuning before a 5-seed run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PILOT_SEED="${VDS2RAW_PILOT_SEED:-42}"
PRETRAINED_DIR="${VDS2RAW_PRETRAINED_DIR:-$HERE/pretrained_weights}"
CCSA_BACKBONE="$PRETRAINED_DIR/ccsa_vgg16_imagenet1k_v1_backbone.pth"
DSNE_BACKBONE="$PRETRAINED_DIR/dsne_resnetv2_101_a1h_in1k_backbone.pth"

if [ -n "${VDS2RAW_CCSA_SOURCE_CHECKPOINT:-}" ] || \
   [ -n "${VDS2RAW_DSNE_SOURCE_CHECKPOINT:-}" ]; then
  echo "Tuned pilot must retrain source stages; unset both VDS2RAW_*_SOURCE_CHECKPOINT variables." >&2
  exit 1
fi
for checkpoint in "$CCSA_BACKBONE" "$DSNE_BACKBONE"; do
  if [ ! -f "$checkpoint" ]; then
    echo "Missing strict ImageNet backbone checkpoint: $checkpoint" >&2
    exit 1
  fi
done

export VDS2RAW_SEEDS="$PILOT_SEED"
export VDS2RAW_METHOD_ORDER="dsne ccsa mcft3"
export VDS2RAW_CCSA_PRETRAINED_BACKBONE="$CCSA_BACKBONE"
export VDS2RAW_DSNE_PRETRAINED_BACKBONE="$DSNE_BACKBONE"
export VDS2RAW_RUN_COMPLEXITY_BENCHMARK=0

echo "Starting tuned VDS2Raw pilot: seed=$PILOT_SEED, order=$VDS2RAW_METHOD_ORDER"
exec bash "$HERE/scripts/run_vds2raw_background.sh" formal
