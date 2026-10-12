#!/usr/bin/env bash
# One target, one epoch, one JMDA iteration: validate each method under its
# original protocol before committing GPU time to the formal experiment.
# Existing JMDA-Net supervised results are intentionally not rerun here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JMDA_TARGETS="${JMDA_TARGETS:-backlight}"
export JMDA_NUM_ITERATIONS="${JMDA_NUM_ITERATIONS:-1}"
export JMDA_REQUIRE_CUDA="${JMDA_REQUIRE_CUDA:-1}"
export JMDA_CHECK_ALL_PATHS="${JMDA_CHECK_ALL_PATHS:-1}"

echo "[smoke 1/3] JMDA-Net UDA"
bash "$HERE/scripts/run_server.sh" jmda_uda 1

echo "[smoke 2/3] original-protocol UDA comparison methods"
bash "$HERE/scripts/run_server.sh" uda 1

echo "[smoke 3/3] CoOp supervised prompt-learning baseline"
bash "$HERE/scripts/run_server.sh" vlm 1

echo "All one-epoch smoke tests passed."
