#!/usr/bin/env bash
# One target, one epoch, one JMDA iteration: validate every training path
# before committing GPU time to the full four-weather experiment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JMDA_TARGETS="${JMDA_TARGETS:-backlight}"
export JMDA_NUM_ITERATIONS="${JMDA_NUM_ITERATIONS:-1}"
export JMDA_REQUIRE_CUDA="${JMDA_REQUIRE_CUDA:-1}"
export JMDA_CHECK_ALL_PATHS="${JMDA_CHECK_ALL_PATHS:-1}"

echo "[smoke 1/4] SDA methods"
bash "$HERE/scripts/run_server.sh" sda 1

echo "[smoke 2/4] JMDA-Net UDA"
bash "$HERE/scripts/run_server.sh" jmda_uda 1

echo "[smoke 3/4] comparison UDA methods"
bash "$HERE/scripts/run_server.sh" uda 1

echo "[smoke 4/4] CoOp VLM"
bash "$HERE/scripts/run_server.sh" vlm 1

echo "All one-epoch smoke tests passed."
