#!/usr/bin/env bash
# Submit one-target/one-seed multimodal MDD+MCC runs. Each method waits until
# run_on_available_gpu.py finds a candidate GPU with sufficient free VRAM and
# sufficiently low utilization, then binds that child to the selected GPU.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-backlight}"
SEED="${2:-42}"
export UDA_METHODS="MDD,MCC"
export JMDA_TARGETS="$TARGET"
export JMDA_SEED="$SEED"
export UDA_IMAGE_BATCH_SIZE="${UDA_IMAGE_BATCH_SIZE:-16}"
export UDA_MDD_EPOCHS="${UDA_MDD_EPOCHS:-30}"
export UDA_MDD_LR="${UDA_MDD_LR:-0.0025}"
export UDA_MDD_ITERS_PER_EPOCH="${UDA_MDD_ITERS_PER_EPOCH:-100}"
export UDA_MDD_TRADE_OFF="${UDA_MDD_TRADE_OFF:-0.5}"
export UDA_MDD_TRADE_OFF_WARMUP_EPOCHS="${UDA_MDD_TRADE_OFF_WARMUP_EPOCHS:-10}"
export UDA_MCC_EPOCHS="${UDA_MCC_EPOCHS:-22}"
export UDA_MCC_LR="${UDA_MCC_LR:-0.0025}"
export UDA_MCC_ITERS_PER_EPOCH="${UDA_MCC_ITERS_PER_EPOCH:-100}"
export UDA_MCC_TRADE_OFF="${UDA_MCC_TRADE_OFF:-0.2}"
export UDA_MCC_TRADE_OFF_WARMUP_EPOCHS="${UDA_MCC_TRADE_OFF_WARMUP_EPOCHS:-10}"
export JMDA_DETERMINISTIC="${JMDA_DETERMINISTIC:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export JMDA_GPU_SCHEDULER=1
export JMDA_GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-0,1}"
export JMDA_GPU_MAX_UTIL_PERCENT="${JMDA_GPU_MAX_UTIL_PERCENT:-10}"
export JMDA_GPU_STABLE_CHECKS="${JMDA_GPU_STABLE_CHECKS:-3}"
export JMDA_GPU_POLL_SECONDS="${JMDA_GPU_POLL_SECONDS:-30}"
export GPU_MIN_FREE_MDD_MIB="${GPU_MIN_FREE_MDD_MIB:-20000}"
export GPU_MIN_FREE_MCC_MIB="${GPU_MIN_FREE_MCC_MIB:-18000}"
export JMDA_CLASS_NAMES_FILE="${JMDA_CLASS_NAMES_FILE:-$HERE/class.txt}"

echo "Server split: MDD and MCC only."
echo "Target=$JMDA_TARGETS Seed=$JMDA_SEED BatchSize=$UDA_IMAGE_BATCH_SIZE"
echo "MDD: epochs=$UDA_MDD_EPOCHS lr=$UDA_MDD_LR iters/epoch=$UDA_MDD_ITERS_PER_EPOCH trade-off=$UDA_MDD_TRADE_OFF warmup=$UDA_MDD_TRADE_OFF_WARMUP_EPOCHS"
echo "MCC: epochs=$UDA_MCC_EPOCHS lr=$UDA_MCC_LR iters/epoch=$UDA_MCC_ITERS_PER_EPOCH trade-off=$UDA_MCC_TRADE_OFF warmup=$UDA_MCC_TRADE_OFF_WARMUP_EPOCHS"
echo "GPU candidates=$JMDA_GPU_CANDIDATES; util<=$JMDA_GPU_MAX_UTIL_PERCENT%; MDD free>=${GPU_MIN_FREE_MDD_MIB}MiB; MCC free>=${GPU_MIN_FREE_MCC_MIB}MiB"
exec bash "$HERE/scripts/run_server_background.sh" uda
