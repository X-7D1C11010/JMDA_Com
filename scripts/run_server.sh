#!/usr/bin/env bash
# =============================================================================
# JMDA 服务器训练入口 —— SDA / UDA / Supervised 三条协议严格分开，不混合。
#
# 用法:
#   bash scripts/run_server.sh sda [epochs]   # 监督域适应（论文主结果）
#   bash scripts/run_server.sh jmda_uda [epochs] # JMDA-Net 无监督
#   bash scripts/run_server.sh jmda_oracle_shared_uda [epochs] # 共同类 oracle 诊断（不进 UDA 主表）
#   bash scripts/run_server.sh uda            # 其他无监督域适应方法（各用官方预算）
#   bash scripts/run_server.sh supervised [epochs] # MCFT（源域监督，VIS+IR，独立表）
#   bash scripts/run_server_background.sh uda # 后台运行，断开 SSH 后继续
#
# 环境变量:
#   JMDA_DATA_ROOT  数据根目录（默认 /mnt/data/lixiang/data/Data）
#   JMDA_NET_DIR    JMDA-Net 源码目录（默认使用本仓库 JMDA-Net/ 镜像）
#   PYTHON          Python 解释器（默认 python）
#   JMDA_SEED       当前运行的随机种子（默认 42；正式多种子实验由外层循环传入）
#   JMDA_NUM_ITERATIONS  每个目标域的连续随机种子数（默认 5）
#   JMDA_BATCH_SIZE  JMDA UDA 训练批次（默认 32）
#   JMDA_NUM_WORKERS JMDA UDA DataLoader worker 数（默认 0）
#   JMDA_PROJECTION_DIM  JMDA Tensor/SVD 每模态投影维度（默认 64；匹配监督参考）
#   JMDA_UDA_SOURCE_WARMUP_EPOCHS  纯源域预热轮数（默认 10；之后才估计伪对应）
#   JMDA_SVD_STAT_BATCH_SIZE  JMDA UDA SVD 特征统计批次（默认 128；匹配监督日志）
#   JMDA_PREFETCH_FACTOR  每个 worker 的预取批次数（默认 2；workers=0 时不启用）
#   JMDA_PIN_MEMORY  是否为 JMDA UDA DataLoader 启用 pinned memory（0/1，默认 1）
#   JMDA_RUN_INVARIANT_TESTS  JMDA 训练前运行 Tensor/OT 数值门禁（0/1，默认 1）
#   JMDA_MIN_FREE_GPU_MEMORY_MIB  可选的启动前空闲显存门禁（MiB；默认不启用）
#   TORCH_HOME      torchvision/tllib 权重根目录（默认 pretrained_weights/torch）
#   JMDA_GPU_SCHEDULER  是否在每个方法前等待合适GPU（0/1，默认1）
#   JMDA_GPU_MAX_UTIL_PERCENT  允许启动时的最高GPU利用率（默认10）
#   JMDA_GPU_STABLE_CHECKS / JMDA_GPU_POLL_SECONDS  连续空闲检查次数/间隔（默认3/30）
#   JMDA_GPU_CANDIDATES  可选物理GPU编号或UUID，逗号分隔（默认全部）
#   GPU_MIN_FREE_<METHOD>_MIB  覆盖各方法最低空闲显存门槛
#   UDA_DANN_EPOCHS / UDA_DEEPCORAL_EPOCHS / UDA_MDD_EPOCHS /
#   UDA_MCC_EPOCHS  所选方法的训练轮数
#   UDA_METHODS  本次 UDA 方法列表，逗号或空格分隔；默认
#     DANN,DeepCORAL,CDAN。服务器重方法可设为 MDD,MCC
#   CDAN_NUM_ITERATIONS  CDAN 总迭代数（多模态协议默认 2000）
#   UDA_IMAGE_BATCH_SIZE  JMDA式 VIS+IR 编码器每域批次（默认16）
#   UDA_NUM_WORKERS / UDA_PREFETCH_FACTOR  对比方法加载进程/每进程预取（默认8/4）
#   UDA_PIN_MEMORY / UDA_PERSISTENT_WORKERS  对比方法 pinned/persistent worker（默认1/1）
#   DEEPCORAL_ALEXNET_WEIGHTS  可选的离线 torchvision AlexNet 权重文件
#   SUPERVISED_MCFT_EPOCHS  MCFT官方预算（默认500）
#
# 本仓库 JMDA-Net/ 子目录为本文方法（JMDA-Net）的源码镜像，与对比方法目录区分。
# 每次运行的日志 -> server_logs/<protocol>/，效率结果 -> efficiency_results/<protocol>/
# =============================================================================
set -uo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PROTOCOL="${1:-}"
EPOCHS_WAS_EXPLICIT=0
if [ "$#" -ge 2 ]; then
  EPOCHS="$2"
  EPOCHS_WAS_EXPLICIT=1
elif [ "$PROTOCOL" = "supervised" ]; then
  EPOCHS=500
else
  EPOCHS=100
fi
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "epochs 必须是正整数: $EPOCHS"
  exit 1
fi
case "$PROTOCOL" in
  sda|jmda_uda|jmda_oracle_shared_uda|uda|supervised) ;;
  *) echo "用法: bash scripts/run_server.sh {sda|jmda_uda|jmda_oracle_shared_uda|uda|supervised} [epochs]"; exit 1 ;;
esac
if [[ "$PROTOCOL" = "uda" && "$#" -ge 2 ]]; then
  echo "uda 协议不接受统一 epochs 参数；各方法默认使用官方预算。"
  echo "如需覆盖，请设置对应的 UDA_<METHOD>_EPOCHS 或 CDAN_NUM_ITERATIONS。"
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
PYTHON="${PYTHON:-python}"
JMDA_NET_DIR="${JMDA_NET_DIR:-$HERE/JMDA-Net}"
SEED="${JMDA_SEED:-42}"
JMDA_NUM_ITERATIONS="${JMDA_NUM_ITERATIONS:-5}"
JMDA_BATCH_SIZE="${JMDA_BATCH_SIZE:-32}"
JMDA_NUM_WORKERS="${JMDA_NUM_WORKERS:-0}"
JMDA_PROJECTION_DIM="${JMDA_PROJECTION_DIM:-64}"
JMDA_UDA_SOURCE_WARMUP_EPOCHS="${JMDA_UDA_SOURCE_WARMUP_EPOCHS:-10}"
JMDA_SVD_STAT_BATCH_SIZE="${JMDA_SVD_STAT_BATCH_SIZE:-128}"
JMDA_PREFETCH_FACTOR="${JMDA_PREFETCH_FACTOR:-2}"
JMDA_PIN_MEMORY="${JMDA_PIN_MEMORY:-1}"
JMDA_RUN_INVARIANT_TESTS="${JMDA_RUN_INVARIANT_TESTS:-1}"
UDA_DANN_EPOCHS="${UDA_DANN_EPOCHS:-24}"
UDA_DANN_LR="${UDA_DANN_LR:-0.0005}"
UDA_DANN_DOMAIN_LOSS_WEIGHT="${UDA_DANN_DOMAIN_LOSS_WEIGHT:-0.5}"
UDA_DANN_GRL_MAX_ITERATIONS="${UDA_DANN_GRL_MAX_ITERATIONS:-3300}"
UDA_DANN_TARGET_EVAL_INTERVAL="${UDA_DANN_TARGET_EVAL_INTERVAL:-1}"
UDA_DEEPCORAL_EPOCHS="${UDA_DEEPCORAL_EPOCHS:-30}"
UDA_DEEPCORAL_LR="${UDA_DEEPCORAL_LR:-0.0005}"
UDA_DEEPCORAL_MAX_WEIGHT="${UDA_DEEPCORAL_MAX_WEIGHT:-0.3}"
UDA_DEEPCORAL_WARMUP_EPOCHS="${UDA_DEEPCORAL_WARMUP_EPOCHS:-0}"
UDA_DEEPCORAL_HEAD_LR_MULT="${UDA_DEEPCORAL_HEAD_LR_MULT:-10.0}"
UDA_DEEPCORAL_VIS_LR_MULT="${UDA_DEEPCORAL_VIS_LR_MULT:-0.1}"
UDA_DEEPCORAL_IR_LR_MULT="${UDA_DEEPCORAL_IR_LR_MULT:-1.0}"
UDA_DEEPCORAL_TARGET_EVAL_INTERVAL="${UDA_DEEPCORAL_TARGET_EVAL_INTERVAL:-1}"
UDA_MDD_EPOCHS="${UDA_MDD_EPOCHS:-36}"
UDA_MCC_EPOCHS="${UDA_MCC_EPOCHS:-14}"
UDA_MDD_LR="${UDA_MDD_LR:-0.004}"
UDA_MDD_ITERS_PER_EPOCH="${UDA_MDD_ITERS_PER_EPOCH:-1000}"
UDA_MDD_TRADE_OFF="${UDA_MDD_TRADE_OFF:-1.0}"
UDA_MDD_TRADE_OFF_WARMUP_EPOCHS="${UDA_MDD_TRADE_OFF_WARMUP_EPOCHS:-0}"
UDA_MCC_LR="${UDA_MCC_LR:-0.005}"
UDA_MCC_ITERS_PER_EPOCH="${UDA_MCC_ITERS_PER_EPOCH:-1000}"
UDA_MCC_TRADE_OFF="${UDA_MCC_TRADE_OFF:-1.0}"
UDA_MCC_TRADE_OFF_WARMUP_EPOCHS="${UDA_MCC_TRADE_OFF_WARMUP_EPOCHS:-0}"
UDA_MCC_SOURCE_WARMUP_EPOCHS="${UDA_MCC_SOURCE_WARMUP_EPOCHS:-5}"
UDA_MCC_LABEL_SMOOTHING="${UDA_MCC_LABEL_SMOOTHING:-0}"
CDAN_NUM_ITERATIONS="${CDAN_NUM_ITERATIONS:-1400}"
UDA_CDAN_LR="${UDA_CDAN_LR:-0.0005}"
UDA_CDAN_HEAD_LR_MULT="${UDA_CDAN_HEAD_LR_MULT:-3.0}"
UDA_CDAN_AD_LR_MULT="${UDA_CDAN_AD_LR_MULT:-10.0}"
UDA_CDAN_TRADE_OFF="${UDA_CDAN_TRADE_OFF:-1.0}"
UDA_CDAN_SOURCE_WARMUP_ITERATIONS="${UDA_CDAN_SOURCE_WARMUP_ITERATIONS:-0}"
UDA_CDAN_LABEL_SMOOTHING="${UDA_CDAN_LABEL_SMOOTHING:-0.05}"
UDA_CDAN_COEFF_MAX_ITERATIONS="${UDA_CDAN_COEFF_MAX_ITERATIONS:-12000}"
UDA_CDAN_TEST_INTERVAL="${UDA_CDAN_TEST_INTERVAL:-100}"
UDA_IMAGE_BATCH_SIZE="${UDA_IMAGE_BATCH_SIZE:-16}"
UDA_METHODS="${UDA_METHODS:-DANN,DeepCORAL,CDAN}"
UDA_NUM_WORKERS="${UDA_NUM_WORKERS:-8}"
UDA_PREFETCH_FACTOR="${UDA_PREFETCH_FACTOR:-4}"
UDA_PIN_MEMORY="${UDA_PIN_MEMORY:-1}"
UDA_PERSISTENT_WORKERS="${UDA_PERSISTENT_WORKERS:-1}"
DEEPCORAL_ALEXNET_WEIGHTS="${DEEPCORAL_ALEXNET_WEIGHTS:-}"
if [ "$EPOCHS_WAS_EXPLICIT" = "1" ]; then
  SUPERVISED_MCFT_EPOCHS="${SUPERVISED_MCFT_EPOCHS:-$EPOCHS}"
else
  SUPERVISED_MCFT_EPOCHS="${SUPERVISED_MCFT_EPOCHS:-500}"
fi
# Version lock: the lightweight gate recomputes the same portable fingerprint
# from all JMDA core files.  Update this constant only together with reviewed
# source changes; it is intentionally not overridable from the environment.
JMDA_EXPECTED_CODE_FINGERPRINT="90f74dbbad2a858e"
JMDA_MIN_FREE_GPU_MEMORY_MIB="${JMDA_MIN_FREE_GPU_MEMORY_MIB:-}"
JMDA_GPU_SCHEDULER="${JMDA_GPU_SCHEDULER:-1}"
JMDA_GPU_MAX_UTIL_PERCENT="${JMDA_GPU_MAX_UTIL_PERCENT:-10}"
JMDA_GPU_STABLE_CHECKS="${JMDA_GPU_STABLE_CHECKS:-3}"
JMDA_GPU_POLL_SECONDS="${JMDA_GPU_POLL_SECONDS:-30}"
JMDA_GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-}"
GPU_MIN_FREE_DANN_MIB="${GPU_MIN_FREE_DANN_MIB:-16000}"
GPU_MIN_FREE_DEEPCORAL_MIB="${GPU_MIN_FREE_DEEPCORAL_MIB:-16000}"
GPU_MIN_FREE_CDAN_MIB="${GPU_MIN_FREE_CDAN_MIB:-18000}"
GPU_MIN_FREE_MDD_MIB="${GPU_MIN_FREE_MDD_MIB:-20000}"
GPU_MIN_FREE_MCC_MIB="${GPU_MIN_FREE_MCC_MIB:-18000}"
GPU_MIN_FREE_JMDA_MIB="${GPU_MIN_FREE_JMDA_MIB:-20000}"
GPU_MIN_FREE_MCFT_MIB="${GPU_MIN_FREE_MCFT_MIB:-4608}"

require_positive_integer() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name 必须是正整数: $value"
    exit 1
  fi
}

require_nonnegative_integer() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "$name 必须是非负整数: $value"
    exit 1
  fi
}

declare -A UDA_METHOD_ENABLED=()
UDA_METHOD_TEXT="${UDA_METHODS//,/ }"
read -r -a UDA_METHOD_ARRAY <<< "$UDA_METHOD_TEXT"
for method_name in "${UDA_METHOD_ARRAY[@]}"; do
  method_key="${method_name,,}"
  case "$method_key" in
    dann|deepcoral|cdan|mdd|mcc)
      UDA_METHOD_ENABLED["$method_key"]=1
      ;;
    *)
      echo "UDA_METHODS 包含未知方法: $method_name"
      echo "允许: DANN, DeepCORAL, CDAN, MDD, MCC"
      exit 1
      ;;
  esac
done
if [ "${#UDA_METHOD_ENABLED[@]}" -eq 0 ]; then
  echo "UDA_METHODS 至少需要包含一个方法。"
  exit 1
fi
uda_method_enabled() {
  local requested_key="${1,,}"
  [ "${UDA_METHOD_ENABLED[$requested_key]:-0}" = "1" ]
}

if [[ "$PROTOCOL" = "uda" ]]; then
  if uda_method_enabled dann; then
    require_positive_integer "UDA_DANN_EPOCHS" "$UDA_DANN_EPOCHS"
  fi
  if uda_method_enabled deepcoral; then
    require_positive_integer "UDA_DEEPCORAL_EPOCHS" "$UDA_DEEPCORAL_EPOCHS"
  fi
  if uda_method_enabled mdd; then
    require_positive_integer "UDA_MDD_EPOCHS" "$UDA_MDD_EPOCHS"
    require_positive_integer "UDA_MDD_ITERS_PER_EPOCH" "$UDA_MDD_ITERS_PER_EPOCH"
    require_nonnegative_integer "UDA_MDD_TRADE_OFF_WARMUP_EPOCHS" "$UDA_MDD_TRADE_OFF_WARMUP_EPOCHS"
  fi
  if uda_method_enabled mcc; then
    require_positive_integer "UDA_MCC_EPOCHS" "$UDA_MCC_EPOCHS"
    require_positive_integer "UDA_MCC_ITERS_PER_EPOCH" "$UDA_MCC_ITERS_PER_EPOCH"
    require_nonnegative_integer "UDA_MCC_TRADE_OFF_WARMUP_EPOCHS" "$UDA_MCC_TRADE_OFF_WARMUP_EPOCHS"
    require_nonnegative_integer "UDA_MCC_SOURCE_WARMUP_EPOCHS" "$UDA_MCC_SOURCE_WARMUP_EPOCHS"
  fi
  if uda_method_enabled cdan; then
    require_positive_integer "CDAN_NUM_ITERATIONS" "$CDAN_NUM_ITERATIONS"
  fi
  require_positive_integer "UDA_IMAGE_BATCH_SIZE" "$UDA_IMAGE_BATCH_SIZE"
  require_nonnegative_integer "UDA_NUM_WORKERS" "$UDA_NUM_WORKERS"
  require_positive_integer "UDA_PREFETCH_FACTOR" "$UDA_PREFETCH_FACTOR"
  if [[ "$UDA_PIN_MEMORY" != "0" && "$UDA_PIN_MEMORY" != "1" ]]; then
    echo "UDA_PIN_MEMORY 必须是 0 或 1: $UDA_PIN_MEMORY"
    exit 1
  fi
  if [[ "$UDA_PERSISTENT_WORKERS" != "0" && "$UDA_PERSISTENT_WORKERS" != "1" ]]; then
    echo "UDA_PERSISTENT_WORKERS 必须是 0 或 1: $UDA_PERSISTENT_WORKERS"
    exit 1
  fi
fi
if [[ "$PROTOCOL" = "supervised" ]]; then
  require_positive_integer "SUPERVISED_MCFT_EPOCHS" "$SUPERVISED_MCFT_EPOCHS"
fi

if [[ "$PROTOCOL" = "jmda_uda" || "$PROTOCOL" = "jmda_oracle_shared_uda" ]]; then
  require_positive_integer "JMDA_NUM_ITERATIONS" "$JMDA_NUM_ITERATIONS"
  if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
    echo "JMDA_SEED 必须是整数: $SEED"
    exit 1
  fi
  if ! [[ "$JMDA_EXPECTED_CODE_FINGERPRINT" =~ ^[0-9a-f]{16}$ ]]; then
    echo "JMDA_EXPECTED_CODE_FINGERPRINT 必须是16位小写十六进制: $JMDA_EXPECTED_CODE_FINGERPRINT"
    exit 1
  fi
  require_positive_integer "JMDA_BATCH_SIZE" "$JMDA_BATCH_SIZE"
  require_nonnegative_integer "JMDA_NUM_WORKERS" "$JMDA_NUM_WORKERS"
  require_positive_integer "JMDA_PROJECTION_DIM" "$JMDA_PROJECTION_DIM"
  require_nonnegative_integer \
    "JMDA_UDA_SOURCE_WARMUP_EPOCHS" "$JMDA_UDA_SOURCE_WARMUP_EPOCHS"
  if (( JMDA_UDA_SOURCE_WARMUP_EPOCHS >= EPOCHS )); then
    echo "JMDA_UDA_SOURCE_WARMUP_EPOCHS 必须小于总训练轮数 EPOCHS；否则该任务全程只有源域训练，不能标记为 UDA: warmup=$JMDA_UDA_SOURCE_WARMUP_EPOCHS epochs=$EPOCHS"
    exit 1
  fi
  if [[ "$PROTOCOL" = "jmda_oracle_shared_uda" ]] \
      && (( JMDA_UDA_SOURCE_WARMUP_EPOCHS != 10 )); then
    echo "jmda_oracle_shared_uda 在建立目标伪对应前固定进行10轮源域预热: $JMDA_UDA_SOURCE_WARMUP_EPOCHS"
    exit 1
  fi
  require_positive_integer "JMDA_SVD_STAT_BATCH_SIZE" "$JMDA_SVD_STAT_BATCH_SIZE"
  if [[ "$PROTOCOL" = "jmda_oracle_shared_uda" ]] \
      && (( JMDA_BATCH_SIZE != 32 \
            || JMDA_PROJECTION_DIM != 64 \
            || JMDA_SVD_STAT_BATCH_SIZE != 128 )); then
    echo "jmda_oracle_shared_uda 锁定监督参考配置: batch=32 projection_dim=64 SVD_batch=128；当前为 batch=$JMDA_BATCH_SIZE projection_dim=$JMDA_PROJECTION_DIM SVD_batch=$JMDA_SVD_STAT_BATCH_SIZE"
    echo "可通过 JMDA_NUM_WORKERS/JMDA_PREFETCH_FACTOR/JMDA_PIN_MEMORY 提速，但不要改变优化批次或SVD参考配置。"
    exit 1
  fi
  if [[ "$PROTOCOL" = "jmda_oracle_shared_uda" ]] && (( EPOCHS < 25 )); then
    echo "jmda_oracle_shared_uda 至少需要25轮，才能完成10轮源域预热、10轮Tensor/SVD预热和5轮OT权重爬升: epochs=$EPOCHS"
    exit 1
  fi
  require_positive_integer "JMDA_PREFETCH_FACTOR" "$JMDA_PREFETCH_FACTOR"
  case "$JMDA_PIN_MEMORY" in
    0) JMDA_PIN_MEMORY_ARG="--no_pin_memory" ;;
    1) JMDA_PIN_MEMORY_ARG="--pin_memory" ;;
    *)
      echo "JMDA_PIN_MEMORY 必须是 0 或 1: $JMDA_PIN_MEMORY"
      exit 1
      ;;
  esac
  case "$JMDA_RUN_INVARIANT_TESTS" in
    0|1) ;;
    *)
      echo "JMDA_RUN_INVARIANT_TESTS 必须是 0 或 1: $JMDA_RUN_INVARIANT_TESTS"
      exit 1
      ;;
  esac
  if ! JMDA_ACTUAL_CODE_FINGERPRINT="$(
    "$PYTHON" "$HERE/tools/print_jmda_code_fingerprint.py" \
      --jmda-net-dir "$JMDA_NET_DIR"
  )"; then
    echo "无法计算 JMDA 核心源码指纹；尚未启动训练。"
    exit 1
  fi
  if [ "$JMDA_ACTUAL_CODE_FINGERPRINT" != "$JMDA_EXPECTED_CODE_FINGERPRINT" ]; then
    echo "JMDA 核心源码版本不匹配；尚未启动训练。"
    echo "expected=$JMDA_EXPECTED_CODE_FINGERPRINT actual=$JMDA_ACTUAL_CODE_FINGERPRINT"
    echo "请完整同步 JMDA 核心文件及启动/审计工具，禁止混用旧服务器副本。"
    exit 1
  fi
  echo "[OK] JMDA reviewed code fingerprint: $JMDA_ACTUAL_CODE_FINGERPRINT"
fi
if [ -n "$JMDA_MIN_FREE_GPU_MEMORY_MIB" ]; then
  require_nonnegative_integer \
    "JMDA_MIN_FREE_GPU_MEMORY_MIB" "$JMDA_MIN_FREE_GPU_MEMORY_MIB"
fi

LOG_ROOT="$HERE/server_logs/$PROTOCOL"
EFF_DIR="$HERE/efficiency_results/$PROTOCOL"
RESULTS_DIR="$HERE/results"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
SUITE_ID="${JMDA_SUITE_ID:-$RUN_ID}"
CHECKPOINT_ROOT="$HERE/checkpoints/$PROTOCOL/$RUN_ID"
FAIR_LIST_DIR="$HERE/prepared_lists/$SUITE_ID"
TORCH_HOME="${TORCH_HOME:-$HERE/pretrained_weights/torch}"
export TORCH_HOME
if [ -z "$DEEPCORAL_ALEXNET_WEIGHTS" ] \
    && [ -f "$HERE/pretrained_weights/deepcoral_alexnet_owt_4df8aa71.pth" ]; then
  DEEPCORAL_ALEXNET_WEIGHTS="$HERE/pretrained_weights/deepcoral_alexnet_owt_4df8aa71.pth"
fi
export JMDA_DATA_ROOT="$DATA_ROOT"
export JMDA_EFFICIENCY_OUTPUT_DIR="$EFF_DIR"
mkdir -p "$LOG_ROOT" "$EFF_DIR" "$RESULTS_DIR" "$CHECKPOINT_ROOT" "$FAIR_LIST_DIR"

read -r -a TARGETS <<< "${JMDA_TARGETS:-backlight rainy foggy dark}"
VALID_TARGETS=" backlight rainy foggy dark "
for target in "${TARGETS[@]}"; do
  if [[ "$VALID_TARGETS" != *" $target "* ]]; then
    echo "未知目标域: $target（允许: backlight rainy foggy dark）"
    exit 1
  fi
done

if [ ! -d "$DATA_ROOT" ]; then
  echo "数据根不存在: $DATA_ROOT（可用 JMDA_DATA_ROOT 指定）"; exit 1
fi
echo "协议=$PROTOCOL 轮数=$EPOCHS 数据根=$DATA_ROOT"
echo "日志目录=$LOG_ROOT"
echo "生成所有方法共享的精确 VIS/IR 配对样本清单..."
if ! "$PYTHON" "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT" \
  --output-dir "$FAIR_LIST_DIR" \
  --targets "${TARGETS[@]}"; then
  echo "共享配对样本清单生成失败；尚未启动训练。"
  exit 1
fi

PREFLIGHT_ARGS=(
  --protocol "$PROTOCOL"
  --data-root "$DATA_ROOT"
  --jmda-net-dir "$JMDA_NET_DIR"
  --list-dir "$FAIR_LIST_DIR"
  --targets "${TARGETS[@]}"
)
if [ "${JMDA_REQUIRE_CUDA:-1}" = "1" ]; then
  PREFLIGHT_ARGS+=(--require-cuda)
fi
if [ "${JMDA_CHECK_ALL_PATHS:-1}" = "1" ]; then
  PREFLIGHT_ARGS+=(--check-all-paths)
fi
if [ -n "$JMDA_MIN_FREE_GPU_MEMORY_MIB" ]; then
  PREFLIGHT_ARGS+=(
    --min-free-gpu-memory-mib "$JMDA_MIN_FREE_GPU_MEMORY_MIB"
  )
fi
echo "运行公共配置预检..."
if ! "$PYTHON" "$HERE/tools/preflight_server.py" "${PREFLIGHT_ARGS[@]}"; then
  echo "公共配置预检失败；尚未启动任何训练任务。"
  exit 1
fi

if [[ "$JMDA_RUN_INVARIANT_TESTS" = "1" ]] \
   && [[ "$PROTOCOL" = "jmda_uda" || "$PROTOCOL" = "jmda_oracle_shared_uda" ]]; then
  echo "运行 JMDA 协议接线门禁..."
  if ! (cd "$HERE" && "$PYTHON" -m unittest \
    tools.test_jmda_protocol_wiring \
    tools.test_audit_jmda_uda_log); then
    echo "JMDA 协议接线门禁失败；尚未启动训练。"
    exit 1
  fi
  echo "运行 JMDA Tensor/OT 数值门禁..."
  if ! (cd "$JMDA_NET_DIR" && "$PYTHON" -m unittest test_tensor_ot_invariants.py); then
    echo "JMDA Tensor/OT 数值门禁失败；尚未启动训练。"
    exit 1
  fi
fi

FAILURES=0

gpu_min_free_for_tag() {
  case "$1" in
    dann_*)      echo "$GPU_MIN_FREE_DANN_MIB" ;;
    deepcoral_*) echo "$GPU_MIN_FREE_DEEPCORAL_MIB" ;;
    cdan_*)      echo "$GPU_MIN_FREE_CDAN_MIB" ;;
    mdd_*)       echo "$GPU_MIN_FREE_MDD_MIB" ;;
    mcc_*)       echo "$GPU_MIN_FREE_MCC_MIB" ;;
    jmda_net_*)  echo "$GPU_MIN_FREE_JMDA_MIB" ;;
    mcft_*)      echo "$GPU_MIN_FREE_MCFT_MIB" ;;
    *)           echo "16000" ;;
  esac
}

run() {  # run "<log-tag>" <cmd...>
  local tag="$1"; shift
  local log_path="$LOG_ROOT/${tag}_${RUN_ID}.log"
  local started finished elapsed exit_code status record_protocol run_epochs
  local input_modalities model_selection label_space_relation
  local audit_protocol audit_exit audit_status
  local min_free_mib
  local -a launch_command gpu_args
  run_epochs="$EPOCHS"
  if [ "${1:-}" = "--record-epochs" ]; then
    run_epochs="$2"
    shift 2
  fi
  launch_command=("$@")
  min_free_mib="$(gpu_min_free_for_tag "$tag")"
  if [ "$JMDA_GPU_SCHEDULER" = "1" ]; then
    gpu_args=(--min-free-mib "$min_free_mib" \
      --max-util-percent "$JMDA_GPU_MAX_UTIL_PERCENT" \
      --stable-checks "$JMDA_GPU_STABLE_CHECKS" \
      --poll-seconds "$JMDA_GPU_POLL_SECONDS" --tag "$tag")
    if [ -n "$JMDA_GPU_CANDIDATES" ]; then
      gpu_args+=(--gpus "$JMDA_GPU_CANDIDATES")
    fi
    launch_command=("$PYTHON" "$HERE/tools/run_on_available_gpu.py" \
      "${gpu_args[@]}" -- "${launch_command[@]}")
    echo ">>> [$tag] GPU门禁: free>=${min_free_mib}MiB, util<=${JMDA_GPU_MAX_UTIL_PERCENT}% (日志=$log_path)"
  else
    echo ">>> [$tag] GPU门禁已禁用"
  fi
  started="$(date +%s)"
  exit_code=0
  status="success"
  if "${launch_command[@]}" > "$log_path" 2>&1; then
    echo "    done: $tag"
  else
    exit_code=$?
    status="failed"
    echo "    FAILED: $tag (见 $log_path)"
    echo "    ---- error tail ----"
    tail -n 30 "$log_path" || true
    echo "    --------------------"
    FAILURES=$((FAILURES + 1))
  fi
  audit_protocol=""
  if [ "$status" = "success" ]; then
    case "$tag" in
      jmda_net_uda_*) audit_protocol="UDA" ;;
      jmda_net_oracle_shared_uda_*) audit_protocol="ORACLE_SHARED_UDA" ;;
    esac
  fi
  if [ -n "$audit_protocol" ]; then
    if "$PYTHON" "$HERE/tools/audit_jmda_uda_log.py" \
      --log "$log_path" \
      --expected-protocol "$audit_protocol" \
      --expected-projection-dim "$JMDA_PROJECTION_DIM" \
      --expected-warmup-epochs "$JMDA_UDA_SOURCE_WARMUP_EPOCHS" \
      --expected-total-epochs "$EPOCHS" \
      --expected-code-fingerprint "$JMDA_EXPECTED_CODE_FINGERPRINT" \
      --expected-num-iterations "$JMDA_NUM_ITERATIONS" \
      --expected-base-seed "$SEED" \
      --expected-batch-size "$JMDA_BATCH_SIZE" \
      --expected-svd-stat-batch-size "$JMDA_SVD_STAT_BATCH_SIZE" \
      >> "$log_path" 2>&1; then
      audit_status="$(grep -E 'JMDA log audit status:' "$log_path" | tail -n 1 || true)"
      echo "    protocol audit completed: $tag (${audit_status:-status unavailable})"
    else
      audit_exit=$?
      exit_code=$audit_exit
      status="failed"
      FAILURES=$((FAILURES + 1))
      echo "    FAILED protocol audit: $tag (见 $log_path)"
      tail -n 30 "$log_path" || true
    fi
  fi
  finished="$(date +%s)"
  elapsed=$((finished - started))
  record_protocol="$PROTOCOL"
  case "$record_protocol" in
    jmda_uda)
      record_protocol="uda"
      label_space_relation="partial-set"
      ;;
    jmda_oracle_shared_uda)
      record_protocol="oracle_shared_uda"
      label_space_relation="oracle-shared-support"
      ;;
    *)
      label_space_relation="partial-set"
      ;;
  esac
  case "$tag" in
    dann_*|deepcoral_*|cdan_*|mdd_*|mcc_*|jmda_net_*|mcft_*) input_modalities="VIS+IR" ;;
    *) input_modalities="VIS" ;;
  esac
  case "$record_protocol" in
    uda|oracle_shared_uda|supervised) model_selection="final_checkpoint" ;;
    *) model_selection="method_declared" ;;
  esac
  case "$tag" in
    dann_*|deepcoral_*|cdan_*|mdd_*|mcc_*) model_selection="best_target_val_ACC_within_budget" ;;
    mcft_*) model_selection="official_target_best" ;;
  esac
  if ! "$PYTHON" "$HERE/tools/collect_run_result.py" \
    --log "$log_path" \
    --suite-id "$SUITE_ID" \
    --results-dir "$RESULTS_DIR" \
    --protocol "$record_protocol" \
    --run-tag "$tag" \
    --status "$status" \
    --exit-code "$exit_code" \
    --duration-seconds "$elapsed" \
    --epochs "$run_epochs" \
    --seed "$SEED" \
    --efficiency-dir "$EFF_DIR" \
    --input-modalities "$input_modalities" \
    --label-space-relation "$label_space_relation" \
    --model-selection "$model_selection" \
    >> "$log_path" 2>&1; then
    echo "    FAILED result collection: $tag (见 $log_path)"
    FAILURES=$((FAILURES + 1))
  fi
}

# 通用参数（各方法脚本一致）
COMMON=(--epochs "$EPOCHS" --seed "$SEED" --data-dir "$FAIR_LIST_DIR" --efficiency_output_dir "$EFF_DIR" --save-dir "$CHECKPOINT_ROOT")
METHOD_COMMON=(--seed "$SEED" --data-dir "$FAIR_LIST_DIR" --efficiency_output_dir "$EFF_DIR" --save-dir "$CHECKPOINT_ROOT")

if [ "$PROTOCOL" = "sda" ]; then
  # ---- SDA retained baselines ----
  for t in "${TARGETS[@]}"; do
    run "dann_$t"         "$PYTHON" "$HERE/DANN/train_dann_sda.py"             --target "$t" "${COMMON[@]}"
    run "deepcoral_$t"    "$PYTHON" "$HERE/DeepCORAL/train_deepcoral_sda.py"   --target "$t" "${COMMON[@]}"
    run "cdan_$t"         "$PYTHON" "$HERE/CDAN/train_cdan_sda.py"             --target "$t" "${COMMON[@]}"
    run "mdd_$t"          "$PYTHON" "$HERE/MDD/train_mdd_sda.py"               --target "$t" "${COMMON[@]}"
    run "mcc_$t"          "$PYTHON" "$HERE/MCC/train_mcc_sda.py"               --target "$t" "${COMMON[@]}"
    # JMDA-Net-SDA（本文方法，仓库内 JMDA-Net/ 目录）
    run "jmda_net_sda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && JMDA_SDA_LIST_DIR='$FAIR_LIST_DIR' '$PYTHON' main_sda.py --target '$t' --epochs '$EPOCHS' --num-iterations '${JMDA_NUM_ITERATIONS:-5}' --sda-seed '$SEED'"
  done

elif [ "$PROTOCOL" = "jmda_uda" ]; then
  # ---- JMDA-Net UDA：单独放在 SDA 与其他 UDA 方法之间运行 ----
  echo "JMDA加载参数: batch=$JMDA_BATCH_SIZE workers=$JMDA_NUM_WORKERS projection_dim=$JMDA_PROJECTION_DIM source_warmup=$JMDA_UDA_SOURCE_WARMUP_EPOCHS SVD_batch=$JMDA_SVD_STAT_BATCH_SIZE prefetch=$JMDA_PREFETCH_FACTOR pin_memory=$JMDA_PIN_MEMORY"
  for t in "${TARGETS[@]}"; do
    case "$t" in
      backlight) target_cn="逆光" ;;
      rainy)     target_cn="雨天" ;;
      foggy)     target_cn="雾天" ;;
      dark)      target_cn="黑天" ;;
    esac
    run "jmda_net_uda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && '$PYTHON' Ablation/module_ablation.py \
         --source_root '$DATA_ROOT/晴天' --target_root '$DATA_ROOT/$target_cn' \
         --epochs '$EPOCHS' --num_iterations '$JMDA_NUM_ITERATIONS' \
         --batch_size '$JMDA_BATCH_SIZE' \
         --num_workers '$JMDA_NUM_WORKERS' \
         --projection_dim '$JMDA_PROJECTION_DIM' \
         --svd_projection_mode legacy_supervised \
         --basis_transport_mode legacy_supervised \
         --uda_source_warmup_epochs '$JMDA_UDA_SOURCE_WARMUP_EPOCHS' \
         --svd_stat_batch_size '$JMDA_SVD_STAT_BATCH_SIZE' \
         --prefetch_factor '$JMDA_PREFETCH_FACTOR' $JMDA_PIN_MEMORY_ARG \
         --ablation_mode full --no_auto_ablation_hparams \
         --transport_mode sinkhorn --no_adaptive_sinkhorn \
         --ot_sinkhorn_iterations 50 --ot_warmup_epochs 10 --ot_ramp_epochs 5 \
         --tensor_loss_weight 0.12 --adv_loss_weight 0.08 \
         --lr_feature 5e-6 --lr_other 2e-4 --weight_decay 5e-4 \
         --class_conditional_ot --balance_svd_classes \
         --target_label_ratio 0 --target_cls_weight 0 \
         --label_space_mode partial_set \
         --visual_finetuning_mode legacy_supervised \
         --no_sync_discriminator_lr \
         --report_strategy last --base_seed '$SEED' \
         --checkpoint_dir '$CHECKPOINT_ROOT/jmda_net_uda_$t'"
  done

elif [ "$PROTOCOL" = "jmda_oracle_shared_uda" ]; then
  # ---- 共同类别支持 Oracle：独立预处理清单提供类支持，仅作 partial-set 负迁移诊断 ----
  # 该协议使用独立日志/结果/checkpoint，严禁并入严格 UDA 主表。
  echo "JMDA加载参数: batch=$JMDA_BATCH_SIZE workers=$JMDA_NUM_WORKERS projection_dim=$JMDA_PROJECTION_DIM source_warmup=$JMDA_UDA_SOURCE_WARMUP_EPOCHS SVD_batch=$JMDA_SVD_STAT_BATCH_SIZE prefetch=$JMDA_PREFETCH_FACTOR pin_memory=$JMDA_PIN_MEMORY"
  for t in "${TARGETS[@]}"; do
    case "$t" in
      backlight) target_cn="逆光" ;;
      rainy)     target_cn="雨天" ;;
      foggy)     target_cn="雾天" ;;
      dark)      target_cn="黑天" ;;
    esac
    if ! oracle_shared_labels="$(
      "$PYTHON" "$HERE/tools/read_target_class_support.py" \
        --manifest "$FAIR_LIST_DIR/paired_protocol_manifest.json" \
        --target "$t"
    )"; then
      echo "无法从独立配对清单读取 $t 的目标类别支持；未启动该任务。"
      exit 1
    fi
    echo "目标域 $t 独立类别支持(original ids)=$oracle_shared_labels"
    run "jmda_net_oracle_shared_uda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && '$PYTHON' Ablation/module_ablation.py \
         --source_root '$DATA_ROOT/晴天' --target_root '$DATA_ROOT/$target_cn' \
         --epochs '$EPOCHS' --num_iterations '$JMDA_NUM_ITERATIONS' \
         --batch_size '$JMDA_BATCH_SIZE' \
         --num_workers '$JMDA_NUM_WORKERS' \
         --projection_dim '$JMDA_PROJECTION_DIM' \
         --svd_projection_mode legacy_supervised \
         --basis_transport_mode legacy_supervised \
         --uda_source_warmup_epochs '$JMDA_UDA_SOURCE_WARMUP_EPOCHS' \
         --svd_stat_batch_size '$JMDA_SVD_STAT_BATCH_SIZE' \
         --prefetch_factor '$JMDA_PREFETCH_FACTOR' $JMDA_PIN_MEMORY_ARG \
         --ablation_mode full --no_auto_ablation_hparams \
         --transport_mode sinkhorn --no_adaptive_sinkhorn \
         --ot_sinkhorn_iterations 50 --ot_warmup_epochs 10 --ot_ramp_epochs 5 \
         --tensor_loss_weight 0.12 --adv_loss_weight 0.08 \
         --lr_feature 5e-6 --lr_other 2e-4 --weight_decay 5e-4 \
         --class_conditional_ot --balance_svd_classes \
         --target_label_ratio 0 --target_cls_weight 0 \
         --label_space_mode oracle_shared \
         --oracle_shared_labels '$oracle_shared_labels' \
         --target_train_pair_list '$FAIR_LIST_DIR/${t}_train_pairs.tsv' \
         --uda_pseudo_class_pairing \
         --visual_finetuning_mode legacy_supervised \
         --no_sync_discriminator_lr \
         --report_strategy last --base_seed '$SEED' \
         --checkpoint_dir '$CHECKPOINT_ROOT/jmda_net_oracle_shared_uda_$t'"
  done

elif [ "$PROTOCOL" = "uda" ]; then
  # ---- VIS+IR exact-paired multimodal extensions; final epoch, no checkpoints ----
  RESNET18_WEIGHT="$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
  if [ ! -f "$RESNET18_WEIGHT" ]; then
    echo "缺少 JMDA VIS ResNet-18 ImageNet 权重: $RESNET18_WEIGHT"
    echo "请先在本地运行 tools/download_official_uda_weights.py 并上传 pretrained_weights。"
    exit 1
  fi
  UDA_COMMON=(--seed "$SEED" --data-dir "$FAIR_LIST_DIR" --efficiency_output_dir "$EFF_DIR" --save-dir "$CHECKPOINT_ROOT" --num-workers "$UDA_NUM_WORKERS" --prefetch-factor "$UDA_PREFETCH_FACTOR" --multimodal --enhanced-augmentation --no-save-model)
  if [ "$UDA_PIN_MEMORY" = "1" ]; then
    UDA_COMMON+=(--pin-memory)
  else
    UDA_COMMON+=(--no-pin-memory)
  fi
  if [ "$UDA_PERSISTENT_WORKERS" = "1" ]; then
    UDA_COMMON+=(--persistent-workers)
  else
    UDA_COMMON+=(--no-persistent-workers)
  fi
  echo "UDA selected methods: $UDA_METHODS"
  echo "UDA multimodal protocol: JMDA non-tensor VIS ResNet-18 + IR U-Net, 512+512 direct concatenation, batch=$UDA_IMAGE_BATCH_SIZE, CDAN iterations=$CDAN_NUM_ITERATIONS, report peak target-val ACC, no checkpoint"
  echo "UDA DataLoader: workers=$UDA_NUM_WORKERS prefetch=$UDA_PREFETCH_FACTOR pin_memory=$UDA_PIN_MEMORY persistent_workers=$UDA_PERSISTENT_WORKERS"
  for t in "${TARGETS[@]}"; do
    if uda_method_enabled dann; then
      run "dann_$t" --record-epochs "$UDA_DANN_EPOCHS" "$PYTHON" "$HERE/DANN/train_dann.py" --target "$t" --epochs "$UDA_DANN_EPOCHS" --lr "$UDA_DANN_LR" --domain-loss-weight "$UDA_DANN_DOMAIN_LOSS_WEIGHT" --grl-max-iterations "$UDA_DANN_GRL_MAX_ITERATIONS" --lr-scheduler cosine --min-lr-ratio 0.2 --target-eval-interval "$UDA_DANN_TARGET_EVAL_INTERVAL" --backbone jmda --batch-size "$UDA_IMAGE_BATCH_SIZE" "${UDA_COMMON[@]}"
    fi
    if uda_method_enabled deepcoral; then
      run "deepcoral_$t" --record-epochs "$UDA_DEEPCORAL_EPOCHS" "$PYTHON" "$HERE/DeepCORAL/train_deepcoral.py" --target "$t" --epochs "$UDA_DEEPCORAL_EPOCHS" --lr "$UDA_DEEPCORAL_LR" --coral-max-weight "$UDA_DEEPCORAL_MAX_WEIGHT" --coral-warmup-epochs "$UDA_DEEPCORAL_WARMUP_EPOCHS" --head-lr-mult "$UDA_DEEPCORAL_HEAD_LR_MULT" --vis-lr-mult "$UDA_DEEPCORAL_VIS_LR_MULT" --ir-lr-mult "$UDA_DEEPCORAL_IR_LR_MULT" --weight-decay 0.0005 --label-smoothing 0.05 --lr-scheduler constant --min-lr-ratio 0.1 --target-eval-interval "$UDA_DEEPCORAL_TARGET_EVAL_INTERVAL" --backbone jmda --source-batch-size "$UDA_IMAGE_BATCH_SIZE" --target-batch-size "$UDA_IMAGE_BATCH_SIZE" --test-batch-size "$UDA_IMAGE_BATCH_SIZE" "${UDA_COMMON[@]}"
    fi
    if uda_method_enabled cdan; then
      run "cdan_$t"       "$PYTHON" "$HERE/CDAN/train_cdan.py"           --target "$t" --num-iterations "$CDAN_NUM_ITERATIONS" --test-interval "$UDA_CDAN_TEST_INTERVAL" --lr "$UDA_CDAN_LR" --head-lr-mult "$UDA_CDAN_HEAD_LR_MULT" --ad-lr-mult "$UDA_CDAN_AD_LR_MULT" --trade-off "$UDA_CDAN_TRADE_OFF" --source-warmup-iterations "$UDA_CDAN_SOURCE_WARMUP_ITERATIONS" --label-smoothing "$UDA_CDAN_LABEL_SMOOTHING" --coeff-max-iter "$UDA_CDAN_COEFF_MAX_ITERATIONS" --backbone JMDA --batch-size "$UDA_IMAGE_BATCH_SIZE" "${UDA_COMMON[@]}"
    fi
    if uda_method_enabled mdd; then
      run "mdd_$t" --record-epochs "$UDA_MDD_EPOCHS" "$PYTHON" "$HERE/MDD/train_mdd.py" --target "$t" --epochs "$UDA_MDD_EPOCHS" --lr "$UDA_MDD_LR" --iters-per-epoch "$UDA_MDD_ITERS_PER_EPOCH" --trade-off "$UDA_MDD_TRADE_OFF" --trade-off-warmup-epochs "$UDA_MDD_TRADE_OFF_WARMUP_EPOCHS" --backbone jmda --batch-size "$UDA_IMAGE_BATCH_SIZE" "${UDA_COMMON[@]}"
    fi
    if uda_method_enabled mcc; then
      run "mcc_$t" --record-epochs "$UDA_MCC_EPOCHS" "$PYTHON" "$HERE/MCC/train_mcc.py" --target "$t" --epochs "$UDA_MCC_EPOCHS" --lr "$UDA_MCC_LR" --iters-per-epoch "$UDA_MCC_ITERS_PER_EPOCH" --trade-off "$UDA_MCC_TRADE_OFF" --trade-off-warmup-epochs "$UDA_MCC_TRADE_OFF_WARMUP_EPOCHS" --source-warmup-epochs "$UDA_MCC_SOURCE_WARMUP_EPOCHS" --label-smoothing "$UDA_MCC_LABEL_SMOOTHING" --backbone jmda --batch-size "$UDA_IMAGE_BATCH_SIZE" "${UDA_COMMON[@]}"
    fi
  done

else
  # ---- Supervised 独立表：MCFT 是源域监督多模态分类，不是域适应方法 ----
  for t in "${TARGETS[@]}"; do
    run "mcft_$t" --record-epochs "$SUPERVISED_MCFT_EPOCHS" \
      "$PYTHON" "$HERE/MCFT/train_mcft.py" --target "$t" \
      --epochs "$SUPERVISED_MCFT_EPOCHS" "${METHOD_COMMON[@]}"
  done
fi

echo "阶段结束（$PROTOCOL）：失败任务数=$FAILURES。"
if [ "$FAILURES" -ne 0 ]; then
  exit 1
fi
echo "全部任务成功（$PROTOCOL）。"
