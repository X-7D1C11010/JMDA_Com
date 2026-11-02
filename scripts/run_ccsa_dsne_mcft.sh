#!/usr/bin/env bash
# Sequential CCSA -> d-SNE -> MCFT experiment suite.
#
# Usage:
#   bash scripts/run_ccsa_dsne_mcft.sh formal
#   bash scripts/run_ccsa_dsne_mcft.sh smoke
#
# Formal mode keeps the released budgets (80/20/500 epochs). Smoke mode runs
# one batch for one epoch to validate the complete data/training/result path.
set -uo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in
  formal|smoke) ;;
  *)
    echo "用法: bash scripts/run_ccsa_dsne_mcft.sh {formal|smoke}"
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
OUTPUT_ROOT="${JMDA_OFFICIAL_OUTPUT_ROOT:-$HERE/runs/ccsa_dsne_mcft}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
SUITE_ID="${JMDA_SUITE_ID:-ccsa_dsne_mcft_${MODE}_${RUN_ID}}"
RUN_ROOT="$OUTPUT_ROOT/$SUITE_ID"
LIST_ROOT="$RUN_ROOT/prepared_lists"
PAIRED_LIST_DIR="$LIST_ROOT/paired"
LOG_DIR="$RUN_ROOT/logs"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
EFFICIENCY_DIR="$RUN_ROOT/efficiency"
RESULTS_DIR="$RUN_ROOT/results"
SUMMARY_DIR="$RUN_ROOT/summary"

read -r -a TARGETS <<< "${JMDA_TARGETS:-backlight rainy foggy dark}"
read -r -a SEEDS <<< "${JMDA_SEEDS:-42}"
read -r -a METHOD_ORDER <<< "${JMDA_METHOD_ORDER:-ccsa dsne mcft}"
VALID_TARGETS=" backlight rainy foggy dark "
for target in "${TARGETS[@]}"; do
  if [[ "$VALID_TARGETS" != *" $target "* ]]; then
    echo "未知目标域: $target（允许: backlight rainy foggy dark）"
    exit 1
  fi
done
if [ "${#METHOD_ORDER[@]}" -lt 1 ] || [ "${#METHOD_ORDER[@]}" -gt 3 ]; then
  echo "JMDA_METHOD_ORDER 必须包含一至三个方法: ccsa dsne mcft。"
  exit 1
fi
CCSA_ORDER_COUNT=0; DSNE_ORDER_COUNT=0; MCFT_ORDER_COUNT=0
for method in "${METHOD_ORDER[@]}"; do
  case "$method" in
    ccsa) CCSA_ORDER_COUNT=$((CCSA_ORDER_COUNT + 1)) ;;
    dsne) DSNE_ORDER_COUNT=$((DSNE_ORDER_COUNT + 1)) ;;
    mcft) MCFT_ORDER_COUNT=$((MCFT_ORDER_COUNT + 1)) ;;
    *) echo "JMDA_METHOD_ORDER 中存在未知方法: $method"; exit 1 ;;
  esac
done
if [ "$CCSA_ORDER_COUNT" -gt 1 ] || [ "$DSNE_ORDER_COUNT" -gt 1 ] \
  || [ "$MCFT_ORDER_COUNT" -gt 1 ]; then
  echo "JMDA_METHOD_ORDER 不能包含重复方法。"
  exit 1
fi
for seed in "${SEEDS[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "随机种子必须是非负整数: $seed"
    exit 1
  fi
done

if [ ! -d "$DATA_ROOT" ]; then
  echo "数据根不存在: $DATA_ROOT（可用 JMDA_DATA_ROOT 指定）"
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
  echo "Python 不可用: $PYTHON_BIN（可用 PYTHON 指定）"
  exit 1
fi
if ! DATA_ROOT_CANONICAL="$("$PYTHON_BIN" -c \
  'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' \
  "$DATA_ROOT")"; then
  echo "无法规范化数据根路径: $DATA_ROOT"
  exit 1
fi

CCSA_EPOCHS="${SDA_CCSA_EPOCHS:-80}"
DSNE_EPOCHS="${SDA_DSNE_EPOCHS:-20}"
MCFT_EPOCHS="${SUPERVISED_MCFT_EPOCHS:-500}"
CCSA_SOURCE_EPOCHS="${SDA_CCSA_SOURCE_EPOCHS:-50}"
DSNE_SOURCE_EPOCHS="${SDA_DSNE_SOURCE_EPOCHS:-50}"
DSNE_LAYERS="${SDA_DSNE_LAYERS:-101}"
NUM_WORKERS="${JMDA_NUM_WORKERS:-2}"
MCFT_EVAL_FREQUENCY="${MCFT_EVAL_FREQUENCY:-10}"
CCSA_EARLY_PATIENCE="${SDA_CCSA_EARLY_STOPPING_PATIENCE:-8}"
CCSA_EARLY_MIN_EPOCHS="${SDA_CCSA_EARLY_STOPPING_MIN_EPOCHS:-5}"
DSNE_EARLY_PATIENCE="${SDA_DSNE_EARLY_STOPPING_PATIENCE:-6}"
DSNE_EARLY_MIN_EPOCHS="${SDA_DSNE_EARLY_STOPPING_MIN_EPOCHS:-10}"
MCFT_EARLY_PATIENCE="${SUPERVISED_MCFT_EARLY_STOPPING_PATIENCE:-30}"
MCFT_EARLY_MIN_EPOCHS="${SUPERVISED_MCFT_EARLY_STOPPING_MIN_EPOCHS:-50}"
GPU_MAX_UTIL="${JMDA_GPU_MAX_UTIL_PERCENT:-10}"
GPU_STABLE_CHECKS="${JMDA_GPU_STABLE_CHECKS:-3}"
GPU_POLL_SECONDS="${JMDA_GPU_POLL_SECONDS:-30}"
GPU_CANDIDATES="${JMDA_GPU_CANDIDATES:-}"
CCSA_MIN_FREE_MIB="${JMDA_CCSA_MIN_FREE_MIB:-7168}"
DSNE_MIN_FREE_MIB="${JMDA_DSNE_MIN_FREE_MIB:-12288}"
MCFT_MIN_FREE_MIB="${JMDA_MCFT_MIN_FREE_MIB:-4608}"
CCSA_LABELS_PER_CLASS="${SDA_CCSA_LABELS_PER_CLASS:-0}"
DSNE_LABELS_PER_CLASS="${SDA_DSNE_LABELS_PER_CLASS:-0}"
SDA_SUBSET_SEED=42
PROTOCOL_AUDIT_SHOTS=10
CCSA_BATCH_SIZE="${SDA_CCSA_BATCH_SIZE:-256}"
DSNE_BATCH_SIZE="${SDA_DSNE_BATCH_SIZE:-32}"
MCFT_BATCH_SIZE="${SUPERVISED_MCFT_BATCH_SIZE:-64}"
CCSA_EXTRA=(--batch-size "$CCSA_BATCH_SIZE" --source-epochs "$CCSA_SOURCE_EPOCHS" --early-stopping-patience "$CCSA_EARLY_PATIENCE" --early-stopping-min-epochs "$CCSA_EARLY_MIN_EPOCHS")
DSNE_EXTRA=(--batch-size "$DSNE_BATCH_SIZE" --source-epochs "$DSNE_SOURCE_EPOCHS" --nlayers "$DSNE_LAYERS" --fn --early-stopping-patience "$DSNE_EARLY_PATIENCE" --early-stopping-min-epochs "$DSNE_EARLY_MIN_EPOCHS")
MCFT_EXTRA=(--batch-size "$MCFT_BATCH_SIZE" --eval-frequency "$MCFT_EVAL_FREQUENCY" --early-stopping-patience "$MCFT_EARLY_PATIENCE" --early-stopping-min-epochs "$MCFT_EARLY_MIN_EPOCHS")
if [ "$MODE" = "smoke" ]; then
  CCSA_EPOCHS=1
  DSNE_EPOCHS=1
  MCFT_EPOCHS=1
  CCSA_SOURCE_EPOCHS=1
  DSNE_SOURCE_EPOCHS=1
  NUM_WORKERS=0
  CCSA_BATCH_SIZE=2
  DSNE_BATCH_SIZE=2
  MCFT_BATCH_SIZE=2
  DSNE_LAYERS=18
  MCFT_EVAL_FREQUENCY=1
  CCSA_EARLY_PATIENCE=0
  DSNE_EARLY_PATIENCE=0
  MCFT_EARLY_PATIENCE=0
  CCSA_EXTRA=(--batch-size "$CCSA_BATCH_SIZE" --source-batch-size "$CCSA_BATCH_SIZE" --micro-batch-size 1 --source-epochs "$CCSA_SOURCE_EPOCHS" --max-train-batches 1 --no-pretrained --early-stopping-patience 0)
  DSNE_EXTRA=(--batch-size "$DSNE_BATCH_SIZE" --source-batch-size "$DSNE_BATCH_SIZE" --source-epochs "$DSNE_SOURCE_EPOCHS" --max-train-batches 1 --nlayers "$DSNE_LAYERS" --fn --no-pretrained --early-stopping-patience 0)
  MCFT_EXTRA=(--batch-size "$MCFT_BATCH_SIZE" --max-train-batches 1 --eval-frequency "$MCFT_EVAL_FREQUENCY" --early-stopping-patience 0)
fi

for value_name in CCSA_EPOCHS DSNE_EPOCHS MCFT_EPOCHS NUM_WORKERS \
  CCSA_BATCH_SIZE DSNE_BATCH_SIZE MCFT_BATCH_SIZE MCFT_EVAL_FREQUENCY \
  CCSA_LABELS_PER_CLASS DSNE_LABELS_PER_CLASS \
  CCSA_EARLY_PATIENCE CCSA_EARLY_MIN_EPOCHS \
  DSNE_EARLY_PATIENCE DSNE_EARLY_MIN_EPOCHS \
  MCFT_EARLY_PATIENCE MCFT_EARLY_MIN_EPOCHS \
  GPU_MAX_UTIL GPU_STABLE_CHECKS CCSA_MIN_FREE_MIB DSNE_MIN_FREE_MIB \
  MCFT_MIN_FREE_MIB; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name 必须是非负整数: $value"
    exit 1
  fi
done
if [ "$CCSA_EPOCHS" -eq 0 ] || [ "$DSNE_EPOCHS" -eq 0 ] || [ "$MCFT_EPOCHS" -eq 0 ]; then
  echo "训练轮数必须大于 0。"
  exit 1
fi
if [ "$CCSA_LABELS_PER_CLASS" -eq 0 ]; then
  CCSA_TARGET_LABEL_POLICY="all"
else
  CCSA_TARGET_LABEL_POLICY="${CCSA_LABELS_PER_CLASS}-shot seed${SDA_SUBSET_SEED}"
fi
if [ "$DSNE_LABELS_PER_CLASS" -eq 0 ]; then
  DSNE_TARGET_LABEL_POLICY="all"
else
  DSNE_TARGET_LABEL_POLICY="${DSNE_LABELS_PER_CLASS}-shot seed${SDA_SUBSET_SEED}"
fi
if [ "$MODE" = "formal" ] && \
   { { [ "$CCSA_ORDER_COUNT" -gt 0 ] && [ "$CCSA_LABELS_PER_CLASS" -ne 0 ]; } \
     || { [ "$DSNE_ORDER_COUNT" -gt 0 ] && [ "$DSNE_LABELS_PER_CLASS" -ne 0 ]; }; }; then
  echo "正式对比要求 CCSA、d-SNE、MCFT 使用全部 target_train 标签；请将两个 *_LABELS_PER_CLASS 设为 0。"
  exit 1
fi
if [ "$CCSA_BATCH_SIZE" -eq 0 ] || [ "$DSNE_BATCH_SIZE" -eq 0 ] \
  || [ "$MCFT_BATCH_SIZE" -eq 0 ] || [ "$MCFT_EVAL_FREQUENCY" -eq 0 ]; then
  echo "批大小和 MCFT 评估间隔必须大于 0。"
  exit 1
fi

if ! mkdir -p "$PAIRED_LIST_DIR" "$LOG_DIR" "$ARTIFACT_DIR" \
  "$EFFICIENCY_DIR" "$RESULTS_DIR" "$SUMMARY_DIR"; then
  echo "无法创建实验输出目录: $RUN_ROOT"
  exit 1
fi
export JMDA_DATA_ROOT="$DATA_ROOT_CANONICAL"
export JMDA_EFFICIENCY_OUTPUT_DIR="$EFFICIENCY_DIR"

{
  echo "suite_id=$SUITE_ID"
  echo "mode=$MODE"
  echo "data_root=$DATA_ROOT_CANONICAL"
  echo "targets=${TARGETS[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "method_order=${METHOD_ORDER[*]}"
  echo "source_domain=晴天"
  echo "input_modalities=VIS+IR for CCSA,d-SNE,MCFT"
  echo "target_label_policy=target-known classes filter sunny source"
  echo "ccsa_target_labels=$CCSA_TARGET_LABEL_POLICY"
  echo "dsne_target_labels=$DSNE_TARGET_LABEL_POLICY"
  echo "ccsa_epochs=$CCSA_EPOCHS"
  echo "ccsa_source_epochs=$CCSA_SOURCE_EPOCHS"
  echo "ccsa_batch_size=$CCSA_BATCH_SIZE"
  echo "dsne_epochs=$DSNE_EPOCHS"
  echo "dsne_source_epochs=$DSNE_SOURCE_EPOCHS"
  echo "dsne_batch_size=$DSNE_BATCH_SIZE"
  echo "dsne_layers=$DSNE_LAYERS"
  echo "mcft_epochs=$MCFT_EPOCHS"
  echo "mcft_batch_size=$MCFT_BATCH_SIZE"
  echo "early_stopping=CCSA:${CCSA_EARLY_MIN_EPOCHS}/${CCSA_EARLY_PATIENCE},dSNE:${DSNE_EARLY_MIN_EPOCHS}/${DSNE_EARLY_PATIENCE},MCFT:${MCFT_EARLY_MIN_EPOCHS}/${MCFT_EARLY_PATIENCE}"
  echo "gpu_gate=max_util:${GPU_MAX_UTIL}%,stable_checks:${GPU_STABLE_CHECKS},poll_seconds:${GPU_POLL_SECONDS},candidates:${GPU_CANDIDATES:-all}"
  echo "gpu_min_free_mib=CCSA:${CCSA_MIN_FREE_MIB},dSNE:${DSNE_MIN_FREE_MIB},MCFT:${MCFT_MIN_FREE_MIB}"
} > "$RUN_ROOT/run_config.txt"

echo "实验目录: $RUN_ROOT"
echo "数据根: $DATA_ROOT_CANONICAL"
echo "目标域: ${TARGETS[*]}"
echo "随机种子: ${SEEDS[*]}"
echo "为 CCSA、d-SNE、MCFT 生成同一套精确 VIS/IR 配对清单..."
if ! "$PYTHON_BIN" "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT_CANONICAL" \
  --output-dir "$PAIRED_LIST_DIR" \
  --targets "${TARGETS[@]}" \
  --modality paired \
  --sda-shots "$PROTOCOL_AUDIT_SHOTS" \
  --sda-seed "$SDA_SUBSET_SEED"; then
  echo "清单生成失败；尚未启动训练。"
  exit 1
fi

PREFLIGHT_COMMON=(
  --protocol sda
  --data-root "$DATA_ROOT_CANONICAL"
  --jmda-net-dir "$HERE/JMDA-Net"
  --targets "${TARGETS[@]}"
)
if [ "${JMDA_REQUIRE_CUDA:-1}" = "1" ]; then
  PREFLIGHT_COMMON+=(--require-cuda)
fi
if [ "${JMDA_CHECK_ALL_PATHS:-1}" = "1" ]; then
  PREFLIGHT_COMMON+=(--check-all-paths)
fi
if [ -n "${JMDA_MIN_FREE_GPU_MEMORY_MIB:-}" ]; then
  PREFLIGHT_COMMON+=(--min-free-gpu-memory-mib "$JMDA_MIN_FREE_GPU_MEMORY_MIB")
fi
echo "运行数据、类别空间、路径和 CUDA 预检..."
if ! "$PYTHON_BIN" "$HERE/tools/preflight_server.py" "${PREFLIGHT_COMMON[@]}" \
  --list-dir "$PAIRED_LIST_DIR" --required-modality paired; then
  echo "预检失败；尚未启动训练。"
  exit 1
fi

FAILURES=0

run_one() {
  local method="$1"
  local protocol="$2"
  local target="$3"
  local seed="$4"
  local epochs="$5"
  local modalities="$6"
  shift 6
  local tag="${method}_${target}"
  local log_path="$LOG_DIR/${method}_${target}_seed${seed}.log"
  local started finished elapsed exit_code status
  local min_free_mib
  case "$method" in
    ccsa) min_free_mib="$CCSA_MIN_FREE_MIB" ;;
    dsne) min_free_mib="$DSNE_MIN_FREE_MIB" ;;
    mcft) min_free_mib="$MCFT_MIN_FREE_MIB" ;;
    *) echo "未知 GPU 门控方法: $method"; return 1 ;;
  esac
  local gpu_args=(--min-free-mib "$min_free_mib"
    --max-util-percent "$GPU_MAX_UTIL"
    --stable-checks "$GPU_STABLE_CHECKS"
    --poll-seconds "$GPU_POLL_SECONDS" --tag "$tag")
  if [ -n "$GPU_CANDIDATES" ]; then
    gpu_args+=(--gpus "$GPU_CANDIDATES")
  fi

  echo ">>> [$method] target=$target seed=$seed epochs=$epochs (GPU gate: free>=${min_free_mib} MiB, util<=${GPU_MAX_UTIL}%)"
  started="$(date +%s)"
  exit_code=0
  status="success"
  if "$PYTHON_BIN" "$HERE/tools/run_on_available_gpu.py" \
      "${gpu_args[@]}" -- "$@" > "$log_path" 2>&1; then
    echo "    完成；日志: $log_path"
  else
    exit_code=$?
    status="failed"
    FAILURES=$((FAILURES + 1))
    echo "    失败；日志: $log_path"
    echo "    ---- error tail ----"
    tail -n 30 "$log_path" || true
    echo "    --------------------"
  fi
  finished="$(date +%s)"
  elapsed=$((finished - started))

  local budget_args=()
  if [ "$method" = "dsne" ] && [ "$status" = "success" ]; then
    budget_args=(--budget-results "$ARTIFACT_DIR/dsne_${target}_seed${seed}_budget_results.json")
  fi
  if ! "$PYTHON_BIN" "$HERE/tools/collect_run_result.py" \
    --log "$log_path" \
    --suite-id "$SUITE_ID" \
    --results-dir "$RESULTS_DIR" \
    --protocol "$protocol" \
    --run-tag "$tag" \
    --status "$status" \
    --exit-code "$exit_code" \
    --duration-seconds "$elapsed" \
    --epochs "$epochs" \
    --seed "$seed" \
    --efficiency-dir "$EFFICIENCY_DIR" \
    --input-modalities "$modalities" \
    --label-space-relation "partial-set-target-known" \
    --model-selection "final_checkpoint" \
    "${budget_args[@]}" \
    >> "$log_path" 2>&1; then
    echo "    结果收集失败: $log_path"
    FAILURES=$((FAILURES + 1))
  fi
}

COMMON_PAIRED_ARGS=(
  --data-dir "$PAIRED_LIST_DIR"
  --save-dir "$ARTIFACT_DIR"
  --efficiency_output_dir "$EFFICIENCY_DIR"
  --num-workers "$NUM_WORKERS"
)

CCSA_CHECKPOINT_ARGS=()
DSNE_CHECKPOINT_ARGS=()
if [ -n "${CCSA_SOURCE_CHECKPOINT:-}" ] && [ -n "${CCSA_PRETRAINED_BACKBONE:-}" ]; then
  echo "CCSA_SOURCE_CHECKPOINT 与 CCSA_PRETRAINED_BACKBONE 只能指定一个。"
  exit 1
fi
if [ -n "${DSNE_SOURCE_CHECKPOINT:-}" ] && [ -n "${DSNE_BACKBONE_CHECKPOINT:-}" ]; then
  echo "DSNE_SOURCE_CHECKPOINT 与 DSNE_BACKBONE_CHECKPOINT 只能指定一个。"
  exit 1
fi
if [ -n "${CCSA_SOURCE_CHECKPOINT:-}" ]; then
  CCSA_CHECKPOINT_ARGS=(--source-checkpoint "$CCSA_SOURCE_CHECKPOINT")
elif [ -n "${CCSA_PRETRAINED_BACKBONE:-}" ]; then
  CCSA_CHECKPOINT_ARGS=(--pretrained-backbone "$CCSA_PRETRAINED_BACKBONE")
fi
if [ -n "${DSNE_SOURCE_CHECKPOINT:-}" ]; then
  DSNE_CHECKPOINT_ARGS=(--source-checkpoint "$DSNE_SOURCE_CHECKPOINT")
elif [ -n "${DSNE_BACKBONE_CHECKPOINT:-}" ]; then
  DSNE_CHECKPOINT_ARGS=(--pretrained-backbone "$DSNE_BACKBONE_CHECKPOINT")
fi

# Method-major order is configurable so the largest model can claim an idle GPU first.
for method in "${METHOD_ORDER[@]}"; do
  for target in "${TARGETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      case "$method" in
        ccsa)
          run_one "ccsa" "sda" "$target" "$seed" "$CCSA_EPOCHS" "VIS+IR" \
            "$PYTHON_BIN" "$HERE/CCSA/train_ccsa.py" \
            --target "$target" --epochs "$CCSA_EPOCHS" --seed "$seed" \
            --target-labels-per-class "$CCSA_LABELS_PER_CLASS" \
            --sda-seed "$SDA_SUBSET_SEED" \
            "${COMMON_PAIRED_ARGS[@]}" "${CCSA_EXTRA[@]}" \
            "${CCSA_CHECKPOINT_ARGS[@]}"
          ;;
        dsne)
          run_one "dsne" "sda" "$target" "$seed" "$DSNE_EPOCHS" "VIS+IR" \
            "$PYTHON_BIN" "$HERE/d-SNE/train_dsne.py" \
            --target "$target" --epochs "$DSNE_EPOCHS" --seed "$seed" \
            --target-labels-per-class "$DSNE_LABELS_PER_CLASS" \
            --sda-seed "$SDA_SUBSET_SEED" \
            "${COMMON_PAIRED_ARGS[@]}" "${DSNE_EXTRA[@]}" \
            "${DSNE_CHECKPOINT_ARGS[@]}"
          ;;
        mcft)
          run_one "mcft" "supervised" "$target" "$seed" "$MCFT_EPOCHS" "VIS+IR" \
            "$PYTHON_BIN" "$HERE/MCFT/train_mcft.py" \
            --target "$target" --epochs "$MCFT_EPOCHS" --seed "$seed" \
            "${COMMON_PAIRED_ARGS[@]}" "${MCFT_EXTRA[@]}"
          ;;
      esac
    done
  done
done

COMPLEXITY_ARGS=()
if [ "$MODE" = "smoke" ]; then
  COMPLEXITY_ARGS=(--smoke)
fi
if [ "${JMDA_RUN_COMPLEXITY_BENCHMARK:-1}" = "1" ]; then
  echo ">>> [complexity] current CCSA/d-SNE/MCFT implementations"
  COMPLEXITY_GPU_ARGS=(--min-free-mib "$DSNE_MIN_FREE_MIB"
    --max-util-percent "$GPU_MAX_UTIL"
    --stable-checks "$GPU_STABLE_CHECKS"
    --poll-seconds "$GPU_POLL_SECONDS" --tag complexity)
  if [ -n "$GPU_CANDIDATES" ]; then
    COMPLEXITY_GPU_ARGS+=(--gpus "$GPU_CANDIDATES")
  fi
  if ! "$PYTHON_BIN" "$HERE/tools/run_on_available_gpu.py" \
    "${COMPLEXITY_GPU_ARGS[@]}" -- \
    "$PYTHON_BIN" "$HERE/tools/benchmark_efficiency.py" \
      --methods ccsa,dsne,mcft --protocol benchmark \
      --benchmark_train_batch_size 1 \
      --efficiency_output_dir "$EFFICIENCY_DIR/microbenchmark" \
      "${COMPLEXITY_ARGS[@]}" \
      > "$LOG_DIR/complexity.log" 2>&1; then
    echo "复杂度基准失败；日志: $LOG_DIR/complexity.log"
    FAILURES=$((FAILURES + 1))
  fi
fi

RUNS_CSV="$RESULTS_DIR/runs.csv"
if [ -f "$RUNS_CSV" ]; then
  if ! "$PYTHON_BIN" "$HERE/tools/summarize_run_results.py" \
    --runs-csv "$RUNS_CSV" --suite-id "$SUITE_ID" \
    --protocol sda --output-dir "$SUMMARY_DIR"; then
    echo "警告: SDA 汇总未生成（可能所有 SDA 运行均失败）。"
  fi
  if ! "$PYTHON_BIN" "$HERE/tools/summarize_run_results.py" \
    --runs-csv "$RUNS_CSV" --suite-id "$SUITE_ID" \
    --protocol supervised --output-dir "$SUMMARY_DIR"; then
    echo "警告: MCFT 汇总未生成（可能所有 MCFT 运行均失败）。"
  fi
fi

echo "训练结束。"
echo "逐次结果: $RUNS_CSV"
echo "汇总结果: $SUMMARY_DIR"
echo "逐轮指标与模型: $ARTIFACT_DIR"
echo "原始日志: $LOG_DIR"
if [ "$FAILURES" -ne 0 ]; then
  echo "共有 $FAILURES 个训练或结果收集步骤失败。"
  exit 1
fi
echo "所有 CCSA、d-SNE、MCFT 任务均成功。"

