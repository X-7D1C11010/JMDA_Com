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
VIS_LIST_DIR="$LIST_ROOT/vis"
PAIRED_LIST_DIR="$LIST_ROOT/paired"
LOG_DIR="$RUN_ROOT/logs"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
EFFICIENCY_DIR="$RUN_ROOT/efficiency"
RESULTS_DIR="$RUN_ROOT/results"
SUMMARY_DIR="$RUN_ROOT/summary"

read -r -a TARGETS <<< "${JMDA_TARGETS:-backlight rainy foggy dark}"
read -r -a SEEDS <<< "${JMDA_SEEDS:-42}"
VALID_TARGETS=" backlight rainy foggy dark "
for target in "${TARGETS[@]}"; do
  if [[ "$VALID_TARGETS" != *" $target "* ]]; then
    echo "未知目标域: $target（允许: backlight rainy foggy dark）"
    exit 1
  fi
done
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
SDA_LABELS_PER_CLASS=10
SDA_SUBSET_SEED=42
CCSA_BATCH_SIZE="${SDA_CCSA_BATCH_SIZE:-256}"
DSNE_BATCH_SIZE="${SDA_DSNE_BATCH_SIZE:-32}"
MCFT_BATCH_SIZE="${SUPERVISED_MCFT_BATCH_SIZE:-64}"
CCSA_EXTRA=(--batch-size "$CCSA_BATCH_SIZE" --source-epochs "$CCSA_SOURCE_EPOCHS")
DSNE_EXTRA=(--batch-size "$DSNE_BATCH_SIZE" --source-epochs "$DSNE_SOURCE_EPOCHS" --nlayers "$DSNE_LAYERS" --fn)
MCFT_EXTRA=(--batch-size "$MCFT_BATCH_SIZE" --eval-frequency "$MCFT_EVAL_FREQUENCY")
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
  CCSA_EXTRA=(--batch-size "$CCSA_BATCH_SIZE" --source-batch-size "$CCSA_BATCH_SIZE" --micro-batch-size 1 --source-epochs "$CCSA_SOURCE_EPOCHS" --max-train-batches 1 --no-pretrained)
  DSNE_EXTRA=(--batch-size "$DSNE_BATCH_SIZE" --source-batch-size "$DSNE_BATCH_SIZE" --source-epochs "$DSNE_SOURCE_EPOCHS" --max-train-batches 1 --nlayers "$DSNE_LAYERS" --fn --no-pretrained)
  MCFT_EXTRA=(--batch-size "$MCFT_BATCH_SIZE" --max-train-batches 1 --eval-frequency "$MCFT_EVAL_FREQUENCY")
fi

for value_name in CCSA_EPOCHS DSNE_EPOCHS MCFT_EPOCHS NUM_WORKERS \
  CCSA_BATCH_SIZE DSNE_BATCH_SIZE MCFT_BATCH_SIZE MCFT_EVAL_FREQUENCY; do
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
if [ "$CCSA_BATCH_SIZE" -eq 0 ] || [ "$DSNE_BATCH_SIZE" -eq 0 ] \
  || [ "$MCFT_BATCH_SIZE" -eq 0 ] || [ "$MCFT_EVAL_FREQUENCY" -eq 0 ]; then
  echo "批大小和 MCFT 评估间隔必须大于 0。"
  exit 1
fi

if ! mkdir -p "$VIS_LIST_DIR" "$PAIRED_LIST_DIR" "$LOG_DIR" "$ARTIFACT_DIR" \
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
  echo "source_domain=晴天"
  echo "target_label_policy=target-known classes filter sunny source"
  echo "sda_subset=10-shot seed42"
  echo "ccsa_epochs=$CCSA_EPOCHS"
  echo "ccsa_source_epochs=$CCSA_SOURCE_EPOCHS"
  echo "ccsa_batch_size=$CCSA_BATCH_SIZE"
  echo "dsne_epochs=$DSNE_EPOCHS"
  echo "dsne_source_epochs=$DSNE_SOURCE_EPOCHS"
  echo "dsne_batch_size=$DSNE_BATCH_SIZE"
  echo "dsne_layers=$DSNE_LAYERS"
  echo "mcft_epochs=$MCFT_EPOCHS"
  echo "mcft_batch_size=$MCFT_BATCH_SIZE"
} > "$RUN_ROOT/run_config.txt"

echo "实验目录: $RUN_ROOT"
echo "数据根: $DATA_ROOT_CANONICAL"
echo "目标域: ${TARGETS[*]}"
echo "随机种子: ${SEEDS[*]}"
echo "生成单模态完整 VIS 清单（CCSA/d-SNE）和精确 VIS/IR 配对清单（MCFT）..."
if ! "$PYTHON_BIN" "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT_CANONICAL" \
  --output-dir "$VIS_LIST_DIR" \
  --targets "${TARGETS[@]}" \
  --modality vis \
  --sda-shots "$SDA_LABELS_PER_CLASS" \
  --sda-seed "$SDA_SUBSET_SEED"; then
  echo "VIS 清单生成失败；尚未启动训练。"
  exit 1
fi
if ! "$PYTHON_BIN" "$HERE/tools/prepare_paired_protocol_lists.py" \
  --data-root "$DATA_ROOT_CANONICAL" \
  --output-dir "$PAIRED_LIST_DIR" \
  --targets "${TARGETS[@]}" \
  --modality paired \
  --sda-shots "$SDA_LABELS_PER_CLASS" \
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
  --list-dir "$VIS_LIST_DIR" --required-modality vis; then
  echo "VIS 预检失败；尚未启动训练。"
  exit 1
fi
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

  echo ">>> [$method] target=$target seed=$seed epochs=$epochs"
  started="$(date +%s)"
  exit_code=0
  status="success"
  if "$@" > "$log_path" 2>&1; then
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

COMMON_VIS_ARGS=(
  --data-dir "$VIS_LIST_DIR"
  --save-dir "$ARTIFACT_DIR"
  --efficiency_output_dir "$EFFICIENCY_DIR"
  --num-workers "$NUM_WORKERS"
)
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

# Method-major order is intentional: CCSA -> d-SNE -> MCFT.
for target in "${TARGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "ccsa" "sda" "$target" "$seed" "$CCSA_EPOCHS" "VIS" \
      "$PYTHON_BIN" "$HERE/CCSA/train_ccsa.py" \
      --target "$target" --epochs "$CCSA_EPOCHS" --seed "$seed" \
      --target-labels-per-class "$SDA_LABELS_PER_CLASS" \
      --sda-seed "$SDA_SUBSET_SEED" \
      "${COMMON_VIS_ARGS[@]}" "${CCSA_EXTRA[@]}" \
      "${CCSA_CHECKPOINT_ARGS[@]}"
  done
done

for target in "${TARGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "dsne" "sda" "$target" "$seed" "$DSNE_EPOCHS" "VIS" \
      "$PYTHON_BIN" "$HERE/d-SNE/train_dsne.py" \
      --target "$target" --epochs "$DSNE_EPOCHS" --seed "$seed" \
      --target-labels-per-class "$SDA_LABELS_PER_CLASS" \
      --sda-seed "$SDA_SUBSET_SEED" \
      "${COMMON_VIS_ARGS[@]}" "${DSNE_EXTRA[@]}" \
      "${DSNE_CHECKPOINT_ARGS[@]}"
  done
done

for target in "${TARGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "mcft" "supervised" "$target" "$seed" "$MCFT_EPOCHS" "VIS+IR" \
      "$PYTHON_BIN" "$HERE/MCFT/train_mcft.py" \
      --target "$target" --epochs "$MCFT_EPOCHS" --seed "$seed" \
      "${COMMON_PAIRED_ARGS[@]}" "${MCFT_EXTRA[@]}"
  done
done

COMPLEXITY_ARGS=()
if [ "$MODE" = "smoke" ]; then
  COMPLEXITY_ARGS=(--smoke)
fi
if [ "${JMDA_RUN_COMPLEXITY_BENCHMARK:-1}" = "1" ]; then
  echo ">>> [complexity] current CCSA/d-SNE/MCFT implementations"
  if ! "$PYTHON_BIN" "$HERE/tools/benchmark_efficiency.py" \
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

