#!/usr/bin/env bash
# =============================================================================
# JMDA 服务器训练入口 —— SDA / UDA 两条协议严格分开，不混合；CoOp 单独 VLM。
#
# 用法:
#   bash scripts/run_server.sh sda [epochs]   # 监督域适应（论文主结果）
#   bash scripts/run_server.sh jmda_uda [epochs] # JMDA-Net 无监督
#   bash scripts/run_server.sh uda [epochs]   # 其他无监督域适应方法
#   bash scripts/run_server.sh vlm [epochs]   # CoOp（独立 VLM 表）
#
# 环境变量:
#   JMDA_DATA_ROOT  数据根目录（默认 /mnt/data/lixiang/data/Data）
#   JMDA_NET_DIR    JMDA-Net 源码目录（默认使用本仓库 JMDA-Net/ 镜像）
#   PYTHON          Python 解释器（默认 python）
#   JMDA_SEED       当前运行的随机种子（默认 42；正式多种子实验由外层循环传入）
#   JMDA_CLASS_NAMES_FILE  14类真实英文名称文件（AD-CLIP/CoOp 必需）
#   TRANSADAPTER_BACKBONE_WEIGHTS  固定的本地 Swin-T 预训练权重（UDA 必需）
#
# 本仓库 JMDA-Net/ 子目录为本文方法（JMDA-Net）的源码镜像，与对比方法目录区分。
# 每次运行的日志 -> server_logs/<protocol>/，效率结果 -> efficiency_results/<protocol>/
# =============================================================================
set -uo pipefail

PROTOCOL="${1:-}"
EPOCHS="${2:-100}"
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "epochs 必须是正整数: $EPOCHS"
  exit 1
fi
case "$PROTOCOL" in
  sda|jmda_uda|uda|vlm) ;;
  *) echo "用法: bash scripts/run_server.sh {sda|jmda_uda|uda|vlm} [epochs]"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
PYTHON="${PYTHON:-python}"
JMDA_NET_DIR="${JMDA_NET_DIR:-$HERE/JMDA-Net}"
SEED="${JMDA_SEED:-42}"

LOG_ROOT="$HERE/server_logs/$PROTOCOL"
EFF_DIR="$HERE/efficiency_results/$PROTOCOL"
RESULTS_DIR="$HERE/results"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
SUITE_ID="${JMDA_SUITE_ID:-$RUN_ID}"
CHECKPOINT_ROOT="$HERE/checkpoints/$PROTOCOL/$RUN_ID"
FAIR_LIST_DIR="$HERE/prepared_lists/$SUITE_ID"
CLASS_NAMES_FILE="${JMDA_CLASS_NAMES_FILE:-}"
TRANSADAPTER_WEIGHTS="${TRANSADAPTER_BACKBONE_WEIGHTS:-}"
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
echo "运行公共配置预检..."
if ! "$PYTHON" "$HERE/tools/preflight_server.py" "${PREFLIGHT_ARGS[@]}"; then
  echo "公共配置预检失败；尚未启动任何训练任务。"
  exit 1
fi

FAILURES=0

run() {  # run "<log-tag>" <cmd...>
  local tag="$1"; shift
  local log_path="$LOG_ROOT/${tag}_${RUN_ID}.log"
  local started finished elapsed exit_code status record_protocol
  local input_modalities model_selection
  echo ">>> [$tag]"
  started="$(date +%s)"
  exit_code=0
  status="success"
  if "$@" > "$log_path" 2>&1; then
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
  finished="$(date +%s)"
  elapsed=$((finished - started))
  record_protocol="$PROTOCOL"
  [ "$record_protocol" = "jmda_uda" ] && record_protocol="uda"
  case "$tag" in
    jmda_net_*|modfinity_*|boomda_*) input_modalities="VIS+IR" ;;
    *) input_modalities="VIS" ;;
  esac
  case "$record_protocol" in
    uda|vlm) model_selection="final_checkpoint" ;;
    *) model_selection="method_declared" ;;
  esac
  "$PYTHON" "$HERE/tools/collect_run_result.py" \
    --log "$log_path" \
    --suite-id "$SUITE_ID" \
    --results-dir "$RESULTS_DIR" \
    --protocol "$record_protocol" \
    --run-tag "$tag" \
    --status "$status" \
    --exit-code "$exit_code" \
    --duration-seconds "$elapsed" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --efficiency-dir "$EFF_DIR" \
    --input-modalities "$input_modalities" \
    --label-space-relation "partial-set" \
    --model-selection "$model_selection" \
    >> "$log_path" 2>&1 || echo "    WARN: result collection failed for $tag"
}

# 通用参数（各方法脚本一致）
COMMON=(--epochs "$EPOCHS" --seed "$SEED" --data-dir "$FAIR_LIST_DIR" --efficiency_output_dir "$EFF_DIR" --save-dir "$CHECKPOINT_ROOT")

if [ "$PROTOCOL" = "sda" ]; then
  # ---- SDA：所有方法使用同一源标签 + 同一目标标注子集（共享 list）----
  for t in "${TARGETS[@]}"; do
    run "dann_$t"         "$PYTHON" "$HERE/DANN/train_dann_sda.py"             --target "$t" "${COMMON[@]}"
    run "deepcoral_$t"    "$PYTHON" "$HERE/DeepCORAL/train_deepcoral_sda.py"   --target "$t" "${COMMON[@]}"
    run "cdan_$t"         "$PYTHON" "$HERE/CDAN/train_cdan_sda.py"             --target "$t" "${COMMON[@]}"
    run "mdd_$t"          "$PYTHON" "$HERE/MDD/train_mdd_sda.py"               --target "$t" "${COMMON[@]}"
    run "mcc_$t"          "$PYTHON" "$HERE/MCC/train_mcc_sda.py"               --target "$t" "${COMMON[@]}"
    run "modfinity_$t"    "$PYTHON" "$HERE/Modfinity/train_modfinity_sda.py"   --target "$t" "${COMMON[@]}"
    run "boomda_$t"       "$PYTHON" "$HERE/Boomda/train_boomda_sda.py"         --target "$t" "${COMMON[@]}"
    # JMDA-Net-SDA（本文方法，仓库内 JMDA-Net/ 目录）
    run "jmda_net_sda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && JMDA_SDA_LIST_DIR='$FAIR_LIST_DIR' '$PYTHON' main_sda.py --target '$t' --epochs '$EPOCHS' --num-iterations '${JMDA_NUM_ITERATIONS:-5}' --sda-seed '$SEED'"
  done

elif [ "$PROTOCOL" = "jmda_uda" ]; then
  # ---- JMDA-Net UDA：单独放在 SDA 与其他 UDA 方法之间运行 ----
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
         --epochs '$EPOCHS' --num_iterations '${JMDA_NUM_ITERATIONS:-5}' \
         --ablation_mode full --no_auto_ablation_hparams \
         --transport_mode sinkhorn \
         --target_label_ratio 0 --target_cls_weight 0 \
         --report_strategy last --base_seed '$SEED' \
         --checkpoint_dir '$CHECKPOINT_ROOT/jmda_net_uda_$t'"
  done

elif [ "$PROTOCOL" = "uda" ]; then
  # ---- 其他 UDA 对比方法 ----
  for t in "${TARGETS[@]}"; do
    run "dann_$t"         "$PYTHON" "$HERE/DANN/train_dann.py"             --target "$t" "${COMMON[@]}"
    run "deepcoral_$t"    "$PYTHON" "$HERE/DeepCORAL/train_deepcoral.py"   --target "$t" "${COMMON[@]}"
    run "cdan_$t"         "$PYTHON" "$HERE/CDAN/train_cdan.py"             --target "$t" "${COMMON[@]}"
    run "mdd_$t"          "$PYTHON" "$HERE/MDD/train_mdd.py"               --target "$t" "${COMMON[@]}"
    run "mcc_$t"          "$PYTHON" "$HERE/MCC/train_mcc.py"               --target "$t" "${COMMON[@]}"
    run "transadapter_$t" "$PYTHON" "$HERE/TransAdapter/train_transadapter.py" --target "$t" --backbone-weights "$TRANSADAPTER_WEIGHTS" "${COMMON[@]}"
    run "adclip_$t"       "$PYTHON" "$HERE/AD-CLIP/train_adclip.py"         --target "$t" --epochs "$EPOCHS" --seed "$SEED" --data-dir "$FAIR_LIST_DIR" --save-dir "$CHECKPOINT_ROOT" --class-names-file "$CLASS_NAMES_FILE"
    run "modfinity_$t"    "$PYTHON" "$HERE/Modfinity/train_modfinity.py"   --target "$t" "${COMMON[@]}"
    run "boomda_$t"       "$PYTHON" "$HERE/Boomda/train_boomda_jmda.py"    --target "$t" "${COMMON[@]}"
  done

else
  # ---- VLM 独立表：CoOp 不是 UDA 方法 ----
  for t in "${TARGETS[@]}"; do
    run "coop_$t" "$PYTHON" "$HERE/CoOp/train_coop.py" --target "$t" --epochs "$EPOCHS" --seed "$SEED" --data-dir "$FAIR_LIST_DIR" --save-dir "$CHECKPOINT_ROOT" --class-names-file "$CLASS_NAMES_FILE"
  done
fi

echo "阶段结束（$PROTOCOL）：失败任务数=$FAILURES。"
if [ "$FAILURES" -ne 0 ]; then
  exit 1
fi
echo "全部任务成功（$PROTOCOL）。"
