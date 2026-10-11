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
#
# 本仓库 JMDA-Net/ 子目录为本文方法（JMDA-Net）的源码镜像，与对比方法目录区分。
# 每次运行的日志 -> server_logs/<protocol>/，效率结果 -> efficiency_results/<protocol>/
# =============================================================================
set -uo pipefail

PROTOCOL="${1:-}"
EPOCHS="${2:-100}"
case "$PROTOCOL" in
  sda|jmda_uda|uda|vlm) ;;
  *) echo "用法: bash scripts/run_server.sh {sda|jmda_uda|uda|vlm} [epochs]"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${JMDA_DATA_ROOT:-/mnt/data/lixiang/data/Data}"
PYTHON="${PYTHON:-python}"
JMDA_NET_DIR="${JMDA_NET_DIR:-$HERE/JMDA-Net}"
SEED=42

LOG_ROOT="$HERE/server_logs/$PROTOCOL"
EFF_DIR="$HERE/efficiency_results/$PROTOCOL"
RESULTS_DIR="$HERE/results"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
export JMDA_DATA_ROOT="$DATA_ROOT"
export JMDA_EFFICIENCY_OUTPUT_DIR="$EFF_DIR"
mkdir -p "$LOG_ROOT" "$EFF_DIR" "$RESULTS_DIR"

TARGETS=(backlight rainy foggy dark)

if [ ! -d "$DATA_ROOT" ]; then
  echo "数据根不存在: $DATA_ROOT（可用 JMDA_DATA_ROOT 指定）"; exit 1
fi
echo "协议=$PROTOCOL 轮数=$EPOCHS 数据根=$DATA_ROOT"
echo "日志目录=$LOG_ROOT"

run() {  # run "<log-tag>" <cmd...>
  local tag="$1"; shift
  local log_path="$LOG_ROOT/${tag}_${RUN_ID}.log"
  local started finished elapsed exit_code status record_protocol
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
  fi
  finished="$(date +%s)"
  elapsed=$((finished - started))
  record_protocol="$PROTOCOL"
  [ "$record_protocol" = "jmda_uda" ] && record_protocol="uda"
  "$PYTHON" "$HERE/tools/collect_run_result.py" \
    --log "$log_path" \
    --results-dir "$RESULTS_DIR" \
    --protocol "$record_protocol" \
    --run-tag "$tag" \
    --status "$status" \
    --exit-code "$exit_code" \
    --duration-seconds "$elapsed" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --efficiency-dir "$EFF_DIR" \
    >> "$log_path" 2>&1 || echo "    WARN: result collection failed for $tag"
}

# 通用参数（各方法脚本一致）
COMMON=(--epochs "$EPOCHS" --seed "$SEED" --efficiency_output_dir "$EFF_DIR")

if [ "$PROTOCOL" = "sda" ]; then
  # ---- SDA：所有方法使用同一源标签 + 同一目标标注子集（共享 list）----
  for t in "${TARGETS[@]}"; do
    run "source_only_$t"  "$PYTHON" "$HERE/SDA_Baselines/train_source_only_sda.py" --target "$t" "${COMMON[@]}"
    run "joint_$t"        "$PYTHON" "$HERE/SDA_Baselines/train_joint_training.py"  --target "$t" "${COMMON[@]}"
    run "target_only_$t"  "$PYTHON" "$HERE/SDA_Baselines/train_target_only.py"    --target "$t" "${COMMON[@]}"
    run "dann_$t"         "$PYTHON" "$HERE/DANN/train_dann_sda.py"             --target "$t" "${COMMON[@]}"
    run "deepcoral_$t"    "$PYTHON" "$HERE/DeepCORAL/train_deepcoral_sda.py"   --target "$t" "${COMMON[@]}"
    run "cdan_$t"         "$PYTHON" "$HERE/CDAN/train_cdan_sda.py"             --target "$t" "${COMMON[@]}"
    run "mdd_$t"          "$PYTHON" "$HERE/MDD/train_mdd_sda.py"               --target "$t" "${COMMON[@]}"
    run "mcc_$t"          "$PYTHON" "$HERE/MCC/train_mcc_sda.py"               --target "$t" "${COMMON[@]}"
    run "modfinity_$t"    "$PYTHON" "$HERE/Modfinity/train_modfinity_sda.py"   --target "$t" "${COMMON[@]}"
    run "boomda_$t"       "$PYTHON" "$HERE/Boomda/train_boomda_sda.py"         --target "$t" "${COMMON[@]}"
    # JMDA-Net-SDA（本文方法，仓库内 JMDA-Net/ 目录）
    run "jmda_net_sda_$t" bash -c \
      "cd '$JMDA_NET_DIR' && JMDA_SDA_LIST_DIR='$HERE/data_lists' '$PYTHON' main_sda.py --target '$t' --epochs '$EPOCHS' --sda-seed '$SEED'"
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
         --report_strategy last"
  done

elif [ "$PROTOCOL" = "uda" ]; then
  # ---- 其他 UDA 对比方法 ----
  for t in "${TARGETS[@]}"; do
    run "dann_$t"         "$PYTHON" "$HERE/DANN/train_dann.py"             --target "$t" "${COMMON[@]}"
    run "deepcoral_$t"    "$PYTHON" "$HERE/DeepCORAL/train_deepcoral.py"   --target "$t" "${COMMON[@]}"
    run "cdan_$t"         "$PYTHON" "$HERE/CDAN/train_cdan.py"             --target "$t" "${COMMON[@]}"
    run "mdd_$t"          "$PYTHON" "$HERE/MDD/train_mdd.py"               --target "$t" "${COMMON[@]}"
    run "mcc_$t"          "$PYTHON" "$HERE/MCC/train_mcc.py"               --target "$t" "${COMMON[@]}"
    run "modfinity_$t"    "$PYTHON" "$HERE/Modfinity/train_modfinity.py"   --target "$t" "${COMMON[@]}"
    run "boomda_$t"       "$PYTHON" "$HERE/Boomda/train_boomda_jmda.py"    --target "$t" "${COMMON[@]}"
    run "transadapter_$t" "$PYTHON" "$HERE/TransAdapter/train_transadapter.py" --target "$t" "${COMMON[@]}"
    run "adclip_$t"       "$PYTHON" "$HERE/AD-CLIP/train_adclip.py"         --target "$t" --epochs "$EPOCHS" --seed "$SEED"
  done

else
  # ---- VLM 独立表：CoOp 不是 UDA 方法 ----
  for t in "${TARGETS[@]}"; do
    run "coop_$t" "$PYTHON" "$HERE/CoOp/train_coop.py" --target "$t" --epochs "$EPOCHS" --seed "$SEED"
  done
fi

echo "全部任务结束（$PROTOCOL）。失败项见上方 FAILED 行。"
