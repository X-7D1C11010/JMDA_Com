# Sourced by run_server.sh. Pin both the CLI settings and the matching
# trainer revisions, because current trainers apply different multimodal defaults.

declare -A HISTORICAL_UDA_ENTRY

prepare_historical_trainer() {
  local variant="$1" method="$2" revision="$3" entry="$4" evidence="$5"
  shift 5
  local snapshot="$FAIR_LIST_DIR/historical_code/$RUN_ID/$method"
  mkdir -p "$snapshot" || return 1
  if ! git -C "$HERE" archive "$revision" -- "$entry" tools "$@" |
      tar -xf - -C "$snapshot"; then
    echo "Unable to prepare $method revision $revision. Ensure the server checkout contains its Git history." >&2
    return 1
  fi
  HISTORICAL_UDA_ENTRY["$method"]="$snapshot/$entry"
  "$PYTHON" -B "$HERE/tools/prepare_historical_uda_variant.py" \
    "$snapshot/$entry" "$method" "$variant" || return 1
  echo "[$variant] $method revision=$revision reference_run=$evidence"
}

require_historical_resnet() {
  local weight_id="$1" filename="$2"
  if [ ! -f "$TORCH_HOME/hub/checkpoints/$filename" ]; then
    echo "Missing ImageNet weights: $TORCH_HOME/hub/checkpoints/$filename" >&2
    echo "Populate that cache or run tools/download_official_uda_weights.py --only $weight_id for the default cache." >&2
    return 1
  fi
}

historical_method_enabled() {
  local variant="$1" method="$2"
  uda_method_enabled "$method" || return 1
  case "$variant:$method" in
    dann_capped:dann|dann_ir_pretrain:dann|coral_lr5:deepcoral|sampling:deepcoral|cdan_source_only:cdan|cdan_grl_cap:cdan|mdd_ramp8:mdd|mdd_ramp4:mdd|mcc_source_only:mcc|mcc_joint_zero:mcc|mcc_stronger:mcc|mcc_stronger_bn:mcc|ir_pretrain:dann|ir_pretrain:mdd|ir_pretrain:mcc) return 0 ;;
    dann_capped:*|dann_ir_pretrain:*|coral_lr5:*|sampling:*|cdan_source_only:*|cdan_grl_cap:*|mdd_ramp8:*|mdd_ramp4:*|mcc_source_only:*|mcc_joint_zero:*|mcc_stronger:*|mcc_stronger_bn:*|ir_pretrain:*) return 1 ;;
  esac
  return 0
}

prepare_historical_uda() {
  local variant="$1"
  export JMDA_DETERMINISTIC=1
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  HISTORICAL_UDA_ENTRY=()

  if historical_method_enabled "$variant" dann || historical_method_enabled "$variant" mdd ||
      historical_method_enabled "$variant" mcc; then
    require_historical_resnet resnet18 resnet18-f37072fd.pth || return 1
  fi
  if historical_method_enabled "$variant" deepcoral || historical_method_enabled "$variant" cdan; then
    require_historical_resnet resnet50 resnet50-0676ba61.pth || return 1
  fi

  if historical_method_enabled "$variant" dann; then
    prepare_historical_trainer "$variant" dann 0813a41c3b08f90a0d415a461bdf17279f414aab \
      DANN/train_dann.py 20261107_012813_2448535 DANN/models || return 1
  fi
  if historical_method_enabled "$variant" deepcoral; then
    prepare_historical_trainer "$variant" deepcoral a40910e56fc9fd5bfee80c56d06683116f5a8b81 \
      DeepCORAL/train_deepcoral.py 20261104_134252 DeepCORAL/models.py || return 1
  fi
  if historical_method_enabled "$variant" cdan; then
    prepare_historical_trainer "$variant" cdan 6c37ef71727540cce43cf7e8c60651acefaf2dc3 \
      CDAN/train_cdan.py 20261104_095733 CDAN/pytorch || return 1
  fi
  if historical_method_enabled "$variant" mdd; then
    prepare_historical_trainer "$variant" mdd d933f67bfaf88c6ab950ef8056dfc1f30fdcb2d1 \
      MDD/train_mdd.py 20261105_101246_1939778 Transfer-Learning-Library/tllib || return 1
  fi
  if historical_method_enabled "$variant" mcc; then
    prepare_historical_trainer "$variant" mcc d9db6baa1e88d58ee5e6ca2630e97b5b67f474cb \
      MCC/train_mcc.py 20261105_114303_2771708 Transfer-Learning-Library/tllib || return 1
  fi
}

run_historical_uda_target() {
  local variant="$1" target="$2"
  local coral_head_mult=10 cdan_updates=1200 cdan_weight=1 mdd_weight=0.45 mdd_ramp=12
  local mcc_weight=0.02 mcc_ramp=10 mcc_warmup=2
  if [ "$variant" = "tuned" ]; then
    coral_head_mult=5
    cdan_updates=1400
    mdd_weight=0.30
    mdd_ramp=8
    mcc_weight=0.006
    mcc_ramp=3
  fi
  case "$variant" in
    coral_lr5|sampling) coral_head_mult=5 ;;
    bn_probe) coral_head_mult=5; mdd_weight=0.30; mdd_ramp=8 ;;
    weighted|audit) coral_head_mult=5; mdd_weight=0.30; mdd_ramp=8; mcc_weight=0.20 ;;
    mcc_stronger|mcc_stronger_bn) mcc_weight=0.20 ;;
    cdan_source_only) cdan_weight=0 ;;
    mdd_ramp8) mdd_weight=0.30; mdd_ramp=8 ;;
    mdd_ramp4) mdd_weight=0.30; mdd_ramp=4 ;;
    mcc_source_only) mcc_weight=0; mcc_warmup=14 ;;
    mcc_joint_zero) mcc_weight=0 ;;
  esac
  local EFF_DIR="$EFF_DIR/$SUITE_ID/$target"
  mkdir -p "$EFF_DIR" || return 1
  local -a common=(
    --target "$target" --seed "$SEED" --data-dir "$FAIR_LIST_DIR"
    --efficiency_output_dir "$EFF_DIR" --save-dir "$CHECKPOINT_ROOT"
    --multimodal --enhanced-augmentation --no-save-model
    --eval-num-workers 0 --prefetch-factor 4 --pin-memory --persistent-workers
  )
  if historical_method_enabled "$variant" dann; then
    run "dann_$target" --record-epochs 36 "$PYTHON" -B -u "${HISTORICAL_UDA_ENTRY[dann]}" \
      "${common[@]}" --num-workers 8 --backbone jmda --batch-size 16 \
      --epochs 36 --lr 0.0003 --domain-loss-weight 0.35 --grl-max-iterations 5000 \
      --lr-scheduler constant --min-lr-ratio 0.1 --target-eval-interval 1
  fi
  if historical_method_enabled "$variant" deepcoral; then
    run "deepcoral_$target" --record-epochs 30 "$PYTHON" -B -u "${HISTORICAL_UDA_ENTRY[deepcoral]}" \
      "${common[@]}" --num-workers 2 --backbone resnet50 \
      --source-batch-size 16 --target-batch-size 16 --test-batch-size 16 \
      --epochs 30 --lr 0.0005 --head-lr-mult "$coral_head_mult" --coral-max-weight 0.3 --coral-warmup-epochs 0 \
      --weight-decay 0.0005 --label-smoothing 0.05 --lr-scheduler constant \
      --min-lr-ratio 0.05 --target-eval-interval 1
  fi
  if historical_method_enabled "$variant" cdan; then
    run "cdan_$target" --record-epochs 0 "$PYTHON" -B -u "${HISTORICAL_UDA_ENTRY[cdan]}" \
      "${common[@]}" --num-workers 2 --backbone ResNet50 --batch-size 16 \
      --method CDAN+E --num-iterations "$cdan_updates" --test-interval 100 --lr 0.0005 \
      --head-lr-mult 3 --ad-lr-mult 10 --trade-off "$cdan_weight" --coeff-max-iter 10000 --coeff-alpha 10
  fi
  if historical_method_enabled "$variant" mdd; then
    run "mdd_$target" --record-epochs 36 "$PYTHON" -B -u "${HISTORICAL_UDA_ENTRY[mdd]}" \
      "${common[@]}" --num-workers 8 --backbone jmda --batch-size 16 \
      --epochs 36 --iters-per-epoch 80 --lr 0.0025 --trade-off "$mdd_weight" --trade-off-warmup-epochs "$mdd_ramp"
  fi
  if historical_method_enabled "$variant" mcc; then
    run "mcc_$target" --record-epochs 14 "$PYTHON" -B -u "${HISTORICAL_UDA_ENTRY[mcc]}" \
      "${common[@]}" --num-workers 8 --backbone jmda --batch-size 16 \
      --epochs 14 --iters-per-epoch 100 --lr 0.002 --trade-off "$mcc_weight" \
      --source-warmup-epochs "$mcc_warmup" --trade-off-warmup-epochs "$mcc_ramp" --label-smoothing 0.10
  fi
  "$PYTHON" -B "$HERE/tools/efficiency_common.py" "$EFF_DIR" || return 1
  if [ -f "$EFF_DIR/efficiency_summary.csv" ]; then
    echo "Efficiency summary: $EFF_DIR/efficiency_summary.csv"
  fi
}

run_historical_uda_suite() {
  local requested="$1" phase target method SEED
  local base_run_id="$RUN_ID" base_suite_id="$SUITE_ID"
  local RUN_ID="$base_run_id" SUITE_ID="$base_suite_id"
  local seeds_csv="${UDA_HISTORICAL_SEEDS:-${JMDA_SEED:-42}}"
  if [[ ! "$seeds_csv" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Invalid historical UDA seed list: $seeds_csv" >&2
    return 1
  fi
  local -a seeds
  IFS=, read -r -a seeds <<< "$seeds_csv"
  local -a phases=("$requested")
  case "$requested" in
    compare) phases=(historical_best mcc_source_only tuned) ;;
    diagnose) phases=(coral_lr5 sampling historical_best cdan_source_only cdan_grl_cap
                      mcc_source_only mcc_joint_zero dann_capped dann_ir_pretrain mdd_ramp8 mdd_ramp4) ;;
    focused) phases=(bn_probe mcc_stronger_bn) ;;
    historical_best|tuned|weighted|audit|bn_probe|mcc_stronger|mcc_stronger_bn|dann_capped|dann_ir_pretrain|coral_lr5|sampling|cdan_source_only|cdan_grl_cap|mdd_ramp8|mdd_ramp4|mcc_source_only|mcc_joint_zero|augmentation|ir_pretrain|ema_teacher) ;;
    *) echo "Unknown historical UDA variant: $requested" >&2; return 1 ;;
  esac
  for phase in "${phases[@]}"; do
    # The first MCC comparison must measure adaptation against source-only.
    # Run its new weight schedule separately after examining that comparison.
    local -A UDA_METHOD_ENABLED=()
    for method in "${UDA_METHOD_ARRAY[@]}"; do
      case "$requested:$phase:${method,,}" in
        compare:tuned:mcc|diagnose:historical_best:dann|diagnose:historical_best:deepcoral|diagnose:historical_best:mdd) continue ;;
      esac
      UDA_METHOD_ENABLED["${method,,}"]=1
    done
    RUN_ID="$base_run_id"
    SUITE_ID="$base_suite_id"
    if [ "${#phases[@]}" -gt 1 ]; then
      RUN_ID="${base_run_id}_$phase"
      SUITE_ID="${base_suite_id}_$phase"
    fi
    prepare_historical_uda "$phase" || return 1
    if [ "${#HISTORICAL_UDA_ENTRY[@]}" -eq 0 ]; then
      if [ "${#phases[@]}" -gt 1 ]; then continue; fi
      echo "No selected method supports $phase" >&2
      return 1
    fi
    local phase_run_id="$RUN_ID"
    for SEED in "${seeds[@]}"; do
      if [ "${#seeds[@]}" -gt 1 ]; then RUN_ID="${phase_run_id}_seed$SEED"; fi
      echo "Historical UDA variant=$phase suite=$SUITE_ID seed=$SEED"
      for target in "${TARGETS[@]}"; do
        run_historical_uda_target "$phase" "$target" || return 1
      done
    done
  done
}
