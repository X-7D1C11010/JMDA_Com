#!/usr/bin/env bash
# Append CDAN's five-seed study without repeating the MDD/DANN/Deep CORAL jobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}" >&2; exit 1 ;; esac
export VDS2RAW_DA_METHODS=cdan
export VDS2RAW_SEEDS="42 43 44 45 46"
export VDS2RAW_DA_PROFILE=optimized
export VDS2RAW_DA_BATCH_SIZE=16
export VDS2RAW_NUM_WORKERS=2
export JMDA_DETERMINISTIC=1
unset VDS2RAW_DA_EPOCHS
echo "VDS2Raw CDAN+E UDA: seeds=42 43 44 45 46; five sequential runs."
echo "Formal budget: max 100 epochs/seed, batch/domain=16; CE plateau LR reduction and CE early stopping."
echo "Validation CE: LR x0.5 after 12 stale epochs (min scale .1); stop after 20 stale epochs, minimum 50 epochs."
echo "Best target_val ACC/F1 selects the state; target_test evaluated once after training."
exec bash "$HERE/scripts/run_server_vds2raw_da_background.sh" "$MODE"
