#!/usr/bin/env bash
# One queued sequence: MDD, DANN, Deep CORAL, regularized CDAN; five seeds each.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}" >&2; exit 1 ;; esac
export VDS2RAW_DA_METHODS="mdd dann deepcoral cdan"
export VDS2RAW_SEEDS="42 43 44 45 46"
export VDS2RAW_DA_PROFILE=optimized
export VDS2RAW_DA_BATCH_SIZE=16
export VDS2RAW_NUM_WORKERS=2
export JMDA_DETERMINISTIC=1
unset VDS2RAW_DA_EPOCHS
echo "VDS2Raw UDA: 4 methods x seeds 42..46 = 20 sequential runs."
echo "MDD=250 epochs; DANN/Deep CORAL=100; CDAN=max100 with validation-CE plateau LR/early stopping."
exec bash "$HERE/scripts/run_server_vds2raw_da_background.sh" "$MODE"
