#!/usr/bin/env bash
# Repeat the validated VDS UDA settings for three methods and five fixed seeds.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
case "$MODE" in formal|smoke) ;; *) echo "usage: $0 {formal|smoke}" >&2; exit 1 ;; esac

export VDS2RAW_DA_METHODS="mdd dann deepcoral"
export VDS2RAW_SEEDS="42 43 44 45 46"
# The optimized profile changes MDD/MCC only. DANN and Deep CORAL retain
# their validated baseline settings, including AIS LR multiplier 1.
export VDS2RAW_DA_PROFILE=optimized
export VDS2RAW_DA_BATCH_SIZE=16
export VDS2RAW_NUM_WORKERS=2
export JMDA_DETERMINISTIC=1
# Ignore a stale global epoch override: formal defaults must be MDD=250,
# DANN=100 and Deep CORAL=100. The smoke entry still explicitly uses 2.
unset VDS2RAW_DA_EPOCHS

echo "VDS2Raw UDA: MDD, DANN, Deep CORAL; seeds=42 43 44 45 46; 15 sequential runs."
echo "Formal budgets: MDD=250 epochs; DANN=100; Deep CORAL=100; batch/domain=16."
echo "Per-run held-out metrics and across-seed mean/std are recorded in summary/."
exec bash "$HERE/scripts/run_server_vds2raw_da_background.sh" "$MODE"
