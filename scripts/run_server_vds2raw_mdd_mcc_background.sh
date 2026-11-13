#!/usr/bin/env bash
# The optimized VDS-only profile; never changes the weather experiment settings.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VDS2RAW_DA_METHODS="mdd mcc"
export VDS2RAW_DA_PROFILE=optimized
exec bash "$HERE/scripts/run_server_vds2raw_da_background.sh" "${1:-formal}"
