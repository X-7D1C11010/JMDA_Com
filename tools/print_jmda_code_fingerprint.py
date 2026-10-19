"""Print the portable fingerprint of the JMDA training implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .audit_jmda_uda_log import compute_training_code_fingerprint
except ImportError:
    from audit_jmda_uda_log import compute_training_code_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jmda-net-dir", required=True, type=Path)
    args = parser.parse_args()
    print(compute_training_code_fingerprint(args.jmda_net_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
