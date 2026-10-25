"""Command-line VDS2Raw integrity audit and train-only statistics builder."""

import argparse
import json
import sys
from pathlib import Path

import timm
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vds2raw.dataset import audit_dataset, load_or_compute_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--report-file", required=True)
    parser.add_argument(
        "--sampled-check",
        action="store_true",
        help="Skip per-NPZ schema checking (formal runs should not use this).",
    )
    args = parser.parse_args()

    audit = audit_dataset(args.data_root, check_every_sample=not args.sampled_check)
    statistics = load_or_compute_statistics(args.data_root, args.stats_file)
    payload = {
        "audit": audit,
        "normalization": statistics.to_dict(),
        "runtime": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    report_path = Path(args.report_file).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] VDS2Raw protocol audit: {report_path}")


if __name__ == "__main__":
    main()
