"""Read one target's independently prepared original-label support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--target",
        required=True,
        choices=("backlight", "rainy", "foggy", "dark"),
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    support = payload.get("target_class_support", {}).get(args.target, {})
    labels = support.get("original_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(
            f"Missing target_class_support.{args.target}.original_labels in "
            f"{args.manifest}."
        )
    labels = sorted({int(label) for label in labels})
    compact_map = support.get("compact_label_map")
    expected_map = {str(label): index for index, label in enumerate(labels)}
    if compact_map != expected_map:
        raise ValueError(
            f"Invalid compact label map for {args.target}: "
            f"expected {expected_map}, found {compact_map}."
        )
    print(",".join(map(str, labels)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
