"""Read-only VDS input/protocol audit. Never trains or changes dataset files."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from vds2raw.dataset import audit_dataset, TRAIN_PARTITIONS


def audit_inputs(root):
    root = Path(root).resolve()
    audit = audit_dataset(root)
    schema = json.loads((root/"feature_schema.json").read_text(encoding="utf-8"))
    names = schema["ais_feature_names"]
    if (names[:4] != ["heading_sin", "heading_cos", "length_z", "width_z"]
            or len(names) != 13 or not all(name.startswith("nav_") for name in names[4:])):
        raise ValueError("Unexpected AIS fields; inspect the feature schema before training")
    if schema["train_partitions"] != list(TRAIN_PARTITIONS):
        raise ValueError("Normalization schema includes non-training partitions")
    images, ais_groups = defaultdict(list), defaultdict(list)
    for path in sorted((root/"samples").rglob("*.npz")):
        partition, label = path.relative_to(root/"samples").parts[:2]
        with np.load(path, allow_pickle=False) as sample:
            images[hashlib.sha256(sample["rgb"].tobytes()+sample["nir"].tobytes()).hexdigest()].append((partition, label))
            ais_groups[hashlib.sha256(sample["ais"].tobytes()+sample["ais_mask"].tobytes()).hexdigest()].append((partition, label))
    cross = lambda groups: [items for items in groups.values() if len({partition for partition, _ in items}) > 1]
    image_cross, ais_cross = cross(images), cross(ais_groups)
    if image_cross:
        raise ValueError(f"Found {len(image_cross)} exact RGB+NIR duplicate groups across partitions")
    audit.update(model_ais_feature_names=names, excluded_ais_proxy_features=[],
                 ais_mask_usage="13 validity values plus 13 gated features",
                 exact_image_duplicate_groups_across_partitions=0,
                 exact_AIS_duplicate_groups_across_partitions=len(ais_cross),
                 AIS_duplicate_groups_with_different_labels=sum(len({label for _, label in items}) > 1 for items in ais_cross),
                 AIS_duplicate_interpretation="repeated structured features alone do not establish sample leakage",
                 score_adjustment_performed=False, training_started=False)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_inputs(args.data_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("VDS_INPUT_AUDIT: "+json.dumps(report), flush=True)
