"""Validate seven isolated CUDA smoke flows; never trains on real datasets."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_mcc_weather_v2_snapshot import prepare_snapshot
from tools.test_vds2raw_adaptation import fixture as vds_fixture
from tools.test_weather_vds_core import fixture as weather_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT/"results"/f"adaptation_v4_synthetic_{time.strftime('%Y%m%d_%H%M%S')}"))
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = prepare_snapshot(ROOT)
    weights = ROOT/"pretrained_weights/torch/hub/checkpoints/resnet18-f37072fd.pth"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    environment = {**os.environ, "JMDA_DATA_ROOT": "", "JMDA_DETERMINISTIC": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    completed = []
    with tempfile.TemporaryDirectory(prefix="adaptation_v4_fixture_") as temporary:
        temporary = Path(temporary)
        vds = temporary/"vds"
        vds.mkdir()
        vds_fixture(vds)
        # A correct target loader must work without an NPZ label field.
        import numpy as np
        for path in (vds/"samples/target_train").rglob("*.npz"):
            with np.load(path) as item:
                contents = {key: item[key] for key in item.files if key != "label"}
            np.savez(path, **contents)
        weather_root = temporary/"weather"
        weather_root.mkdir()
        lists = weather_fixture(weather_root)
        jobs = [("vds", "mcc"), ("vds", "cdan")]+[("weather", method) for method in ("dann", "deepcoral", "cdan", "mdd", "mcc")]
        for kind, method in jobs:
            tag = f"{kind}_{method}"
            destination = temporary/tag
            command = [sys.executable, "-B", "-u", str(snapshot/("vds2raw/train_adaptation.py" if kind == "vds" else "weather/train_vds_core.py")),
                       "--method", method, "--save-dir", str(destination), "--weights", str(weights),
                       "--epochs", "4", "--batch-size", "4", "--num-workers", "2", "--cpu-threads", "2",
                       "--device", "cuda", "--max-train-batches", "1", "--adaptation-warmup-epochs", "0", "--adaptation-ramp-epochs", "1"]
            if kind == "vds":
                command += ["--data-root", str(vds), "--skip-audit", "--ais-fields", "heading_size",
                            "--profile", "mcc_v3" if method == "mcc" else "optimized", "--modality-diagnostics"]
                directory = destination/f"{method}_seed42"
            else:
                command += ["--data-dir", str(lists), "--target", "backlight", "--weather-profile", "balanced_v4",
                            "--freeze-bn-after-epochs", "3"]
                directory = destination/f"{method}_backlight_seed42"
            with (output/f"{tag}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=snapshot, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Synthetic {tag} failed; inspect {output/tag}.log")
            record = json.loads((directory/"result.json").read_text())
            complete = json.loads((directory/"completeness.json").read_text())
            assert record["smoke"] and complete["status"] == "PASS"
            assert record["modality_diagnostics"]["inference_independent_of_label_and_path"]
            if kind == "vds":
                assert complete["ais_dimensions"] == complete["ais_mask_dimensions"] == 4
                assert complete["test_evaluations"] == 1
            else:
                evaluations = [json.loads(s) for s in (directory/"evaluation.jsonl").read_text().splitlines()]
                assert all(set(row["auxiliary_branches"]) == {"VIS", "IR"} for row in evaluations)
                policy = [json.loads(s) for s in (directory/"normalization_policy.jsonl").read_text().splitlines()]
                assert policy[-1]["scope"] == "encoders" and policy[-1]["BN_running_stats_frozen"]
            for filename in ("result.json", "run_config.json", "completeness.json", "modality_diagnostics.json"):
                shutil.copyfile(directory/filename, output/f"{tag}_{filename}")
            completed.append({"kind": kind, "method": method, "status": "PASS", "synthetic": True})
            print(f"PASS synthetic {tag}", flush=True)
    (output/"validation.json").write_text(json.dumps({"status": "PASS", "snapshot": str(snapshot),
        "real_dataset_training_started": False, "accuracy_claim": False, "checks": completed}, indent=2)+"\n")
    print(f"Seven synthetic CUDA checks passed: {output}", flush=True)


if __name__ == "__main__":
    main()
