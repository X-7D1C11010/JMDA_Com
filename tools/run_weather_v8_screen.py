"""Reproducible local seed42 candidate queues; not a formal weather release."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_mcc_weather_v2_snapshot import prepare_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("linear", "fda", "domain", "zoom"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--weights", type=Path, default=Path.home()/".cache/torch/hub/checkpoints/convnext_small-0c510722.pth")
    args = p.parse_args()
    root = args.project_root.resolve()
    data = root/"runs/weather_local_screen/balanced_v4_seed42_20261114_005133/prepared_exact_pairs"
    if not data.is_dir() or not args.weights.is_file():
        raise FileNotFoundError("Prepared paired lists and ConvNeXt-Small weights are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"candidate_plan.json").exists():
        raise FileExistsError("Use a new output directory; previous trials must remain immutable")
    snapshot = prepare_snapshot(root)
    common = [sys.executable, "-B", "-u", str(snapshot/"weather/train_vds_core.py"),
              "--data-dir", str(data), "--weights", str(args.weights.resolve()),
              "--image-backbone", "convnext_small", "--feature-normalization", "unit",
              "--epochs", "60", "--batch-size", "16", "--seed", "42",
              "--weight-decay", ".002", "--gate-lr-multiplier", ".01", "--gate-weight-decay", "0",
              "--source-class-balance", "--adaptation-warmup-epochs", "3", "--adaptation-ramp-epochs", "10",
              "--num-workers", "2", "--cpu-threads", "2", "--cache-evaluation",
              "--modality-diagnostics", "--log-feature-moments", "--early-stopping-patience", "0"]
    if args.suite == "linear":
        variants = [("mcc_linear", "mcc", ".02"), ("mdd_linear", "mdd", ".2")]
        commands = [(name, common+["--method", method, "--target", "backlight", "--lr", ".1",
                                  "--trade-off", weight, "--classifier-layout", "linear",
                                  "--encoder-training-scope", "frozen", "--initial-vis-weight", ".85",
                                  "--stop-after-epochs", "20", "--save-dir", str(output/name)])
                    for name, method, weight in variants]
    elif args.suite == "fda":
        commands = [(name, common+["--method", "deepcoral", "--target", "dark", "--lr", ".01",
                                  "--trade-off", ".05", "--encoder-training-scope", "last_stages",
                                  "--image-lr-override", ".0001", "--initial-vis-weight", ".4",
                                  "--fda-beta", beta, "--fda-probability", ".5",
                                  "--stop-after-epochs", "30", "--save-dir", str(output/name)])
                    for name, beta in (("deepcoral_last_stages_clean", "0"), ("deepcoral_last_stages_fda01", ".01"))]
    elif args.suite == "domain":
        # Dark Deep CORAL's measured image DA/CE gradient ratio was ~.01.
        # Probe a feature-domain adversary with a stronger relative signal,
        # retaining the source classifier scale established by the linear test.
        name = "dann_last_stages_fda01"
        commands = [(name, common+["--method", "dann", "--target", "dark", "--lr", ".1",
                                   "--optimizer", "sgd", "--classifier-layout", "linear",
                                   "--trade-off", ".5", "--encoder-training-scope", "last_stages",
                                   "--image-lr-override", ".0003", "--initial-vis-weight", ".4",
                                   "--fda-beta", ".01", "--fda-probability", ".5",
                                   "--stop-after-epochs", "30", "--save-dir", str(output/name)])]
    else:
        commands = [(name, common+["--method", "deepcoral", "--target", "dark", "--lr", ".01",
                                  "--trade-off", ".05", "--encoder-training-scope", "last_stages",
                                  "--image-lr-override", ".0001", "--initial-vis-weight", ".4",
                                  "--source-zoom-min-area", ".25", "--source-zoom-probability", ".75",
                                  "--fda-beta", beta, "--fda-probability", ".5",
                                  "--stop-after-epochs", "30", "--save-dir", str(output/name)])
                    for name, beta in (("deepcoral_centerzoom_clean", "0"), ("deepcoral_centerzoom_fda01", ".01"))]
    plan = {"suite": args.suite, "seed": 42, "snapshot": str(snapshot),
            "target_training_labels_used": False, "formal_release": False,
            "candidates": [{"name": name, "argv": argv} for name, argv in commands]}
    (output/"candidate_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JMDA_DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for name, argv in commands:
        print("START="+name, flush=True)
        with (output/(name+".log")).open("x", encoding="utf-8") as stream:
            subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("COMPLETE="+name, flush=True)


if __name__ == "__main__":
    main()
