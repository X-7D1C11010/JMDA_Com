"""Build an immutable runtime copy from a Git checkout; no ZIP or training needed."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

PACKAGES = ("tools", "vds2raw", "weather", "Transfer-Learning-Library/tllib")
EXTRA_FILES = (
    "DeepCORAL/models.py", "CDAN/pytorch/loss.py",
    "scripts/run_pending_adaptation_v2.sh", "scripts/run_vds2raw_da_methods.sh",
    "scripts/run_weather_vds_core_methods.sh",
)
REQUIRED_FILES = EXTRA_FILES + (
    "vds2raw/__init__.py", "vds2raw/adaptation.py", "vds2raw/train_adaptation.py",
    "vds2raw/dataset.py", "vds2raw/common.py", "vds2raw/summarize_adaptation.py",
    "weather/__init__.py", "weather/train_vds_core.py", "weather/summarize_vds_core.py",
    "tools/run_on_available_gpu.py", "tools/prepare_paired_protocol_lists.py",
    "tools/uda_multimodal.py", "tools/uda_data.py", "tools/data_paths.py",
    "tools/reproducibility.py", "tools/efficiency_common.py", "tools/classification_metrics.py",
    "tools/audit_vds2raw_inputs.py",
    "Transfer-Learning-Library/tllib/alignment/mdd.py",
    "Transfer-Learning-Library/tllib/self_training/mcc.py",
)
MANIFEST = "runtime_snapshot_manifest.json"


def _within(path, parent):
    return path.resolve().is_relative_to(parent.resolve())


def collect_sources(project_root):
    project_root = Path(project_root).resolve()
    missing = [name for name in REQUIRED_FILES if not (project_root/name).is_file()]
    if missing:
        raise FileNotFoundError("Git checkout is incomplete; sync these files: "+", ".join(missing))
    paths = {project_root/name for name in EXTRA_FILES}
    for package in PACKAGES:
        paths.update((project_root/package).rglob("*.py"))
    payload = {}
    for path in sorted(paths):
        if not _within(path, project_root):
            raise ValueError(f"Source file is outside the project: {path}")
        payload[path.relative_to(project_root).as_posix()] = path.read_bytes().replace(b"\r\n", b"\n")
    return payload


def _verified(directory, hashes):
    if directory.is_symlink() or not directory.is_dir():
        return False
    try:
        manifest = json.loads((directory/MANIFEST).read_text(encoding="utf-8"))
        if manifest.get("files") != hashes:
            return False
        actual_code = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
                       if p.is_file() and p.suffix in (".py", ".sh")}
        if actual_code != set(hashes):
            return False
        return all(not (directory/name).is_symlink() and
                   hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected
                   for name, expected in hashes.items())
    except (OSError, ValueError):
        return False


def prepare_snapshot(project_root):
    project_root = Path(project_root).resolve()
    payload = collect_sources(project_root)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payload.items())}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    base = project_root/"experiment_updates"
    if not _within(base, project_root):
        raise ValueError(f"Snapshot destination is outside the project: {base}")
    base.mkdir(exist_ok=True)
    stem = "mcc_weather_v2_"+fingerprint[:16]
    for candidate in sorted(base.glob(stem+"*")):
        if _verified(candidate, hashes):
            print(f"[SNAPSHOT] verified existing code copy: {candidate}", file=sys.stderr)
            return candidate.resolve()

    staging = Path(tempfile.mkdtemp(prefix=".prepare_mcc_weather_", dir=base))
    try:
        for name, data in payload.items():
            path = staging/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (staging/MANIFEST).write_text(json.dumps({
            "format": 1, "source": "Git checkout", "project_root": str(project_root),
            "code_sha256": fingerprint, "files": hashes,
            "training_started": False, "weights_and_results_copied": False,
        }, indent=2), encoding="utf-8")
        if not _verified(staging, hashes):
            raise RuntimeError("Runtime snapshot verification failed")
        destination = base/stem
        for _ in range(4):
            if destination.exists():
                if _verified(destination, hashes):
                    return destination.resolve()
                destination = base/(stem+"_"+uuid.uuid4().hex[:8])
            # Both resolved targets are checked before moving the directory.
            if not _within(staging, base) or not _within(destination, base):
                raise ValueError("Snapshot move escaped experiment_updates")
            try:
                staging.rename(destination)
                print(f"[SNAPSHOT] created {len(payload)} code files: {destination}", file=sys.stderr)
                return destination.resolve()
            except OSError:
                if not destination.exists():
                    raise
        raise RuntimeError("Could not publish runtime snapshot after concurrent updates")
    finally:
        if staging.exists():
            # This is only the temporary directory created above, within base.
            if not _within(staging, base) or not staging.name.startswith(".prepare_mcc_weather_"):
                raise ValueError("Refusing to remove an unexpected temporary path")
            shutil.rmtree(staging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    print(prepare_snapshot(parser.parse_args().project_root))
