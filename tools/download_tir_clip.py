"""Download and structurally verify the official CLIP ViT-B/16 archive."""

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TIR.clip.clip import _MODELS, _download
from TIR.vds2raw_adapter import load_clip_checkpoint_strict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "pretrained_weights" / "clip"),
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    path = Path(_download(_MODELS["ViT-B/16"], str(output_dir))).resolve()
    model = load_clip_checkpoint_strict(path)
    del model
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    expected = _MODELS["ViT-B/16"].split("/")[-2]
    if digest != expected:
        raise RuntimeError(f"SHA-256 mismatch: expected {expected}, got {digest}")
    print(f"[OK] {path}")
    print(f"SHA256={digest}")


if __name__ == "__main__":
    main()
