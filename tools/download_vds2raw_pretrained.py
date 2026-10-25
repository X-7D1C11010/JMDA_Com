"""Download and export portable ImageNet backbones for offline servers."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import certifi
import requests
import timm
import torch
import torchvision
from safetensors.torch import load_file
from tqdm import tqdm

VGG16_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
VGG16_SHA256_PREFIX = "397923af"
DSNE_COMMIT = "155e705ee1fa3b6d63e84880b7aadd1a6e45df21"
DSNE_URL = (
    "https://huggingface.co/timm/resnetv2_101.a1h_in1k/resolve/"
    f"{DSNE_COMMIT}/model.safetensors"
)
DSNE_SHA256 = "2622f2e318304f35c9401e7e989635c302d76e07ea5b4a49c2ee6b2093bdddad"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url, destination, expected_sha256):
    """Download with certifi instead of the potentially broken Windows store."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination).startswith(expected_sha256):
        print(f"Using verified download: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(30, 120),
        allow_redirects=True,
        verify=certifi.where(),
    ) as response:
        response.raise_for_status()
        resumed = offset > 0 and response.status_code == 206
        if not resumed:
            offset = 0
        total = int(response.headers.get("content-length", 0)) + offset
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as stream, tqdm(
            total=total or None,
            initial=offset,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination.name,
        ) as progress:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    stream.write(block)
                    progress.update(len(block))

    actual = sha256(partial)
    if not actual.startswith(expected_sha256):
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    partial.replace(destination)
    print(f"Verified download SHA-256: {actual}")
    return destination


def save_payload(path, method, model_name, state_dict):
    torch.save(
        {
            "format": "vds2raw-portable-backbone-v1",
            "method": method,
            "model_name": model_name,
            "state_dict": state_dict,
        },
        path,
    )
    return {
        "method": method,
        "model_name": model_name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export VDS2Raw CCSA/d-SNE ImageNet backbones"
    )
    parser.add_argument("--output-dir", default="pretrained_weights")
    parser.add_argument(
        "--method", choices=("all", "ccsa", "dsne"), default="all"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []

    if args.method in ("all", "ccsa"):
        path = output_dir / "ccsa_vgg16_imagenet1k_v1_backbone.pth"
        if not path.exists():
            download = download_file(
                VGG16_URL,
                output_dir / ".download_cache" / "vgg16-397923af.pth",
                VGG16_SHA256_PREFIX,
            )
            state = torch.load(download, map_location="cpu", weights_only=True)
            feature_state = {
                key.removeprefix("features."): value
                for key, value in state.items()
                if key.startswith("features.")
            }
            if not feature_state:
                raise RuntimeError("downloaded VGG-16 file contains no feature tensors")
            save_payload(
                path,
                "CCSA",
                "torchvision/vgg16/IMAGENET1K_V1/features",
                feature_state,
            )
            del state, feature_state
            gc.collect()
            download.unlink(missing_ok=True)
        files.append(
            {
                "method": "CCSA",
                "model_name": "torchvision/vgg16/IMAGENET1K_V1/features",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    if args.method in ("all", "dsne"):
        path = output_dir / "dsne_resnetv2_101_a1h_in1k_backbone.pth"
        if not path.exists():
            download = download_file(
                DSNE_URL,
                output_dir / ".download_cache" / "dsne_resnetv2_101.safetensors",
                DSNE_SHA256,
            )
            downloaded_state = load_file(str(download), device="cpu")
            model = timm.create_model(
                "resnetv2_101.a1h_in1k",
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
            target_state = model.state_dict()
            compatible = {
                key: value
                for key, value in downloaded_state.items()
                if key in target_state and value.shape == target_state[key].shape
            }
            missing = sorted(set(target_state) - set(compatible))
            if missing:
                raise RuntimeError(
                    "downloaded d-SNE backbone is missing tensors: "
                    + ", ".join(missing[:5])
                )
            model.load_state_dict(compatible, strict=True)
            save_payload(
                path,
                "d-SNE",
                f"timm/resnetv2_101.a1h_in1k@{DSNE_COMMIT}",
                model.state_dict(),
            )
            del downloaded_state, compatible, target_state, model
            gc.collect()
            download.unlink(missing_ok=True)
        files.append(
            {
                "method": "d-SNE",
                "model_name": f"timm/resnetv2_101.a1h_in1k@{DSNE_COMMIT}",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "runtime": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
        },
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[OK] portable pretrained backbones: {manifest_path}")


if __name__ == "__main__":
    main()
