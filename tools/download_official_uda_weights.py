"""Download all additional weights needed by the UDA baselines.

Run this on a machine with fast internet, then upload the whole output folder
to the server repository's ``pretrained_weights`` directory.
"""

import argparse
import hashlib
import json
import os
import shutil
import ssl
import tempfile
import urllib.request


WEIGHTS = (
    {
        "id": "resnet50",
        "method": "Enhanced DANN/DeepCORAL/CDAN/MDD/MCC (ResNet-50)",
        "url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "relative_path": os.path.join(
            "torch", "hub", "checkpoints", "resnet50-0676ba61.pth"
        ),
        "sha256_prefix": "0676ba61",
    },
    {
        "id": "resnet18",
        "method": "CDAN/MDD/MCC",
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "relative_path": os.path.join(
            "torch", "hub", "checkpoints", "resnet18-f37072fd.pth"
        ),
        "sha256_prefix": "f37072fd",
    },
    {
        "id": "alexnet",
        "method": "DeepCORAL",
        "url": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
        "relative_path": "deepcoral_alexnet_owt_4df8aa71.pth",
        "sha256_prefix": "4df8aa71",
    },
)

OPTIONAL_WEIGHTS = ({
    "id": "convnext_small",
    "method": "Qualified weather ConvNeXt-Small candidates",
    "url": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
    "relative_path": os.path.join("torch", "hub", "checkpoints", "convnext_small-0c510722.pth"),
    "sha256_prefix": "0c510722",
},)

WEIGHTS_BY_ID = {spec["id"]: spec for spec in WEIGHTS+OPTIONAL_WEIGHTS}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path, spec):
    if not os.path.isfile(path):
        return False
    digest = sha256(path)
    if "sha256" in spec:
        return digest == spec["sha256"]
    return digest.startswith(spec["sha256_prefix"])


def cached_weight_candidates(spec):
    """Yield standard torch-hub cache paths that may already hold a weight."""
    filename = os.path.basename(spec["url"])
    roots = []
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        roots.append(os.path.expanduser(torch_home))
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "torch"))
    seen = set()
    for root in roots:
        candidate = os.path.abspath(os.path.join(root, "hub", "checkpoints", filename))
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def certifi_ssl_context():
    """Use certifi directly, bypassing malformed Windows certificate stores."""
    try:
        import certifi
    except ImportError as exc:
        raise RuntimeError(
            "HTTPS certificate setup failed and certifi is not installed. "
            "Install it with: python -m pip install certifi"
        ) from exc
    return ssl.create_default_context(cafile=certifi.where())


def download_https(url, destination):
    request = urllib.request.Request(
        url, headers={"User-Agent": "JMDA-official-weight-downloader/1.0"}
    )
    context = certifi_ssl_context()
    with urllib.request.urlopen(request, context=context, timeout=120) as response, \
            open(destination, "wb") as output:
        total = int(response.headers.get("Content-Length", "0") or 0)
        received = 0
        next_report = 64 * 1024 * 1024
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            received += len(chunk)
            if received >= next_report:
                if total:
                    print(f"  received {received / 1024**2:.0f}/"
                          f"{total / 1024**2:.0f} MiB")
                else:
                    print(f"  received {received / 1024**2:.0f} MiB")
                next_report += 64 * 1024 * 1024


def download(spec, output_dir):
    destination = os.path.join(output_dir, spec["relative_path"])
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if is_valid(destination, spec):
        print(f"[OK cached] {spec['method']}: {destination}")
        return destination

    for cached in cached_weight_candidates(spec):
        if is_valid(cached, spec):
            shutil.copy2(cached, destination)
            print(f"[OK copied] {spec['method']}: {cached} -> {destination}")
            return destination

    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(destination) + ".", suffix=".part",
        dir=os.path.dirname(destination),
    )
    os.close(fd)
    try:
        print(f"[DOWNLOAD] {spec['method']}: {spec['url']}")
        download_https(spec["url"], temporary)
        if not is_valid(temporary, spec):
            raise RuntimeError(
                f"SHA-256 verification failed for {spec['method']}: {sha256(temporary)}"
            )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    print(f"[OK] {spec['method']}: {destination}")
    return destination


def ensure_weight(weight_id, output_dir):
    """Return a verified local official checkpoint, downloading if missing."""
    try:
        spec = WEIGHTS_BY_ID[weight_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown weight id {weight_id!r}; expected one of "
            f"{sorted(WEIGHTS_BY_ID)}"
        ) from exc
    return download(spec, os.path.abspath(output_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "pretrained_weights"),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(WEIGHTS_BY_ID),
        help=("download only the named checkpoints; the official single-modal "
              "suite needs: --only resnet50 alexnet"),
    )
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    records = []
    selected = (WEIGHTS if not args.only
                else tuple(WEIGHTS_BY_ID[name] for name in args.only))
    for spec in selected:
        path = download(spec, output_dir)
        records.append({
            "method": spec["method"],
            "path": os.path.relpath(path, output_dir),
            "sha256": sha256(path),
            "url": spec["url"],
        })
    manifest_path = os.path.join(output_dir, "official_uda_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"weights": records}, handle, ensure_ascii=False, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
