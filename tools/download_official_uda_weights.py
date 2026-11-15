"""Download all additional weights needed by the UDA baselines.

Run this on a machine with fast internet, then upload the whole output folder
to the server repository's ``pretrained_weights`` directory.
"""

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
import shutil
import ssl
import time
import urllib.error
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
}, {
    "id": "unip_tiny",
    "method": "Rainy CDAN infrared UNIP-tiny",
    "url": "https://huggingface.co/casiatao/UNIP/resolve/main/unip_vit_tiny_by_ibot_large.pth",
    "relative_path": os.path.join("unip", "unip_vit_tiny_by_ibot_large.pth"),
    "sha256": "d5ebe70319a1d5a5fd7c2ed4fdfc6518f34c1c2020c25a57175fcf4191805f59",
})

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


def download_https(url, destination, timeout=120):
    offset = os.path.getsize(destination) if os.path.isfile(destination) else 0
    headers = {"User-Agent": "JMDA-official-weight-downloader/2.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(
        url, headers=headers
    )
    context = certifi_ssl_context()
    with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
        partial = response.getcode() == 206
        if partial and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise OSError("Server returned an inconsistent Content-Range; refusing to append")
        # If Range is ignored (HTTP 200), restart instead of appending a whole
        # file to the partial one. Never publish before the final hash check.
        received = offset if partial else 0
        length = int(response.headers.get("Content-Length", "0") or 0)
        total = received + length if length else 0
        print(f"  connected; resume={received / 1024**2:.1f} MiB", flush=True)
        next_report, last_report = received + 8 * 1024**2, time.monotonic()
        with open(destination, "ab" if partial else "wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                now = time.monotonic()
                if received >= next_report or now-last_report >= 5:
                    suffix = f"/{total / 1024**2:.1f}" if total else ""
                    print(f"  received {received / 1024**2:.1f}{suffix} MiB", flush=True)
                    next_report, last_report = received + 8 * 1024**2, now
        if total and received != total:
            raise OSError(f"Incomplete download: received {received}/{total} bytes")


def require_weight(weight_id, output_dir):
    """Offline preflight: validate a file without downloading or importing torch."""
    spec = WEIGHTS_BY_ID[weight_id]
    path = os.path.abspath(os.path.join(output_dir, spec["relative_path"]))
    if not is_valid(path, spec):
        raise FileNotFoundError(
            f"Missing or invalid {weight_id} weight: {path}. "
            f"Run: python -u tools/download_official_uda_weights.py --only {weight_id}")
    print(f"[OK verified] {weight_id}: {path}", flush=True)
    return path


@contextmanager
def weight_lock(destination):
    """Serialize writers of a resumable partial file, including other jobs."""
    with open(destination+".lock", "a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"\0"); stream.flush()
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.25)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def download(spec, output_dir, retries=5, timeout=120):
    destination = os.path.join(output_dir, spec["relative_path"])
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with weight_lock(destination):
        return _download_locked(spec, output_dir, retries, timeout)


def _download_locked(spec, output_dir, retries, timeout):
    if retries < 1 or timeout <= 0:
        raise ValueError("retries and timeout must be positive")
    destination = os.path.join(output_dir, spec["relative_path"])
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if is_valid(destination, spec):
        print(f"[OK cached] {spec['method']}: {destination}")
        return destination

    temporary = destination + ".part"
    for cached in cached_weight_candidates(spec):
        if is_valid(cached, spec):
            shutil.copy2(cached, temporary)
            if is_valid(temporary, spec):
                os.replace(temporary, destination)
                print(f"[OK copied] {spec['method']}: {cached} -> {destination}")
                return destination
            os.remove(temporary)

    for attempt in range(1, retries+1):
        try:
            if not is_valid(temporary, spec):
                print(f"[DOWNLOAD {attempt}/{retries}] {spec['method']}: {spec['url']}", flush=True)
                download_https(spec["url"], temporary, timeout=timeout)
            if not is_valid(temporary, spec):
                digest = sha256(temporary)
                os.remove(temporary)
                raise OSError(f"SHA-256 verification failed for {spec['method']}: {digest}; restarting")
            os.replace(temporary, destination)
            break
        except (OSError, http.client.HTTPException) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416 and os.path.isfile(temporary):
                os.remove(temporary)
            if attempt == retries:
                retained = (f"Partial file retained at {temporary}; rerun to resume. " if os.path.isfile(temporary)
                            else "No incomplete file was published; rerun to retry. ")
                raise RuntimeError(
                    f"Weight download failed after {retries} attempts: {spec['id']}. "
                    f"{retained}Last error: {exc}") from exc
            delay = min(2**(attempt-1), 15)
            print(f"[RETRY] {type(exc).__name__}: {exc}; retry in {delay}s", flush=True)
            time.sleep(delay)
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
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--check-only", action="store_true", help="Verify all selected local weights; never use the network")
    args = parser.parse_args()
    if args.retries < 1 or args.timeout <= 0:
        parser.error("retries and timeout must be positive")
    output_dir = os.path.abspath(args.output_dir)
    if not args.check_only:
        os.makedirs(output_dir, exist_ok=True)
    records = []
    selected = (WEIGHTS if not args.only
                else tuple(WEIGHTS_BY_ID[name] for name in args.only))
    for spec in selected:
        path = (require_weight(spec["id"], output_dir) if args.check_only
                else download(spec, output_dir, retries=args.retries, timeout=args.timeout))
        records.append({
            "method": spec["method"],
            "path": os.path.relpath(path, output_dir),
            "sha256": sha256(path),
            "url": spec["url"],
        })
    if args.check_only:
        print(f"[PREFLIGHT PASS] {len(records)} local weights verified; no network access", flush=True)
        return
    manifest_path = os.path.join(output_dir, "official_uda_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"weights": records}, handle, ensure_ascii=False, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
