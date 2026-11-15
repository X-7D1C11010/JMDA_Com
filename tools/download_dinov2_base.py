"""Download the public official DINOv2-B backbone for an opt-in diagnostic."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request
import certifi

ROOT=Path(__file__).resolve().parents[1]
URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"


def main():
    output=ROOT/"pretrained_weights/dinov2_vitb14_pretrain.pth"
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        print("CACHED="+str(output),flush=True)
    else:
        part=output.with_suffix(".download_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+".part")
        context=ssl.create_default_context(cafile=certifi.where())
        request=urllib.request.Request(URL,headers={"User-Agent":"JMDA-local-research/1.0"})
        with urllib.request.urlopen(request,context=context,timeout=30) as response,part.open("xb") as stream:
            expected=int(response.headers.get("Content-Length","0"))
            if expected and expected<300_000_000:raise ValueError("Unexpected public backbone size")
            done=0;last_report=0
            while True:
                block=response.read(1024*1024)
                if not block:break
                stream.write(block);done+=len(block)
                if done-last_report>=16*1024*1024:
                    print(f"DOWNLOADED_MiB={done/1024**2:.1f}; total_MiB={expected/1024**2:.1f}",flush=True);last_report=done
            if expected and expected!=done:raise IOError("Incomplete backbone download")
        part.rename(output)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    record={"source":"official public DINOv2-B/14 backbone", "source_page":"https://github.com/facebookresearch/dinov2",
        "url":URL,"bytes":output.stat().st_size,"sha256":digest,"is_student_training_checkpoint":False}
    (ROOT/"pretrained_weights/dinov2_base_provenance.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print("CHECKPOINT_SHA256="+digest,flush=True)


if __name__=="__main__":main()
