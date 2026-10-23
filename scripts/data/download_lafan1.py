"""Fetch the pinned public 40-clip G1 LAFAN1 release; CPU-only, no credentials needed."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"


def fetch(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/g1"
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = [e for e in json.load(response) if e["path"].endswith(".csv")]
    if len(entries) != 40:
        raise ValueError(f"Expected 40 clips, got {len(entries)}")

    def download(entry):
        name = Path(entry["path"]).name
        dest = output / name
        if dest.exists():
            raw = dest.read_bytes()
        else:
            source = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{entry['path']}"
            with urllib.request.urlopen(source, timeout=120) as response:
                raw = response.read()
        if len(raw) != entry["size"]:
            raise ValueError(f"Size mismatch: {name}")
        if "lfs" in entry:
            valid = hashlib.sha256(raw).hexdigest() == entry["lfs"]["oid"]
        else:
            valid = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == entry["oid"]
        if not valid:
            raise ValueError(f"Source hash mismatch: {name}; refusing to overwrite")
        if not dest.exists():
            temporary = dest.with_suffix(".csv.part")
            temporary.write_bytes(raw)
            temporary.replace(dest)
        return {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "source_oid": entry["oid"]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = sorted(pool.map(download, entries), key=lambda e: e["name"])
    manifest = {"dataset": DATASET, "revision": REVISION, "motion_license": "CC-BY-NC-ND-4.0 (upstream)", "files": files}
    (output / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(files)} clips under {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    fetch(parser.parse_args().output)
