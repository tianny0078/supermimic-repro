"""Record enough provenance to distinguish experiments and refuse changed datasets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

from supermimic.data.motion_convert import g1_xml_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_manifest(files: list[Path], device: str) -> dict:
    root = Path(__file__).resolve().parents[3]
    data = [{"name": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)} for f in sorted(map(Path, files))]
    identity = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    source_files = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").rglob("*.py"))
    source_files += [root / "pyproject.toml", root / "uv.lock"]
    source = hashlib.sha256()
    for path in source_files:
        source.update(str(path.relative_to(root)).encode())
        source.update(bytes.fromhex(sha256_file(path)))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        revision = os.getenv("SUPERMIMIC_SOURCE_REVISION")
    packages = {}
    for package in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "numpy"):
        packages[package] = importlib.metadata.version(package)
    return {"dataset_sha256": identity, "dataset": data, "source_sha256": source.hexdigest(),
            "robot_xml_sha256": sha256_file(g1_xml_path()),
            "git_revision": revision, "packages": packages, "python": platform.python_version(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(device), "precision": "fp32 with TF32 matmul enabled",
            "evaluation_protocol": "full-reference-v1"}
