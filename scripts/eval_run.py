"""Evaluate periodic checkpoints and the final checkpoint using one Slurm allocation."""
from __future__ import annotations

import json
from pathlib import Path

import tyro
from eval_benchmark import main as evaluate

from supermimic.artifacts import write_json


def main(run_dir: Path, npz_dir: Path = Path("data/npz/lafan1"), device: str = "cuda:0") -> None:
    checkpoints = sorted(run_dir.glob("ckpt_[0-9]*.pt"))
    latest = run_dir / "ckpt_latest.pt"
    if latest.exists():
        checkpoints.append(latest)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints under {run_dir}")
    entries = []
    seen_iterations = set()
    for checkpoint in reversed(checkpoints):
        if checkpoint.stem.removeprefix("ckpt_").isdigit() and int(checkpoint.stem[5:]) in seen_iterations:
            continue
        # The final checkpoint is evaluated first, and duplicate periodic
        # checkpoints of that same iteration are omitted from the curve.
        report_path = run_dir / f"benchmark_{checkpoint.stem}.json"
        evaluate(ckpt=checkpoint, npz_dir=npz_dir, device=device, out=report_path)
        report = json.loads(report_path.read_text())
        iteration = report["metadata"]["checkpoint_iteration"]
        if iteration not in seen_iterations:
            entries.append({"iteration": iteration, "train_elapsed_s": report["metadata"]["train_elapsed_s"],
                            "checkpoint": checkpoint, "checkpoint_sha256": report["metadata"]["checkpoint_sha256"],
                            "report": report_path, **report["summary"]})
            seen_iterations.add(iteration)
        # Write incrementally so an interrupted evaluation keeps completed results.
        write_json(run_dir / "evaluation_index.json", {"checkpoints": sorted(entries, key=lambda e: e["iteration"])})


if __name__ == "__main__":
    tyro.cli(main)
