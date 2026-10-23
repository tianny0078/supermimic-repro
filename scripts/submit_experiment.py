"""Freeze a clean local commit and submit it through Slurm; no GitHub access needed."""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path


def freeze_source(repo: Path, output: Path) -> tuple[Path, str]:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo)

    if git("status", "--porcelain").strip():
        raise ValueError("Commit local changes before submitting a reproducible snapshot")
    revision = git("rev-parse", "HEAD").decode().strip()
    parent = output / "sources"
    parent.mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix=revision[:12] + "-", dir=parent))
    with tarfile.open(fileobj=io.BytesIO(git("archive", "HEAD"))) as archive:
        archive.extractall(snapshot, filter="data")
    (snapshot / "SOURCE_REVISION").write_text(revision + "\n")
    (snapshot / "data").symlink_to((output / "data").resolve(), target_is_directory=True)
    return snapshot, revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["tests", "pilot", "train", "evaluate", "evaluate-run"])
    parser.add_argument("--name")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--time", help="Explicit Slurm time limit (required for train)")
    parser.add_argument("--afterok", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--anchor-goal", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/mnt/data") / os.environ["USER"] / "supermimic-repro")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.kind == "train" and (not args.time or not args.name or args.max_hours is None or args.max_hours <= 0):
        parser.error("train requires --name, --time and a positive --max-hours measured from a pilot")
    if args.kind == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    if args.kind == "evaluate-run" and args.run_dir is None:
        parser.error("evaluate-run requires --run-dir")
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    snapshot, revision = freeze_source(repo, output)
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    script = {"tests": "tests", "pilot": "train", "train": "train",
              "evaluate": "evaluate", "evaluate-run": "evaluate_run"}[args.kind]
    command = ["sbatch", "--parsable", f"--output={logs / (args.kind + '-%j.log')}"]
    if args.kind == "tests":
        command += ["--cpus-per-task=4", "--mem=24G"]
    if args.time:
        command += ["--time=" + args.time]
    if args.afterok:
        command += ["--dependency=afterok:" + ":".join(args.afterok)]
    command += [f"scripts/slurm/{script}.sbatch"]
    if args.kind in ("pilot", "train"):
        command += ["--mode", "paper" if args.kind == "train" else "pilot"]
        if args.name:
            command += ["--name", args.name]
        if args.max_hours is not None:
            command += ["--max-hours", str(args.max_hours)]
        if args.resume:
            command += ["--resume", str(args.resume.resolve())]
        if args.anchor_goal:
            command += ["--anchor-goal"]
    elif args.kind == "evaluate":
        command += ["--ckpt", str(args.checkpoint.resolve())]
    elif args.kind == "evaluate-run":
        command += ["--run-dir", str(args.run_dir.resolve())]
    environment = {**os.environ, "SUPERMIMIC_OUTPUT_ROOT": str(output)}
    record = {"revision": revision, "snapshot": str(snapshot), "kind": args.kind, "command": command,
              "submitted_utc": dt.datetime.now(dt.UTC).isoformat()}
    if not args.dry_run:
        response = subprocess.check_output(command, cwd=snapshot, env=environment, text=True).strip()
        record["job_id"] = response.split(";")[0]
        submissions = output / "submissions"
        submissions.mkdir(exist_ok=True)
        (submissions / f"{record['job_id']}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
