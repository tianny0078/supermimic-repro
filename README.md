# SuperMimic reproduction

Paper-driven reproduction of [SuperMimic](https://arxiv.org/abs/2610.28480) on Unitree G1, initially using all 40 LAFAN1 clips.

Work in progress. No independently measured convergence result is claimed yet. The implementation starts from the MIT-licensed [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) baseline; see `NOTICE.md` for provenance and `docs/paper-contract.md` for paper settings and unresolved choices.

All GPU simulation, tests, training and evaluation on the development server must run through Slurm. CPU data conversion does not need a GPU.

## Install and data

Linux, Python 3.12, CUDA-capable NVIDIA GPU and a driver compatible with CUDA 13 are required by the pinned dependency lock. Setup uses a managed Python with development headers for Triton, without changing the system Python.

```bash
bash scripts/setup.sh
bash scripts/prepare_lafan1.sh
```

The pinned source release has 40 CSV clips (30 Hz). Downloads are checked against source object hashes and recorded in a SHA256 manifest; conversion produces a 50 Hz library. Raw shared data defaults to `/mnt/data/datasets/lafan1_g1/raw`. Environments, caches, converted data and outputs default to `/mnt/data/$USER/`. Override the variables documented in the scripts on another machine. No dataset or checkpoint is included in Git. See `NOTICE.md` for data licensing.

## Tests and pilot on the development server

Run submissions from the repository root. Slurm assigns the physical GPU; these scripts do not bind a card index.

```bash
python3 scripts/submit_experiment.py tests
python3 scripts/submit_experiment.py pilot --afterok TEST_JOB_ID
```

The submission helper requires a clean local commit and freezes it before queueing; no GitHub connection is needed. It reuses data through a symlink and records each submission under the output directory. `--dry-run` prints the submission without requesting resources.

The pilot keeps the paper's 4096 environments, network widths, k=8, H=32 and optimizer batches; it runs 240 iterations. It is a correctness/throughput check, not a convergence experiment. GPU tests also include a smaller end-to-end training and resume check. CPU-only tests can run with:

```bash
CUDA_VISIBLE_DEVICES='' /mnt/data/$USER/supermimic-repro-env/bin/python -m pytest -q
```

## Train and evaluate

Choose a finite Slurm time budget from measured pilot throughput. For example, after confirming a three-hour run is appropriate, request overhead and a graceful checkpoint signal:

```bash
python3 scripts/submit_experiment.py train --name lafan1-reference-full-state-seed0 \
  --max-hours 3 --time 03:30:00 --afterok TEST_JOB_ID

python3 scripts/submit_experiment.py evaluate-run --afterok TRAIN_JOB_ID \
  --run-dir /mnt/data/$USER/supermimic-repro/runs/lafan1-reference-full-state-seed0
```

Default actor proprioception is 93 dimensions per frame, history 10. The paper leaves exact future-goal encoding unclear; the default reference-only goal and optional `--anchor-goal` hypothesis must remain separately labelled. Evaluation reports all 40 clips from their first-frame reset, full-reference normalized returns, failure counts and per-clip details. Early failure contributes zero over the unplayed suffix.

The default critic receives the full 71-value state; the actor's 93-value proprioception is unchanged. See [goal observability](docs/goal-observability.md) for the unresolved global-drift feedback issue in the reference-only goal.

Each run saves `config.json`, `manifest.json`, `metrics.jsonl`, `status.json`, TensorBoard events and atomic checkpoints. Metrics include steady-window throughput, first-update time including compilation, Torch memory and losses. Training reward fraction is labelled separately from full-clip benchmark return. Nonfinite gradients abort before optimizer updates. Manifests include dataset/source hashes, pinned package versions, GPU and Slurm job. Resume rejects a changed motion library and model/environment/optimizer settings; replay and simulator are rebuilt, so it is not bit-for-bit continuation. `SIGUSR1`/`SIGTERM` request checkpointed exit at the next iteration boundary.

`evaluate-run` evaluates the final and periodic checkpoints and writes `evaluation_index.json`. Reports are atomic JSON, support path-valued configs, verify that motions match the checkpoint's training data, and record training iteration/time plus checkpoint/source/data hashes.

See [paper contract](docs/paper-contract.md), [community comparison](docs/community-review.zh-CN.md) and [experiment log](docs/experiments.md) for details and actual validation status.
