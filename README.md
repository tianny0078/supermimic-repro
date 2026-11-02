# SuperMimic reproduction

Paper-driven reproduction of [SuperMimic](https://arxiv.org/abs/2610.28480) on Unitree G1, initially using all 40 LAFAN1 clips.

Work in progress. The three-hour current-orientation run reached **0.807746 full-reference return, with 3/40 failed clips**. At matched progress (32000 collection steps / 12784 updates), it scored 0.799138 versus 0.569920 for reference-only. Both are single-seed training-library results, below the paper's 0.82 LAFAN1 target. The implementation starts from the MIT-licensed [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) baseline; see [NOTICE.md](NOTICE.md) for provenance and the [paper contract](docs/paper-contract.md) for explicit settings and unresolved choices.

All GPU simulation, tests, training and evaluation on the development server must run through Slurm. CPU data conversion does not need a GPU.

The corrected orientation source passed **45 GPU regression tests** in Slurm 2370; pilot, training and complete checkpoint evaluation finished in jobs 2371–2373. The earlier reference-only run used jobs 742/743. See [experiment results](docs/experiments.md) for the measured scope and identities. New GPU work is currently **on hold at the user's request**; publishing documentation does not start jobs.

## Audit and next steps

- [English source audit: top three repositories by GitHub stars](docs/top3-code-audit-2026-11-01.en.md) — pinned November 1 snapshot, implementation gaps, metric differences, and evidence links.
- [English implementation and validation plan](docs/reproduction-plan.en.md) — phased work, acceptance gates, controlled comparisons, and execution status.
- [Chinese audit](docs/top3-code-audit-2026-11-01.zh-CN.md) and [machine-readable evidence](docs/top3-code-audit-2026-11-01.json).

The audit identifies missing input-parity profiles, normalization, WM-only warm-up, critic features, clip-end rules, and packed-library evaluation. These are findings and planned work, not newly implemented improvements. It also explains why some community scores cannot be directly compared with our complete-reference metric.

## Install and data

Linux, Python 3.12, CUDA-capable NVIDIA GPU and a driver compatible with CUDA 13 are required by the pinned dependency lock. Setup uses a managed Python with development headers for Triton, without changing the system Python.

```bash
bash scripts/setup.sh
bash scripts/prepare_lafan1.sh
```

The pinned source release has 40 CSV clips (30 Hz). Downloads are checked against source object hashes and recorded in a SHA256 manifest; conversion produces a 50 Hz library. Raw shared data defaults to `/mnt/data/datasets/lafan1_g1/raw`; converted data and outputs live under `/mnt/data/$USER/`. Environments and compilation caches use `/home/$USER/` on NVMe. Slurm jobs stage frozen source and LAFAN1 inputs in per-job `/dev/shm` scratch (counted in host RAM) and clean it on exit. Override the variables documented in the scripts on another machine. No dataset or checkpoint is included in Git. See `NOTICE.md` for data licensing.

## Tests and pilot on the development server

Run submissions from the repository root. Slurm assigns the physical GPU; these scripts do not bind a card index.

```bash
python3 scripts/submit_experiment.py tests
python3 scripts/submit_experiment.py pilot --afterok TEST_JOB_ID
```

The submission helper requires a clean local commit and freezes it before queueing; no GitHub connection is needed. It reuses data through a symlink and records each submission under the output directory. `--dry-run` prints the submission without requesting resources.

The pilot keeps the paper's 4096 environments, network widths, k=8, H=32 and optimizer batches; it runs 240 iterations. It is a correctness/throughput check, not a convergence experiment. GPU tests also include a smaller end-to-end training and resume check. CPU-only tests can run with:

```bash
CUDA_VISIBLE_DEVICES='' /home/$USER/.venvs/supermimic-repro/bin/python -m pytest -q
```

## Train and evaluate

Choose a finite Slurm time budget from measured pilot throughput. For example, after confirming a three-hour run is appropriate, request overhead and a graceful checkpoint signal:

```bash
python3 scripts/submit_experiment.py train --name lafan1-reference-full-state-seed0 \
  --max-hours 3 --time 03:30:00 --afterok TEST_JOB_ID

python3 scripts/submit_experiment.py evaluate-run --afterok TRAIN_JOB_ID \
  --run-dir /mnt/data/$USER/supermimic-repro/runs/lafan1-reference-full-state-seed0
```

Default actor proprioception is 93 dimensions per frame, history 10. The paper leaves exact future-goal encoding unclear. The default goal is reference-only (680 values); `--orientation-goal` appends only current reference-to-robot torso rotation (6 values, total 686), while `--anchor-goal` appends position and rotation (9, total 689). These flags are mutually exclusive and results remain separately labelled. Evaluation reports all 40 clips from their first-frame reset, full-reference normalized returns, failure counts and per-clip details. Early failure contributes zero over the unplayed suffix.

For the orientation-only ablation, use a fresh run, for example `train --orientation-goal --name lafan1-ori6-current-full-state-seed0 --max-hours 3 --time 03:30:00`. Keep the remaining settings fixed; this is an explicit goal hypothesis, not a claim that the authors used this encoding. Checkpoint evaluation and playback recover the selected goal from the saved config.

The default critic receives the full 71-value state; the actor's 93-value proprioception is unchanged. See [goal observability](docs/goal-observability.md) for the unresolved global-drift feedback issue in the reference-only goal.

Each run saves `config.json`, `manifest.json`, `metrics.jsonl`, `status.json`, TensorBoard events and atomic checkpoints. Metrics include steady-window throughput, first-update time including compilation, Torch memory and losses. Training reward fraction is labelled separately from full-clip benchmark return. Nonfinite gradients abort before optimizer updates. Manifests include dataset/source hashes, pinned package versions, GPU and Slurm job. Resume rejects a changed motion library and model/environment/optimizer settings; replay and simulator are rebuilt, so it is not bit-for-bit continuation. `SIGUSR1`/`SIGTERM` request checkpointed exit at the next iteration boundary.

`evaluate-run` evaluates the final and periodic checkpoints and writes `evaluation_index.json`. Reports are atomic JSON, support path-valued configs, verify that motions match the checkpoint's training data, and record training iteration/time plus checkpoint/source/data hashes.

See [paper contract](docs/paper-contract.md), [community comparison](docs/community-review.zh-CN.md) and [experiment log](docs/experiments.md) for details and actual validation status.
