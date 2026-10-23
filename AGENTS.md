# Project instructions

Before GPU work on this development server, read `/home/SERVER_GPU_RULES.md` and use Slurm for every GPU operation, including simulation, tests, inference and rendering. Do not bypass scheduling or alter device/cgroup settings. Preserve Slurm's device visibility.

Use `docs/paper-contract.md` as the experiment contract. Distinguish explicit paper settings, unresolved assumptions, reduced smoke/pilot workloads and measured convergence. Keep reference-only and relative-goal results separate. Do not silently modify observations, reward, termination or evaluation to obtain a higher score.

Preserve upstream MIT attribution in `LICENSE` and `NOTICE.md`. Do not commit datasets, weights, credentials or generated training output. Record source/data hashes and the Slurm job for every reported run. GPU tests and training run through `scripts/slurm/`; CPU-only tests may use `CUDA_VISIBLE_DEVICES=''`.
