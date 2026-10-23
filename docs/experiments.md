# Experiment record

All entries describe this repository. Dates use America/Los_Angeles. Upstream scores are not our results.

## 2026-10-22: setup and initial corrections

Pinned LAFAN1 G1 source revision `ce1572906efe6157840e8474d5a0d7aa87481e74`: 40 original CSV clips, verified against source hashes, converted to 50 Hz; 2.451 h and zero conversion failures. The raw source SHA256 manifest and converted-data hashes are retained with the data/runs.

Slurm 704 passed CUDA checks on NVIDIA H200 NVL (139.80 GiB), torch 2.14.1+cu130, CUDA 13.0 and Warp 1.18.0. Initial tests exposed missing Python headers for Triton and an output-directory issue; managed CPython 3.12.15 and directory creation fixed those. A logged-reward regression initially reused older random-action replay while assuming zero previous actions; its fixture now isolates new transitions. Incomplete/cancelled attempts are not counted as passing suites.

## 2026-10-23: verified pipeline

Validated source: `9623b1ee29d9b3903d5d888552dc0ed9e560132c`.

- Slurm **730**: **32 passed**, including GPU simulation/FK, real logged reward versus replay-next-state analytic reward, WM overfit, actor update, checkpoint and resume. Runtime 60.36 s.
- Slurm **731**: completed a full-scale 240-iteration pilot (4096 envs; paper network widths/batches; k=8, H=32). First-update compilation dominates the initial window; subsequent throughput was **3.7–4.4 iterations/s**. The initial 50-iteration window took about 283 s including compilation/startup. These measurements are specific to the recorded implementation and hardware.
- Slurm **732**: simulated the complete 40-clip evaluation, then failed while writing JSON because checkpoint configs contain `Path` objects. The short policy failed all clips; it is a workflow artifact, not a convergence result. The missing report is being regenerated after fixing serialization.

## Current implementation changes

Reports now handle path-valued configs and are written atomically. Metrics/status JSON and training memory/throughput are recorded, nonfinite gradients abort updates, and a local-commit snapshot submission tool replaces ad hoc archive commands. Complete-run evaluation records periodic/final checkpoints and source/data identity.

The critic now uses the complete 71-value state stated in Table A.4; the previous pilot used heading-invariant features. Actor proprioception remains 93 values per frame. The old pilot is not resumed as an interchangeable checkpoint for the new critic configuration. Reference-only versus relative-goal assumptions are documented in `goal-observability.md`.

The previous pilot implies roughly 0.23–0.27 s/iteration after compilation. A first three-hour training budget targets about 40k iterations, with a 3.5-hour Slurm allocation for startup/checkpoint overhead. Full 320k-iteration training would take about 20–24 hours at that throughput and has not been claimed necessary or completed. New GPU regression checks precede formal training; evaluation follows training as a separate allocation.

## Publication

GitHub reauthorization succeeded. README initialization and complete source publication on main were verified. The code tree of published commit `cb2ce51a5fc2cd8f9947acf3689bc206bf69af7a` exactly matches local source revision `7e4be07085ce1d7d7b202ae53d90673b77e63623`. Pre-publication local history is retained locally. No convergence claim is made by publishing the implementation.

## New regression and formal training

- Slurm **740**: **38 GPU tests passed** in 61.37 s, including full-state critic translation sensitivity, atomic JSON report round trips and frozen submission snapshots.
- Slurm **741**: regenerated and saved the old 240-iteration pilot's 40-clip report. Macro return 0.0014522, 40/40 failed; this is an expected unconverged workflow artifact, not a trained benchmark result.
- Slurm **742**: fresh formal reference-only/full-state-critic run started. Budget: 3 training hours, 3.5-hour Slurm allocation, one H200, 8 CPUs, 64 GiB host RAM. Actor proprioception remains 93 values/frame. No resume from the old critic configuration.
- Initial health check passed 200 iterations with finite metrics/gradients, approximately 3.5 it/s and 17.6 GiB peak Torch allocation. The memory figure excludes Warp/driver allocations. It does not establish convergence.
- Slurm **743**: queued after successful training to evaluate final/periodic checkpoints and write an evaluation index.
- Machine-readable evidence is in `experiments/2026-10-23/validation.json`; live metrics/checkpoints are under the output run directory and are not committed.
