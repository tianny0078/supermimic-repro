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

## 2026-10-25: completed run and community comparison

Slurm **742** training and **743** evaluation have completed. The run `lafan1-reference-full-state-seed0` reached its three-hour budget at 32150 collection steps / 12844 update groups. All 643 training metric records were finite. Its source SHA256 is `8e93b3bbfe1abfb21b5fccd12e38985d33f8ac492289fb85b122ebcdfb309f93`; dataset SHA256 is `39e17fa7ced964ec04cb226a7ad9b1b206c26cacf096daa82c5ecd5b4ebe1f83`.

| Checkpoint | Collection steps | Update groups | Full-reference macro return | Failed / 40 |
|---|---:|---:|---:|---:|
| Best evaluated | 32000 | 12784 | 0.569920 | 3 |
| Final | 32150 | 12844 | 0.545565 | 7 |

The best checkpoint's reference-length-weighted return is 0.577598. It completed 37 motions; failures were `fallAndGetUp1_subject4`, `fallAndGetUp1_subject5`, and `fallAndGetUp2_subject3`. These results do not reach the paper's 0.82 target. The best checkpoint SHA256 is `290cf8ca3c8f2949d07eaf06a8e20e5d5ef232d74ba27bf08b19666eeff833a2`. Full inspection artifacts are retained under `/mnt/data/tianyuan/supermimic-repro/reports/reference-full-state-seed0/`; checkpoints and per-clip reports remain in the run directory.

The [community code audit](community-code-audit.zh-CN.md) records 53 GitHub search matches plus MorphMimic, separates full-reference versus executed-step return, and identifies current/per-future-frame relative orientation as distinct goal hypotheses. The current pure-reference goal must not be described as the only paper-compatible interpretation. No new orientation-goal training was launched during this review.

## 2026-10-28: orientation-only implementation and queued experiment

Added `--orientation-goal`: append only six current reference-to-robot torso rotation values to the original 680-value goal. Proprioception remains 930, with no robot-position feedback. Simulation and imagined rollouts use the same rotation convention; imagination recomputes it from predicted FK poses without detaching state gradients. Evaluation and playback restore this mode from the checkpoint. The legacy nine-value `--anchor-goal` remains separate; conflicting flags fail before startup. No other learning, physics or reward settings were changed for the ablation. See the [implementation comparison](implementation-comparison.zh-CN.md).

Ten targeted CPU checks passed, as did shell syntax and diff checks. Slurm **2322**, source `28e2fcd`, passed 40 tests, failed one and skipped four: end-to-end orientation training exposed a missing flag forwarded to the environment (680-vs-686 input mismatch); raw-CSV conversion tests were skipped because staging only copied converted inputs. Both issues were corrected in source **`5c0a006296f5db6bc91c82320c190d31263210af`**. The first attempt is not counted as a passing suite. Its simulator-vs-mjlab rotation, imagined-vs-FK rotation, and orientation/translation gradient checks passed.

The corrected frozen source is queued through Slurm: **2370** regression → **2371** full-scale 240-collection-step pilot → **2372** fresh three-hour `lafan1-ori6-current-full-state-seed0` training → **2373** final/periodic checkpoint evaluation on all 40 motions. Each dependency requires successful completion. At submission, regression was pending for resources and the later jobs were pending dependencies; no orientation convergence score is available. Machine-readable setup and job/resource details are in [orientation-ablation.json](../experiments/2026-10-28/orientation-ablation.json).

Current server storage rules require environments/caches on NVMe and repeated job input reads in fast scratch. The same installed Python 3.12.15/packages were copied to `/home/tianyuan/.venvs/supermimic-repro`, with managed Python on NVMe. Slurm stages source/converted motions (plus raw CSVs for tests) in a unique job RAM directory and removes it on exit. Weights, logs and final reports remain on bulk storage. Baseline artifacts were retained.

## Orientation-only experiment: completed results

The queued sequence above subsequently completed. Frozen training source: `5c0a006296f5db6bc91c82320c190d31263210af`; source-content SHA256: `10748e166b271e71e85b29fe8e98834f36ce31fa61be0422d162e3024ee9b777`. The published orientation implementation is also represented by commit `539f69dcd0537e392766366cd79130c254aae9e8`.

- Slurm **2370**: **45 regression tests passed**, no skips, 83.49 s.
- Slurm **2371**: full-scale 240-collection-step pilot completed, 80 update groups.
- Slurm **2372**: fresh three-hour orientation-only training, seed 0, one H200, 4096 environments, 37450 collection steps and 14964 update groups.
- Slurm **2373**: final and periodic checkpoints evaluated on all 40 motions with `full-reference-v1`.

| Configuration / checkpoint | Collection steps | Update groups | Macro return | Failed clips |
|---|---:|---:|---:|---:|
| Reference-only, best evaluated | 32000 | 12784 | 0.569920 | 3/40 |
| Orientation-only, matched progress | 32000 | 12784 | 0.799138 | 2/40 |
| Orientation-only, final | 37450 | 14964 | 0.807746 | 3/40 |

The final orientation checkpoint's reference-length-weighted return is 0.809841. Failed clips: `dance1_subject1` at 107.76 s, `fallAndGetUp1_subject4` at 141.32 s, and `walk3_subject5` at 71.02 s. Final checkpoint SHA256: `1403e1ee309e5c95931f745a13c22b73e78cb643be51441c04da81e5c1ec137a`. Dataset SHA256 is unchanged from reference-only: `39e17fa7ced964ec04cb226a7ad9b1b206c26cacf096daa82c5ecd5b4ebe1f83`. Robot XML SHA256: `febdcbeffbbf84051556ae41a5ac1b43fb479a5d76bdb3f54824dbc2721c20aa`.

The 749 metric records contained finite values; peak Torch allocated memory was approximately 17.64 GiB. This is a single-seed goal-encoding ablation on training motions. The final score remains below 0.82 and does not establish paper equivalence, unseen-motion generalization, a PPO speedup, BONES coverage, or real-robot readiness.

Artifacts remain under `/mnt/data/tianyuan/supermimic-repro/runs/lafan1-ori6-current-full-state-seed0/`. Raw metrics, weights, motion data, and generated videos are not committed to Git.

## Simulator recordings and November 1 source audit

Recorder implementation: `c00a98a1934fec59496486a152cd7a96b45f45be`. Slurm **2422** recorded actual closed-loop policy rollouts and rendered comparison videos with the same final orientation checkpoint and motion library. The 48.16 s highlights contain selected excerpts, not complete-motion validation. A separate 11.08 s diagnostic shows the first tracking failure in that fresh replay, `dance1_subject1` at 107.44 s, and holds the true terminal pose for one second. Its timestamp can differ from the original benchmark because of floating-point simulation variation. Reference global XY/yaw is preserved in the comparison videos.

The [English star-ranked source audit](top3-code-audit-2026-11-01.en.md), [Chinese report](top3-code-audit-2026-11-01.zh-CN.md), and [evidence index](top3-code-audit-2026-11-01.json) document implementation gaps and limited isolated CPU checks. The [English follow-up plan](reproduction-plan.en.md) records acceptance gates. **New GPU execution is on hold at the user's request.** This documentation update schedules no work and adds no new training result.
