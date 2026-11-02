# Source audit of the three most-starred SuperMimic reproductions

Audit date: November 1, 2026, America/Los_Angeles. Repository metadata was captured at approximately 04:07 UTC on November 2. Stars are a dated popularity snapshot, not a correctness ranking.

The local implementation reviewed was `c00a98a1934fec59496486a152cd7a96b45f45be`. The measured orientation-only training run used frozen source `5c0a006296f5db6bc91c82320c190d31263210af`. See the [evidence index](top3-code-audit-2026-11-01.json), [Chinese report](top3-code-audit-2026-11-01.zh-CN.md), and [implementation and validation plan](reproduction-plan.en.md).

## Finding

This repository implements the main SuperMimic training algorithm, but it does not yet provide a configuration equivalent to the successful high-star LAFAN1 reproductions. Important gaps include trajectory-goal encoding, critic error features, running normalization, world-model-only warm-up, clip-end handling, and training throughput. A separate interface gap was confirmed with isolated CPU calls: checkpoints trained with a packed motion library cannot be evaluated directly through the existing NPZ evaluation entrypoint.

The audit distinguishes missing implementations, existing but disabled or unvalidated options, and alternative designs that require experiments. A community choice is not automatically the authors' setting, and a code difference does not establish its effect on return. No training implementation was changed during this audit, and no third-party checkpoint was executed.

## Ranking and scope

GitHub's official REST repository search for `supermimic`, sorted by stars descending, reported 119 matches. The first 100 results were sufficient to identify the leaders. Searches for `supermimic fork:true stars:>200` and `super-mimic stars:>200` confirmed the same three code repositories. The connected GitHub search timed out, so the snapshot was obtained through GitHub's public REST API.

| Rank | Repository | Stars | Reviewed commit | Backend |
|---|---|---:|---|---|
| 1 | [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | 770 | `0525a23a2dcfa61f8543f98b84c075fe66d9d286` | Isaac Lab 3 / Newton / MuJoCo Warp |
| 2 | [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | 356 | `ba6105349069bf040f1fdad60873d13138d505a2` | mjlab / MuJoCo Warp |
| 3 | [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | 220 | `faee48a875579c45b3a41e6399b05ad014641ebf` | UniLab: mjwarp / MuJoCo / Motrix |

`miao-org/awesome-supermimic` had 420 stars but is a resource index, so it was excluded from the implementation ranking. Text snapshots of 148 source, configuration, and documentation files were saved locally. Focused reading covered training loops, networks, MDPs, motion libraries, replay, termination, evaluation, and result records. The download count is not a claim that every file received a complete review.

Uni substantially follows the vibed network and learner design and acknowledges reference implementations. Shared choices are therefore not independent causal evidence. Cuboi is this project's upstream; its current Python source was compared file by file with the local implementation. External links below are pinned to the reviewed commits. Local evidence paths and source hashes are recorded in the JSON index; raw third-party snapshots are not vendored into this repository.

## Reported results and comparability

All external numbers in this table are author reports, not independent measurements on this server.

| Implementation / configuration | Reported result | What the code establishes |
|---|---|---|
| vibed, LAFAN1 | 0.824 with 1/40 failures at 1.24 h; 0.829 with 0/40 at 1.47 h on an RTX PRO 6000 | Full `N−1` reference denominator; extra orientation observation, bf16, disabled self-collisions, and 500 WM-only rounds. [Milestones][z-results], [evaluation][z-eval]. |
| Cuboi, LAFAN1 with 9D anchor feedback | 0.904 with 3/40 failures at 3.3 h; reference-only runs about 0.56–0.59 | The evaluator still divides by executed steps, so these returns must not be ranked against our full-reference return. [Results][c-results], [evaluator][c-eval]. |
| Cuboi, BONES-SEED | About 0.93, 145/129491 failures, four RTX 5090s for 7.1 h | Different motion set and executed-step denominator. The author reports failures on all 54 flip clips. Similar dataset counts do not prove matching motion IDs. [Results][c-results]. |
| Uni, LAFAN1 | 0.827227 with 0/40 failures at checkpoint 12000 | Full `N−1` denominator; 99D observations and a 730D goal. [Results][u-results], [evaluator][u-eval]. |
| Uni, optimized BONES-SEED | 0.9395 with 193/129487 failures at 8000 iterations | Uses EMA 0.995, `no_sync`, and learning rate 4e−4; this is distinct from the earlier 2e−4 baseline and from the paper's motion set. [Configuration][u-config], [EMA checkpoint selection][u-checkpoint]. |

Our controlled seed-0 runs use the same 40 LAFAN1 training motions and `full-reference-v1`. At 32000 collection steps / 12784 update groups, reference-only achieved **0.569920, 3/40 failures**, while current relative torso orientation achieved **0.799138, 2/40**. The latter ended at 37450 collection steps / 14964 update groups / three training hours with **0.807746, 3/40**. These measure training-library coverage, not unseen-motion generalization. See the [experiment log](experiments.md).

## Implementation findings

### 1. Equivalent trajectory and observation inputs are missing — P1

vibed and Uni append six values of current reference-to-robot **pelvis** rotation to each observation. Their 99D observations form a 990D ten-frame history. Their goal contains ten 73D future frames: reference displacement relative to the current reference heading (3), rotation (6), local linear and angular velocity (3 each), joint position (29), and joint velocity (29). [Observation][z-obs], [vibed goal][z-goal], [Uni goal][u-goal].

Our completed orientation run keeps a 930D proprioceptive history and appends only the current **torso** relative rotation to a 680D reference goal, yielding 686D. It lacks an equivalent displacement/rotation trajectory mode, places orientation in another network branch, and does not provide its ten-frame history. Adding six values did not establish input parity. [Local goal][o-goal], [imagination][o-imagine], [policy][o-policy].

Add an explicit community-comparison profile with matching pelvis convention, history, noise, scaling, and reference timing in both simulation and imagination. Preserve the existing 680/686/689 modes. The paper does not enumerate the goal completely, so this remains a community interpretation. Reference trajectory displacement also does not provide the robot's current global XY tracking error.

### 2. Critic tracking-error features are only partially available and disabled — P1

vibed and Uni supply joint error (29D), body-position error (42D), root-position error (3D), and relative rotation (6D), in addition to invariant state, goal, and simulation context. With their goal, the total critic input is 914D. They also normalize this input with running statistics. [vibed][z-critic], [Uni][u-critic].

Our formal run uses raw full state (71D), goal (686D), and context (33D): 790D total. `root_error_features=False`. The optional four-value root-position/yaw augmentation already exists, but it is neither enabled in the reported run nor equivalent to the complete 80D error vector. [Policy configuration][o-policy], [critic inputs][o-trainer].

Raw robot world coordinates are not explicit robot-to-reference errors. The current goal omits reference absolute XY; receiving the full robot state does not by itself establish that the critic can uniquely infer the root-position reward. Test error features separately rather than treating them as a proven explanation of the gap.

In the final run, the executed-step mean root-position reward was 0.0010865 of its 0.01 maximum (about 10.9%), while root-orientation reward was 0.0096254 of 0.01 (about 96.3%). Orientation feedback improved heading, but substantial path drift remains. These are survived-step component means, not additional full-reference scores.

### 3. Running normalization and WM LayerNorm are missing — P1

vibed and Uni normalize actor observations, goals, critic inputs, and WM inputs/output increments with running statistics. Their WM hidden layers include LayerNorm, and their error normalization uses increment statistics with per-component floors. [vibed networks][z-norm], [Uni networks][u-norm], [WM error][u-wmerr].

We use fixed, manually selected input/output/error scales and no RunningNorm or WM LayerNorm. We also predict velocity increments plus pose corrections over explicit integration, whereas the other two predict local state increments directly. These are substantive parameterization differences despite matching hidden widths. [State representation][o-state], [MLP][o-dynamics].

Make normalization, LayerNorm, and state parameterization independently configurable. Save statistics in checkpoints; update them from real collection and freeze them during imagination and evaluation. Because error units differ, our NSE cannot be compared directly with the paper's or a community implementation's 0.007–0.011.

### 4. A world-model-only warm-up stage is missing — P1

vibed and Uni initially collect 64 vectorized steps and run 500 WM-only update rounds before actor/critic learning. Each round contains ten WM minibatches; 500 rounds is not 500 minibatches. [vibed settings][z-warm], [Uni loop][u-loop].

We start alternating WM and policy updates after 40 collection steps, without a separate WM-only option. Early policy updates may therefore use poorly fitted dynamics and termination gradients. This is a plausible mechanism, not a measured causal finding. [Local loop][o-loop].

### 5. Clip-end semantics need explicit implementation and validation — P1

vibed and Uni intend to sample imagination starts with enough reference frames for the H=32 rollout. vibed's latest goal padding also holds the final pose with zero velocity for **synthetic frames beyond the clip**, preserving the actual final recorded frame's velocities. Uni's reviewed goal code does not yet include that padding fix. [Start filtering][z-start], [vibed padding][z-padding], [Uni goal][u-goal].

Our start sampler does not enforce remaining clip length. Imagination clamps reference indices to the last frame but continues reward, survival, and critic bootstrap calculations; goal padding repeats the final recorded nonzero velocity. [Rollout][o-imagine], [goal window][o-goal].

An isolated CPU call confirmed that synthetic future queries still contain joint velocity 2 when the final stored frame has velocity 2. The behavior is confirmed; its effect on return is unmeasured. Compare valid-start filtering, explicit end masks/bootstrap rules, and stationary synthetic padding. Do not overwrite real final-frame velocities or silently change benchmark reference data. Short BONES clips make this especially relevant.

Uni's `no_sync` start selector oversamples twice and selects valid entries first, without guaranteeing a complete valid batch for every possible motion library. A replacement should define a safe fallback for short clips rather than copy that assumption blindly.

### 6. Packed-library training and evaluation do not connect — P1, CPU-confirmed

Compact fp16 storage, online FK, `build_library.py`, and `DataCfg.library` already exist. However, training from a packed `.pt` library records that container in the training manifest. `eval_benchmark.py` discovers NPZ files and compares their names/hashes with that manifest, rejecting them before simulation. It also does not forward a packed-library mode to the evaluation environment. [Training][o-loop], [evaluation entrypoint][o-eval-cli], [library builder][o-build].

The existing evaluator function was called with file/checkpoint/simulator dependencies replaced by CPU stubs. The NPZ-to-NPZ control passed manifest validation and reached the simulator stub. The packed-to-NPZ case raised `ValueError` before reaching it. No GPU checkpoint was loaded and no simulator ran.

Uni supports packed-library training and evaluation that restores the recorded library selection. [Library][u-goal], [evaluation][u-eval]. We need a packed-library evaluator, per-motion IDs/source hashes, an explicit account of fp16 quantization, and complete-library validation. Local compact-library unit support is not evidence that this workflow already works end to end.

Our BONES filter also predates Cuboi's restored acrobatics keywords. Its revised set still differs from the paper, so it must not be copied simply to make counts look closer or remove failures. [Upstream filter][c-filter]. No local full BONES training/evaluation has been completed.

### 7. Several throughput optimizations are missing — P2, benefit unmeasured

Our WM sampler constructs a window mask over 4096×1000 replay rows for every minibatch, then samples from that mask with multinomial and host reads. vibed and Uni directly sample start ages and mask invalid suffixes after episode ends. [Our replay][o-replay], [vibed][z-replay], [Uni][u-replay].

We explicitly synchronize after collection, WM, and policy work; policy metrics also frequently become Python floats. Our optimizers do not request fused Adam. Uni includes fused Adam, small-matrix arithmetic helpers, and paths that remove some host reads. [Our loop][o-loop], [policy update][o-trainer], [Uni optimizers][u-learner], [geometry][u-geometry].

Recomputing existing orientation-run logs after collection step 10000 gives median seconds per collection step of approximately 0.028 for simulation, 0.076 for WM work, and 0.180 for policy/critic work. These are phase timings including their sampling and update overhead, not individual kernel measurements. Profile under Slurm before attributing the gap to a call or claiming a speedup. Preserve terminal samples, episode boundaries, and ring-buffer correctness when changing replay. Uni's `no_sync` flag also does not eliminate every synchronization; metric conversion still occurs.

## Alternative choices that require separate experiments

| Choice | Community implementation | Local status | Interpretation |
|---|---|---|---|
| Imagined termination | vibed/Uni use sigmoid-softened analytic thresholds | Learned termination head and BCE | A selectable analytic mode is missing; either approach can fit the paper's allowed design. |
| Precision | bf16 MLPs, fp32 state/FK arithmetic | fp32 with TF32 | Optional acceleration is missing; retaining TF32 follows the explicit paper setting. |
| EMA policy | Uni's current profile uses 0.995, saves EMA for evaluation and live weights separately | Missing | Report live and EMA evaluations separately. |
| Learning rate | Uni's optimized default is 4e−4; earlier baseline 2e−4 | 2e−4 | Test independently of EMA and normalization. |
| Assets, contact, reward anchor | vibed disables self-collision and uses a pelvis anchor; Uni follows the same pelvis MDP family | mjlab FULL_COLLISION; torso-relative body reward, pelvis global reward/termination | Audit asset/contact/anchor equivalence; disabling self-collision is not a required correction. |
| Reset perturbations | Enabled by default in vibed/Uni | Implemented but disabled in our formal runs | An untested training-distribution choice, not a missing feature; benchmark resets remain exact. |
| Return-based sampler | Primarily episode-start-bin updates and mean-reward bootstrap approximations | Updates visited bins; without `bootstrap_fn`, excludes steps near truncation | All use a 50/40/10 mixture but are not identical algorithms. The local bootstrap callback is not connected. |
| Iteration unit | An update round with about 2.5 collection steps | A collection step with about 0.4 update rounds | UTD matches; total budget and LR schedule units do not. Add explicit counters and a selectable schedule clock. |

vibed's 14000 iterations include 500 WM-only rounds: approximately 13500 actor updates and 35000 vectorized collection steps, plus initial filling. Our matched checkpoint has 32000 collection steps / 12784 updates, and the final run 37450 / 14964. The learning amounts are comparable in order of magnitude; comparing the labels “14k” and “37k” alone is misleading.

## Capabilities to preserve and validation still missing

Already implemented: k=8 WM fitting, H=32 BPTT, critic tail bootstrap, GAE, 5×4 critic fitting, separate clean/executed actions, ten-frame proprioception, return-based resets, domain randomization, compilation, distributed gradient averaging, compact-library components, per-clip failure times, and real simulator video recording. Preserve full-reference `N−1` scoring, zero reward for failed suffixes, source/data/robot identities, atomic checkpoints, and boundary/gradient regression coverage.

Cuboi still divides the actor objective by H and WM loss by k before optimization, and uses executed-step scoring. Those choices were changed locally and must not be restored by a mechanical upstream merge. We already accept `--clip-list` shards, but lack Cuboi's newly added shard merger. A local merger should reject duplicate/missing motion IDs and inconsistent provenance and use our full-reference aggregation. [Upstream merger][c-merge].

Uni adds cross-backend policy-I/O contracts, saved-library restoration, and interactive Viser playback. We have mjlab and recorded video, but no equivalent cross-backend contract or live pause/step/environment-switching UI. These are useful extensions, not the first explanation to test for return 0.8077. [Contract][u-contract].

Outstanding experimental validation includes repeated seeds, full BONES coverage, PPO-Sim, H×critic, paper-aligned push statistics, and real-robot transfer. The top three repositories do not all complete these either: Cuboi explicitly lists PPO, velocity interfaces, and real deployment as unfinished, while vibed's main result covers LAFAN1. Existing `eval_push.py`, no-critic, compact-library, and distributed modules are not evidence of completed corresponding experiments.

## Follow-up

Use the [phased plan](reproduction-plan.en.md). First close evaluation and boundary gaps, then run controlled comparisons of normalization, WM warm-up, critic errors, and trajectory inputs. Profile before throughput work; test analytic termination, EMA, precision, and reset noise separately. Lock motion IDs before large-library claims.

**Execution remains on hold at the user's request.** This publication records findings and planned acceptance gates; it does not launch jobs, claim fixes are implemented, or claim improved results.

[z-results]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/MILESTONES.md#L21
[z-eval]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/evaluate.py#L84
[c-results]: https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/README.md#L9
[c-eval]: https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/src/supermimic/eval/benchmark.py#L97
[u-results]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/README.md#L254
[u-eval]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/tasks/motion_tracking/g1/supermimic/evaluate.py#L82
[u-config]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/conf/supermimic/config.yaml#L27
[u-checkpoint]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/learner.py#L571
[z-obs]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/mdp.py#L95
[z-goal]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/motion.py#L100
[u-goal]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/tasks/motion_tracking/g1/supermimic/motion.py#L296
[o-goal]: ../src/supermimic/envs/mdp/motion_library.py#L226
[o-imagine]: ../src/supermimic/imagination/rollout.py#L36
[o-policy]: ../src/supermimic/policy/nets.py#L20
[z-critic]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/mdp.py#L163
[u-critic]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/tasks/motion_tracking/g1/supermimic/mdp.py#L249
[o-trainer]: ../src/supermimic/imagination/policy_trainer.py#L56
[z-norm]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/networks.py#L24
[u-norm]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/networks.py#L34
[u-wmerr]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/learner.py#L279
[o-state]: ../src/supermimic/world_model/state_spec.py#L47
[o-dynamics]: ../src/supermimic/world_model/dynamics.py#L9
[z-warm]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/train.py#L37
[u-loop]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/learner.py#L632
[o-loop]: ../src/supermimic/training/loop.py#L35
[z-start]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/train.py#L159
[z-padding]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/motion.py#L104
[o-eval-cli]: ../scripts/eval_benchmark.py#L22
[o-build]: ../scripts/data/build_library.py#L18
[c-filter]: https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/scripts/data/filter_bones_seed.py#L29
[o-replay]: ../src/supermimic/data/replay_buffer.py#L97
[z-replay]: https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/buffer.py#L41
[u-replay]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/replay.py#L59
[u-learner]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/learner.py#L155
[u-geometry]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/rl/algos/supermimic/geometry.py#L91
[c-merge]: https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/scripts/merge_benchmark_shards.py#L16
[u-contract]: https://github.com/Motphys/Uni-SuperMimic/blob/faee48a875579c45b3a41e6399b05ad014641ebf/src/unisupermimic/sim2sim.py#L142
