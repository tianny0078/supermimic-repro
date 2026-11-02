# SuperMimic reproduction: implementation and validation plan

This plan follows the [November 1 source audit](top3-code-audit-2026-11-01.en.md). It preserves the [paper contract](paper-contract.md), explicit community interpretations, and the existing measured baselines.

**Status: proposed; new GPU work is on hold at the user's request.** Publishing this document does not submit training, tests, simulation, rendering, or evaluations. No implementation below should be described as completed solely because it appears in this plan. Resume GPU execution only after the user instructs us to continue.

## Current evidence

Both formal runs use seed 0, the same 40 LAFAN1 training motions, full-state critic, and full-reference scoring. Dataset identity: `39e17fa7ced964ec04cb226a7ad9b1b206c26cacf096daa82c5ecd5b4ebe1f83`.

| Configuration | Collection steps | Update groups | Full-reference macro return | Failed clips |
|---|---:|---:|---:|---:|
| Reference-only, best evaluated | 32000 | 12784 | 0.569920 | 3/40 |
| Current torso orientation, matched progress | 32000 | 12784 | 0.799138 | 2/40 |
| Current torso orientation, final | 37450 | 14964 | 0.807746 | 3/40 |

The reference-only jobs were Slurm 742/743. Orientation source was `5c0a006296f5db6bc91c82320c190d31263210af`; Slurm 2370 passed 45 regression tests, 2371 completed the full-scale pilot, 2372 trained for three hours, and 2373 evaluated checkpoints. These are single-seed results, below the paper's LAFAN1 0.82 target and without an independent PPO comparison. They do not reproduce the paper's large-library or real-robot claims.

## Invariants

- Preserve old checkpoints, reports, data identities, and experiment names. Configuration changes create new runs; incompatible inputs cannot resume an old model as the same experiment.
- Keep `full-reference-v1`: exact first-frame reset, no evaluation observation/action/reset noise or DR, friction 1, full `N−1` reference denominator, failed suffix reward zero, macro and reference-length-weighted returns.
- Record source revision/hash, data/motion IDs, robot assets, Slurm job, precision, schedule clock, and checkpoint identity. Distinguish training-episode reward from complete-clip evaluation.
- Treat 680D reference-only, 686D current-torso-orientation, 689D anchor-pose, and any new community profile as separate interpretations. Preserve the actor's real information requirements and gradients through state-dependent imagined inputs.
- All GPU work uses Slurm and the maintained `/home/SERVER_GPU_RULES.md`. Jobs use finite budgets, measured resource needs, fast input scratch, persistent bulk-storage outputs, and cleanup. No queued placeholder allocation or automatic retry while execution is on hold.

## Phase 1 — Make counters and comparison profiles explicit

**Implementation:** expose vectorized collection steps, real transitions, WM update groups, WM minibatches, actor optimizer steps, critic optimizer steps, imagined transitions, and elapsed time. Add an explicit LR/total-budget clock; retain the old collection-based schedule for existing configurations. WM-only rounds must not increment actor-update counts.

Define named profiles for the existing baseline and community comparisons. Each records goal layout, pelvis/torso convention, observation history, critic features, normalization, termination model, precision, warm-up, reset distribution, and asset/contact settings. A complete community profile is a reference condition, not a single-variable causal experiment.

**Acceptance:** arithmetic and schedule-boundary checks cover cold starts and resumes; old config loading retains its original meaning. Every checkpoint/evaluation exposes both collection and update counters. A profile diff identifies every changed field before training.

## Phase 2 — Close evaluation and clip-boundary gaps

**Packed-library evaluation:** accept the same packed library used by training. Store per-motion IDs and source identities in addition to the container hash. Define how fp16 packed states relate to NPZ originals; do not silently compare scores computed against different reference values. Preserve dataset verification rather than removing the check that currently rejects packed-to-NPZ evaluation.

Add sharded evaluation merging with duplicate/missing-ID checks, consistent checkpoint/data/asset/protocol identities, and full-reference aggregation. Longer-first scheduling is an optional throughput improvement that must preserve results.

**Clip ends:** expose a documented choice among sufficient-remaining-length starts, explicit end masks/bootstrap rules, and held-pose extension. For synthetic frames beyond a clip, stationary padding must not modify recorded final-frame velocities. Handle clips shorter than H without loops that cannot fill a batch. Use the same rules for goals, reward references, and imagination.

**Acceptance:** CPU boundary checks cover final frames, mixed clip lengths, H-short clips, ring-buffer wrap, and immutable reference data. Once execution resumes, compare NPZ and packed paths under a declared precision convention and run complete 40-motion evaluation. Sharded and unsharded summaries must agree within a justified tolerance. A change in clip-end training semantics gets a new experiment ID.

## Phase 3 — Add learning components behind separate switches

| Component | Implementation | Acceptance before an experiment |
|---|---|---|
| Running normalization | Actor observation/goal, critic input, WM input/output statistics; optional WM LayerNorm | Save/restore statistics; collection-only updates; frozen imagination/evaluation; finite outputs and gradients. |
| WM-only warm-up | Configurable fill steps and WM-only rounds; community reference: 64 and 500 | Correct ten-minibatch-per-round counting; actor/critic weights and optimizers unchanged during WM-only work. |
| Critic errors | Optional joint/body/root/rotation error vector, independent of the existing 4D option | Independent numerical checks of frames and dimensions; useful state gradients; no extra actor information leak. |
| Community trajectory inputs | 99D observations with pelvis-relative orientation history and ten 73D future goals | Simulator/imagination agreement, noise and scale parity, independent timing checks, input/config validation. |
| Dynamics parameterization | Optional local state-increment model alongside current explicit integration | Coordinate/integration/error tests; report normalization units; no numerical comparison of incompatible NSE definitions. |

Build targeted checks for actual coordinate, boundary, checkpoint, or gradient risks. Do not add tests that only restate configuration assignments. Keep state/FK/reward arithmetic precise enough for meaningful simulator comparisons.

## Phase 4 — Run controlled LAFAN1 experiments after execution resumes

Begin with the completed orientation configuration as the measured reference. Add one new factor at a time: normalization, WM-only warm-up, critic error features, trajectory inputs, or clip-end semantics. If a later branch is based on an improved configuration, record that parent and do not present the sequence as an independent one-factor comparison against the original baseline.

For each candidate, first run targeted regressions and a full-scale short pilot. Use actual throughput to set a finite Slurm budget with compilation/checkpoint headroom. The earlier three-hour budget is a comparison anchor, not an instruction to launch all variants or reserve many GPUs.

Evaluate saved checkpoints on all 40 motions. Report curves against **both real environment transitions and elapsed training time**, plus actor updates and imagined transitions. Report root/path errors and failure times alongside return. Repeat promising comparisons with at least three explicitly recorded seeds before making a robustness claim. Keep seed-0 exploratory results labelled as such.

**Progress gate:** reach and retain the paper's 0.82 LAFAN1 checkpoint-evaluation target; report any failures separately. Reaching 0.82 once is not proof of paper equivalence or full reproduction. Record time to threshold and any later regressions, rather than selecting only a favorable checkpoint.

## Phase 5 — Profile and optimize throughput

Profile replay sampling, WM fitting, imagined forward/backward, critic fitting, metrics, and simulator synchronization under a Slurm allocation. Existing phase medians suggest policy/critic and WM work deserve attention, but do not identify a kernel bottleneck.

Test direct replay indexing/masking, reduced host reads, fused Adam, and compile scope independently. Retain terminal-transition supervision, no cross-reset windows, correct history, and ring-buffer bounds. Distinguish a sampler speedup from a change in the training-data distribution. Log compilation separately from steady-state timing.

**Acceptance:** correct boundary behavior, finite losses/gradients, comparable checkpoint evaluation, and measured end-to-end speed improvement. Do not report isolated-kernel throughput as total training speed.

## Phase 6 — Optional algorithm and systems ablations

After the reference pipeline is stable, separately evaluate analytic soft termination, EMA, bf16 MLPs, reset perturbations, and LR changes. Keep fp32 integration/FK where required. Save live and EMA weights separately and evaluate both. Self-collision changes require an explicit physics profile and cannot be counted as a pure implementation acceleration.

Add PPO-Sim with matching motions, actor observations, rewards, termination, DR, network capacity, and complete-clip evaluation. Define its rollout batch and collection count explicitly. Compare equal environment-interaction budgets and equal wall-clock budgets; optimizer iterations across algorithms are not interchangeable. This baseline is unimplemented and unrun locally.

H×critic and push recovery are subsequent validation tasks. The existing no-critic option and `eval_push.py` are starting points, not completed ablations. State the push protocol and any departure from the paper's unspecified settling criterion.

## Phase 7 — Large-library and deployment validation

Lock a BONES-SEED motion-ID manifest and explain unresolved differences from the paper's 129514 motions before training. Inspect difficult categories rather than dropping failures to match counts. Validate packed reference precision, full-library evaluation, and shard aggregation first; size RAM/VRAM and runtime from a pilot.

Report full-library coverage, per-category failures, full-reference returns, and separately defined held-out evaluation. Do not equate training-library coverage with generalization or small selected video clips with full-motion success.

Cross-backend policy-I/O contracts, interactive playback, export, velocity interfaces, and real-robot deployment remain later work. Real-robot readiness is not established by the current simulation videos or this plan.

## Status and completion evidence

| Work item | Current status |
|---|---|
| Star-ranked source audit and evidence index | Completed; no third-party GPU execution |
| Packed-to-NPZ validation-path and goal-padding CPU probes | Completed; limited isolated checks |
| Existing seed-0 baseline and orientation experiment | Completed; results above |
| Real simulator recording tools and result webpage | Implemented and recorded previously |
| Phases 1–7 above | Proposed; not implemented or scheduled by this publication |
| New GPU submissions | On hold |

Each future change should include a focused source diff, new config/profile identity, relevant validation, source/data hashes, and an updated measured-results record. Mark gates complete only when their evidence exists.
