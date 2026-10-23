# Paper contract and implementation decisions

This is an unofficial reproduction, adapted from CuboiLeo/SuperMimic-Reproduced (MIT). The objective is to match every explicit paper setting and expose choices the paper does not specify. A matching score alone is not sufficient evidence of reproduction. No convergence claim has been verified here yet.

Source: [paper v1](https://arxiv.org/pdf/2610.28480), sections 3, 4, Appendices A–F. Community implementation comparisons are in `community-review.zh-CN.md`; their numbers are external reports and may use different metrics.

## Explicit settings

| Item | Paper | Implementation |
|---|---|---|
| Robot / timing | G1 29 DoF, 50 Hz, 4×5 ms substeps | mjlab G1 + MuJoCo Warp |
| Actor observation | q−q0, qdot, body gyro, projected gravity, previous action; history 10 | 93 dimensions per frame; 930 total; no root position, linear velocity or relative pose appended to this observation |
| Goal | 10 future reference frames | Default reference-only goal; exact encoding is unspecified (see below) |
| World model | MLP 2048,2048,2048; k=8 autoregressive window | Learned dynamics and termination, analytic observation/reward |
| Motion encoder / decoder | Each 2048,1024,512,256; tanh latent 32 | Same widths |
| Critic | 256,128; inputs state, goal, sim context | Complete 71-value state, same widths, no extra root-error input by default |
| Policy update | H=32, γ=.99, actor Adam β=(.9,.95) | Differentiate through fixed WM weights; critic tail value |
| Critic update | λ=.95; 5 epochs, 4 minibatches | Targets fixed per imagined rollout |
| Batches | 4096 envs; 10×2048 WM windows; 4096 imagined rollouts | Per GPU, unchanged in full-scale pilot |
| Collection | Executed action noise std .1; FIFO 1000 steps/env | Executed noisy action stored separately |
| Schedule | UTD .4; LR 2e−4 → .03× over 250k; 320k iterations | One iteration is one vectorized simulator step; 2 updates per 5 steps |
| Clipping | WM/critic norm 1; actor norm 1 then element .1 | Applied after summing temporal objective, without dividing by H/k |
| Precision | TF32 | fp32 tensors, TF32 matmul enabled; no bf16 autocast |
| Termination | pelvis z error >.25 m or gravity-z difference >.8 | Same; no end-effector failure |
| Reward | Appendix A.2; dt=.02; no collision penalty; joint-limit −2 | Positive maximum .1 per control interval |
| Evaluation | Every complete clip, first-frame reset; no noise/DR, friction 1 | Failure leaves zero reward for remaining reference intervals; report both macro and reference-length-weighted means |

## Underspecified choices (must not be presented as known author settings)

1. **Goal encoding:** Table A.1 specifies proprioception but does not fully enumerate `g_t`. Default: 68 reference-only features per frame (joints 29, velocities 29, gravity 3, angular velocity 3, heading-frame linear velocity 3, height 1). This omits an absolute XY/yaw target. The optional `--anchor-goal` appends relative torso pose to the goal, retaining the same 93-dimensional proprioception. These are different hypotheses, not proven author settings. Never silently select the higher-scoring hypothesis.
2. **State representation:** 71 floating-base/joint state values; WM inputs are heading-invariant and scaled. Outputs combine learned velocity increments and pose corrections with integration. State normalization scales are an implementation choice, so our reported WM prediction error cannot automatically be compared with the paper's .007–.011 without aligning normalization.
3. **Termination head:** learned 64×64 MLP on root-error features with BCE; reward and observation analytic. The paper allows learned or analytic heads but does not specify these hidden sizes.
4. **Reward anchor:** torso anchor for body-relative rewards, pelvis for root rewards/termination. The imported baseline used torso for its global root rewards too; this repository changes those global terms to pelvis. The paper does not list every tracked link and asset collision detail. Record robot assets and keep self-collision behavior from the pinned mjlab asset.
5. **Reset initialization:** reference pose and velocity, no additional pose/velocity perturbation in the paper profile. Perturbation magnitudes are not specified by the paper. Train episodes time out at 10 seconds; this duration is also an implementation choice, not a reported paper constant.
6. **Simulator ordering:** reset robot state at reference frame f, target f+1 at the next interval. Reward and termination use the current target before advancing to the next. The last interval is detected at target==last, avoiding an extra repeated final target. Upstream mjlab evaluates reward/termination on derived quantities stale by one 5 ms physics substep, then forwards state for observation/replay. We register an always-false first termination term as a lifecycle bridge: forward current state and refresh body-relative targets before evaluating any reward/failure. Its extra forward costs time, which must be included in measured throughput. A regression test compares the actual logged reward with analytic heads on the replayed next state; post-step FK tests alone do not validate this alignment.
7. **Clip ends in imagination:** inherited behavior clamps reference at the final frame and retains bootstrap. This needs a separate ablation against masking the clip end; the paper does not specify the boundary convention.
8. **Reset sampler:** paper gives the 50/40/10 mixture and temperature 25; EMA update rates and treatment of truncated return estimates are inherited implementation choices.
9. **Replay startup:** collect 40 steps before optimization; no separate world-model pretraining. Startup and resumed runs both refill replay before updating. Resume restores model/optimizer/RNG/counters, rebuilds simulator and replay, and is not bit-for-bit continuation.

See `goal-observability.md` for the reference-only goal's inability to observe global drift. Since 2026-10-23, the critic defaults to the complete state from Table A.4, rather than sharing the WM's heading-invariant features. The old pilot used the latter and is not an interchangeable training checkpoint for this configuration.

## Changes from the imported baseline

- Evaluation divides by the entire reference interval count; early failure no longer shortens the denominator. Weighting uses reference length rather than survived steps. Protocol is labelled `full-reference-v1` to prevent mixing older reports.
- Detect final target before repeating the last interval.
- Eq. 1 and Eq. 2 use temporal sums before clipping. Baseline divided WM loss by k and actor objective by H.
- Disable the critic's four additional root-error inputs by default; keep an explicit opt-in for comparison.
- Preserve all critic minibatch samples when batch size is not divisible by minibatch count.
- Record source/dataset hashes, dependencies, hardware and Slurm job in each run; reject changed datasets and mismatched model/environment configs on resume; write checkpoints atomically.
- Refill replay on resume instead of immediately sampling an empty new buffer.

## Acceptance gates

1. CPU/GPU FK and frame/velocity conventions; clean/noisy executed action distinction; replay never crosses reset/ring boundaries.
2. Compare simulator logged reward/termination and imagined analytic heads at the same time/reference, including terminal transition.
3. Finite-difference actor gradient check with the WM frozen; terminal critic keeps state gradient; GAE stops bootstrap on termination.
4. H200 full-scale pilot at paper batch/model sizes; record compilation cost separately and estimate training time from measured throughput.
5. Full 40-clip evaluation, per-clip failure times and full-reference denominator; fixed manifest; repeated seeds after the pipeline is correct. Reference-only and relative-goal experiments remain separate.
6. Expand to BONES-SEED only after LAFAN1 issues are understood; pin filtering IDs and verify counts rather than comparing unmatched subsets.
