# Goal encoding and state information

The paper specifies 93 proprioceptive values per frame and ten future reference frames as `g_t`, but does not enumerate the coordinates used for those future targets. Therefore our reference-only and relative-goal variants are hypotheses, not known author settings.

## A limitation of the default 68-value reference encoding

The default reference-only goal contains joint position/velocity, projected gravity, body angular velocity, heading-frame linear velocity and root height. It contains no global reference XY/yaw or reference-to-robot pose error.

Consider two robot states with identical joint configurations, velocities, roll/pitch and previous action, but differing by a horizontal translation against the same reference. The 93-value proprioception is identical; the reference-only goal is identical; the global root-position reward is different. A finite observation history cannot observe a constant global translation either. Thus this goal interpretation has no direct feedback for global XY drift. An analogous ambiguity exists for absolute yaw. This does not establish a flaw in the paper: its actual `g_t` may encode the relative pose information that is missing from this interpretation.

The optional `--anchor-goal` appends current reference-to-robot torso position/orientation to `g_t`; it does not change the 93-value proprioception. It is explicitly labelled because its robot-pose availability and noise model are additional assumptions. A better score with this variant cannot by itself identify the author's implementation.

## Critic state

Table A.4 specifies state, goal and simulator context as critic inputs. The inherited implementation used the world model's heading-invariant state features, dropping XY/yaw, and optionally compensated with four privileged root-error values. Neither change is necessary to represent the paper's stated input directly.

The default critic now receives the complete 71-value state (floating-base pose, joints and velocities), goal and simulator context, with no extra four-value root-error input. The old canonical representation remains an explicit `CriticCfg.state_features="canonical"` option. A regression check verifies that the default path retains sensitivity to root translation while the old path discards it.

The first formal run uses the complete-state critic and reference-only goal. Relative-goal runs, if performed later, must have separate configs, manifests, checkpoints and evaluation reports. Early results should be interpreted with this observability limitation, rather than silently adding inputs to reach a target score.
