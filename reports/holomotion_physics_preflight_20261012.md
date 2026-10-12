# HoloMotion physics preflight (2026-10-12)

## Scope

One seed (`42`), one CUDA environment, one named K1 walking clip, 20 PPO
iterations and 32 simulation steps per iteration.  This is a wiring and trend
check, not an algorithm-quality result.

The controlled comparison is:

- `matched`: original HoloMotion actor, physics randomization and independent
  left/right foot rewards enabled, no physics network;
- `physics`: the same task plus the deployable Dynamics/Teacher residual.

`vanilla` uses the original task and is reported only as a compatibility
reference.  It is not an isolated algorithm comparison because its task and
reward differ.

## Aggregate results

| run | mean reward (all 20) | mean reward (last 5) | done count | last-5 ratio | last-5 clip fraction | last-5 analytic KL |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | -3.3335 | -2.4426 | 58 | 0.9835 | 0.1167 | 0.01354 |
| matched | -3.0224 | -3.2515 | 57 | 0.9942 | 0.0979 | 0.01412 |
| physics | -1.9408 | -1.8613 | 39 | 0.9878 | 0.1573 | 0.01399 |

The physics model completed three updates (four optimizer steps each) at
iterations 5, 11 and 17.  Its total loss decreased from `3.4099` to `3.2827`
to `2.9416`.  Mean confidence remained deliberately low, moving from `0.0222`
on the first rollout to `0.0250` on the last rollout.

These numbers are a positive signal for a longer matched comparison, but are
not sufficient evidence of improvement: the budget is tiny, there is only one
seed/environment/clip, and the policy trajectories diverge after their first
updates.

## Defects exposed by the preflight

1. Episode resets allowed reference starts too close to the clip end.  The
   sampler/task now terminate or reset while preserving the complete future
   reference window.
2. The physics actor returned log-probabilities as `[B,T]` while rollout values
   were `[B,T,1]`, silently producing a `[B,T,T]` PPO ratio.  The wrapper now
   preserves the scalar event axis.  After the repair the physics ratio,
   clipping and gradient norms returned to the baseline scale.
3. Physics mode previously changed the task at the same time as the network.
   `--matched-physics-task` now permits a fair no-physics control.
4. The runner did not record left/right reward terms.  New rollouts and logs
   now include mean reward terms for subsequent experiments.

## Artifacts

- `runs/holomotion_preflight_20261012_vanilla_v3`
- `runs/holomotion_preflight_20261012_matched`
- `runs/holomotion_preflight_20261012_physics_v2`

The earlier `physics` run (without `_v2`) is invalid because it contains the
log-probability broadcasting defect and must not be used for comparison.
