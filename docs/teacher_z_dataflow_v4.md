# Teacher-z v4 data flow

> Retired: this 173-D physics-only contract does not match the intended
> Teacher-to-Student experiment.  Use `teacher_z_dataflow_v5.md`; v4
> checkpoints must not be resumed as v5.

This is the privileged simulation experiment in which a frozen Physics Teacher
supplies a 32-D observation to a PPO Actor.  It is intentionally distinct from
the legacy Student/confidence/action-residual path.

```text
K1 simulator
  |-- robot proprioception + executed-action history -----------+
  |-- exact L/R contact, slip, forces + friction/mass --> Teacher v4 --> z[32]
  |-- exact contact/object/parameter privilege ----------------------------+
  |                                                                        |
reference motion --> Actor reference branch                                |
robot observation --> Actor state branch                                   |
z[32] -----------> Actor physics branch -----------------------------------+
                         |
                         +--> one PPO action[22] --> K1 simulator

robot/reference Critic observation[843] + physical privilege[98]
                         --> asymmetric Critic --> value/advantage
reward + value + Actor log-prob --> PPO update (Actor and Critic only)
```

## Fixed contracts

- Teacher input: 173 values.  This is 72 robot state/action-history values, 3
  base linear-velocity values, 28 exact left/right contact values, 68 values for
  four masked object slots, and 2 randomized friction/mass values.
- Teacher input contains no HoloMotion Critic tensor and no current/future
  reference target.
- Teacher latent: 32 values.
- Actor: the existing 506-value HoloMotion observation plus the separately
  encoded 32-value latent.  The MLP concatenates state, reference and physics
  branches; it does not add z to the state token.
- Critic: the existing 843-value observation plus 98 exact physical values.
  Critic privilege never enters the Actor.
- Action: exactly one 22-value Actor output.  There is no action residual,
  Student Dynamics feature, confidence gate, or second action head in this
  mode.

## Two training phases

1. Train Teacher v4 with a frozen, known-good source Actor.  Its supervised
   heads predict next normalized state delta, left/right contact probability,
   contact load/tangential load/slip/utilization, contact mode, and direct
   multi-step state deltas.
2. Freeze the Teacher and train a new PPO Actor/Critic from scratch.  PPO never
   updates the Teacher.  The Teacher latent changes with simulator physics and
   becomes an Actor observation.

Legacy Teacher v1-v3 checkpoints remain loadable for their original experiments
but are rejected by the v4 direct-observation mode.  A v4 Teacher checkpoint is
therefore required before starting phase 2.
