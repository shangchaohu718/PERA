# Teacher-z v5 shared Critic data flow

Teacher v5 is the privileged upper-bound model that will later supervise a
non-privileged Student.  The Teacher Encoder and asymmetric Critic receive the
same information tensor at every time step.

```text
HoloMotion Critic observation[843] ----+
                                       +--> shared privilege tensor[941]
physical privilege[98] ----------------+          |             |
                                                  |             +--> Critic --> V(s)
                                                  +--> split branch encoders
                                                       843->128, 98->128
                                                       concat 256->128
                                                       Teacher history --> grouped z[64]

Actor observation/reference[506] + z[64] --> PPO Actor --> action[22] --> simulator

later: deployable history --> Student --> z_student, supervised by frozen z_teacher
```

## Shared 941-D contract

- 843-D original HoloMotion Critic tensor:
  - 417-D current full robot state;
  - 46-D current reference/anchor state;
  - 380-D ten-frame future reference.
- 98-D physical privilege:
  - 28-D exact left/right contact position, velocity, normal, normal/tangential
    force, contact flag, slip speed, and friction utilization;
  - 68-D four masked external-object slots;
  - 2-D randomized friction and trunk-mass parameters.

The Teacher Transformer uses up to 64 causal time steps.  Before attention,
the 843-D state/reference block and 98-D physical block are independently
normalized and projected to 128 dimensions, concatenated, and fused from 256
back to 128 dimensions.  This gives physical truth equal pre-fusion capacity
without changing the information shared with the Critic.  The information
channels remain identical even when Teacher and Critic use independent
architectures and gradients.

The production K1 grouped experiment is configured by
`configs/teachers/k1_transformer_shared_critic_grouped_z64.yaml`.  Its latent
is routed rather than merely widened: contact 8D, force 16D, kinematics 16D,
and dynamics 24D.  The Actor receives their concatenation as one 64-D
observation; task-specific Teacher heads are restricted to their declared
groups during Teacher pretraining.  The earlier
`k1_transformer_shared_critic_z32.yaml` remains an explicit shared-z32
ablation and its checkpoints are not interchangeable with grouped z64.
The grouped configuration also enables a `0.1` parameter-reconstruction loss
from `dynamics24` to the randomized friction/mass pair; without this explicit
term, next-state loss alone can learn to ignore those two coordinates.

K1 keeps two stable contact slots (left foot, right foot).  G1 uses four
(left/right foot and left/right wrist), so G1's 126-D physics suffix must not
be copied into K1: K1's suffix remains 98D and its full shared tensor remains
941D.

The Teacher prediction decoder additionally receives the current executed
action to learn its physical consequence.  That action does not enter the
pre-action latent `z_t` and is not part of the shared 941-D Encoder contract.

## Training phases

1. Freeze a known-good K1 Actor and train Teacher v5 from scratch using
   next-state, contact, force, slip, contact-mode, and multi-step targets.
2. Freeze Teacher v5 and train a privileged PPO Actor with `z_teacher` as an
   observation.  This tests whether an informative physical latent helps
   control.
3. Train a deployable Student from non-privileged robot history to imitate the
   frozen Teacher latent/predictions, then replace `z_teacher` with `z_student`.

Teacher v4 used a 173-D physics-only Encoder input and is retired for this
research question.  Its partial checkpoints must not be resumed as v5.
