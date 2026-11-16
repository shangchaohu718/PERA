# Implementation map

PERA expands a frozen G1 MoE controller with task-specific FFN experts and competitive routing.

| Component | Code | Role |
|---|---|---|
| Public interface | `src/pera/__main__.py` | Explicit training/evaluation commands; no default jobs or downloads |
| G1 base | `src/blinkmimic/networks.py`, `moe_policy.py` | MoE-MLP Actor, eight Top-2 public experts and one shared FFN |
| Base learning | `src/blinkmimic/train.py`, `env.py`, `mdp.py` | Real MuJoCo-Warp collection, world model, policy learning and tracking rewards |
| Expansion | `src/pera/routing.py` | One new FFN/router, actual 9-way Top-2 selection, frozen public state and cached feature replay |
| Task context | `src/pera/context.py` | Canonical context-aware FFN and competitive routing, shared with independent training |
| Expert bank | `src/pera/switching.py`, `configs/expert_bank.json` | One public Actor, 30 installed experts, explicit mutually exclusive switches and self-contained inference export |
| Shared-world demo | `src/pera/multi_robot.py`, `showcase.py`, `interaction_showcase.py`, `configs/showcase.json` | Independent robot histories, grouped dispatch through one Actor, one shared MuJoCo physics state |
| Fusion ablation | `src/pera/fusion.py` | Separate continuous-fusion comparison; not competitive admission |
| Robot contract | `src/pera/robot.py`, `configs/robot.json` | Joint order, defaults, action scales, reference and sensor metadata |
| Martial PPO | `reproduction/experiments/blink_competitive_martial_ppo_v1_20261106/` | Isaac Lab rollout, separate Critic, 24 Actor updates per iteration |
| Reward definitions | `reproduction/roundhouse_impl.py`, `configs/martial_isaac.yaml` | Original retained martial reward code and recipe |
| Multi-teacher distillation | `reproduction/experiments/g1_competitive_suite31_v1/` | BC, online DAgger, teacher adapters, actual update accounting and retention evaluation |
| External task bridges | `src/modumotion_g1/`, selected `reproduction/scripts/` | Named joints, sensor context, teacher native physics and action conversion |
| Stairs A/B | `reproduction/experiments/blink_stairs_routing_ab_20261106/` | Matched competitive/fusion training and evaluation |
| Full-Actor extension | `reproduction/experiments/blink_stairs_full_finetune_20261106/` | Explicitly unfreezes public weights; separate checkpoint format |
| Transfer extension | `reproduction/experiments/martial_transfer_v1_20261106/` | Fixed-checkpoint unseen-motion study; additional data required |
| Historical integration | `integrations/HoloMotion/` | Selected HoloMotion modules for Isaac/reference compatibility, not the primary MLP base |

The date-bearing directories preserve the original experiment identities and checkpoint schemas. Public commands use stable workflow names. Compatibility wrappers delegate to the canonical routing module; there is only one native base runtime.

Upstream source repositories remain external and pinned in `resources/upstreams.json`. Original weights, motion arrays, meshes, generated replay, logs and videos are excluded. `provenance/source_files.json` maps retained source origins to the final files; `provenance/VALIDATION.json` records checks and their limits.
