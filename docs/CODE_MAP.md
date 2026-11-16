# Implementation map

This open-source release of PERA is based on the official HoloMotion G1 v1.4 weights.

| Component | Code | Role |
|---|---|---|
| Public interface | `src/pera/__main__.py` | Scene demonstrations, checks and explicit experiment entry points |
| Official base | `src/modumotion_g1/official.py`, `integrations/HoloMotion/` | Load the official Actor and preserve its observation/action contract |
| Task FFN/router | `src/modumotion_g1/domain_moe.py` | Matching domain parameters and original checkpoint routing |
| Expert bank | `src/pera/holomotion_bank.py` | One Actor holding 25 experts; at most one domain active per call |
| Robot memory | `src/pera/multi_robot.py`, `holomotion_showcase.py` | Shared weights with independent temporal caches |
| Scene runtime | `src/pera/interaction_showcase.py`, `multirate_showcase.py`, `compact_showcase.py` | Scene physics, task observations, action mapping and rendering |
| Fifteen-scene suite | `tools/replay_suite.py`, `configs/reproduction_suite15.json` | Sequential replay, measured reports and videos |
| Motion and geometry helpers | `src/blinkmimic/motion.py`, `kinematics.py`, `mdp.py`, `rotations.py`, `sensing.py` | Motion loading, kinematics, observations and sensor conventions |
| External task bridges | `src/modumotion_g1/`, selected `reproduction/experiments/` | Native teacher interfaces, task context and expert-training adapters |
| Distribution checks | `tools/build_scene_release.py`, `verify_scene_files.py`, `check_holomotion_release.py` | Hash-verified resources and archive construction |
| Project website | `website/` | Editable website and individual recorded task demonstrations |

Base-policy pretraining is excluded. The remaining `blinkmimic` modules provide runtime utilities and compatibility for retained expert experiments; their original attribution is preserved in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). They do not select the public Actor used by the HoloMotion demonstrations.

Historical competitive routing, A/B experiments and fixed-checkpoint evaluation code are retained separately from the official-base scene implementation. Their formats and protocols are not interchangeable. Current release scope is recorded in [RELEASE_SCOPE.json](../provenance/RELEASE_SCOPE.json).
