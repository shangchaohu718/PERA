# PERA website: source audit and implementation plan

Snapshot: 2026-11-05. Public-facing name corrected to **PERA** on 2026-11-06, matching the repository **PERA-g1**; no expansion of the acronym is invented.

## Findings before implementation

| Item | Available evidence | Treatment on the website |
| --- | --- | --- |
| G1 model | Actual 29-DoF G1 MJCF and STL meshes in `HoloMotion/assets/robots/unitree/G1/29dof/` | Re-render these assets; no AI robot images or downloaded reference-site media |
| Public Actor | `checkpoints/holomotion_v1.4`, README, `modumotion_g1/official.py` | Explicit attribution to the verified HoloMotion G1 v1.4 base |
| Modular experts | `modumotion_g1/domain_moe.py`, including `set_domain_enabled` and anchor group mixture | Architecture uses frozen public weights, new FFN/router and explicit bypass; no unsupported preservation guarantee |
| Teacher route | `scripts/instinct_dagger.py`, `reports/instinct_dagger_v1/run/final_summary.json` | Separate BC + DAgger route, frozen official InstinctMJ teacher, no PPO or world model in this experiment |
| PPO route | `reports/roundhouse_expert_20261105/summary.json` and local training implementation | Separate PPO martial-arts expert; zero teacher action queries; separate critic updates |
| Flat-floor rollouts | Baseline and expert NPZ files under `reports/kungfubot_original_20261105/` and `reports/roundhouse_expert_20261105/` | Real recorded robot states, not reference poses; full recorded episode per clip |
| Terrain rollout | `reports/velocity_course_v11/seed7002/physical_replay.npz`, `scene.mjb`, reports | Complete 50 cm climb segment from one successful continuous course, original terrain retained; 0.6 s controller entry blend disclosed |
| Results | Seven matched clip metrics; independent held-out climbing report; 21 BC/DAgger round loss records | Downloadable snapshot, exact source SHA256 hashes, CSV, units and evaluation limits |
| Old-skill retention / ablations | Software off-switch check and frozen-weight checks exist; reported expert-on regressions on other motions | All seven results exposed, including regressions. Physical retention and ablations labeled unreported |
| Paper, arXiv, BibTeX, authors | No verified PERA paper draft or author list found in inspected project documentation | Forthcoming/pending, no invented URLs, IDs, author affiliations or citation |
| Recovery expert | No validated PERA result selected in this audit | Planned; references and unrelated teachers are not treated as trained PERA skills |

The supplied brief cautions against *assuming* HoloMotion is the base. Inspection confirms that it **is** the public Actor used in the reported modular-expert experiments. A separate MLP-MoE / world-model research branch also exists; it is not combined with the reported PPO/DAgger experiments or presented as the same algorithm.

## Visual direction

White/off-white, Apple-like dark gray typography, restrained #0071E3 blue, large titles and generous space. No third-party reference media. Hero emphasizes an actual G1, with truthful public-policy labeling. Navigation: Overview → Method → Skills → Results → Research. Skills use a keyboard-accessible tab list; the diagram has independent inference/training views. Data are deliberately secondary to the robot and the research narrative.

## Implementation sequence and outcome

1. Audit real code, available trajectories, checkpoints, metrics and reference sites.
2. Build an independent white studio renderer; preview actual robot and feet before batch rendering. Preserve originals and lossless masters.
3. Implement Vite + React with configured content, native SVG, lazy local H.264 media, responsive CSS, reduced-motion support and semantic keyboard controls.
4. Export evidence with hashes, include all seven metric comparisons, expose missing claims explicitly; test browser interactions, sizes and links.
5. Build static output, include GitHub Pages Actions workflow and deployment instructions. Existing private repository rejects Pages setup with HTTP 422: current account plan does not support Pages for this repository. No repository visibility was changed.

## Design references

Inspected the public [SuperMimic](https://supermimic.github.io/), [MorphMimic](https://lzyang2000.github.io/morphmimic/), and [Apple](https://www.apple.com/) pages. Used their narrative restraint, emphasis on robot motion and typographic hierarchy as references. No layouts or media are copied.

## Media caveats

These are re-renders of archived closed-loop physics states, **not fresh policy evaluations**. The flat-floor source simulator is Isaac Lab; the web renderer is MuJoCo. Playing a recorded state in a different renderer does not revalidate physical behavior. Nearest-frame temporal sampling converts 50 or 25 Hz to 30 fps at 1× time, with no pose interpolation. Full episodes are looped as recordings; their end poses do not necessarily join seamlessly. The 5.04 s kick is intentionally not lengthened to satisfy a nominal 6–12 s recommendation.
