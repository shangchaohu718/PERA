# HoloMotion release status

## Scene replay update — 2026-11-16

The missing scene runtime sources are now supplied as a licensed, hashed subset in `vendor/`. The seven-region 12-second student replay passed using only checkout-local project sources. All seven task traces exactly matched the prior demonstration. External model paths are supported; no official base weights were added to this folder.

See [scene replay instructions](../SCENE_REPRODUCTION.md), the exact environment in `requirements/scene-replay.txt`, and [current verification](../../provenance/SCENE_REPRODUCTION_CHECK.json). The scene archive has also passed a complete replay after extraction into an independent directory, with exact task-trace parity. Fresh dependency installation, public asset retrieval and full training reproduction remain separate, unverified gates. The earlier audit below is retained as history.

## Historical audit

Checked locally on 2026-11-13. **The community reproduction release is incomplete.**
The selected version is official HoloMotion G1 v1.4 `model_14000`, with matching
frozen-base FFN/router plugins. The historical self-trained MoE-MLP and its
30-expert bundle are excluded by [release selection](../../configs/release.json).

## Available locally

- Twelve matching task experts are staged locally; the required official-base
  config is also present. Model identities are in
  [the weight manifest](../../resources/holomotion_weights_manifest.json).
- Official Actor/Critic weights are not bundled. Users download them directly
  from the pinned official v1.4.0 release following
  [the download guide](../HOLOMOTION_DOWNLOAD.md). URLs, revision, sizes and SHA256
  are recorded in [holomotion_base.json](../../resources/holomotion_base.json).
- Before removing the two official weight copies, all 380 declared resources
  (1,926,644,302 bytes) passed SHA256 verification, and the Actor with 12 plugins
  loaded from release-local paths on CPU. Its public digest and frozen parameters
  were verified, and a synthetic public forward was finite. This remains a prior
  network-load check, not a claim that official weights are still in the folder.
- Seven selected tasks have a local physical replay using the separate native
  research tree; [the evidence](../../provenance/CLEAN_CAMERA_SHOWCASE_CHECK.json)
  includes actual teacher-free control and robot-memory isolation checks.

```bash
python tools/verify_weights.py --manifest resources/holomotion_weights_manifest.json
python tools/check_holomotion_release.py
```

The second command currently exits with code 2 because runtime source files are
missing from the standalone folder and official weights must be installed by the user. It starts no simulator or network transfer.
Even a future static pass will not establish physical or training reproduction.

## Required before release

| Item | Current state | Completion criterion |
|---|---|---|
| Standalone runtime | Demo still imports ATHLETE, ULTRA, InstinctMJ, mjlab, instinct_rl and an ULTRA adapter through a native-root path | Supply pinned source/patches and close imports using the release checkout only |
| Models and task assets | 12 local plugins and scene assets; official base uses verified upstream-only downloads | Publish permitted plugin/scene resources and installation instructions; keep official weights external |
| Clean installation | Existing local environment only | Install documented dependencies in an isolated checkout and replay without access to the research tree |
| HoloMotion training | Historical competitive CLI recipes still target the self-trained base | Package actual official-base teacher preflight, DAgger/PPO configs, references, encoders, evaluation and resume contracts; run end-to-end checks |
| GitHub delivery | Private repository has an empty main tree and one legacy draft release without assets | Review and explicitly authorize the correct source/model release; verify retrieval as a new user |

The checked remote was `shangchaohu718/PERA-g1`, commit
`182417bf1624e49b059036360c3544c4460f6cf3`, with empty tree
`4b825dc642cb6eb9a060e54bf8d69288fbee4904`. The only draft was
`base-g1-11000-v1`, which describes the historical self-trained base and had no
attachments. It is not a HoloMotion release. Nothing was uploaded, committed or published during this audit. The two newly
staged official weight copies were then removed at the user's request; original
research checkpoints remain intact.

## Model and experiment boundaries

The 12 configured plugins are not a complete inventory of all historical
HoloMotion experts. Seven appear in the current shared-world demonstration.
Neither count establishes acceptance of every skill. The HoloMotion plugins use
continuous anchor-group mixtures with native public Top-2, not the distinct
self-trained global competitive-router experiment. Reference completion,
physical survival, task success, teacher comparison and old-motion retention
must be evaluated separately. Current videos include labeled episode loops.

Full machine-readable findings:
[HOLOMOTION_RELEASE_AUDIT.json](../../provenance/history/HOLOMOTION_RELEASE_AUDIT.json).
