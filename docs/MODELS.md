# Models

PERA uses the official HoloMotion G1 v1.4 `model_14000` Actor as a frozen public base.

| Resource | Distribution |
|---|---|
| Official Actor, config and Critic | Download from the original publisher; [pinned links and verification](HOLOMOTION_DOWNLOAD.md) |
| `checkpoints/holomotion_experts/pera_25skills.experts.pt` | One file containing all 25 matching FFNs/routers, supplied in the private scene release |
| References, frozen context encoders, normalization and scene geometry | Paired scene archive; [resource guide](RESOURCES.md) |

The expert bank is 52,976,259 bytes. SHA256:

```text
280c79290c1a8ed7fc559602f0ca7415625d995a2f248a158310d55d756f500f
```

Every expert is tied to public digest `898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d` and its original input/action/context contract. The loader checks identities and tensor shapes. Do not substitute the historical self-trained base or competitive expert bank.

## One checkpoint

The expert-only file and external official base assemble one Actor in memory. To save a complete single checkpoint for local use:

```bash
python tools/materialize_holomotion_bank.py \
  --experts checkpoints/holomotion_experts/pera_25skills.experts.pt \
  --base-dir /path/to/holomotion-v1.4 \
  --output /path/outside/PERA/pera_holomotion_25skills.pt
```

Pass that file with `--bank /path/outside/PERA/pera_holomotion_25skills.pt` to either replay launcher; `--base-dir` is then unnecessary. Official weights remain external to the distributed repository and archive.

All 25 experts stay resident; each call selects zero or one. Other domain FFNs do not execute. All-off restores the exact public path. Robots share weights and keep separate temporal caches. Routing retains checkpoint anchor-group blending and the native public Top-2; it is not a global Top-2 competition among all 25 domains.

The [skill catalogue](HOLOMOTION_SKILLS.zh-CN.md) distinguishes the 15 packaged scene replays from the remaining historical task experts. Network-output parity has been checked for all 25; that alone is not a physical success-rate evaluation.

[Archived model notes](history/MODELS_20261113.md) describe earlier research families and are not the current release recipe.
