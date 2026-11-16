# Optional training adapters

This release validates inference for the fifteen packaged scenes. It does not claim a clean-checkout reproduction of expert training. The remaining `modumotion_g1` and `reproduction` code includes native observation builders, teacher interfaces and expert-training adapters; some are also used by student inference.

The public `pera` CLI has no training or full-Actor fine-tuning command. The historical self-trained workflows are archived; see [history](HISTORY.md).

## Instinct teacher prerequisites

`vendor/` contains only the source subset needed by student replay. It intentionally does not include `instinct_rl.modules`, the full task registry or all training algorithms. Native teacher construction requires complete pinned source checkouts, not an expansion of the inference subset by guessing missing imports.

Preview their exact revisions first:

```bash
python tools/fetch_upstreams.py --plan --only InstinctMJ instinct_rl mjlab
```

When explicitly obtaining those sources, `--fetch` places them under `third_party/`. That command downloads source only; it does not install CUDA/Torch or fetch teacher weights. Use a separate native training process with these full source roots before importing `mjlab`, `instinct_mj` or `instinct_rl`. Teacher checkpoints and task data must also match their original contracts. A source download alone does not make training reproducible.

The parkour teacher constructor now checks for these full source trees and reports the missing prerequisite before attempting its teacher imports. Scene replay never constructs this teacher and needs no `instinct_rl.modules`.

## Isaac Lab prerequisites

The optional stair PPO reward registration requires an external compatible Isaac Sim / Isaac Lab runtime and the original task assets. It is not a pip dependency of the MuJoCo scene environment. Use the upstream installation procedure in a separate environment; do not mix its simulator dependencies into scene replay. No Isaac Lab package or CUDA distribution is installed automatically.

Teacher preflight, task reward equivalence, frozen public/context hashes, real Actor optimizer-step counts, teacher-free student evaluation and old-motion retention all require their own validation before a training result can be claimed.
