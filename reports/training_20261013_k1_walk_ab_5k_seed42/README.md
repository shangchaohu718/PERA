# K1 single-motion HoloMotion / Physics A/B snapshot

Snapshot time: `2026-10-13T17:35:49+08:00`

This directory is a fixed, reviewable snapshot of the run in
`runs/k1_walk_ab_5k_seed42`. The live Physics run continued after this snapshot,
so its archived metrics are intentionally marked partial. Model checkpoints are
not included because the local checkpoint set is about 2.4 GiB; the metrics and
exact run configuration are included instead.

## Experiment

- Motion: `walk1_subject1_reference_named.npz`
- Seed: `42`
- Environments: `512`
- Steps per environment: `32`
- Planned iterations: `5000`
- Actor: randomly initialized 16-expert HoloMotion policy
- Baseline: matched task/randomization without the Physics residual
- Physics: Set encoder, GRU dynamics history, calibrated confidence gate, and a
  trainable action-conditioned residual

See `ab_plan.json` and the two `*_mjlab_runner.json` files for exact commands and
runtime settings.

## Snapshot results

| Metric | Baseline | Physics |
| --- | ---: | ---: |
| Archived iterations | 0-4999 (complete) | 0-1559 (partial) |
| Last-100 mean reward | 2.7722 | 1.1943 |
| Last-100 mean done count | 94.32 | 235.46 |
| Last-500 mean reward | 2.7447 | 0.4677 |

At the matched 1460-1559 training window, Baseline had mean reward `1.5645`
and mean done count `174.23`; Physics had mean reward `1.1943` and mean done
count `235.46`.

The Physics learner had completed 260 dynamics updates. At iteration 1559 its
logged values were:

- next-state prediction NLL: `-7.1901`
- mean predicted state standard deviation: `0.3649`
- left/right contact BCE: `0.5521` / `0.4147`
- system-identification NLL: `-0.0575`
- held-out calibration skill: `0.00368`

The calibrated gate first crossed its `0.05` skill threshold at iteration 1463
with skill `0.06185`. Confidence was nonzero for iterations 1464-1469, but its
mean was only about `0.00065`. The next calibration result was `0.00468`, so the
gate closed again. This is evidence of a brief threshold crossing, not yet a
stable or material Physics contribution.

## Logs

- `baseline_metrics.jsonl.gz`: all 5000 Baseline metric records
- `physics_metrics_partial.jsonl.gz`: Physics records through iteration 1559

To inspect a compressed log:

```bash
gzip -dc reports/training_20261013_k1_walk_ab_5k_seed42/baseline_metrics.jsonl.gz | less
```

SHA-256:

```text
eadacb5326f507467bb5db2e8e530baa9ea59ead581872f424ee65df7f05631e  baseline_metrics.jsonl.gz
71262ed7b95242c02216ead0451be6e233395bf5bd9c7bc956922733113c916b  physics_metrics_partial.jsonl.gz
```
