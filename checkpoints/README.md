# PERA K1 Teacher Checkpoints

The binary checkpoints are published as assets of the
[`k1-teacher-checkpoints-v1`](https://github.com/shangchaohu718/PERA/releases/tag/k1-teacher-checkpoints-v1)
release so that normal Git clones stay small.

## Assets

| Asset | Motion / role | Size (bytes) | SHA256 |
|---|---|---:|---|
| `k1_walk1_subject1_mlp_model_2499.pt` | K1 walking policy; complete PERA training checkpoint | 2,665,397 | `5ee1cc96b67bc4672b7093fc90ff3938a863bb63a64b682795e954b71537199b` |
| `k1_moonwalk_f3a_model99_frozen.pt` | Frozen deterministic Moonwalk F3A policy for Teacher collection | 931,845 | `5784fec6d6011d0f478e6286d25d5032531d8e75f66d6c09ecb267a1fd068133` |
| `k1_moonwalk_v13_direction_model999_frozen.pt` | Frozen deterministic direction-gated Moonwalk policy for Teacher collection | 932,213 | `6526dd5d7f4ad2e3f62d69931cd99d6e4c6dfc9942434e8dd17ded66a192feff` |
| `k1_moonwalk_v13_f2r1_model2929_frozen.pt` | Frozen deterministic pelvis-relative slide policy for Teacher collection | 931,957 | `680d439e91e732f463b82fab51af232c493e57d1d558016d12d6017963f9fbd6` |

Total binary payload: 5,461,412 bytes (about 5.21 MiB).

The Moonwalk held-out policy is intentionally not included in the training set.
A verified K1 dance checkpoint was not available at publication time and must not
be substituted with the incompatible G1/HoloMotion dance checkpoint.

## Download

```bash
gh release download k1-teacher-checkpoints-v1 \
  --repo shangchaohu718/PERA \
  --dir checkpoints/downloaded
```

Verify the files against the SHA256 values above before loading them. PyTorch
checkpoint files use pickle-based serialization; only load files from a trusted
release.
