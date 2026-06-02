# FlowCLIP tests

Lightweight regression and smoke tests for the optical-flow modality.

## Run

```bash
pip install -r requirements-dev.txt
pytest -q
```

All tests run on **CPU** and require no dataset.

## What's covered

| File | Scope | Needs CLIP weights? |
|------|-------|---------------------|
| `test_flow_encoder_unit.py` | `FlowCLIP` 2→3 flow projection, forward output shape, dtype-cast guard — via a stub encoder | No (fast) |
| `test_flow_encoder_clip.py` | End-to-end flow path through real CLIP ViT-B/16 returns a `(B, 512)` embedding | Yes — **skips** if `clip`/weights/network unavailable |
| `test_configs.py` | Every `configs/**/*.yaml` parses with `yaml.safe_load`; flow config declares `use_flow`/`flow_tmpl`/`flow_root` | No |

`test_configs.py` is a regression guard for the PyYAML 6+ crash
(`TypeError: load() missing 1 required positional argument: 'Loader'`) that
previously broke `test.py` and `train.py`.

## Scope / limits

These tests verify the flow **architecture and config plumbing**. They do
**not** measure HMDB-51 / UCF101 / K400 accuracy — end-to-end evaluation
needs a GPU, extracted frames + optical flow, and a fine-tuned checkpoint.
See [`../results/README.md`](../results/README.md) for how to reproduce
reported numbers.
