# Results

This file separates what has been **measured in this repository** from
numbers **inherited from upstream ActionCLIP** and from results that are
**not yet produced**. Nothing here is a placeholder dressed up as a
measurement.

## 1. Verified in this repo

### Flow-encoder smoke test (CPU)

`FlowCLIP.forward` maps a `(B, 2, H, W)` optical-flow tensor through the
learned 2→3 projection and CLIP's ViT-B/16 image encoder to a `(B, 512)`
embedding. Full pytest output: [`test_run.log`](test_run.log).

```
17 passed in 5.46s
tests/test_flow_encoder_clip.py::test_flow_forward_returns_clip_embedding PASSED
tests/test_flow_encoder_unit.py::test_flow_proj_is_2_to_3_conv PASSED
...
```

Reproduce:

```bash
pip install -r requirements-dev.txt
pytest -q
```

This verifies the **architecture and config plumbing**, not task accuracy.

## 2. Inherited ActionCLIP baselines (NOT FlowCLIP results)

The Kinetics-400 / HMDB-51 / UCF101 tables in the top-level
[`README.md`](../README.md#pretrained-models) are the **RGB-only numbers
reported by the original ActionCLIP paper** (Wang et al., "Table 6"). They
are reproduced here as a reference baseline for the RGB stream. They do
**not** reflect the optical-flow (RGB+Flow) contribution of this fork.

- **K400, ViT-B/32, 8-frame (78.36% top-1 / 94.26% top-5)** — verifiable
  from the committed training log [`logs/ViT32_8F_K400.log`](../logs/ViT32_8F_K400.log)
  (final epoch 49). Note this is the upstream authors' run (`modality: RGB`,
  `type: clip_tem`, wandb project `wmm/clip_tem`, Aug 2021), reproduced here,
  not a run from this fork.
- **K400 ViT-B/16 rows and the HMDB-51 / UCF101 rows** — no training log or
  checkpoint is committed; the HMDB-51 / UCF101 links are upstream
  placeholders.

## 3. FlowCLIP (RGB+Flow) — not yet measured

No end-to-end RGB+Flow accuracy has been produced yet. Doing so requires,
beyond the smoke test above:

1. Extracted RGB frames + pre-computed optical-flow frames
   (`datasets/flow_utils.py::compute_optical_flow`, or the Colab notebook's
   flow cell).
2. A fine-tuned flow checkpoint (the public model zoo ships RGB checkpoints
   only).
3. A GPU run of:
   ```bash
   python test.py --config configs/hmdb51/hmdb_flow.yaml
   ```
   (now fixed — previously crashed with
   `TypeError: load() missing 1 required positional argument: 'Loader'`
   from `yaml.load`; the scripts now use `yaml.safe_load`).

When that run completes, drop its log here as `hmdb51_flow.log` and add the
RGB-vs-RGB+Flow comparison (including the learned fusion weight α) to the
table below.

### HMDB-51 RGB+Flow (split 1, ViT-B/16)

| stream | top-1 | top-5 | α | status |
|--------|-------|-------|---|--------|
| RGB only | — | — | — | to be measured |
| RGB + Flow | — | — | — | to be measured |

> The claim that "the flow stream improves accuracy on motion-intensive
> classes" is a **hypothesis pending the measurement above**, not a
> reported result.
