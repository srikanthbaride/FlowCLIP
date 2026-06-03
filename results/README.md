# Results

This file separates what has been **measured in this repository** from
numbers **inherited from upstream ActionCLIP** and from results that are
**not yet produced**. Nothing here is a placeholder dressed up as a
measurement. §4 (added 2026-06) is a *preliminary, proof-of-concept*
encoding comparison — measured here, but explicitly **not**
publication-grade, and distinct from the end-to-end run still pending in §3.

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

## 4. Encoding comparison — frozen CLIP + linear probe (preliminary, proof-of-concept)

A separate, lighter-weight experiment from the end-to-end fine-tuning in §3.
Here the CLIP ViT-B/16 backbone is **frozen** and only a linear
(logistic-regression) probe is trained on cached features, so any difference
between rows reflects the **flow encoding**, not the training budget. This is
**not** the fine-tuned RGB+Flow result of §3 (which remains unmeasured); it is a
controlled probe of *how to represent flow* for a frozen vision-language encoder.

**Setup.** HMDB-51 proof-of-concept split (1,428 train / 612 test, 51 classes,
seed = 1); A100 High-RAM on Colab; optical flow via Farneback; CLIP ViT-B/16
features L2-normalized (512-d); logistic-regression probe (`max_iter=2000`,
`C=1.0`); late fusion = concatenation of L2-normed RGB + flow features (1024-d).
Notebook: [`experiments/FlowCLIP_encoding_comparison.ipynb`](../experiments/FlowCLIP_encoding_comparison.ipynb).

Encodings compared:

- `rgb` — 8 sampled frames → CLIP → mean-pool (RGB-only baseline)
- `flow_xy` — flow_x, flow_y, magnitude packed into 3 channels (original FlowCLIP style)
- `flow_hsv` — HSV / Middlebury color-coded flow (angle → hue, magnitude → value) → RGB image
- `flow_temporal` — flow magnitude at start / mid / end packed into RGB channels

### HMDB-51 (proof-of-concept split, seed 1, ViT-B/16, frozen backbone + linear probe) — top-1 %

| encoding | flow-only | RGB + flow (fused) |
|----------|-----------|--------------------|
| RGB only (baseline) | — | **65.69** |
| `flow_xy` | 15.69 | 62.91 |
| `flow_hsv` | **17.97** | **63.24** |
| `flow_temporal` | 12.09 | 62.58 |

**Findings.**

1. Among the flow encodings, `flow_hsv` is best in **both** the flow-only and
   fused settings — consistent with the hypothesis that color-coding flow matches
   the RGB distribution the frozen CLIP encoder was pretrained on.
2. **No flow encoding beats the RGB-only baseline yet** (63.24 < 65.69). In a
   frozen-encoder regime the flow stream does not overcome the distribution gap.
   This is reported as a precise negative result, not omitted.

**Caveats (read before citing).** Single proof-of-concept split, one seed
(`seed=1`); top-1 only; no confidence intervals or multi-seed variance, so the
small gaps between flow encodings are **directional, not publication-grade**.
Run by S. Baride on Colab (A100), 2026-06. A publication-grade version needs the
official 3-split HMDB-51 protocol (and UCF-101), multiple seeds, and ideally the
end-to-end fine-tuning of §3 rather than a frozen-backbone probe.
