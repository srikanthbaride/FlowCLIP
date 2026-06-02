# Experiments

## `FlowCLIP_encoding_comparison.ipynb`

A proof-of-concept comparison of **optical-flow encodings** for a **frozen CLIP ViT-B/16**
backbone on HMDB51. Each encoding is evaluated as a **linear probe** on fixed CLIP features,
so differences reflect the *encoding* rather than the training budget.

Encodings compared:

| name | input representation |
|------|----------------------|
| `rgb` | RGB frames (baseline) |
| `flow_xy` | `[flow_x, flow_y, magnitude]` → CLIP (the current 2→3 approach) |
| `flow_hsv` | HSV/Middlebury colour-coded flow → CLIP (**literature-recommended**) |
| `flow_temporal` | flow magnitude at start/mid/end packed into R/G/B → CLIP |

Plus late fusion `rgb + <flow>`. The notebook prints a results table and saves
`results.csv` / `results.json`.

**Run:** open in Colab → set GPU runtime → Run all. ~1–3 hrs (tune `MAX_PER_CLASS` in the config cell).

**Hypothesis under test:** `flow_hsv` should beat the naive `flow_xy`, because flow's value is its
*appearance-invariance* (which colour-coding preserves) and a 3-channel colour image matches the RGB
distribution the frozen CLIP encoder was pretrained on, whereas a raw 2→3 projection is out-of-distribution
for a frozen encoder. No prior work benchmarks this on a frozen CLIP ViT — this notebook is the test.

> Proof-of-concept settings: fixed per-class split (seed=1), not the official 3-split protocol, and
> Farneback flow by default. The comparison across encodings is fair (same split, same flow). Switch to
> `FLOW_METHOD='tvl1'`, `MAX_PER_CLASS=None`, and the official 3-split files for publishable numbers.
