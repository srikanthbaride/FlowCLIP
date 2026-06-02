"""End-to-end smoke test of the flow path through the real CLIP ViT encoder.

This is the ported version of the Colab notebook's smoke cell. It loads
CLIP ViT-B/16 on CPU and asserts that FlowCLIP.forward returns a
(batch, 512) embedding. It is skipped automatically when the ``clip``
package or its pretrained weights are unavailable (e.g. offline CI), so the
rest of the suite still runs.
"""
import pytest

torch = pytest.importorskip("torch")
clip = pytest.importorskip("clip")

from modules.Flow_Encoder import FlowCLIP

CLIP_EMBED_DIM = 512


@pytest.fixture(scope="module")
def clip_model():
    try:
        model, _ = clip.load("ViT-B/16", device="cpu", jit=False)
    except Exception as exc:  # network failure, missing weights, etc.
        pytest.skip(f"CLIP ViT-B/16 unavailable: {exc}")
    return model.eval()


def test_flow_forward_returns_clip_embedding(clip_model):
    flow_encoder = FlowCLIP(clip_model).to("cpu").eval()
    with torch.no_grad():
        out = flow_encoder(torch.randn(4, 2, 224, 224))
    assert out is not None, "FlowCLIP.forward returned None"
    assert out.shape == (4, CLIP_EMBED_DIM), f"unexpected shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "embedding contains NaN/Inf"
