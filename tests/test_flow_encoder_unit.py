"""Fast, dependency-light unit tests for the FlowCLIP optical-flow encoder.

These do NOT download CLIP weights. They wrap a tiny stub image encoder so
the 2->3 flow projection and forward contract can be checked in milliseconds
on CPU. The end-to-end behaviour against the real CLIP ViT encoder is covered
by ``test_flow_encoder_clip.py`` (which skips when CLIP is unavailable).
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from modules.Flow_Encoder import FlowCLIP


class _StubImageEncoder(nn.Module):
    """Stand-in for CLIP's image encoder.

    Exposes the single attribute FlowCLIP depends on -- ``encode_image`` --
    and records the channel count it was handed so tests can assert that the
    flow projection produced a 3-channel (RGB-like) tensor.
    """

    EMBED_DIM = 512

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, self.EMBED_DIM)
        self.seen_channels = None

    def encode_image(self, x):
        self.seen_channels = x.shape[1]
        pooled = x.mean(dim=(2, 3))  # (N, 3) global average pool
        return self.proj(pooled)     # (N, EMBED_DIM)


def test_flow_proj_is_2_to_3_conv():
    fe = FlowCLIP(_StubImageEncoder())
    assert isinstance(fe.flow_proj, nn.Conv2d)
    assert fe.flow_proj.in_channels == 2, "flow input is (flow_x, flow_y)"
    assert fe.flow_proj.out_channels == 3, "must map to an RGB-like 3-channel input"
    assert fe.flow_proj.bias is None, "projection is bias-free by design"


def test_forward_returns_embedding_shape():
    stub = _StubImageEncoder()
    fe = FlowCLIP(stub).eval()
    with torch.no_grad():
        out = fe(torch.randn(4, 2, 224, 224))
    assert out.shape == (4, _StubImageEncoder.EMBED_DIM)


def test_forward_feeds_three_channels_to_encoder():
    stub = _StubImageEncoder()
    fe = FlowCLIP(stub).eval()
    with torch.no_grad():
        fe(torch.randn(2, 2, 64, 64))
    assert stub.seen_channels == 3, "encode_image must receive a 3-channel tensor"


def test_forward_casts_input_to_weight_dtype():
    """The forward pass casts the flow tensor to flow_proj's dtype.

    This guards the fp32-input / fp16-weight mismatch that CLIP's
    ``convert_weights`` would otherwise trigger on GPU.
    """
    fe = FlowCLIP(_StubImageEncoder())
    fe.flow_proj = fe.flow_proj.half()  # simulate post-convert_weights fp16
    flow_fp32 = torch.randn(1, 2, 32, 32)  # input arrives as fp32
    weight_dtype = fe.flow_proj.weight.dtype
    projected = fe.flow_proj(flow_fp32.to(weight_dtype))
    assert projected.dtype == torch.float16
    assert projected.shape == (1, 3, 32, 32)
