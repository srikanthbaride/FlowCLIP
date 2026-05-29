# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# Optical flow modality extension.

import torch
import torch.nn as nn


class FlowCLIP(nn.Module):
    """Wraps CLIP's image encoder to accept 2-channel optical flow (u, v).

    A learned 2->3 Conv2d projection maps flow channels to an RGB-like
    3-channel representation before passing to CLIP's encode_image.
    The projection weight is initialised with Xavier uniform so that it
    starts in a reasonable scale relative to the CLIP encoder.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Projects (flow_x, flow_y) to 3-channel input for CLIP's ViT encoder
        self.flow_proj = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.flow_proj.weight)

    def forward(self, flow):
        """
        Args:
            flow: FloatTensor of shape (N, 2, H, W).
                  Channel 0 = flow_x, Channel 1 = flow_y.
                  Should already be normalised (mean=0.5, std=0.5 applied by
                  the dataset transform).
        Returns:
            Tensor of shape (N, embed_dim) from CLIP's image encoder.
        """
        # Cast input to match flow_proj's current dtype.
        # Before convert_weights this is fp32; after it is fp16.
        # Using weight.dtype avoids a RuntimeError from a fp32-input / fp16-weight
        # mismatch while remaining compatible with encode_image's native dtype.
        weight_dtype = self.flow_proj.weight.dtype
        flow_3ch = self.flow_proj(flow.to(weight_dtype))
        