"""FOMOJPEGFusion with a cheaper entry: optionally no fusion 1x1 conv and/or a depthwise separable stem.

On the 40x40 block grid the backbone stem (a full 3x3 conv, 24 -> 8 channels at
alpha 0.1) and the fusion module's 1x1 output conv are the costliest layers
before the head. Here the gated luma/chroma features [L', C'] go straight to the
stem, so the stem does the luma/chroma mixing that the fusion conv did. With
separable_stem the stem becomes a 3x3 depthwise conv + BN + ReLU6 and a 1x1
conv + BN + ReLU6. expand_ratio replaces MobileNetV2's expansion factor 6 in the
backbone blocks (48 -> 8 * expand_ratio hidden channels at alpha 0.1). The cross
gating, loss and post-processing are FOMOJPEGFusion's (head_channels can shrink the head).
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2

from .fomo import FOMOHead
from .fomo_jpeg_fusion import FOMOJPEGFusion


class FOMOJPEGFusionLite(FOMOJPEGFusion):
    """FOMOJPEGFusion whose fusion output is the concat of its gated branches
    (luma_channels + chroma_channels), with a depthwise separable stem if separable_stem."""

    def __init__(self, input_channels, luma_inputs, luma_channels=16, chroma_channels=8,
                 separable_stem=True, fusion_conv=False, expand_ratio=6, **kwargs):
        if "fusion_channels" in kwargs:
            raise ValueError("FOMO_JPEGFusionLite's fusion width is luma_channels + chroma_channels")
        if type(expand_ratio) is not int or expand_ratio < 1:
            raise ValueError("expand_ratio must be a positive integer")
        channels = luma_channels + chroma_channels
        super().__init__(input_channels, luma_inputs, luma_channels, chroma_channels,
                         fusion_channels=channels, **kwargs)
        # fusion_conv keeps FOMOJPEGFusion's 1x1 conv over [L', C'] (channels -> channels).
        if not fusion_conv:
            self.fusion.fuse = nn.Identity()
        if expand_ratio != 6:
            self._set_expand_ratio(expand_ratio, kwargs.get("width_mult", 0.35), kwargs.get("cut_block", 6))
        if separable_stem:
            stem = self.backbone[0][0]
            self.backbone[0] = nn.Sequential(
                nn.Conv2d(channels, channels, stem.kernel_size, stem.stride, stem.padding, groups=channels,
                          bias=False),
                nn.BatchNorm2d(channels), nn.ReLU6(),
                nn.Conv2d(channels, stem.out_channels, 1, bias=False),
                nn.BatchNorm2d(stem.out_channels), nn.ReLU6())

    def _set_expand_ratio(self, expand_ratio, width_mult, cut_block):
        """Rebuild the blocks after the stem with MobileNetV2's t=6 stages at expand_ratio.

        The cut layer (the next block's expansion) and so the head input shrink with it;
        the stem, block_0 (t=1) and the stride handling are as in FOMO.
        """
        setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
                   [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        setting = [[expand_ratio if t == 6 else t, c, n, s] for t, c, n, s in setting]
        features = mobilenet_v2(weights=None, width_mult=width_mult, inverted_residual_setting=setting).features
        block = features[cut_block + 1]
        layers = [*features[1:cut_block + 1], block.conv[0]]
        if self.stride == 1:  # block_grid_input
            for module in nn.Sequential(*layers).modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)
        self.backbone = nn.Sequential(self.backbone[0], *layers)
        self.out_channels = block.conv[0][0].out_channels
        self.head = FOMOHead(self.out_channels, self.num_classes, self.head.conv.out_channels)
