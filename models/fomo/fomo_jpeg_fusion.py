"""FOMO with a JPEG luma/chroma fusion module before the backbone.

The input is the dataloader's DCT block grid: the first luma_inputs channels
are Y zigzag coefficients, the rest Cb/Cr. JPEGFusion processes them in two
branches and lets each gate the other, then feeds fusion_channels to the
unchanged FOMO backbone and head. Labels, loss and post-processing are FOMO's.
"""

import torch.ao.nn.quantized as nnq
import torch.ao.quantization as aq
import torch.nn as nn
import torch.nn.functional as F

from .fomo import FOMO


def _conv_bn_relu6(in_channels, out_channels, kernel_size=1, groups=1):
    # Conv and BN are siblings so fuse_model folds them, as in the backbone.
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2,
                                   groups=groups, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU6())


class JPEGFusion(nn.Module):
    """Luma/chroma fusion of DCT block-grid input.

    Luma: 1x1 conv across a block's frequencies, then a 3x3 depthwise conv across
    neighbouring blocks. Chroma: 1x1 conv, colour being a per-block cue. Each
    branch is gated by the other with a residual,
    L' = L + L * hardsigmoid(conv(C)) and C' = C + C * hardsigmoid(conv(L)),
    so a face cell can require both skin colour and face texture, while either
    branch still passes when the other is silent (e.g. grayscale input).
    A 1x1 conv over [L', C'] gives out_channels.

    hardsigmoid(z) = ReLU6(z + 3) / 6 is built from a conv whose bias starts at
    +3, a ReLU6 and a 1/6 scalar multiply: nn.Hardsigmoid would get a fixed-range
    fake quantizer that the INT24/INT16/INT8 stage switching cannot rescale.
    """

    def __init__(self, luma_inputs, chroma_inputs, luma_channels=16, chroma_channels=8, out_channels=24,
                 luma_kernel=3):
        super().__init__()
        if min(luma_inputs, chroma_inputs, luma_channels, chroma_channels, out_channels) < 1:
            raise ValueError("JPEGFusion needs positive luma and chroma inputs and channel counts")
        if luma_kernel < 1 or luma_kernel % 2 == 0:
            raise ValueError("luma_kernel must be a positive odd number")
        self.luma_inputs = luma_inputs
        # luma_kernel sets how many neighbouring blocks the depthwise conv sees (3 or 5, ...).
        self.luma = nn.Sequential(_conv_bn_relu6(luma_inputs, luma_channels),
                                  _conv_bn_relu6(luma_channels, luma_channels, luma_kernel, groups=luma_channels))
        self.chroma = _conv_bn_relu6(chroma_inputs, chroma_channels)
        self.gate_from_chroma = self._gate(chroma_channels, luma_channels)
        self.gate_from_luma = self._gate(luma_channels, chroma_channels)
        self.fuse = _conv_bn_relu6(luma_channels + chroma_channels, out_channels)
        # FloatFunctional gives the elementwise ops their own fake-quant observers in QAT.
        self.luma_scale, self.chroma_scale = nnq.FloatFunctional(), nnq.FloatFunctional()
        self.luma_mul, self.luma_add = nnq.FloatFunctional(), nnq.FloatFunctional()
        self.chroma_mul, self.chroma_add = nnq.FloatFunctional(), nnq.FloatFunctional()
        self.cat = nnq.FloatFunctional()

    @staticmethod
    def _gate(in_channels, out_channels):
        conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.constant_(conv.bias, 3.0)  # ReLU6(z + 3): a silent branch gives a 0.5 gate.
        return nn.Sequential(conv, nn.ReLU6())

    def forward(self, x):
        luma = self.luma(x[:, :self.luma_inputs])
        chroma = self.chroma(x[:, self.luma_inputs:])
        luma_gate = self.luma_scale.mul_scalar(self.gate_from_chroma(chroma), 1 / 6)
        chroma_gate = self.chroma_scale.mul_scalar(self.gate_from_luma(luma), 1 / 6)
        gated_luma = self.luma_add.add(luma, self.luma_mul.mul(luma, luma_gate))
        gated_chroma = self.chroma_add.add(chroma, self.chroma_mul.mul(chroma, chroma_gate))
        return self.fuse(self.cat.cat([gated_luma, gated_chroma], dim=1))


class FOMOJPEGFusion(FOMO):
    """FOMO whose backbone reads JPEGFusion(input) instead of the raw coefficients.

    input_channels is the dataloader's kY + kCb + kCr and luma_inputs its kY;
    the backbone's first conv takes fusion_channels. Other arguments are FOMO's.
    loss_type "focal" replaces each cell's cross entropy -log p_t by
    -(1 - p_t)^focal_gamma log p_t (Lin et al.), keeping FOMO's per-cell weights
    and mean, so easy background cells fade and hard ones (skin, hands) dominate.
    """

    def __init__(self, input_channels, luma_inputs, luma_channels=16, chroma_channels=8,
                 fusion_channels=24, luma_kernel=3, loss_type="ce", focal_gamma=2.0, **kwargs):
        if not 0 < luma_inputs < input_channels:
            raise ValueError("luma_inputs must leave Cb/Cr channels in input_channels")
        if loss_type not in ("ce", "focal") or focal_gamma < 0:
            raise ValueError("loss_type must be 'ce' or 'focal' and focal_gamma >= 0")
        super().__init__(input_channels=fusion_channels, **kwargs)
        self.fusion = JPEGFusion(luma_inputs, input_channels - luma_inputs,
                                 luma_channels, chroma_channels, fusion_channels, luma_kernel)
        self.loss_type = loss_type
        self.focal_gamma = float(focal_gamma)

    def loss(self, logits, meta):
        if self.loss_type == "ce":
            return super().loss(logits, meta)
        labels, weights = self.targets(logits, meta)
        log_p = F.log_softmax(logits.float(), dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
        ce = -log_p
        focal = (1 - log_p.exp()).pow(self.focal_gamma) * ce
        loss_focal = (focal * weights).sum() / weights.numel()
        positive = labels > 0
        # loss_object/loss_background stay plain cross entropy, comparable with the "ce" runs.
        states = {"loss_focal": loss_focal,
                  "loss_object": ce[positive].mean() if positive.any() else ce.new_zeros(()),
                  "loss_background": ce[(labels == 0) & (weights > 0)].mean()}
        return loss_focal, states

    def fuse_model(self, qat=False):
        super().fuse_model(qat)
        fuse = aq.fuse_modules_qat if qat else aq.fuse_modules
        for module in list(self.fusion.modules()):
            children = list(module.named_children())
            for (name, child), (next_name, next_child) in zip(children, children[1:]):
                if isinstance(child, nn.Conv2d) and isinstance(next_child, nn.BatchNorm2d):
                    fuse(module, [name, next_name], inplace=True)

    def train_forward(self, x, meta=None):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % self.stride, 0, (-h) % self.stride), mode="constant", value=0)
        return self.dequant(self.head(self.backbone(self.fusion(self.quant(x)))))
