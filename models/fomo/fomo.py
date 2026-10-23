"""FOMO (Faster Objects, More Objects) after Edge Impulse's constrained object detection.

MobileNetV2 is cut at the ReLU6 after the expansion layer of Keras block
``block_{cut_block}`` (block_6_expand_relu by default, 1/8 resolution). A 1x1
conv with 32 filters and ReLU, then a 1x1 conv to background + classes, gives
per-cell logits. Each object is labelled only in the cell holding its box
centre (label_mode="center", as the original), or in every cell whose centre
lies inside its box plus the centre cell (label_mode="box"), or in the cells of
its box whose block is at least half covered by the object mask plus the centre
cell (label_mode="mask", needs meta["gt_mask"] [B, 1, H', W'] coverage);
object cells are weighted by object_weight in the cross entropy.

block_grid_input=True is for inputs already on an 8x8 block grid (DCT
coefficients): every stride-2 convolution before the cut runs at stride 1, so
each output cell is one 8x8 block, matching the RGB model's 1/8 grid.
"""

import numpy as np
import torch
import torch.ao.quantization as aq
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torchvision.models import mobilenet_v2


class FOMOHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, head_channels=32):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, head_channels, 1)
        self.relu = nn.ReLU()
        # Logits for background (index 0) and each foreground class.
        self.logits = nn.Conv2d(head_channels, num_classes + 1, 1)

    def forward(self, x):
        return self.logits(self.relu(self.conv(x)))


class FOMO(nn.Module):
    """Per-cell classifier returning [B, num_classes + 1, H/stride, W/stride] logits.

    Post-processing (see post_process) only changes how cells become objects:
    postprocess "components" groups 8-connected cells with probability >=
    group_threshold (default score_threshold) and keeps groups whose peak
    reaches score_threshold, so fragments of one object joined by weaker cells
    become one; "peaks" makes every local maximum >= score_threshold an object,
    peaks at least peak_radius cells apart, so touching objects stay separate.
    """

    def __init__(self, width_mult=0.35, cut_block=6, num_classes=1, head_channels=32,
                 object_weight=100.0, score_threshold=0.5, input_channels=3,
                 block_grid_input=False, label_mode="center",
                 postprocess="components", group_threshold=None, peak_radius=2):
        super().__init__()
        if label_mode not in ("center", "box", "mask"):
            raise ValueError("label_mode must be 'center', 'box' or 'mask'")
        if postprocess not in ("components", "peaks"):
            raise ValueError("postprocess must be 'components' or 'peaks'")
        if group_threshold is not None and not 0 < group_threshold < 1:
            raise ValueError("group_threshold must be in (0, 1)")
        if type(peak_radius) is not int or peak_radius < 1:
            raise ValueError("peak_radius must be a positive integer")
        self.postprocess = postprocess
        self.group_threshold = None if group_threshold is None else float(group_threshold)
        self.peak_radius = peak_radius
        if width_mult <= 0:
            raise ValueError("width_mult must be positive")
        if type(cut_block) is not int or not 1 <= cut_block <= 16:
            raise ValueError("cut_block must be a Keras block number in [1, 16]")
        if object_weight <= 0 or not 0 < score_threshold < 1:
            raise ValueError("object_weight must be positive and score_threshold in (0, 1)")
        if type(input_channels) is not int or input_channels < 1:
            raise ValueError("input_channels must be a positive integer")
        features = mobilenet_v2(weights=None, width_mult=width_mult).features
        stem = features[0][0]
        if input_channels != stem.in_channels:
            features[0][0] = nn.Conv2d(input_channels, stem.out_channels, stem.kernel_size,
                                       stem.stride, stem.padding, bias=False)
        # Keras expanded_conv is features[1], so block_k is features[k + 1].
        block = features[cut_block + 1]
        self.backbone = nn.Sequential(*features[:cut_block + 1], block.conv[0])
        self.stride = 2 * int(np.prod([features[i].stride for i in range(1, cut_block + 1)]))
        if block_grid_input:
            for module in self.backbone.modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)
            self.stride = 1
        self.out_channels = block.conv[0][0].out_channels
        self.head = FOMOHead(self.out_channels, num_classes, head_channels)
        self.num_classes = num_classes
        self.object_weight = float(object_weight)
        self.score_threshold = float(score_threshold)
        self.label_mode = label_mode
        self.quant = aq.QuantStub()
        self.dequant = aq.DeQuantStub()
        self.epoch = 0

    def fuse_model(self, qat=False):
        """Fuse Conv+BN (ReLU6 cannot be fused) and the head Conv+ReLU."""
        fuse = aq.fuse_modules_qat if qat else aq.fuse_modules
        # Snapshot before fusion changes the module hierarchy.
        for module in list(self.backbone.modules()):
            children = list(module.named_children())
            for (name, child), (next_name, next_child) in zip(children, children[1:]):
                if isinstance(child, nn.Conv2d) and isinstance(next_child, nn.BatchNorm2d):
                    fuse(module, [name, next_name], inplace=True)
        fuse(self.head, ["conv", "relu"], inplace=True)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, x, meta):
        return self.train_forward(x, meta) if self.training else self.infer_forward(x, meta)

    def train_forward(self, x, meta=None):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % self.stride, 0, (-h) % self.stride), mode="constant", value=0)
        return self.dequant(self.head(self.backbone(self.quant(x))))

    def infer_forward(self, x, meta):
        return self.post_process(self.train_forward(x, meta), meta)

    def targets(self, logits, meta):
        """Class labels [B, H', W'] of each object's cells (see label_mode) and per-cell loss weights."""
        batch, _, grid_h, grid_w = logits.shape
        labels = torch.zeros(batch, grid_h, grid_w, dtype=torch.long)
        weights = torch.ones(batch, grid_h, grid_w)

        def cells(boxes, mode, coverage=None):
            """(rows, cols) of each box's cells: the centre cell, plus every cell centred inside (box)
            or every cell inside that is at least half covered by the mask (mask)."""
            result = []
            for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float32).reshape(-1, 4) / self.stride:
                cx = min(max(int((x1 + x2) / 2), 0), grid_w - 1)
                cy = min(max(int((y1 + y2) / 2), 0), grid_h - 1)
                mask = np.zeros((grid_h, grid_w), dtype=bool)
                mask[cy, cx] = True
                if mode in ("box", "mask"):
                    # Cell (r, c) is centred at (c + 0.5, r + 0.5).
                    c0, c1 = max(int(np.ceil(x1 - 0.5)), 0), min(int(np.floor(x2 - 0.5)) + 1, grid_w)
                    r0, r1 = max(int(np.ceil(y1 - 0.5)), 0), min(int(np.floor(y2 - 0.5)) + 1, grid_h)
                    inside = np.zeros_like(mask)
                    inside[r0:r1, c0:c1] = True
                    mask |= inside & (coverage >= 0.5) if mode == "mask" else inside
                result.append(np.nonzero(mask))
            return result

        coverages = None
        if self.label_mode == "mask":
            coverages = np.asarray(meta["gt_mask"].cpu())[:, 0]
            if coverages.shape[1:] != (grid_h, grid_w):
                raise ValueError(f"gt_mask grid {coverages.shape[1:]} disagrees with logits {(grid_h, grid_w)}")
        for index in range(batch):
            coverage = coverages[index] if coverages is not None else None
            for (cy, cx), label in zip(cells(meta["gt_bboxes"][index], self.label_mode, coverage),
                                       meta["gt_labels"][index]):
                labels[index, cy, cx] = int(label) + 1
                weights[index, cy, cx] = self.object_weight
            # A crowd/ignored region is neither object nor background.
            # Ignored regions have no mask of their own; their whole box is excluded.
            ignore_mode = "box" if self.label_mode == "mask" else self.label_mode
            for cy, cx in cells(meta["gt_bboxes_ignore"][index], ignore_mode):
                labels[index, cy, cx] = 0
                weights[index, cy, cx] = 0
        return labels.to(logits.device), weights.to(logits.device)

    def loss(self, logits, meta):
        labels, weights = self.targets(logits, meta)
        loss = F.cross_entropy(logits, labels, reduction="none")
        # Weighted mean over all cells, as a Keras weighted cross entropy.
        loss_ce = (loss * weights).sum() / weights.numel()
        positive = labels > 0
        states = {"loss_ce": loss_ce,
                  "loss_object": loss[positive].mean() if positive.any() else loss.new_zeros(()),
                  "loss_background": loss[(labels == 0) & (weights > 0)].mean()}
        return loss_ce, states

    def _object_cells(self, probability):
        """[(rows, cols, score)] of each object on one class's probability map (see class doc)."""
        group = self.score_threshold if self.group_threshold is None else self.group_threshold
        components, count = ndimage.label(probability >= group, structure=np.ones((3, 3)))
        if self.postprocess == "components":
            objects = []
            for component in range(1, count + 1):
                ys, xs = np.nonzero(components == component)
                score = probability[ys, xs].max()
                if score >= self.score_threshold:
                    objects.append((ys, xs, score))
            return objects
        size = 2 * self.peak_radius + 1
        peaks = (probability == ndimage.maximum_filter(probability, size=size, mode="constant", cval=0.0))
        peaks &= probability >= self.score_threshold
        # A flat top of equal maxima is one peak: keep its first cell.
        plateaus, count = ndimage.label(peaks, structure=np.ones((3, 3)))
        objects = []
        for plateau in range(1, count + 1):
            pys, pxs = np.nonzero(plateaus == plateau)
            py, px = pys[0], pxs[0]
            # The object is the window around the peak, within the peak's group of cells.
            window = np.zeros_like(peaks)
            window[max(py - self.peak_radius, 0):py + self.peak_radius + 1,
                   max(px - self.peak_radius, 0):px + self.peak_radius + 1] = True
            ys, xs = np.nonzero(window & (components == components[py, px]))
            objects.append((ys, xs, probability[py, px]))
        return objects

    @torch.no_grad()
    def post_process(self, logits, meta):
        """Turn each class's probability map into objects (see the class doc for postprocess).

        Returns {image_id: {class: [{"centroid", "bbox_xyxy", "score"}]}} in
        original image coordinates; the box only encloses the object's cells.
        """
        probabilities = logits.float().softmax(dim=1).cpu().numpy()
        results = {}
        for index, image_id in enumerate(meta["img_info"]["id"]):
            inverse = np.linalg.inv(meta["warp_matrix"][index])
            width = float(meta["img_info"]["width"][index])
            height = float(meta["img_info"]["height"][index])
            detections = {}
            for label in range(self.num_classes):
                probability = probabilities[index, label + 1]
                objects = []
                for ys, xs, score in self._object_cells(probability):
                    scores = probability[ys, xs]
                    centre = np.array([(np.average(xs, weights=scores) + 0.5) * self.stride,
                                       (np.average(ys, weights=scores) + 0.5) * self.stride, 1.0])
                    corners = np.array([[xs.min() * self.stride, ys.min() * self.stride, 1.0],
                                        [(xs.max() + 1) * self.stride, (ys.max() + 1) * self.stride, 1.0]])
                    centre = inverse @ centre
                    corners = (inverse @ corners.T).T
                    x1, x2 = sorted(np.clip(corners[:, 0], 0, width))
                    y1, y2 = sorted(np.clip(corners[:, 1], 0, height))
                    objects.append({"centroid": [float(np.clip(centre[0], 0, width)),
                                                 float(np.clip(centre[1], 0, height))],
                                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                                    "score": float(score)})
                detections[label] = sorted(objects, key=lambda item: -item["score"])
            results[int(image_id)] = detections
        return results
