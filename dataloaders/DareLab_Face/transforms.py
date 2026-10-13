"""Synchronized RGB image, box and valid-region transforms."""

import random

import numpy as np
import torch
from PIL import Image


class DareLabFaceTransform:
    """Letterbox to (height, width); optionally flip training images."""

    def __init__(self, input_size=(320, 320), flip_probability=0.0):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        # Encoder downsamples by 8; decoder must reconstruct the same size.
        # Detector handles its own additional feature padding.
        if (len(input_size) != 2
                or any(not isinstance(n, int) or n <= 0 or n % 8 for n in input_size)):
            raise ValueError("input_size must contain positive multiples of 8 (H, W)")
        if not 0 <= flip_probability <= 1:
            raise ValueError("flip_probability must be in [0, 1]")
        self.input_size = tuple(input_size)
        self.flip_probability = flip_probability

    def __call__(self, image, boxes, ignored_boxes):
        image = image.convert("RGB")
        src_w, src_h = image.size
        dst_h, dst_w = self.input_size
        scale = min(dst_w / src_w, dst_h / src_h)
        resized_w = max(1, min(dst_w, round(src_w * scale)))
        resized_h = max(1, min(dst_h, round(src_h * scale)))
        left, top = (dst_w - resized_w) // 2, (dst_h - resized_h) // 2
        canvas = Image.new("RGB", (dst_w, dst_h), (0, 0, 0))
        canvas.paste(image.resize((resized_w, resized_h), Image.Resampling.BILINEAR),
                     (left, top))
        valid = np.zeros((dst_h, dst_w), dtype=np.float32)
        valid[top:top + resized_h, left:left + resized_w] = 1
        matrix = np.array([[resized_w / src_w, 0, left],
                           [0, resized_h / src_h, top], [0, 0, 1]], dtype=np.float32)
        if self.flip_probability and random.random() < self.flip_probability:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            valid = valid[:, ::-1].copy()
            # Boxes use continuous edge coordinates, hence W rather than W-1.
            flip = np.array([[-1, 0, dst_w], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            matrix = flip @ matrix

        def transform_boxes(values):
            values = np.asarray(values, dtype=np.float32).reshape(-1, 4)
            corners = values[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
            xy = corners @ matrix[:2, :2].T + matrix[:2, 2]
            result = np.concatenate((xy.min(axis=1), xy.max(axis=1)), axis=1)
            result[:, [0, 2]] = result[:, [0, 2]].clip(0, dst_w)
            result[:, [1, 3]] = result[:, [1, 3]].clip(0, dst_h)
            return np.ascontiguousarray(result, dtype=np.float32)

        pixels = np.array(canvas, dtype=np.float32) / 255.0
        target = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
        return {
            "img": target.clone(),
            "reconstruction_target": target,
            "valid_mask": torch.from_numpy(valid[None].copy()),
            "gt_bboxes": transform_boxes(boxes),
            "gt_bboxes_ignore": transform_boxes(ignored_boxes),
            "warp_matrix": matrix,
        }
