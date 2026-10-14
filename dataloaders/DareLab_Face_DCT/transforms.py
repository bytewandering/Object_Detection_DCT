"""Synchronized DCT coefficient, box and valid-region transforms."""

import random

import cv2
import numpy as np
import torch

BLOCK = 8
AUGMENTATION_KEYS = {"flip", "scale", "translate", "brightness", "contrast", "saturation"}


class PixelAugmentation:
    """Augment the RGB image before JPEG encoding, as Object_Detection_RGB does.

    Same keys, sampling and order as DareLabFaceTransform of Object_Detection_RGB:
    flip probability, scale range about the centre, translate as a fraction of
    the size, then brightness offset, contrast and saturation ranges. Uncovered
    pixels are filled with 128, which encodes to all-zero coefficients.
    Returns the uint8 image and the original-to-augmented pixel matrix.
    """

    def __init__(self, augmentation):
        augmentation = dict(augmentation)
        unknown = set(augmentation) - AUGMENTATION_KEYS
        if unknown:
            raise ValueError(f"Unknown augmentation keys: {sorted(unknown)}")
        self.flip = float(augmentation.get("flip", 0.0))
        self.scale = tuple(augmentation.get("scale", (1.0, 1.0)))
        self.translate = float(augmentation.get("translate", 0.0))
        self.brightness = float(augmentation.get("brightness", 0.0))
        self.contrast = tuple(augmentation.get("contrast", (1.0, 1.0)))
        self.saturation = tuple(augmentation.get("saturation", (1.0, 1.0)))
        if not 0 <= self.flip <= 1 or not 0 <= self.translate < 1 or self.brightness < 0:
            raise ValueError("flip must be in [0, 1], translate in [0, 1) and brightness >= 0")
        for name in ("scale", "contrast", "saturation"):
            low, high = getattr(self, name)
            if not 0 < low <= high:
                raise ValueError(f"{name} must be [low, high] with 0 < low <= high")
        self.geometric = self.flip > 0 or self.scale != (1.0, 1.0) or self.translate > 0
        self.photometric = (self.brightness > 0 or self.contrast != (1.0, 1.0)
                            or self.saturation != (1.0, 1.0))

    def _geometric_matrix(self, width, height):
        """Continuous-coordinate affine: flip, scale about the centre, then translate."""
        matrix = np.eye(3, dtype=np.float64)
        if random.random() < self.flip:
            # Boxes use continuous edge coordinates, hence W rather than W-1.
            matrix = np.array([[-1, 0, width], [0, 1, 0], [0, 0, 1]], dtype=np.float64) @ matrix
        scale = random.uniform(*self.scale)
        centre = np.array([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]], dtype=np.float64)
        resize = np.diag([scale, scale, 1.0])
        matrix = centre @ resize @ np.linalg.inv(centre) @ matrix
        shift = np.array([[1, 0, random.uniform(-self.translate, self.translate) * width],
                          [0, 1, random.uniform(-self.translate, self.translate) * height],
                          [0, 0, 1]], dtype=np.float64)
        return shift @ matrix

    def _photometric(self, image):
        pixels = image.astype(np.float32) / 255.0
        if self.brightness > 0:
            pixels += random.uniform(-self.brightness, self.brightness)
        pixels *= random.uniform(*self.contrast)
        pixels = np.clip(pixels, 0.0, 1.0)
        if self.saturation != (1.0, 1.0):
            hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(*self.saturation), 0.0, 1.0)
            pixels = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return np.clip(np.round(pixels * 255.0), 0, 255).astype(np.uint8)

    def __call__(self, image):
        height, width = image.shape[:2]
        matrix = np.eye(3, dtype=np.float64)
        if self.geometric:
            matrix = self._geometric_matrix(width, height)
            # cv2 samples at pixel centres (index i covers [i, i + 1)), so shift by half a pixel.
            half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float64)
            pixel_matrix = np.linalg.inv(half) @ matrix @ half
            image = cv2.warpAffine(image, pixel_matrix[:2], (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
        if self.photometric:
            image = self._photometric(image)
        return image, matrix


class DareLabFaceDCTTransform:
    """Assemble [kY + kCb + kCr, H/8, W/8] input and map boxes onto the block grid.

    DCT blocks cannot be resized, so the native block grid is kept and boxes are
    expressed in block-grid coordinates (pixels / 8). pixel_matrix is the pixel
    augmentation applied before encoding (identity without augmentation); boxes
    follow it, are clipped and, when augmented, dropped if narrower or shorter
    than 2 pixels, as in the RGB pipeline.
    """

    def __call__(self, coefficients, boxes, ignored_boxes, pixel_matrix=None):
        y, cb, cr = coefficients["Y"], coefficients["Cb"], coefficients["Cr"]
        grid_h, grid_w = y.shape[1:]
        augmented = pixel_matrix is not None
        block = np.array([[1 / BLOCK, 0, 0], [0, 1 / BLOCK, 0], [0, 0, 1]], dtype=np.float64)
        matrix = block @ pixel_matrix if augmented else block
        # Each chroma block covers 2x2 (4:2:0) or 1x2 (4:2:2) luma blocks.
        factor_h, factor_w = grid_h // cb.shape[1], grid_w // cb.shape[2]
        cb, cr = (component.repeat(factor_h, axis=1).repeat(factor_w, axis=2) for component in (cb, cr))
        feature = np.concatenate((y, cb, cr), axis=0)

        def transform_boxes(values):
            values = np.asarray(values, dtype=np.float32).reshape(-1, 4)
            corners = values[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
            xy = corners @ matrix[:2, :2].T + matrix[:2, 2]
            result = np.concatenate((xy.min(axis=1), xy.max(axis=1)), axis=1)
            result[:, [0, 2]] = result[:, [0, 2]].clip(0, grid_w)
            result[:, [1, 3]] = result[:, [1, 3]].clip(0, grid_h)
            if augmented:
                minimum = 2 / BLOCK
                keep = (result[:, 2] - result[:, 0] >= minimum) & (result[:, 3] - result[:, 1] >= minimum)
                result = result[keep]
            return np.ascontiguousarray(result, dtype=np.float32)

        return {
            "img": torch.from_numpy(np.ascontiguousarray(feature, dtype=np.float32)),
            "valid_mask": torch.ones(1, grid_h, grid_w),
            "gt_bboxes": transform_boxes(boxes),
            "gt_bboxes_ignore": transform_boxes(ignored_boxes),
            "warp_matrix": matrix.astype(np.float32),
        }
