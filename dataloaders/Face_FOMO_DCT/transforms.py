"""Letterbox the original photo to the fixed input size and build the DCT sample."""

import cv2
import numpy as np
import torch
from PIL import Image

from ..DareLab_Face_DCT.transforms import BLOCK, DareLabFaceDCTTransform, PixelAugmentation  # noqa: F401

FILL = 128  # Encodes to all-zero coefficients after the -128 level shift.


def load_letterboxed(path, input_size):
    """Decode a JPEG into a [H, W, 3] uint8 canvas of input_size = (W, H).

    The image keeps its aspect ratio, is centred and the rest is filled with 128.
    Returns (canvas, matrix, (width, height)); matrix maps continuous original
    pixel coordinates to canvas pixel coordinates.
    """
    target_w, target_h = input_size
    with Image.open(path) as source:
        width, height = source.size
        scale = min(target_w / width, target_h / height)
        new_w = max(1, round(width * scale))
        new_h = max(1, round(height * scale))
        # JPEG draft decodes at 1/2, 1/4 or 1/8 scale, never below the requested size.
        source.draft("RGB", (new_w, new_h))
        image = np.asarray(source.convert("RGB"))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    offset_x, offset_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas = np.full((target_h, target_w, 3), FILL, dtype=np.uint8)
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = image
    matrix = np.array([[new_w / width, 0, offset_x],
                       [0, new_h / height, offset_y],
                       [0, 0, 1]], dtype=np.float64)
    return canvas, matrix, (width, height)


class FaceFOMODCTTransform(DareLabFaceDCTTransform):
    """DareLabFaceDCTTransform plus a valid_mask of the blocks the photo covers.

    pixel_matrix maps original pixels to the encoded canvas (letterbox, then
    augmentation), so gt boxes, warp_matrix and valid_mask are all relative to
    the original photo. Boxes under 2 canvas pixels are dropped.
    """

    def __call__(self, coefficients, boxes, ignored_boxes, pixel_matrix=None, image_size=None):
        sample = super().__call__(coefficients, boxes, ignored_boxes, pixel_matrix)
        if image_size is not None:
            grid_h, grid_w = sample["valid_mask"].shape[1:]
            corners = np.array([[0, 0], [image_size[0], 0], [0, image_size[1]], image_size], dtype=np.float64)
            matrix = sample["warp_matrix"].astype(np.float64)
            xy = corners @ matrix[:2, :2].T + matrix[:2, 2]
            x0, y0 = np.floor(xy.min(axis=0)).astype(int).clip(0, (grid_w, grid_h))
            x1, y1 = np.ceil(xy.max(axis=0)).astype(int).clip(0, (grid_w, grid_h))
            mask = torch.zeros(1, grid_h, grid_w)
            mask[:, y0:y1, x0:x1] = 1
            sample["valid_mask"] = mask
        return sample
