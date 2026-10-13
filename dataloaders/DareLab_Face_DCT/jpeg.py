"""JPEG-style block DCT encoding, identical to Adaptive_Compression/tools/generate_dct_dataset.py.

Used to encode augmented training images on the fly; without augmentation the
result equals the stored coefficient PNGs exactly.
"""

import numpy as np

BLOCK = 8


def _zigzag_indices(size=BLOCK):
    indices = []
    for diagonal in range(2 * size - 1):
        coordinates = []
        row_start = max(0, diagonal - size + 1)
        row_end = min(size - 1, diagonal)
        for row in range(row_start, row_end + 1):
            coordinates.append((row, diagonal - row))
        if diagonal % 2 == 0:
            coordinates.reverse()
        indices.extend(row * size + col for row, col in coordinates)
    return np.asarray(indices, dtype=np.int64)


def _dct_matrix(size=BLOCK):
    """Orthonormal DCT-II matrix, identical to scipy.fft.dct(norm='ortho')."""
    k = np.arange(size)[:, None]
    n = np.arange(size)[None, :]
    matrix = np.cos(np.pi * (2 * n + 1) * k / (2 * size)) * np.sqrt(2.0 / size)
    matrix[0] /= np.sqrt(2.0)
    return matrix


ZIGZAG = _zigzag_indices()
DCT = _dct_matrix()


def rgb_to_ycbcr(rgb):
    """JFIF full-range BT.601 conversion, rounded to uint8 as libjpeg does."""
    r, g, b = rgb.astype(np.float64).transpose(2, 0, 1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return [np.clip(np.round(c), 0, 255) for c in (y, cb, cr)]


def downsample_420(channel):
    """Average each non-overlapping 2x2 area and round to an integer sample."""
    return np.round((channel[0::2, 0::2] + channel[0::2, 1::2]
                     + channel[1::2, 0::2] + channel[1::2, 1::2]) / 4.0)


def block_dct(channel, count=64):
    """Return the first count zigzag coefficients with shape [count, H/8, W/8]."""
    height, width = channel.shape
    blocks = (channel - 128.0).reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK)
    blocks = blocks.transpose(0, 2, 1, 3)
    coefficients = DCT @ blocks @ DCT.T
    coefficients = coefficients.reshape(*coefficients.shape[:2], -1)[..., ZIGZAG[:count]]
    return coefficients.transpose(2, 0, 1)


def quantize(coefficients, table):
    """Divide by the step and round half away from zero, as libjpeg does."""
    # Drop float noise so exact .5 ties (e.g. DC = sum / 64) round consistently.
    scaled = np.round(coefficients / table[:, None, None], 6)
    return np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)


def encode(rgb, counts, tables=None):
    """uint8 RGB [H, W, 3] -> {component: float32 [k, h, w]} as stored in the dataset.

    counts maps Y/Cb/Cr to the number of leading zigzag coefficients; tables maps
    them to zigzag-ordered quantization steps, or is None for unquantized data.
    """
    height, width = rgb.shape[:2]
    if height % (2 * BLOCK) or width % (2 * BLOCK):
        raise ValueError(f"Image size {width}x{height} must be divisible by 16 for 4:2:0")
    y, cb, cr = rgb_to_ycbcr(rgb)
    planes = {"Y": y, "Cb": downsample_420(cb), "Cr": downsample_420(cr)}
    result = {}
    for name, plane in planes.items():
        coefficients = block_dct(plane, counts[name])
        if tables is None:
            coefficients = np.round(coefficients)
        else:
            coefficients = quantize(coefficients, np.asarray(tables[name], dtype=np.float64)[:counts[name]])
        result[name] = coefficients.astype(np.float32)
    return result
