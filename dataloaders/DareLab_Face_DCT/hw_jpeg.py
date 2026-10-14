"""Bit-exact model of the FPGA JPEG front end (jpeg_dct_original_opt.v + jpeg_quant_zigzag.v).

Same interface as jpeg.encode, so hardware coefficients can replace the float
pipeline. Color conversion and chroma subsampling are not in the Verilog: color
stays as in jpeg.py, chroma is 4:2:0 as in jpeg.py or 4:2:2 (each horizontal
pair averaged as (a + b + 1) >> 1, Cb/Cr grid [H/8, W/16]). Per 8x8 block:

  x = sample - 128                       (row DCT input is signed 8-bit)
  row DCT:    t = (x @ M.T) >>> 9        (integer constants, floor shift)
  column DCT: C = (M @ t) >>> 9          (C ~= 2 x orthonormal DCT)
  quantize:   p = C * floor(4096 / (2 Q))   (q_coeff, 12 fractional bits)
  final:      q = sign(p) * ((|p| + 2048) >> 12)   or   p >> 12

The butterfly in jpeg_dct_original_opt is exactly linear before its single
`>>> 9`, so one pass equals an integer matrix product with M[k][n] =
round(256 sqrt(2) cos((2n + 1) k pi / 16)) and DC row 256 (the `<<< 8` path).
The Verilog leaves the final rounding of q_coeff open; final_rounding picks it.
"""

import numpy as np

from .jpeg import BLOCK, ZIGZAG, downsample_420, rgb_to_ycbcr

DCT_SHIFT = 9
Q_FRAC = 12
# jpeg_quant_zigzag quality 75 tables in natural (row-major) order, as written in
# the Verilog. Chroma [18] and [19] are 20 and 33 there; Annex K at Q75 gives 28 and 50.
LUMA_Q75 = np.array([
    8, 6, 5, 8, 12, 20, 26, 31, 6, 6, 7, 10, 13, 29, 30, 28,
    7, 7, 8, 12, 20, 29, 35, 28, 7, 9, 11, 15, 26, 44, 40, 31,
    9, 11, 19, 28, 34, 55, 52, 39, 12, 18, 28, 32, 41, 52, 57, 46,
    25, 32, 39, 44, 52, 61, 60, 51, 36, 46, 48, 49, 56, 50, 52, 50])
CHROMA_Q75 = np.array([
    9, 9, 12, 24, 50, 50, 50, 50, 9, 11, 13, 33, 50, 50, 50, 50,
    12, 13, 20, 33, 50, 50, 50, 50, 24, 33] + [50] * 38)
TABLES = {"Y": LUMA_Q75, "Cb": CHROMA_Q75, "Cr": CHROMA_Q75}
FINAL_ROUNDING = ("half_away", "floor")
# Chroma samples per (vertical, horizontal) luma samples.
CHROMA_FACTOR = {"420": (2, 2), "422": (1, 2)}


def _dct_matrix():
    k = np.arange(BLOCK)[:, None]
    n = np.arange(BLOCK)[None, :]
    matrix = np.rint(256 * np.sqrt(2) * np.cos(np.pi * (2 * n + 1) * k / (2 * BLOCK))).astype(np.int64)
    matrix[0] = 256
    return matrix


DCT = _dct_matrix()
# c2, c0, c3, 256, c4, c1, c5 of jpeg_dct_original_opt.
assert sorted(set(np.abs(DCT).flat)) == [71, 139, 201, 256, 301, 334, 355]
INV_Q = {name: (1 << Q_FRAC) // (2 * table) for name, table in TABLES.items()}


def block_dct(channel):
    """Integer samples [H, W] -> hardware coefficients [H/8, W/8, 64], natural order."""
    height, width = channel.shape
    blocks = (channel.astype(np.int64) - 128).reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK)
    blocks = blocks.transpose(0, 2, 1, 3)
    rows = (blocks @ DCT.T) >> DCT_SHIFT
    # din_col is 14-bit; the row pass stays within +-512.
    assert np.abs(rows).max() < 1 << 13
    coefficients = (DCT @ rows) >> DCT_SHIFT
    return coefficients.reshape(*coefficients.shape[:2], -1)


def quantize(coefficients, name, final_rounding="half_away"):
    """Hardware coefficients [..., 64] (natural order) -> integer quantized values."""
    scaled = coefficients * INV_Q[name]
    if final_rounding == "half_away":
        return np.sign(scaled) * ((np.abs(scaled) + (1 << (Q_FRAC - 1))) >> Q_FRAC)
    if final_rounding == "floor":
        return scaled >> Q_FRAC
    raise ValueError(f"final_rounding must be one of {FINAL_ROUNDING}")


def downsample_422(channel):
    """Average each horizontal pair of integer samples, rounding half up."""
    channel = channel.astype(np.int64)
    return (channel[:, 0::2] + channel[:, 1::2] + 1) >> 1


def encode(rgb, counts, final_rounding="half_away", subsampling="420"):
    """uint8 RGB [H, W, 3] -> {component: float32 [k, h, w]}, like jpeg.encode."""
    if subsampling not in CHROMA_FACTOR:
        raise ValueError(f"subsampling must be one of {tuple(CHROMA_FACTOR)}")
    height, width = rgb.shape[:2]
    factor_h, factor_w = CHROMA_FACTOR[subsampling]
    if height % (factor_h * BLOCK) or width % (factor_w * BLOCK):
        raise ValueError(f"Image size {width}x{height} must be divisible by "
                         f"{factor_w * BLOCK}x{factor_h * BLOCK} for {subsampling}")
    y, cb, cr = rgb_to_ycbcr(rgb)
    downsample = downsample_420 if subsampling == "420" else downsample_422
    planes = {"Y": y, "Cb": downsample(cb), "Cr": downsample(cr)}
    result = {}
    for name, plane in planes.items():
        quantized = quantize(block_dct(plane), name, final_rounding)
        result[name] = quantized[..., ZIGZAG[:counts[name]]].transpose(2, 0, 1).astype(np.float32)
    return result
