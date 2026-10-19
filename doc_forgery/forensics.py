"""Classic image-forensics features, implemented from scratch.

Four feature maps, each returned as an Image:

- ``ela``: Error Level Analysis approximation. The image is run through a
  JPEG-like 8x8 DCT roundtrip at two quality levels and the absolute
  difference is taken. Regions with a different compression history
  (double-compressed splices, re-saved patches) respond differently than
  the background.
- ``noise_variance_map``: local variance of the Laplacian response.
  Spliced content shot or saved under different conditions carries a
  different noise signature.
- ``block_boundary_score``: for each 8x8 block, how much stronger the
  pixel discontinuities are across the block edges than inside the
  block. Copy-move pastes drag their own block grid with them, so the
  seam lights up when the paste is not grid-aligned.
- ``edge_gradient_inconsistency``: local coefficient of variation of the
  Sobel gradient magnitude. Re-rendered or differently sharpened regions
  have a different edge texture.
"""

from __future__ import annotations

import math

from .image import Image, zeros


# -- 8x8 DCT ---------------------------------------------------------------

_COS8 = [
    [math.cos(math.pi * (2 * n + 1) * k / 16) for n in range(8)]
    for k in range(8)
]
_S8 = [math.sqrt(1 / 8)] + [math.sqrt(2 / 8)] * 7


def _dct_1d(v):
    return [sum(v[n] * _COS8[k][n] for n in range(8)) * _S8[k]
            for k in range(8)]


def _idct_1d(v):
    return [sum(v[k] * _S8[k] * _COS8[k][n] for k in range(8))
            for n in range(8)]


def dct2(block):
    """2D DCT-II (orthonormal) of an 8x8 block."""
    tmp = [_dct_1d(row) for row in block]
    cols = [[tmp[r][c] for r in range(8)] for c in range(8)]
    out = [_dct_1d(col) for col in cols]
    return [[out[c][r] for c in range(8)] for r in range(8)]


def idct2(block):
    """Inverse 2D DCT of an 8x8 block."""
    tmp = [_idct_1d(row) for row in block]
    cols = [[tmp[r][c] for r in range(8)] for c in range(8)]
    out = [_idct_1d(col) for col in cols]
    return [[out[c][r] for c in range(8)] for r in range(8)]


# Standard JPEG luminance quantization table.
_Q_BASE = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]


def _scaled_quant_table(quality):
    q = max(1, min(100, int(quality)))
    scale = 5000 / q if q < 50 else 200 - 2 * q
    return [
        [max(1, min(255, int((b * scale + 50) // 100))) for b in row]
        for row in _Q_BASE
    ]


def jpeg_roundtrip(img: Image, quality=85) -> Image:
    """Simulate one JPEG save/load cycle at the given quality.

    8x8 blocks go through level shift, DCT, quantization with the
    scaled luminance table, dequantization, and inverse DCT. Edges are
    padded by replication so any image size works.
    """
    q = _scaled_quant_table(quality)
    w, h = img.width, img.height
    out = zeros(w, h)
    for by in range(0, h, 8):
        for bx in range(0, w, 8):
            block = [[0.0] * 8 for _ in range(8)]
            for dy in range(8):
                for dx in range(8):
                    sx = min(bx + dx, w - 1)
                    sy = min(by + dy, h - 1)
                    block[dy][dx] = img.pixels[sy][sx] * 255.0 - 128.0
            coef = dct2(block)
            rec = [
                [round(coef[dy][dx] / q[dy][dx]) * q[dy][dx]
                 for dx in range(8)]
                for dy in range(8)
            ]
            spatial = idct2(rec)
            for dy in range(8):
                for dx in range(8):
                    x, y = bx + dx, by + dy
                    if x < w and y < h:
                        v = (spatial[dy][dx] + 128.0) / 255.0
                        out.pixels[y][x] = min(1.0, max(0.0, v))
    return out


def ela(img: Image, q_high=95, q_low=88) -> Image:
    """Error Level Analysis approximation.

    Re-quantize the image at two JPEG-like quality levels and take the
    absolute difference. Regions whose compression history differs from
    the background change differently between the two re-quantizations.
    """
    a = jpeg_roundtrip(img, q_high)
    b = jpeg_roundtrip(img, q_low)
    return Image(
        img.width,
        img.height,
        [
            [abs(ra - rb) for ra, rb in zip(row_a, row_b)]
            for row_a, row_b in zip(a.pixels, b.pixels)
        ],
    )


# -- local noise variance --------------------------------------------------

_LAPLACIAN = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]


def noise_variance_map(img: Image, window=3) -> Image:
    """Local variance of the Laplacian response.

    Flat regions give ~0 response; noisy regions give a high-variance
    response. A splice with a different noise level stands out.
    """
    w, h = img.width, img.height
    lap = zeros(w, h)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            v = (
                img.pixels[y - 1][x] + img.pixels[y + 1][x]
                + img.pixels[y][x - 1] + img.pixels[y][x + 1]
                - 4.0 * img.pixels[y][x]
            )
            lap.pixels[y][x] = v
    out = zeros(w, h)
    r = window // 2
    for y in range(r, h - r):
        for x in range(r, w - r):
            s = s2 = 0.0
            n = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    v = lap.pixels[y + dy][x + dx]
                    s += v
                    s2 += v * v
                    n += 1
            m = s / n
            out.pixels[y][x] = max(0.0, s2 / n - m * m)
    return out


# -- block-boundary artifacts ----------------------------------------------


def block_boundary_score(img: Image, block=8) -> Image:
    """Per-block seam score on the 8x8 grid.

    For each block, compare the mean absolute discontinuity across its
    edges with the mean discontinuity of adjacent pixels inside it.
    A copy-move paste that is not grid-aligned creates fresh seams that
    score high; a pristine JPEG grid scores near zero.
    """
    w, h = img.width, img.height
    out = zeros(w, h)
    for by in range(0, h, block):
        for bx in range(0, w, block):
            bnd = bnd_n = 0.0
            inn = inn_n = 0.0
            for dy in range(block):
                for dx in range(block):
                    x, y = bx + dx, by + dy
                    if x >= w or y >= h:
                        continue
                    v = img.pixels[y][x]
                    if x + 1 < w:
                        d = abs(v - img.pixels[y][x + 1])
                        if dx + 1 == block:
                            bnd += d
                            bnd_n += 1
                        else:
                            inn += d
                            inn_n += 1
                    if y + 1 < h:
                        d = abs(v - img.pixels[y + 1][x])
                        if dy + 1 == block:
                            bnd += d
                            bnd_n += 1
                        else:
                            inn += d
                            inn_n += 1
            score = bnd / max(1.0, bnd_n) - inn / max(1.0, inn_n)
            score = max(0.0, score)
            for dy in range(block):
                for dx in range(block):
                    x, y = bx + dx, by + dy
                    if x < w and y < h:
                        out.pixels[y][x] = score
    return out


# -- edge-gradient inconsistency -------------------------------------------

_SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
_SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def edge_gradient_inconsistency(img: Image, window=3) -> Image:
    """Local coefficient of variation of the Sobel gradient magnitude.

    Re-rendered text or re-sharpened patches have a different edge
    texture than the surrounding card; the normalized local spread of
    gradient magnitudes picks that up independent of overall contrast.
    """
    w, h = img.width, img.height
    mag = zeros(w, h)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = gy = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    v = img.pixels[y + dy][x + dx]
                    gx += v * _SOBEL_X[dy + 1][dx + 1]
                    gy += v * _SOBEL_Y[dy + 1][dx + 1]
            mag.pixels[y][x] = math.hypot(gx, gy)
    out = zeros(w, h)
    r = window // 2
    for y in range(r, h - r):
        for x in range(r, w - r):
            s = s2 = 0.0
            n = 0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    v = mag.pixels[y + dy][x + dx]
                    s += v
                    s2 += v * v
                    n += 1
            m = s / n
            var = max(0.0, s2 / n - m * m)
            out.pixels[y][x] = math.sqrt(var) / (m + 1e-9)
    return out


def forgery_heatmap(img: Image) -> Image:
    """Average of the four min-max normalized forensic maps.

    A quick single-map visualization of "where the signal looks wrong".
    """
    maps = [
        ela(img),
        noise_variance_map(img),
        block_boundary_score(img),
        edge_gradient_inconsistency(img),
    ]
    from .image import normalize01  # local import: image never imports forensics

    norms = [normalize01(m) for m in maps]
    w, h = img.width, img.height
    return Image(
        w,
        h,
        [
            [
                sum(n.pixels[y][x] for n in norms) / len(norms)
                for x in range(w)
            ]
            for y in range(h)
        ],
    )
