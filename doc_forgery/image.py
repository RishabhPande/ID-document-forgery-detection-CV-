"""Minimal grayscale image type built on nested lists of floats.

Pixel values live in [0, 1]. Everything here is stdlib-only on purpose:
the point of this repo is signal-level forensics you can read and audit,
not a dependency tree.
"""

from __future__ import annotations

import math
import random


class Image:
    """Grayscale image: a width x height grid of floats in [0, 1]."""

    def __init__(self, width, height, pixels=None, fill=0.0):
        self.width = int(width)
        self.height = int(height)
        if pixels is None:
            self.pixels = [
                [float(fill)] * self.width for _ in range(self.height)
            ]
        else:
            if len(pixels) != self.height or any(
                len(row) != self.width for row in pixels
            ):
                raise ValueError("pixel grid does not match width x height")
            self.pixels = [[float(v) for v in row] for row in pixels]

    def copy(self) -> "Image":
        return Image(self.width, self.height, self.pixels)

    def get(self, x, y, default=0.0) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixels[y][x]
        return default

    def set(self, x, y, value) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = float(value)

    def values(self) -> list:
        return [v for row in self.pixels for v in row]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Image({self.width}x{self.height})"


def full(width, height, value) -> Image:
    return Image(width, height, fill=value)


def zeros(width, height) -> Image:
    return Image(width, height, fill=0.0)


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def mean(img: Image) -> float:
    vals = img.values()
    return sum(vals) / len(vals)


def std(img: Image) -> float:
    vals = img.values()
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def normalize01(img: Image) -> Image:
    """Min-max normalize into [0, 1]; flat images map to zeros."""
    vals = img.values()
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return zeros(img.width, img.height)
    return Image(
        img.width,
        img.height,
        [[(v - lo) / (hi - lo) for v in row] for row in img.pixels],
    )


def crop(img: Image, x0, y0, x1, y1) -> Image:
    """Half-open crop [x0, x1) x [y0, y1), clipped to the image."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    return Image(
        x1 - x0,
        y1 - y0,
        [row[x0:x1] for row in img.pixels[y0:y1]],
    )


def paste(dst: Image, src: Image, x, y) -> Image:
    """Return a copy of dst with src overlaid at (x, y), clipped."""
    out = dst.copy()
    for sy in range(src.height):
        for sx in range(src.width):
            dx, dy = x + sx, y + sy
            if 0 <= dx < out.width and 0 <= dy < out.height:
                out.pixels[dy][dx] = src.pixels[sy][sx]
    return out


def fill_rect(img: Image, x0, y0, w, h, value) -> Image:
    """Return a copy of img with a filled rectangle."""
    out = img.copy()
    for y in range(y0, y0 + h):
        for x in range(x0, x0 + w):
            out.set(x, y, value)
    return out


def add_gaussian_noise(img: Image, sigma: float, seed: int) -> Image:
    """Additive Gaussian noise via Box-Muller; seedable for reproducibility."""
    rng = random.Random(seed)
    out = img.copy()
    for y in range(img.height):
        row_in, row_out = img.pixels[y], out.pixels[y]
        for x in range(img.width):
            u1 = rng.random() or 1e-12
            u2 = rng.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            row_out[x] = clamp01(row_in[x] + sigma * z)
    return out


def save_ppm(img: Image, path: str) -> None:
    """Write a plain (ASCII, P3) PPM: valid UTF-8, easy to inspect."""
    with open(path, "w", encoding="ascii") as f:
        f.write(f"P3\n{img.width} {img.height}\n255\n")
        for row in img.pixels:
            f.write(
                " ".join(
                    f"{_byte(v)} {_byte(v)} {_byte(v)}" for v in row
                )
                + "\n"
            )


def _byte(v: float) -> int:
    return min(255, max(0, int(round(clamp01(v) * 255))))


def load_ppm(path: str) -> Image:
    """Read a P3 (ASCII) or P6 (binary) PPM back into an Image."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"P3", b"P6"):
            raise ValueError(f"not a PPM file: {path}")
        dims = _ppm_tokens(f, 2)
        width, height = int(dims[0]), int(dims[1])
        maxval = int(_ppm_tokens(f, 1)[0])
        if magic == b"P3":
            vals = [int(t) for t in _ppm_tokens(f, width * height * 3)]
            px = [
                vals[i * 3] / maxval
                for i in range(width * height)
            ]
        else:
            raw = f.read(width * height * 3)
            if len(raw) < width * height * 3:
                raise ValueError(f"truncated PPM data: {path}")
            px = [raw[i * 3] / maxval for i in range(width * height)]
    return Image(
        width,
        height,
        [px[r * width:(r + 1) * width] for r in range(height)],
    )


def _ppm_tokens(f, count):
    tokens = []
    while len(tokens) < count:
        line = f.readline()
        if not line:
            break
        line = line.split(b"#", 1)[0]
        tokens.extend(line.split())
    return tokens


def save_heatmap(values: Image, path: str) -> None:
    """Write a normalized blue-to-red heatmap of a feature map (P3 PPM)."""
    norm = normalize01(values)
    with open(path, "w", encoding="ascii") as f:
        f.write(f"P3\n{norm.width} {norm.height}\n255\n")
        for row in norm.pixels:
            parts = []
            for v in row:
                r = _byte(v)
                b = _byte(1.0 - v)
                parts.append(f"{r} 0 {b}")
            f.write(" ".join(parts) + "\n")
