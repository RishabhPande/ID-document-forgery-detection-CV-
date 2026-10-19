"""Synthetic ID-card generator with planted forgery variants.

Every card is rendered from block glyphs. Names, ID numbers, and dates of
birth are randomly generated and bear no relation to real people. Nothing
here touches real PII or real identity documents.

Three variants:

- pristine: uniform noise + a single JPEG pass (label 0).
- copy-move: the finished card's ID-number strip is cloned over a smooth
  blank area at a spot whose 8x8 grid is misaligned with the source
  (label 1). The pasted strip keeps the source's compression grid, so a
  seam lights up in the block-boundary and ELA features.
- spliced: the ID-number field is re-rendered with a different number,
  slightly blurred, given a harsher JPEG pass and heavier noise applied
  *after* compression, then pasted back (label 1). Different noise +
  different compression history + softer edges: the classic splice
  tells, caught by the noise map, ELA, and gradient inconsistency.

Every document ships with a pixel-level forgery mask and a label.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .forensics import jpeg_roundtrip
from .image import Image, add_gaussian_noise, crop, fill_rect, full, paste, zeros

CARD_W, CARD_H = 128, 80

# Field geometry: (x0, y0, x1, y1), half-open.
PHOTO_BOX = (8, 18, 38, 52)
ID_VALUE_BOX = (46, 49, 100, 56)   # 9 digits x 6px = 54 wide, 7 tall
DOB_VALUE_BOX = (46, 71, 94, 78)   # 8 digits x 6px = 48 wide, 7 tall

_NAMES = [
    "AVA SHARMA", "LIAM VERMA", "MIA IYER", "NOAH GUPTA",
    "ZARA KHAN", "ARYA NAIR", "KIA RAO", "NEIL DAS",
]

# 5x7 block font: '#' = ink, '.' = paper. Covers A-Z, 0-9, space and a
# few punctuation marks used on the card.
_FONT = {
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "B": ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."],
    "C": [".####", "#....", "#....", "#....", "#....", "#....", ".####"],
    "D": ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "F": ["#####", "#....", "#....", "####.", "#....", "#....", "#...."],
    "G": [".####", "#....", "#....", "#.###", "#...#", "#...#", ".###."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "I": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"],
    "J": ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "L": ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
    "M": ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
    "N": ["#...#", "##..#", "##..#", "#.#.#", "#..##", "#..##", "#...#"],
    "O": [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
    "Q": [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "S": [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
    "V": ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."],
    "W": ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "Z": ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
    "0": [".###.", "#..##", "#.#.#", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
    "-": [".....", ".....", ".....", "#####", ".....", ".....", "....."],
    ".": [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."],
    "/": ["....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."],
    ":": [".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."],
}


def draw_text(img: Image, text: str, x: int, y: int, color=0.15) -> None:
    """Render block-glyph text onto img in place (clipped)."""
    for i, ch in enumerate(text.upper()):
        glyph = _FONT.get(ch, _FONT[" "])
        for gy in range(7):
            row = glyph[gy]
            for gx in range(5):
                if row[gx] == "#":
                    img.set(x + i * 6 + gx, y + gy, color)


def _draw_photo(img: Image, x: int, y: int, w: int, h: int) -> None:
    """Placeholder portrait: gray frame, head circle, shoulders."""
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            img.set(xx, yy, 0.74)
    for xx in range(x, x + w):
        img.set(xx, y, 0.40)
        img.set(xx, y + h - 1, 0.40)
    for yy in range(y, y + h):
        img.set(x, yy, 0.40)
        img.set(x + w - 1, yy, 0.40)
    cx, cy, r = x + w // 2, y + h // 2 - 4, 7
    for yy in range(y + 1, y + h - 1):
        for xx in range(x + 1, x + w - 1):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r:
                img.set(xx, yy, 0.55)
    for yy in range(cy + 6, y + h - 1):
        half = (yy - (cy + 6)) // 2 + 4
        for xx in range(cx - half, cx + half + 1):
            img.set(xx, yy, 0.62)


def _render_card(rng: random.Random):
    """Render a pristine card; returns (image, field dict)."""
    img = full(CARD_W, CARD_H, 0.96)
    for y in range(12):  # header band
        for x in range(CARD_W):
            img.pixels[y][x] = 0.30
    draw_text(img, "ID", 6, 2, color=0.92)
    draw_text(img, "IDENTITY CARD", 20, 2, color=0.92)
    for x in range(CARD_W):  # border
        img.pixels[0][x] = img.pixels[CARD_H - 1][x] = 0.35
    for y in range(CARD_H):
        img.pixels[y][0] = img.pixels[y][CARD_W - 1] = 0.35

    _draw_photo(img, *PHOTO_BOX[:2], PHOTO_BOX[2] - PHOTO_BOX[0],
                PHOTO_BOX[3] - PHOTO_BOX[1])

    name = rng.choice(_NAMES)
    number = "".join(rng.choice("0123456789") for _ in range(9))
    dob = "%02d%02d%04d" % (
        rng.randint(1, 28), rng.randint(1, 12), rng.randint(1975, 2005))

    draw_text(img, "NAME", 46, 18, color=0.45)
    draw_text(img, name, 46, 27, color=0.15)
    draw_text(img, "ID NO", 46, 40, color=0.45)
    draw_text(img, number, *ID_VALUE_BOX[:2], color=0.15)
    draw_text(img, "DOB", 46, 62, color=0.45)
    draw_text(img, dob, *DOB_VALUE_BOX[:2], color=0.15)
    return img, {"name": name, "number": number, "dob": dob}


def _mask_rect(x0, y0, x1, y1) -> Image:
    mask = zeros(CARD_W, CARD_H)
    for y in range(y0, y1):
        for x in range(x0, x1):
            mask.pixels[y][x] = 1.0
    return mask


def _box_blur(img: Image) -> Image:
    """One pass of 3x3 box blur (clipped edges)."""
    out = img.copy()
    for y in range(1, img.height - 1):
        for x in range(1, img.width - 1):
            s = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    s += img.pixels[y + dy][x + dx]
            out.pixels[y][x] = s / 9.0
    return out


# Copy-move destination: blank card area right of the photo, misaligned
# with the 8x8 grid (66 % 8 = 2, 18 % 8 = 2) so the pasted strip drags a
# shifted compression grid onto a smooth background.
COPYMOVE_SRC = ID_VALUE_BOX
COPYMOVE_DST = (66, 18)


@dataclass
class Document:
    image: Image
    mask: Image   # 1.0 on forged pixels, 0.0 elsewhere
    label: int    # 0 = pristine, 1 = forged
    kind: str     # "pristine" | "copy-move" | "spliced"


class CardGenerator:
    """Seedable generator of synthetic ID cards with planted forgeries."""

    KINDS = ("pristine", "copy-move", "spliced")

    def __init__(self, seed=0):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate(self, kind=None) -> Document:
        kind = kind or self.rng.choice(self.KINDS)
        img, _fields = _render_card(self.rng)

        if kind == "pristine":
            img = self._finish(img)
            return Document(img, zeros(CARD_W, CARD_H), 0, kind)

        if kind == "copy-move":
            # Clone the ID-number text strip onto the finished card over a
            # smooth blank area, at a grid-misaligned spot. Same pixels,
            # shifted 8x8 grid, hot texture on a cold background: the seam
            # and the misplaced content are both tells.
            img = self._finish(img)
            x0, y0, x1, y1 = COPYMOVE_SRC
            strip = crop(img, x0, y0, x1, y1)
            dx, dy = COPYMOVE_DST
            img = paste(img, strip, dx, dy)
            mask = _mask_rect(dx, dy, dx + strip.width, dy + strip.height)
            return Document(img, mask, 1, kind)

        if kind == "spliced":
            # Re-render the ID number with a different value, softer
            # edges, a harsher JPEG pass, and heavier noise applied
            # after compression so it survives, then paste it back.
            x0, y0, x1, y1 = ID_VALUE_BOX
            w, hh = x1 - x0, y1 - y0
            img = fill_rect(img, x0, y0, w, hh, 0.96)
            patch = full(w, hh, 0.96)
            forged_number = "".join(
                self.rng.choice("0123456789") for _ in range(9))
            draw_text(patch, forged_number, 0, 0, color=0.15)
            patch = _box_blur(_box_blur(patch))
            patch = jpeg_roundtrip(patch, 60)
            patch = add_gaussian_noise(
                patch, 0.10, self.rng.randrange(1 << 30))
            img = paste(img, patch, x0, y0)
            img = self._finish(img)
            mask = _mask_rect(x0, y0, x1, y1)
            return Document(img, mask, 1, kind)

        raise ValueError(f"unknown kind: {kind}")

    def _finish(self, img: Image) -> Image:
        """Global capture pipeline: light sensor noise + one JPEG pass."""
        img = add_gaussian_noise(img, 0.015, self.rng.randrange(1 << 30))
        return jpeg_roundtrip(img, 85)

    def dataset(self, n: int) -> list:
        """Balanced-ish dataset: half pristine, quarter per forgery kind."""
        n_pristine = n // 2
        n_forged = n - n_pristine
        kinds = (
            ["pristine"] * n_pristine
            + ["copy-move"] * (n_forged // 2)
            + ["spliced"] * (n_forged - n_forged // 2)
        )
        self.rng.shuffle(kinds)
        return [self.generate(k) for k in kinds]
