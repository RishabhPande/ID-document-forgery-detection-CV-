import pytest

from doc_forgery import (
    CardGenerator,
    Image,
    LogisticRegression,
    StandardScaler,
    add_gaussian_noise,
    crop,
    dct2,
    ela,
    extract_features,
    full,
    idct2,
    jpeg_roundtrip,
    noise_variance_map,
    paste,
    zeros,
)
from doc_forgery.generator import (
    CARD_H,
    CARD_W,
    COPYMOVE_DST,
    COPYMOVE_SRC,
    ID_VALUE_BOX,
)


def _mean_region(img, x0, y0, x1, y1):
    vals = [img.pixels[y][x]
            for y in range(y0, y1) for x in range(x0, x1)]
    return sum(vals) / len(vals)


def test_dct_roundtrip_constant_block():
    block = [[0.37] * 8 for _ in range(8)]
    rec = idct2(dct2(block))
    err = max(abs(rec[y][x] - 0.37) for y in range(8) for x in range(8))
    assert err < 1e-9


def test_dct_roundtrip_random_block():
    import random
    rng = random.Random(5)
    block = [[rng.random() for _ in range(8)] for _ in range(8)]
    rec = idct2(dct2(block))
    err = max(abs(rec[y][x] - block[y][x])
              for y in range(8) for x in range(8))
    assert err < 1e-9


def test_ela_highlights_spliced_region():
    # Crafted case mirroring the generator: card-like content with one
    # JPEG pass vs a patch with a harsher JPEG history, then a global
    # re-save. ELA must light up the double-compressed patch.
    base = full(64, 64, 0.92)
    for y in range(20, 28):  # fake text bars
        for x in range(8, 56):
            base.pixels[y][x] = 0.2
    for y in range(36, 44):
        for x in range(8, 40):
            base.pixels[y][x] = 0.2
    base = jpeg_roundtrip(base, 85)
    patch = full(24, 16, 0.92)
    for y in range(4, 11):
        for x in range(3, 21):
            patch.pixels[y][x] = 0.2
    patch = jpeg_roundtrip(patch, 55)  # different compression history
    img = paste(base, patch, 20, 20)
    img = jpeg_roundtrip(img, 85)  # global re-save, like the generator

    m = ela(img)
    inside = _mean_region(m, 22, 22, 42, 34)
    background = _mean_region(m, 2, 46, 62, 60)
    assert inside > 5.0 * background


def test_noise_estimator_higher_on_noisy_region():
    img = zeros(64, 32)
    noisy = add_gaussian_noise(crop(img, 0, 0, 32, 32), 0.2, seed=3)
    img = paste(img, noisy, 0, 0)
    m = noise_variance_map(img)
    left = _mean_region(m, 4, 4, 28, 28)
    right = _mean_region(m, 36, 4, 60, 28)
    assert left > 3.0 * right


def test_generator_masks_match():
    gen = CardGenerator(seed=3)

    pristine = gen.generate("pristine")
    assert pristine.label == 0 and pristine.kind == "pristine"
    assert sum(pristine.mask.values()) == 0.0

    moved = gen.generate("copy-move")
    assert moved.label == 1 and moved.kind == "copy-move"
    dx, dy = COPYMOVE_DST
    w = COPYMOVE_SRC[2] - COPYMOVE_SRC[0]
    h = COPYMOVE_SRC[3] - COPYMOVE_SRC[1]
    assert sum(moved.mask.values()) == w * h
    # mask is exactly the pasted strip, nowhere else
    for y in range(CARD_H):
        for x in range(CARD_W):
            expect = 1.0 if (dx <= x < dx + w and dy <= y < dy + h) else 0.0
            assert moved.mask.pixels[y][x] == expect

    spliced = gen.generate("spliced")
    assert spliced.label == 1 and spliced.kind == "spliced"
    x0, y0, x1, y1 = ID_VALUE_BOX
    assert sum(spliced.mask.values()) == (x1 - x0) * (y1 - y0)

    assert pristine.image.width == CARD_W and pristine.image.height == CARD_H


def test_generator_is_deterministic():
    a = CardGenerator(seed=11).dataset(12)
    b = CardGenerator(seed=11).dataset(12)
    for da, db in zip(a, b):
        assert da.kind == db.kind and da.label == db.label
        assert da.image.pixels == db.image.pixels
        assert da.mask.pixels == db.mask.pixels


def test_training_loss_decreases():
    docs = CardGenerator(seed=11).dataset(40)
    X = [extract_features(d.image)[0] for d in docs]
    y = [d.label for d in docs]
    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(len(X[0]), seed=11)
    losses = clf.fit(X, y, lr=0.05, epochs=200)
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.8 * losses[0]
