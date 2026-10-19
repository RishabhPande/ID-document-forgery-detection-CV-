"""doc-forgery: classic image forensics for ID-document forgery detection.

Everything in this package operates on SYNTHETIC data only. Cards are
rendered from block glyphs with random names, numbers, and dates; no
real PII, no real identity documents.
"""

from doc_forgery.evaluate import (
    Metrics,
    evaluate_image_level,
    evaluate_region_level,
    roc_auc,
)
from doc_forgery.forensics import (
    block_boundary_score,
    dct2,
    edge_gradient_inconsistency,
    ela,
    forgery_heatmap,
    idct2,
    jpeg_roundtrip,
    noise_variance_map,
)
from doc_forgery.generator import CARD_H, CARD_W, CardGenerator, Document
from doc_forgery.image import (
    Image,
    add_gaussian_noise,
    crop,
    full,
    load_ppm,
    paste,
    save_heatmap,
    save_ppm,
    zeros,
)
from doc_forgery.model import (
    LogisticRegression,
    StandardScaler,
    extract_features,
)

__all__ = [
    "CARD_H",
    "CARD_W",
    "CardGenerator",
    "Document",
    "Image",
    "LogisticRegression",
    "Metrics",
    "StandardScaler",
    "add_gaussian_noise",
    "block_boundary_score",
    "crop",
    "dct2",
    "edge_gradient_inconsistency",
    "ela",
    "evaluate_image_level",
    "evaluate_region_level",
    "extract_features",
    "forgery_heatmap",
    "full",
    "idct2",
    "jpeg_roundtrip",
    "load_ppm",
    "noise_variance_map",
    "paste",
    "roc_auc",
    "save_heatmap",
    "save_ppm",
    "zeros",
]
