"""Logistic regression trained with SGD, from scratch.

The detector pools each forensic feature map into three statistics
(mean, standard deviation, 99th percentile), giving 12 features per
image, standardizes them, and fits a binary logistic regression with
mini-batch-free SGD.
"""

from __future__ import annotations

import math
import random

from .forensics import (
    block_boundary_score,
    edge_gradient_inconsistency,
    ela,
    noise_variance_map,
)
from .image import Image

FEATURE_NAMES = (
    "ela", "noise_variance", "block_boundary", "gradient_inconsistency",
)


def extract_features(img: Image):
    """Return (feature vector, {name: feature map}) for one image."""
    maps = {
        "ela": ela(img),
        "noise_variance": noise_variance_map(img),
        "block_boundary": block_boundary_score(img),
        "gradient_inconsistency": edge_gradient_inconsistency(img),
    }
    feats = []
    for name in FEATURE_NAMES:
        vals = maps[name].values()
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        s = sorted(vals)
        p99 = s[min(len(s) - 1, int(0.99 * (len(s) - 1)))]
        feats.extend([m, math.sqrt(var), p99])
    return feats, maps


class StandardScaler:
    """Mean/std standardization, fit on train, applied everywhere."""

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        n = len(X)
        d = len(X[0])
        self.mean = [sum(row[j] for row in X) / n for j in range(d)]
        self.scale = []
        for j in range(d):
            var = sum((row[j] - self.mean[j]) ** 2 for row in X) / n
            self.scale.append(math.sqrt(var) or 1e-9)
        return self

    def transform(self, X):
        return [
            [(v - self.mean[j]) / self.scale[j] for j, v in enumerate(row)]
            for row in X
        ]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LogisticRegression:
    """Binary logistic regression, SGD with per-epoch shuffling."""

    def __init__(self, n_features, seed=0):
        rng = random.Random(seed)
        self.w = [rng.uniform(-0.01, 0.01) for _ in range(n_features)]
        self.b = 0.0
        self.seed = seed

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def _proba_one(self, x):
        return self._sigmoid(
            sum(wi * xi for wi, xi in zip(self.w, x)) + self.b)

    def fit(self, X, y, lr=0.05, epochs=300):
        """Run SGD; return the per-epoch average binary-cross-entropy loss."""
        rng = random.Random(self.seed)
        n = len(X)
        losses = []
        for _ in range(epochs):
            order = list(range(n))
            rng.shuffle(order)
            total = 0.0
            for i in order:
                p = min(1 - 1e-9, max(1e-9, self._proba_one(X[i])))
                err = p - y[i]
                total += -(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p))
                for j in range(len(self.w)):
                    self.w[j] -= lr * err * X[i][j]
                self.b -= lr * err
            losses.append(total / n)
        return losses

    def predict_proba(self, X):
        return [self._proba_one(x) for x in X]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]
