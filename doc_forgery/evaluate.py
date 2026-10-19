"""Precision/recall/F1 and ROC-AUC from scratch.

Two levels:

- image level: one score per document vs its label.
- region level (coarse): each image is tiled into cells; a cell is
  positive when the forgery mask covers any of its pixels, and its score
  is the mean heatmap value inside it. This measures localization, not
  just detection.
"""

from __future__ import annotations

from dataclasses import dataclass

from .image import Image


@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    auc: float
    n: int

    def summary(self) -> str:
        return (
            "[eval] n=%d tp=%d fp=%d tn=%d fn=%d "
            "precision=%.3f recall=%.3f f1=%.3f auc=%.3f"
            % (self.n, self.tp, self.fp, self.tn, self.fn,
               self.precision, self.recall, self.f1, self.auc)
        )


def roc_auc(y_true, y_score) -> float:
    """ROC-AUC via the rank (Mann-Whitney) method, ties averaged."""
    n = len(y_true)
    order = sorted(range(n), key=lambda i: y_score[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = sum(r for r, y in zip(ranks, y_true) if y == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _confusion(y_true, y_score, threshold=0.5):
    tp = fp = tn = fn = 0
    for y, s in zip(y_true, y_score):
        pred = 1 if s >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _metrics_from(y_true, y_score, threshold=0.5) -> Metrics:
    tp, fp, tn, fn = _confusion(y_true, y_score, threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return Metrics(tp, fp, tn, fn, precision, recall, f1,
                   roc_auc(y_true, y_score), len(y_true))


def evaluate_image_level(docs, scores, threshold=0.5) -> Metrics:
    """Score whole documents against their labels."""
    return _metrics_from([d.label for d in docs], scores, threshold)


def evaluate_region_level(docs, heatmaps, cell=16, threshold=None) -> Metrics:
    """Score coarse cells against the forgery masks (localization).

    A cell is positive when the forgery mask covers any of its pixels;
    its score is the mean heatmap value inside it. With no threshold
    given, cells scoring above mean + std of all cell scores count as
    flagged: a label-free anomaly cutoff.
    """
    y_true, y_score = [], []
    for doc, heat in zip(docs, heatmaps):
        w, h = doc.image.width, doc.image.height
        for cy in range(0, h, cell):
            for cx in range(0, w, cell):
                label = 0
                total = 0.0
                n = 0
                for y in range(cy, min(cy + cell, h)):
                    for x in range(cx, min(cx + cell, w)):
                        if doc.mask.pixels[y][x] > 0.5:
                            label = 1
                        total += heat.pixels[y][x]
                        n += 1
                y_true.append(label)
                y_score.append(total / max(1, n))
    if threshold is None:
        mean = sum(y_score) / len(y_score)
        var = sum((s - mean) ** 2 for s in y_score) / len(y_score)
        threshold = mean + var ** 0.5
    return _metrics_from(y_true, y_score, threshold)
