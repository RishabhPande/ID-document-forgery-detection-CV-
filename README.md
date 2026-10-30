# ID-document-forgery-detection-CV-

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RishabhPande/ID-document-forgery-detection-CV-/blob/main/examples/quickstart.py)

Classic image forensics for **ID-document forgery detection**, written from scratch in pure Python, zero dependencies.

A forged ID can look perfect to a human reviewer and still fail at the signal level. Spliced text carries a different noise signature and a different compression history than the card around it. A copy-moved region drags its JPEG block grid with it and lands on a spot where that grid does not belong. `doc_forgery` generates synthetic ID cards with planted forgeries (copy-move, spliced fields), then detects them with four hand-built forensic features: Error Level Analysis, local noise variance, 8x8 block-boundary artifacts, and edge-gradient inconsistency, pooled into a from-scratch logistic regression trained with SGD.

> **Disclaimer:** every document in this repo is synthetic. Cards are rendered from block glyphs with randomly generated names, ID numbers, and dates of birth that bear no relation to real people. Nothing here touches real PII or real identity documents.

## Quickstart

```bash
python examples/quickstart.py
```

Generates 120 synthetic cards (60 pristine, 30 copy-move, 30 spliced), trains the detector, prints image-level and coarse region-level metrics, and writes sample PPMs (card, forgery mask, forensic heatmap) to `examples/output/`.

Or as a library:

```python
from doc_forgery import CardGenerator, extract_features, LogisticRegression, StandardScaler

gen = CardGenerator(seed=7)
docs = gen.dataset(120)

X = [extract_features(d.image)[0] for d in docs]
y = [d.label for d in docs]

scaler = StandardScaler()
clf = LogisticRegression(len(X[0]), seed=7)
clf.fit(scaler.fit_transform(X), y)
print(clf.predict_proba(scaler.transform(X[:5])))
```

## How detection works

1. **ELA** (`doc_forgery/forensics.py::ela`): the image goes through a JPEG-like 8x8 DCT roundtrip (DCT implemented by hand, standard luminance quantization tables) at two quality levels; the absolute difference is the map. Regions with a different compression history respond differently than the background.
2. **Noise variance** (`noise_variance_map`): local variance of the Laplacian response. A splice with a different noise signature stands out against the card's uniform sensor noise.
3. **Block-boundary score** (`block_boundary_score`): per 8x8 block, edge discontinuity minus interior discontinuity. A pasted region keeps its source grid, so the seam lights up when the paste is not grid-aligned.
4. **Gradient inconsistency** (`edge_gradient_inconsistency`): local coefficient of variation of the Sobel gradient magnitude. Re-rendered, softer text has a different edge texture than the card's crisp print.
5. **Classify** (`doc_forgery/model.py`): mean, standard deviation, and 99th percentile of each map give 12 features; a from-scratch logistic regression trained with SGD scores the card.

Reference run (`python examples/quickstart.py`, 120 cards, 84 train / 36 test): image-level precision 0.941, recall 0.889, F1 0.914, ROC-AUC 0.969 (n=36 test cards: 16 TP, 1 FP, 17 TN, 2 FN, so these are noisy). Coarse region-level (16px cells scored by the forensic heatmap): ROC-AUC 0.634.

![Three rows (pristine, copy-move, spliced) of synthetic ID cards, each with its ground-truth forged-region mask and the forensic heatmap](docs/gallery.png)

Synthetic cards from `CardGenerator(seed=7)`, with the ground-truth mask and the forensic heatmap for one card of each kind. The heatmap is noisy and weak at region level (ROC-AUC 0.634 above), which is why the detector scores whole cards. Regenerate with `python examples/quickstart.py` then `python examples/plot_gallery.py` (needs matplotlib and Pillow).

## Roadmap

- [ ] Deep CNN detector trained on the same synthetic forgeries (classic features as the baseline to beat)
- [ ] Face-morph detection (blended portraits in the photo field)
- [ ] Video liveness cues (temporal consistency across frames)

## Tests

```bash
python3 -m pytest tests/ -q
```

Covers: DCT roundtrip on constant and random 8x8 blocks, ELA highlighting a crafted splice, the noise estimator on a half-noisy image, generator mask geometry and determinism, and decreasing training loss.

## Part of a fraud-ML series

- [GNN Fraud Ring Detection](https://github.com/RishabhPande/GNN-fraud-ring-detection) - GraphSAGE, GAT, GCN, GIN and LSTM from scratch in pure Python (zero dependencies) for fraud ring detection
- [Agentic Fraud Investigator](https://github.com/RishabhPande/agentic-fraud-investigator) - A multi-agent fraud investigation system in pure Python, zero dependencies.
- [Time-series-anomaly-detection](https://github.com/RishabhPande/Time-series-anomaly-detection) - Time-series anomaly detection, including collective anomalies
- [fraud-eval](https://github.com/RishabhPande/fraud-eval) - Open benchmark for evaluating LLM-based fraud and abuse detectors (20-case smoke test, cost-aware metrics, counterfactual evaluation)
- [fraud-rings](https://github.com/RishabhPande/fraud-rings) - Synthetic identity-fraud ring generator plus graph-based detection, all in pure Python, no dependencies.
