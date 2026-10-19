"""Quickstart: generate forged ID cards, train a forensic detector, score it.

Writes sample PPMs (pristine vs forged + mask + heatmap) to examples/output/.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from doc_forgery import (
    CardGenerator,
    LogisticRegression,
    StandardScaler,
    evaluate_image_level,
    evaluate_region_level,
    extract_features,
    forgery_heatmap,
    save_heatmap,
    save_ppm,
)

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def main():
    gen = CardGenerator(seed=7)
    docs = gen.dataset(120)
    kinds = {}
    for d in docs:
        kinds[d.kind] = kinds.get(d.kind, 0) + 1
    print(f"generated {len(docs)} documents: {kinds}")

    train, test = docs[:84], docs[84:]
    Xtr = [extract_features(d.image)[0] for d in train]
    ytr = [d.label for d in train]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    clf = LogisticRegression(len(Xtr[0]), seed=7)
    losses = clf.fit(Xtr, ytr, lr=0.05, epochs=300)
    print(f"trained logistic regression: loss {losses[0]:.4f} -> "
          f"{losses[-1]:.4f} over {len(losses)} epochs")

    feats_maps = [extract_features(d.image) for d in test]
    Xte = scaler.transform([fm[0] for fm in feats_maps])
    probs = clf.predict_proba(Xte)
    heats = [forgery_heatmap(d.image) for d in test]

    print("image-level:  ", evaluate_image_level(test, probs).summary())
    print("region-level: ", evaluate_region_level(test, heats).summary())

    # Sample outputs: one pristine, one of each forgery kind.
    shown = {}
    for d in docs:
        if d.kind not in shown:
            shown[d.kind] = d
    for kind, d in shown.items():
        save_ppm(d.image, os.path.join(OUT, f"{kind}.ppm"))
        save_ppm(d.mask, os.path.join(OUT, f"{kind}_mask.ppm"))
        save_heatmap(forgery_heatmap(d.image),
                     os.path.join(OUT, f"{kind}_heatmap.ppm"))
    print(f"wrote sample PPMs to {OUT}/")


if __name__ == "__main__":
    main()
