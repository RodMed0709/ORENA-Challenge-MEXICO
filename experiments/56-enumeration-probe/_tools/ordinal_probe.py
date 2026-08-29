"""Rung 50 -- classifier variants for the counting probe: plain multinomial, class-weighted,
and ordinal (Frank & Hall decomposition). PCA dimensionality reduction shared by all three,
fit on the training fold only -- same discipline rung 34 established after its first probe
memorised a 768-row/4096-dim fit perfectly and transferred at 0.25-0.43.

Three variants, in the order the team agreed to test them:
1. `kind="multinomial"` -- rung 34's original recipe, unweighted, replicated on new data.
2. `kind="balanced"` -- same, `class_weight="balanced"`.
3. `kind="ordinal"` -- K-1 binary "is y > threshold" classifiers (Frank & Hall), which respects
   count ordering (predicting 5 when gold is 4 costs less than predicting 12) where the other
   two treat every wrong class as equally wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitResult:
    kind: str
    layer: int
    C: float
    class_weight: str | None
    pca_n_components: int
    scaler: object
    pca: object
    model: object  # LogisticRegression, or dict of them for "ordinal"
    classes: list[int] = field(default_factory=list)


def _reduce(X_train: np.ndarray, X_test: np.ndarray, *, n_components: int = 128, seed: int = 0):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    Zi, Zo = scaler.transform(X_train), scaler.transform(X_test)
    n_comp = min(n_components, Zi.shape[0] - 1, Zi.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit(Zi)
    return scaler, pca, pca.transform(Zi), pca.transform(Zo)


def fit(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, *, kind: str,
        layer: int, C: float = 1.0, class_weight: str | None = None,
        n_components: int = 128, seed: int = 0) -> tuple[FitResult, np.ndarray, np.ndarray]:
    """Fit one (layer, kind, C, class_weight) combination. Returns (fit_result, proba_train,
    proba_test) -- proba columns align with `fit_result.classes`, sorted ascending."""
    from sklearn.linear_model import LogisticRegression

    scaler, pca, Pi, Po = _reduce(X_train, X_test, n_components=n_components, seed=seed)
    classes = sorted(int(c) for c in set(y_train))

    if kind in ("multinomial", "balanced"):
        cw = "balanced" if kind == "balanced" else class_weight
        clf = LogisticRegression(max_iter=3000, C=C, class_weight=cw).fit(Pi, y_train)
        # align proba columns to `classes` -- sklearn only returns columns for labels it saw
        proba_i = _align_proba(clf.predict_proba(Pi), clf.classes_, classes)
        proba_o = _align_proba(clf.predict_proba(Po), clf.classes_, classes)
        result = FitResult(kind=kind, layer=layer, C=C, class_weight=cw,
                           pca_n_components=pca.n_components_, scaler=scaler, pca=pca,
                           model=clf, classes=classes)
        return result, proba_i, proba_o

    if kind == "ordinal":
        thresholds = classes[:-1]
        clfs = {}
        for t in thresholds:
            y_bin = (y_train > t).astype(int)
            if len(set(y_bin)) < 2:
                # every training row is on one side of this threshold (possible for a rare
                # extreme class in a small fold) -- a constant predictor, not an error
                clfs[t] = _ConstantBinary(int(y_bin[0]))
                continue
            clfs[t] = LogisticRegression(max_iter=3000, C=C, class_weight=class_weight).fit(Pi, y_bin)
        model = {"thresholds": thresholds, "clfs": clfs}
        proba_i = _ordinal_proba(model, Pi, classes)
        proba_o = _ordinal_proba(model, Po, classes)
        result = FitResult(kind=kind, layer=layer, C=C, class_weight=class_weight,
                           pca_n_components=pca.n_components_, scaler=scaler, pca=pca,
                           model=model, classes=classes)
        return result, proba_i, proba_o

    raise ValueError(f"unknown kind {kind!r} -- expected multinomial/balanced/ordinal")


class _ConstantBinary:
    """A degenerate 'classifier' for a threshold every training row is on one side of.
    Returns that side with probability 1 -- correct behaviour, not a silent failure, since
    a real logistic fit on a single-class target would raise anyway."""

    def __init__(self, value: int) -> None:
        self._value = value

    def predict_proba(self, X):
        n = X.shape[0]
        out = np.zeros((n, 2))
        out[:, self._value] = 1.0
        return out


def _ordinal_proba(model: dict, X: np.ndarray, classes: list[int]) -> np.ndarray:
    """Frank & Hall reconstruction: P(y=classes[0]) = 1-P(y>t0); P(y=classes[i]) =
    P(y>t_{i-1}) - P(y>t_i); P(y=classes[-1]) = P(y>t_{-1})."""
    thresholds, clfs = model["thresholds"], model["clfs"]
    if not thresholds:  # single-class fold, degenerate but must not crash
        return np.ones((X.shape[0], 1))
    p_gt = np.column_stack([clfs[t].predict_proba(X)[:, 1] for t in thresholds])
    # independently-fit binary classifiers are not guaranteed monotonic in the threshold --
    # enforce P(y>t) non-increasing as t rises, the one algebraic fact they must satisfy.
    p_gt = np.minimum.accumulate(p_gt, axis=1)

    n, k = X.shape[0], len(classes)
    proba = np.zeros((n, k))
    proba[:, 0] = 1.0 - p_gt[:, 0]
    for i in range(1, k - 1):
        proba[:, i] = p_gt[:, i - 1] - p_gt[:, i]
    proba[:, -1] = p_gt[:, -1]
    proba = np.clip(proba, 0.0, None)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # guard an all-zero row (shouldn't happen post-clip, but cheap)
    return proba / row_sums


def _align_proba(proba: np.ndarray, seen_classes: np.ndarray, all_classes: list[int]) -> np.ndarray:
    """sklearn's predict_proba only has columns for labels present in y_train. Re-index to
    the full class list (zero-filling anything unseen) so callers can assume a fixed shape."""
    seen = list(seen_classes)
    out = np.zeros((proba.shape[0], len(all_classes)))
    for j, c in enumerate(all_classes):
        if c in seen:
            out[:, j] = proba[:, seen.index(c)]
    return out


def predict(proba: np.ndarray, classes: list[int]) -> np.ndarray:
    """Argmax decode -- the Bayes-optimal rule under 0-1/exact-match loss (mode, not mean;
    rung 33 already measured `round(E[y])` losing to greedy for exactly this reason)."""
    idx = proba.argmax(axis=1)
    return np.array(classes)[idx]


__all__ = ["fit", "predict", "FitResult"]
