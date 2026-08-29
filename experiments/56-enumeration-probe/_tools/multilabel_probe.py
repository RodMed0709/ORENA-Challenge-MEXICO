"""Rung 50 -- multi-label probe for fo_class: predict the SET of classes present from the
hidden state, one binary one-vs-rest logistic regression per class, same PCA-reduction
discipline as `ordinal_probe.py` (imported, not duplicated).

Vocabulary is built from whatever labels are actually observed in the TRAINING fold, not
hardcoded to the SDK's full 10-name registry -- CAMPAIGN_LOG already recorded zero `fo_class`
gold rows using two of those ten (Mesh, Absorbable Hemostatic Agent), so a fixed 10-way
target would carry two always-negative, uninformative columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ordinal_probe import _reduce  # shared PCA/scaler discipline, not duplicated


def parse_labels(answer: str) -> frozenset[str]:
    """Gold is a comma-separated class list ('Clip, Sponge') or 'none' (never observed for
    `fo_class` per CAMPAIGN_LOG -- guarded anyway, a shipped model could still emit it)."""
    a = str(answer).strip()
    if a.lower() == "none" or not a:
        return frozenset()
    return frozenset(p.strip() for p in a.split(","))


def build_vocab(answers) -> list[str]:
    """Sorted label vocabulary observed across `answers` -- deterministic column order."""
    vocab: set[str] = set()
    for a in answers:
        vocab |= parse_labels(a)
    return sorted(vocab)


def to_binary_matrix(answers, vocab: list[str]) -> np.ndarray:
    """(n, len(vocab)) indicator matrix. A label outside `vocab` (e.g. seen only at read
    time, never at fit time) is silently absent from the matrix -- callers must build
    `vocab` from the UNION of fit and read labels if that matters, never from fit alone
    when scoring read-time rows."""
    out = np.zeros((len(answers), len(vocab)), dtype=int)
    idx = {c: i for i, c in enumerate(vocab)}
    for row, a in enumerate(answers):
        for label in parse_labels(a):
            if label in idx:
                out[row, idx[label]] = 1
    return out


@dataclass
class MultiLabelFit:
    vocab: list[str]
    layer: int
    C: float
    class_weight: str | None
    scaler: object
    pca: object
    clfs: dict  # label -> fitted LogisticRegression or _ConstantBinary


def fit(X_train: np.ndarray, y_train_answers, X_test: np.ndarray, *, vocab: list[str],
        layer: int, C: float = 1.0, class_weight: str | None = None,
        n_components: int = 128, seed: int = 0) -> tuple[MultiLabelFit, np.ndarray, np.ndarray]:
    """One binary logistic regression per label in `vocab`. Returns (fit, proba_train,
    proba_test), each proba shaped (n, len(vocab))."""
    from sklearn.linear_model import LogisticRegression

    from ordinal_probe import _ConstantBinary

    scaler, pca, Pi, Po = _reduce(X_train, X_test, n_components=n_components, seed=seed)
    Y = to_binary_matrix(y_train_answers, vocab)

    clfs = {}
    proba_i = np.zeros((Pi.shape[0], len(vocab)))
    proba_o = np.zeros((Po.shape[0], len(vocab)))
    for j, label in enumerate(vocab):
        y_bin = Y[:, j]
        if len(set(y_bin)) < 2:
            # this label is always-present or always-absent in the training fold (a rare
            # class can do this even with 5,838 rows -- Gallstone/Mesh-scale rarity)
            clf = _ConstantBinary(int(y_bin[0]))
        else:
            clf = LogisticRegression(max_iter=3000, C=C, class_weight=class_weight).fit(Pi, y_bin)
        clfs[label] = clf
        proba_i[:, j] = clf.predict_proba(Pi)[:, 1]
        proba_o[:, j] = clf.predict_proba(Po)[:, 1]

    result = MultiLabelFit(vocab=vocab, layer=layer, C=C, class_weight=class_weight,
                           scaler=scaler, pca=pca, clfs=clfs)
    return result, proba_i, proba_o


def predict_sets(proba: np.ndarray, vocab: list[str], *, threshold: float = 0.5) -> list[frozenset[str]]:
    """Independent per-class thresholding -- each label predicted present iff its own
    probability clears `threshold`. Not mutually exclusive (unlike the ordinal probe): a
    row can predict 0, 1, or several labels, matching what the real question asks for."""
    out = []
    for row in proba:
        out.append(frozenset(vocab[j] for j, p in enumerate(row) if p >= threshold))
    return out


def exact_set_match(pred_sets, gold_answers) -> np.ndarray:
    """The real scoring rule for `fo_class`: predicted set must equal gold set exactly."""
    gold_sets = [parse_labels(a) for a in gold_answers]
    return np.array([p == g for p, g in zip(pred_sets, gold_sets)])


def aggregated_count(pred_sets) -> np.ndarray:
    """`len(predicted set)` -- exact for the `n_classes` number sub-template, an
    approximation (undercounts multi-instance-same-class frames) everywhere else."""
    return np.array([len(s) for s in pred_sets])


__all__ = [
    "parse_labels", "build_vocab", "to_binary_matrix", "fit", "predict_sets",
    "exact_set_match", "aggregated_count", "MultiLabelFit",
]
