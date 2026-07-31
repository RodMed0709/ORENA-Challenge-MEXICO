"""Diagnostic: does the model shortcut position questions via a class->quadrant prior?

Mirrors the diagnostic in "Your other Left! Vision-Language Models Fail to Identify
Relative Positions in Medical Images" (MICCAI 2025, arXiv:2508.00549): that paper found
VLMs answer medical position questions from memorised anatomical priors instead of
reading the image, exposed by testing accuracy on cases where the true position
contradicts the typical one. This module runs the same test on FRAME's own quadrant
questions and existing predictions.

Read-only. No image, no JSONL, no checkpoint touched -- this only reads a run's already
-written results.csv against the raw parquets' question/answer text.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Specimen bag", "Sponge", "External drain", "Needle", "Clip",
           "Specimen", "Silicone loop", "Gallstone"]
_CLASSES_SORTED = sorted(CLASSES, key=len, reverse=True)
_CLASS_RE = {c: re.compile(rf"\b{re.escape(c)}\b", re.IGNORECASE) for c in _CLASSES_SORTED}
_QUAD_RE = re.compile(r"\b(top|bottom)[\s/-]*(left|right)\b", re.IGNORECASE)
_ITEM_RE = re.compile(r"([A-Za-z ]+?)\s*:\s*(top|bottom)[\s/-]*(left|right)", re.IGNORECASE)

# Below this many train observations, "typical quadrant" is too noisy to trust --
# Gallstone has only 7 (class, quadrant) pairs in the whole train split.
MIN_CLASS_N = 20


def find_class(text: str) -> str | None:
    """The first known FO class name matched in *text*, longest-name-first so
    'Specimen bag' does not get shadowed by 'Specimen'."""
    for c in _CLASSES_SORTED:
        if _CLASS_RE[c].search(str(text)):
            return c
    return None


def _quad(v: str, h: str) -> str:
    return f"{v.lower()}/{h.lower()}"


def class_quadrant_rows(audit: pd.DataFrame) -> pd.DataFrame:
    """One row per (qID, class, quadrant) observation, from the audit's 3 transformable
    rules (``flip_audit.audit_dataframe`` output; needs id/question/answer/rule/
    disposition/split/dataset).

    qID is rebuilt as ``<dataset>__<id>`` -- the same convention
    ``frame.ledger.gold_from_frame_parquets`` uses, so this joins directly against any
    run's ``results.csv``.
    """
    out: list[tuple] = []
    tf = audit[audit.disposition == "transformable"]
    for _, row in tf.iterrows():
        qid = f"{row['dataset']}__{row['id']}"
        if row["rule"] == "fixed_quadrant_class":
            m = _QUAD_RE.search(row["question"])
            cls = find_class(row["answer"])
            if m and cls:
                out.append((qid, row["split"], cls, _quad(m.group(1), m.group(2)), row["rule"]))
        elif row["rule"] == "object_center_quadrant":
            cls = find_class(row["question"])
            m = _QUAD_RE.search(row["answer"])
            if m and cls:
                out.append((qid, row["split"], cls, _quad(m.group(1), m.group(2)), row["rule"]))
        elif row["rule"] == "all_object_positions":
            for m in _ITEM_RE.finditer(str(row["answer"])):
                cls = find_class(m.group(1))
                if cls:
                    out.append((qid, row["split"], cls, _quad(m.group(2), m.group(3)), row["rule"]))
    return pd.DataFrame(out, columns=["qID", "split", "class", "quadrant", "rule"])


def typical_quadrant_map(train_pairs: pd.DataFrame) -> dict[str, str]:
    """Each class's single modal (most frequent) quadrant, from TRAIN only.

    Computed on train and applied to test, so the typical/atypical label is never
    circular with the rows accuracy gets measured on.
    """
    counts = train_pairs.groupby("class").size()
    keep = counts[counts >= MIN_CLASS_N].index
    modal: dict[str, str] = {}
    for cls in keep:
        vc = train_pairs.loc[train_pairs["class"] == cls, "quadrant"].value_counts()
        modal[cls] = vc.idxmax()
    return modal


def _video_key(df: pd.DataFrame) -> pd.Series:
    return df["qID"].str.split("__", n=1).str[0] + "|" + df["video"].astype(str)


def label_typicality(test_pairs: pd.DataFrame, modal: dict[str, str]) -> pd.DataFrame:
    """Test rows whose class has a train-derived typical quadrant, tagged typical/atypical.

    A qID can appear more than once (an ``all_object_positions`` list naming several
    classes) -- kept as separate observations, one per named object, matching how the
    source paper compares per-structure rather than per-question.
    """
    out = test_pairs[test_pairs["class"].isin(modal)].copy()
    out["typical_quadrant"] = out["class"].map(modal)
    out["is_typical"] = out["quadrant"] == out["typical_quadrant"]
    return out


def score_against_run(labelled: pd.DataFrame, results_csv: Path, *, n_boot: int = 2000,
                       seed: int = 0) -> dict:
    """Accuracy on typical vs atypical class/quadrant pairs, for one run's results.csv.

    ``correctness`` grades the whole qID. For fixed_quadrant_class/object_center_quadrant
    (one class per qID) that is exact. For all_object_positions (a list answer, several
    classes per qID) it slightly overstates precision on any one named object, since a
    wrong OTHER item in the same list also marks this one "incorrect" -- reported per rule
    too, so the single-item rules can be read on their own as the cleaner cut.
    """
    res = pd.read_csv(results_csv, usecols=["qID", "video", "correctness"])
    merged = labelled.merge(res, on="qID", how="inner")

    def _cell(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"n": 0, "n_videos": 0, "acc": float("nan")}
        vkey = _video_key(sub)
        return {"n": int(len(sub)), "n_videos": int(vkey.nunique()),
                "acc": float(sub["correctness"].mean())}

    typ, atyp = merged[merged.is_typical], merged[~merged.is_typical]
    out: dict = {"typical": _cell(typ), "atypical": _cell(atyp),
                 "n_matched": int(len(merged)), "n_labelled": int(len(labelled))}

    if len(typ) and len(atyp):
        tv = {v: g["correctness"].to_numpy(float) for v, g in typ.groupby(_video_key(typ))}
        av = {v: g["correctness"].to_numpy(float) for v, g in atyp.groupby(_video_key(atyp))}
        rng = np.random.default_rng(seed)
        boots = np.empty(n_boot)
        t_keys, a_keys = list(tv), list(av)
        for b in range(n_boot):
            t_pick = rng.choice(t_keys, size=len(t_keys), replace=True)
            a_pick = rng.choice(a_keys, size=len(a_keys), replace=True)
            t_acc = np.mean([rng.choice(tv[k], size=len(tv[k]), replace=True).mean() for k in t_pick])
            a_acc = np.mean([rng.choice(av[k], size=len(av[k]), replace=True).mean() for k in a_pick])
            boots[b] = a_acc - t_acc
        out["gap_atypical_minus_typical"] = float(atyp["correctness"].mean() - typ["correctness"].mean())
        out["gap_ci_low"] = float(np.percentile(boots, 2.5))
        out["gap_ci_high"] = float(np.percentile(boots, 97.5))
    else:
        out["gap_atypical_minus_typical"] = float("nan")
        out["gap_ci_low"] = float("nan")
        out["gap_ci_high"] = float("nan")
    return out
