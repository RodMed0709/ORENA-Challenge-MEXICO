"""05c engine — per-TEMPLATE count confusion `P(predicted | true)` for `number`.

Sibling of ``number_probe`` (05b). 05b answered the aggregate question — *does the
model count, or emit a constant?* — and closed it: it counts, but its scale saturates
at ~2 (Spearman 0.625; acc 0.803 on the modal value vs 0.232 off it).

This module answers the question 05b could not, because it aggregated: **does the
error have correctable STRUCTURE?** It must be per-template — ``acc_number`` is not
interpretable across the 8 templates (4 degenerate; data card §3), so a LUT fitted on
the pooled slice would calibrate a mixture.

The decisive quantity is the **oracle LUT gain**: the accuracy a best-possible
prediction→prediction remap would reach. It is an OPTIMISTIC upper bound (fitted and
evaluated on the same rows), which is exactly what makes a small value decisive — if
even the oracle cannot buy points, post-hoc calibration is dead. ``transfer_lut_gain``
then asks the harder question: does a LUT fitted on one distribution survive on the
other?

Zero GPU. Input is a scored run's ``inspect.csv``; nothing is re-inferred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from frame.metrics import template_of

from .number_probe import parse_number

__all__ = [
    "load_number_slice",
    "confusion_by_template",
    "oracle_lut_gain",
    "transfer_lut_gain",
]


def load_number_slice(inspect_csv: str | "os.PathLike") -> pd.DataFrame:
    """Load a scored run's ``inspect.csv`` and return only its ``number`` rows.

    Adds ``template`` (timestamp-normalised question, the canonical key), ``dist``
    (ID/OOD from the qID prefix — never from the SDK's all-False ``ood`` column,
    RULES §EVAL), and integer ``true``/``pred`` via 05b's parser.

    Rows where either side fails to parse are kept with NaN so the caller can see how
    many were dropped; ``confusion_by_template`` excludes them and reports the count.
    """
    df = pd.read_csv(inspect_csv)
    df = df[df["answer_format"] == "number"].copy()

    df["template"] = df["question"].map(template_of)
    df["dist"] = np.where(
        df["qID"].astype(str).str.startswith("heico"), "OOD", "ID"
    )
    df["true"] = df["ground_truth"].map(parse_number)
    df["pred"] = df["our_answer"].map(parse_number)
    return df


def confusion_by_template(
    df: pd.DataFrame, *, min_n: int = 30
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return ``{template: P(pred|true) matrix}`` plus a per-template summary.

    Rows are true values, columns predicted values, cells are row-normalised
    probabilities. Templates below ``min_n`` are still returned but flagged
    ``underpowered`` in the summary — a confusion matrix on n=12 is decoration.

    The summary carries the two quantities a calibration decision needs:
    ``argmax_injective`` (do distinct true values have distinct modal predictions? if
    not, a LUT provably trades one error for another) and ``n_true_levels``.
    """
    ok = df.dropna(subset=["true", "pred"])
    mats: dict[str, pd.DataFrame] = {}
    rows = []

    for tpl, g in ok.groupby("template"):
        ct = pd.crosstab(g["true"], g["pred"])
        mats[tpl] = ct.div(ct.sum(axis=1), axis=0)

        argmax = mats[tpl].idxmax(axis=1)
        rows.append(
            {
                "template": tpl,
                "n": len(g),
                "n_true_levels": g["true"].nunique(),
                "acc": float((g["true"] == g["pred"]).mean()),
                # a LUT can only help if distinct truths map to distinct modal preds
                "argmax_injective": bool(argmax.nunique() == len(argmax)),
                "underpowered": len(g) < min_n,
            }
        )

    summary = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return mats, summary


def _fit_lut(g: pd.DataFrame) -> dict[float, float]:
    """Best remap pred→true* on ``g``: send each predicted value to the true value it
    most often co-occurs with. This is the accuracy-optimal LUT for exact-match
    scoring (no MAE credit for being close)."""
    return {
        pred: sub["true"].value_counts().idxmax()
        for pred, sub in g.groupby("pred")
    }


def oracle_lut_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Per-template OPTIMISTIC upper bound on post-hoc calibration.

    The LUT is fitted and evaluated on the same rows, so this OVERSTATES what any real
    calibration could deliver. That is deliberate: it makes a small gain decisive
    against the lever, and a large gain merely permissive (it then has to survive
    ``transfer_lut_gain``).
    """
    ok = df.dropna(subset=["true", "pred"])
    rows = []
    for tpl, g in ok.groupby("template"):
        lut = _fit_lut(g)
        after = g["pred"].map(lut)
        rows.append(
            {
                "template": tpl,
                "n": len(g),
                "acc_before": float((g["true"] == g["pred"]).mean()),
                "acc_oracle": float((g["true"] == after).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["oracle_gain"] = out["acc_oracle"] - out["acc_before"]
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def transfer_lut_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Does a LUT fitted on one distribution survive on the other?

    Fits per template on ID and applies to OOD, and vice versa. This is the only
    generalisation evidence available without new inference, and it targets the risk
    the calibration idea itself names: *the bias may be domain-dependent*. A large
    oracle gain with zero transfer means the LUT is memorising the split, not
    correcting a bias.
    """
    ok = df.dropna(subset=["true", "pred"])
    rows = []
    for tpl, g in ok.groupby("template"):
        for fit_on, apply_to in (("ID", "OOD"), ("OOD", "ID")):
            src, dst = g[g["dist"] == fit_on], g[g["dist"] == apply_to]
            if len(src) == 0 or len(dst) == 0:
                continue
            after = dst["pred"].map(_fit_lut(src))
            rows.append(
                {
                    "template": tpl,
                    "fit_on": fit_on,
                    "apply_to": apply_to,
                    "n_fit": len(src),
                    "n_apply": len(dst),
                    "acc_before": float((dst["true"] == dst["pred"]).mean()),
                    "acc_after": float((dst["true"] == after).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["transfer_gain"] = out["acc_after"] - out["acc_before"]
    return out
