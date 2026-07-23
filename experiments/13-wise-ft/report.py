"""Rung 13 reporting library — imported from a notebook cell, never run by hand.

Turns each α's canonical ``stratified_report`` into (a) the per-α trade-off table that
IS the deliverable, (b) the pre-registered verdict, and (c) the two CSVs.

Three things this module refuses to do, each one a rule rather than a preference:

1. **It never computes a metric.** Every accuracy, floor and margin comes from
   ``frame.metrics.stratified_report``; every paired CI comes from
   ``frame.metrics.paired_delta_ci``. `context/RULES.md` §EVAL rule 1 — re-deriving a
   bucket or a floor beside the module is exactly how rung 07 lost 964 rows.

2. **It never lets an `arm` column reach the shared ledger.** ``frame.ledger`` treats an
   ``arm`` column as a run name (``src/frame/ledger.py:67``), so a per-arm ``RESULTS.csv``
   would file every α as a run of its own and poison the root ledger. The split is
   mechanical here: :func:`write_results` emits a ledger-shaped ``RESULTS.csv`` (one row
   per RUN) and a per-arm ``RESULTS_arms.csv`` (one row per α), the same split rung 06
   made.

3. **It never reads raw accuracy as the result.** The headline is ``bucket_mean`` and the
   per-format read is MARGIN over the template-aware floor (`context/RULES.md` §10–12).

🔴 **Why the paired CI on the accuracy delta IS the CI on the margin delta.**
margin = accuracy − floor, and the floor is a property of the GOLD ANSWERS of the slice,
not of the model. Both arms answer the identical 6252 questions, so the floor is
identical in every cell and cancels exactly in the difference:
``Δmargin = (acc_arm − floor) − (acc_ctrl − floor) = Δacc``. That is why
:func:`number_margin_delta` bootstraps correctness directly instead of bootstrapping a
floor as well — and :func:`build_pairs` asserts the two arms really did answer the same
qIDs, so the cancellation is checked rather than assumed.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── the two CSV schemas ──────────────────────────────────────────────────────
# Ledger-shaped: byte-compatible with experiments/06-vit-lora/RESULTS.csv, which is what
# frame.ledger._ALIASES coalesces (src/frame/ledger.py:66-83).
LEDGER_COLS = [
    "run", "model", "bucket_mean", "acc_ID", "acc_OOD",
    "floor_ID", "floor_OOD", "margin_ID", "margin_OOD",
    "acc_fo_class", "acc_number", "acc_binary", "acc_open_ended", "acc_multiple_choice",
    "n_questions", "date", "notes",
]

# Per-arm: one row per α. The trade-off curve is the deliverable even if nothing wins
# (WAVE_SPEC §Rung 13: "Report every α on every format regardless").
ARM_COLS = [
    "arm", "alpha", "bucket_mean", "delta_bucket_mean",
    "acc_ID", "acc_OOD", "margin_ID", "margin_OOD",
    "bucket_aggregation_ID", "bucket_aggregation_OOD",
    "bucket_object_recognition_ID", "bucket_object_recognition_OOD",
    "number_margin_ID", "number_margin_OOD",
    "fo_class_margin_ID", "fo_class_margin_OOD",
    "binary_margin_ID", "binary_margin_OOD",
    "open_ended_margin_ID", "open_ended_margin_OOD",
    "multiple_choice_margin_ID", "multiple_choice_margin_OOD",
    "number_delta_ID", "number_ci_low_ID", "number_ci_high_ID", "number_nvid_ID",
    "number_delta_OOD", "number_ci_low_OOD", "number_ci_high_OOD", "number_nvid_OOD",
    "n_empty_answers", "n_inference_errors", "max_answer_chars", "mean_answer_chars",
    "verdict", "verdict_why",
]

# The pre-registered rule (context/13-wise-ft/CONTEXT.md §Decision rule, WAVE_SPEC §13).
# Fixed BEFORE any number exists. A threshold edited after seeing a number stops being a
# gate and becomes a story.
BUCKET_MEAN_TOLERANCE = 0.005


# ── slicing the canonical report (read-only accessors, never a re-derivation) ──

def _fmt_cell(strat: dict, fmt: str, dist: str, col: str = "margin") -> float:
    """One cell of ``stratified_report['by_format']``. Missing → NaN, never invented."""
    bf = pd.DataFrame(strat["by_format"])
    if bf.empty:
        return float("nan")
    hit = bf[(bf["answer_format"].astype(str) == fmt) & (bf["distribution"].astype(str) == dist)]
    return float(hit[col].iloc[0]) if len(hit) else float("nan")


def _bucket_cell(strat: dict, group: str, dist: str, col: str = "accuracy") -> float:
    bb = pd.DataFrame(strat["by_bucket"])
    if bb.empty:
        return float("nan")
    hit = bb[(bb["capability_group"].astype(str) == group) & (bb["distribution"].astype(str) == dist)]
    return float(hit[col].iloc[0]) if len(hit) else float("nan")


# ── pairing the two arms ─────────────────────────────────────────────────────

def build_pairs(control_results: pd.DataFrame, arm_results: pd.DataFrame) -> pd.DataFrame:
    """Join rung 06's per-question results to one α's, on qID.

    RAISES unless the two arms answered EXACTLY the same question set. The pairing is
    what buys the power on 8–10 OOD videos (rung 06 README: all four cells cleared the
    power rule only because the delta was paired), and it is also what makes the floor
    cancel in Δmargin — both claims are false if the qID sets differ.
    """
    for name, df in (("control", control_results), ("arm", arm_results)):
        need = {"qID", "video", "answer_format", "correctness"}
        if not need.issubset(df.columns):
            raise KeyError(f"{name} results need {sorted(need)}; got {sorted(df.columns)}")
        if not df["qID"].is_unique:
            raise AssertionError(f"{name} results carry duplicate qIDs — the pairing would double-count")
    only_c = set(control_results["qID"]) - set(arm_results["qID"])
    only_a = set(arm_results["qID"]) - set(control_results["qID"])
    if only_c or only_a:
        raise AssertionError(
            f"arms did not answer the same questions: {len(only_c)} only in control "
            f"(e.g. {sorted(only_c)[:3]}), {len(only_a)} only in arm (e.g. {sorted(only_a)[:3]}). "
            "An unpaired comparison throws away the only power we have and breaks the "
            "floor cancellation in Δmargin."
        )
    pairs = control_results[["qID", "video", "answer_format", "correctness"]].rename(
        columns={"correctness": "correct_a"}
    ).merge(
        arm_results[["qID", "correctness"]].rename(columns={"correctness": "correct_b"}), on="qID"
    )
    pairs["distribution"] = pairs["qID"].astype(str).str.split("__", n=1).str[0].map(
        lambda p: "OOD" if p == "heico" else "ID"
    )
    return pairs


def number_margin_delta(pairs: pd.DataFrame, dist: str, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired, video-clustered CI of the `number` margin delta (arm − control) in ``dist``.

    Delegates the bootstrap to ``frame.metrics.paired_delta_ci`` — the video→question
    two-level scheme the whole project uses. Bootstrapping correctness IS bootstrapping
    the margin (see the module docstring: the floor cancels).
    """
    from frame.metrics import paired_delta_ci  # noqa: PLC0415

    sub = pairs[(pairs["answer_format"].astype(str) == "number") & (pairs["distribution"] == dist)]
    return paired_delta_ci(sub, n_boot=n_boot, seed=seed)


def assert_floor_cancels(control_strat: dict, arm_strat: dict, *, tol: float = 1e-12) -> None:
    """The template-aware floor must be IDENTICAL in both arms, cell by cell.

    It is a function of the gold answers alone, so any difference means the two arms were
    scored on different rows — which would silently turn Δmargin into Δaccuracy plus a
    floor artefact. Cheap to check, fatal to miss.
    """
    bad = []
    for fmt in ("fo_class", "number", "binary", "open_ended", "multiple_choice"):
        for dist in ("ID", "OOD"):
            c, a = _fmt_cell(control_strat, fmt, dist, "floor"), _fmt_cell(arm_strat, fmt, dist, "floor")
            if math.isnan(c) and math.isnan(a):
                continue
            if math.isnan(c) or math.isnan(a) or abs(c - a) > tol:
                bad.append(f"{fmt}/{dist}: control={c} arm={a}")
    if bad:
        raise AssertionError(
            "template-aware floors differ between arms — they cannot, the gold is the same "
            f"eval set: {bad}. Δmargin is only Δaccuracy while this holds."
        )


# ── the degenerate-output guard ──────────────────────────────────────────────

def answer_health(pred_map: dict[str, str]) -> dict:
    """Cheap health read on one α's raw answers.

    A mid-α checkpoint is a model nobody trained and nobody has ever run. It can be
    subtly broken — empty strings, runaway repetition, or ``frame.run``'s
    ``"Inference Error: ..."`` fallback (``src/frame/run.py:55``) — and a broken model
    still yields a scoreable number, because a wrong answer and a garbage answer score
    identically. Reported for EVERY α, so "it degraded gracefully" is a measurement.
    """
    vals = list(pred_map.values())
    lens = [len(v) for v in vals]
    return {
        "n": len(vals),
        "n_empty_answers": sum(1 for v in vals if not v.strip()),
        "n_inference_errors": sum(1 for v in vals if v.startswith("Inference Error:")),
        "max_answer_chars": max(lens) if lens else 0,
        "mean_answer_chars": (sum(lens) / len(lens)) if lens else float("nan"),
    }


# ── the pre-registered decision rule ─────────────────────────────────────────

def decide(row: dict, *, tolerance: float = BUCKET_MEAN_TOLERANCE) -> tuple[str, str]:
    """Apply the pre-registered rule to one α's row. Returns ``(verdict, why)``.

    WIN requires ALL THREE (WAVE_SPEC §Rung 13, frozen in CONTEXT.md before any number):
      (a) ``bucket_mean`` is not lower than rung 06's by more than 0.005;
      (b) the ``number`` margin is STRICTLY higher than rung 06's in BOTH ID and OOD;
      (c) the paired video-clustered CI on the ``number`` margin delta excludes 0 in at
          least one distribution.

    Reports, never decides beyond the rule: a NaN anywhere is INDETERMINATE, not a pass.
    """
    reasons: list[str] = []

    d_bucket = row.get("delta_bucket_mean")
    if d_bucket is None or (isinstance(d_bucket, float) and math.isnan(d_bucket)):
        return "INDETERMINATE", "delta_bucket_mean missing"
    a_ok = d_bucket >= -tolerance
    if not a_ok:
        reasons.append(f"(a) bucket_mean {d_bucket:+.4f} < -{tolerance}")

    d_id, d_ood = row.get("number_delta_ID"), row.get("number_delta_OOD")
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (d_id, d_ood)):
        return "INDETERMINATE", "number margin delta missing in ID or OOD"
    b_ok = d_id > 0 and d_ood > 0
    if not b_ok:
        reasons.append(f"(b) number margin delta ID {d_id:+.4f} / OOD {d_ood:+.4f} — not strictly up in both")

    def _excludes_zero(lo, hi) -> bool:
        if lo is None or hi is None:
            return False
        if isinstance(lo, float) and math.isnan(lo):
            return False
        if isinstance(hi, float) and math.isnan(hi):
            return False
        return lo > 0 or hi < 0

    ci_id = _excludes_zero(row.get("number_ci_low_ID"), row.get("number_ci_high_ID"))
    ci_ood = _excludes_zero(row.get("number_ci_low_OOD"), row.get("number_ci_high_OOD"))
    c_ok = ci_id or ci_ood
    if not c_ok:
        reasons.append("(c) no distribution's paired CI excludes 0")

    if a_ok and b_ok and c_ok:
        return "WIN", f"(a) Δbucket_mean {d_bucket:+.4f}; (b) number margin up ID {d_id:+.4f} / OOD {d_ood:+.4f}; (c) CI excludes 0 in " + ("ID" if ci_id else "OOD")
    return "NO-WIN", "; ".join(reasons)


# ── assembling one α's row ───────────────────────────────────────────────────

def arm_row(
    alpha: float,
    strat: dict,
    control_strat: dict,
    pairs: pd.DataFrame | None = None,
    health: dict | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """One per-arm row: every bucket, every format's margin, the paired `number` CIs,
    the health read and the pre-registered verdict."""
    row: dict = {
        "arm": f"alpha_{alpha:.2f}",
        "alpha": float(alpha),
        "bucket_mean": strat["bucket_mean"],
        "delta_bucket_mean": strat["bucket_mean"] - control_strat["bucket_mean"],
        "acc_ID": strat["acc_ID"],
        "acc_OOD": strat["acc_OOD"],
        "margin_ID": strat["margin_ID"],
        "margin_OOD": strat["margin_OOD"],
    }
    for group in ("aggregation", "object_recognition"):
        for dist in ("ID", "OOD"):
            row[f"bucket_{group}_{dist}"] = _bucket_cell(strat, group, dist, "accuracy")
    for fmt in ("number", "fo_class", "binary", "open_ended", "multiple_choice"):
        for dist in ("ID", "OOD"):
            row[f"{fmt}_margin_{dist}"] = _fmt_cell(strat, fmt, dist, "margin")

    for dist in ("ID", "OOD"):
        d = (
            number_margin_delta(pairs, dist, n_boot=n_boot, seed=seed)
            if pairs is not None
            else {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_videos": 0}
        )
        row[f"number_delta_{dist}"] = d["delta"]
        row[f"number_ci_low_{dist}"] = d["ci_low"]
        row[f"number_ci_high_{dist}"] = d["ci_high"]
        row[f"number_nvid_{dist}"] = d["n_videos"]

    row.update(
        {k: (health or {}).get(k, float("nan"))
         for k in ("n_empty_answers", "n_inference_errors", "max_answer_chars", "mean_answer_chars")}
    )
    row["verdict"], row["verdict_why"] = decide(row)
    return row


def ledger_row(run: str, model: str, strat: dict, date: str, notes: str = "") -> dict:
    """One ledger-shaped row. ``acc_<format>`` is the n-weighted mean over the canonical
    ID/OOD cells — computed by ``frame.ledger._format_overall`` itself, so this file
    never re-derives a number the ledger already knows how to derive."""
    from frame.ledger import _format_overall  # noqa: PLC0415

    by_format = pd.DataFrame(strat["by_format"]).to_dict(orient="records")
    by_bucket = pd.DataFrame(strat["by_bucket"])
    row = {
        "run": run,
        "model": model,
        "bucket_mean": strat["bucket_mean"],
        "acc_ID": strat["acc_ID"],
        "acc_OOD": strat["acc_OOD"],
        "floor_ID": strat["floor_ID"],
        "floor_OOD": strat["floor_OOD"],
        "margin_ID": strat["margin_ID"],
        "margin_OOD": strat["margin_OOD"],
        "n_questions": int(by_bucket["n"].sum()) if len(by_bucket) else 0,
        "date": date,
        "notes": notes,
    }
    for fmt in ("fo_class", "number", "binary", "open_ended", "multiple_choice"):
        row[f"acc_{fmt}"] = _format_overall(by_format, fmt)
    return row


def render(arm_rows: list[dict]) -> pd.DataFrame:
    """The per-α trade-off table, α ascending. This IS the deliverable — the curve is
    reported whether or not any α wins."""
    df = pd.DataFrame(arm_rows, columns=ARM_COLS)
    return df.sort_values("alpha", ignore_index=True) if len(df) else df


def write_results(exp_dir: Path | str, ledger_rows: list[dict], arm_rows: list[dict]) -> dict[str, Path]:
    """Write the split CSVs. See rule 2 in the module docstring for why they are split."""
    exp = Path(exp_dir)
    led = pd.DataFrame(ledger_rows, columns=LEDGER_COLS)
    arms = render(arm_rows)
    if "arm" in led.columns:
        raise AssertionError("RESULTS.csv must not carry an `arm` column — frame.ledger would read it as a run name")
    p_led, p_arm = exp / "RESULTS.csv", exp / "RESULTS_arms.csv"
    led.to_csv(p_led, index=False)
    arms.to_csv(p_arm, index=False)
    logger.info("wrote %s (%d rows) and %s (%d rows)", p_led, len(led), p_arm, len(arms))
    return {"ledger": p_led, "arms": p_arm}


__all__ = [
    "LEDGER_COLS", "ARM_COLS", "BUCKET_MEAN_TOLERANCE", "build_pairs", "number_margin_delta",
    "assert_floor_cancels", "answer_health", "decide", "arm_row", "ledger_row", "render",
    "write_results",
]
