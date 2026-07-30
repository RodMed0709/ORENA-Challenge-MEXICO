"""Reproduction test for the canonical scoring module (src/frame/metrics.py).

Lives INSIDE the owning experiment's ``_tools/`` (repo rule: never a top-level
``tests/`` folder). Two clearly-separated parts:

  PART A — OFFLINE (always runs, pure-python, no pod): synthetic bucket_mean==0.55,
           temporal_grounding n=1 dropped, the leaf-vs-group drop bug + the five
           gates, the HYBRID cross-check helper (assert_matches_sdk_pre_eval), and a
           self-consistency check against rung-05's committed bucket_mean 0.5503
           (needs NO saved predictions).

  PART B — POD/S3-GATED (skips cleanly if artifacts absent): reproduce rung-02's
           REAL, S3-verified bucket_mean==0.5486 / acc_OOD==0.5918 / acc_ID==0.5209
           from the gitignored saved results.csv under experiments/02-lora-sft/runs/
           (present on the pod, or materialised locally by the S3 backfill). This is
           the non-circular check: it reproduces real numbers, not arithmetic
           self-consistency.

Run:  python experiments/02-lora-sft/_tools/test_metrics_canonical.py
No pytest dependency; plain asserts + a __main__ runner printing PASS/SKIP.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── make `frame` (src/) and `focus` (vendored) importable without an install ──
_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT / "src", _ROOT / "vendor" / "orena-focus" / "src"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

from frame import metrics as m  # noqa: E402
from frame.metrics import stratified_report  # noqa: E402

_05_CSV = _ROOT / "experiments" / "05-bottleneck-audit" / "RESULTS.csv"


# ── synthetic fixtures ───────────────────────────────────────────────────────

def _rows(prefix, video, leaf, fmt, n, n_correct):
    """n rows for one bucket; the first n_correct are correct."""
    return [
        {
            "qID": f"{prefix}__{leaf}_{i}",
            "video": video,
            "primary": leaf,
            "answer_format": fmt,
            "correctness": 1 if i < n_correct else 0,
            "ood": False,  # deliberately all-False — must be IGNORED by the module
        }
        for i in range(n)
    ]


def _synthetic_df() -> pd.DataFrame:
    """4 buckets at accuracies [0.60, 0.61, 0.42, 0.57] (mean 0.55) + a
    temporal_grounding bucket of n=1 that bucket_mean must drop."""
    rows: list[dict] = []
    # object_recognition (leaf object_identification): ID 60/100, OOD 61/100
    rows += _rows("lapchole", "l_v1", "object_identification", "number", 100, 60)
    rows += _rows("heico", "h_v1", "object_identification", "number", 100, 61)
    # aggregation (leaf object_aggregation): ID 42/100, OOD 57/100
    rows += _rows("lapchole", "l_v2", "object_aggregation", "number", 100, 42)
    rows += _rows("heico", "h_v2", "object_aggregation", "number", 100, 57)
    # temporal_grounding (leaf temporal_localization): ID n=1 → dropped
    rows += _rows("lapchole", "l_v3", "temporal_localization", "number", 1, 1)
    return pd.DataFrame(rows)


# ── PART A — offline ─────────────────────────────────────────────────────────

def part_a_synthetic_bucket_mean() -> None:
    df = _synthetic_df()
    r = stratified_report(df, n_boot=100, seed=42)

    assert abs(r["bucket_mean"] - 0.55) < 1e-9, f"bucket_mean {r['bucket_mean']!r} != 0.55"

    bb = r["by_bucket"]
    # leaf->group actually happened: no leaf value survives as a bucket name.
    groups = set(bb["capability_group"])
    assert groups == {"object_recognition", "aggregation", "temporal_grounding"}, groups
    # temporal_grounding present with n=1 but excluded from bucket_mean (n<2).
    tg = bb[bb["capability_group"] == "temporal_grounding"]
    assert len(tg) == 1 and int(tg["n"].iloc[0]) == 1, "temporal_grounding n=1 expected"
    kept = bb[bb["n"] >= 2]
    assert len(kept) == 4, f"expected 4 kept buckets, got {len(kept)}"
    print(f"  [A1] synthetic bucket_mean == 0.55 (4 buckets; temporal_grounding n=1 dropped)  PASS")


def part_a_leaf_vs_group_dropbug() -> None:
    df = _synthetic_df()
    # THE BUG: filter a GROUP name against the leaf-valued 'primary' column.
    naive = df[df["primary"] == "object_recognition"]
    assert len(naive) == 0, "group-name filter should match zero leaf-valued rows (the bug)"
    # THE FIX: leaf->group grouping keeps every object_recognition leaf row.
    r = stratified_report(df, n_boot=50)
    bb = r["by_bucket"]
    orec_n = int(bb.loc[bb["capability_group"] == "object_recognition", "n"].sum())
    leaf_n = int((df["primary"] == "object_identification").sum())
    assert orec_n == leaf_n == 200, f"leaf->group must keep all rows: {orec_n} vs {leaf_n}"
    print("  [A2] group-name-vs-leaf filter drops 200 rows; leaf->group keeps all  PASS")


def part_a_gates() -> None:
    clean = _synthetic_df()
    # all gates pass on a clean df
    m.assert_all_rows_grouped(clean)
    m.assert_no_dup_qid(clean)
    m.assert_ood_from_qid(clean)
    exp = {
        (g, d): int(n)
        for (g, d), n in clean.assign(
            capability_group=clean["primary"].map(m._leaf_to_group),
            distribution=clean["qID"].map(m._dist_from_qid),
        ).groupby(["capability_group", "distribution"]).size().items()
    }
    m.assert_bucket_counts(clean, exp)
    m.assert_floors_vs_eval_set(stratified_report(clean, n_boot=50))  # NaN floors → no raise

    # each gate RAISES on its crafted-bad input
    def _raises(fn, *a):
        try:
            fn(*a)
        except (AssertionError, ValueError):
            return True
        return False

    bad_leaf = pd.DataFrame([{"qID": "lapchole__1", "video": "v", "primary": "not_a_capability",
                              "answer_format": "number", "correctness": 1}])
    assert _raises(m.assert_all_rows_grouped, bad_leaf), "all_rows_grouped must raise"
    assert _raises(m.assert_bucket_counts, clean, {("object_recognition", "ID"): 999}), \
        "bucket_counts must raise on wrong count"
    dup = pd.DataFrame([{"qID": "lapchole__1"}, {"qID": "lapchole__1"}])
    assert _raises(m.assert_no_dup_qid, dup), "no_dup_qid must raise"
    foreign = pd.DataFrame([{"qID": "weird__1", "video": "v", "primary": "object_identification",
                             "answer_format": "number", "correctness": 1}])
    assert _raises(m.assert_ood_from_qid, foreign), "ood_from_qid must raise"
    # floors gate: RAISES on an INCONSISTENT floor/margin (margin != accuracy − floor),
    # NOT on a below-floor run (being below the template floor is a documented finding
    # a weak baseline legitimately hits — data card §4b — surfaced as a negative margin).
    inconsistent = {"by_format": pd.DataFrame([{"answer_format": "number", "distribution": "ID",
                                                "accuracy": 0.50, "n": 10, "ci_low": 0.4, "ci_high": 0.6,
                                                "floor": 0.30, "margin": 0.99}])}  # 0.99 != 0.50−0.30
    assert _raises(m.assert_floors_vs_eval_set, inconsistent), "floors gate must raise on margin drift"
    below_but_consistent = {"by_format": pd.DataFrame([{"answer_format": "number", "distribution": "ID",
                                                        "accuracy": 0.10, "n": 10, "ci_low": 0.0, "ci_high": 0.2,
                                                        "floor": 0.50, "margin": -0.40}])}  # below floor, consistent
    m.assert_floors_vs_eval_set(below_but_consistent)  # must NOT raise
    print("  [A3] all 5 gates pass on clean df and RAISE on crafted-bad df  PASS")


def part_a_hybrid_crosscheck() -> None:
    """The HYBRID guard (assert_matches_sdk_pre_eval): our bucket_mean must match
    the vendor pre_eval buckets restricted to n>=2, and RAISE on divergence.

    The vendor buckets_df is simulated here ([group, ood, accuracy, count]) exactly
    as focus.Evaluator.pre_evaluation_score emits it — the same four real buckets
    the synthetic df produces plus the n=1 temporal_grounding bucket the raw vendor
    pre_eval keeps but bucket_mean drops."""
    r = stratified_report(_synthetic_df(), n_boot=50)
    sdk_buckets = pd.DataFrame(
        [
            {"group": "object_recognition", "ood": False, "accuracy": 0.60, "count": 100},
            {"group": "object_recognition", "ood": True, "accuracy": 0.61, "count": 100},
            {"group": "aggregation", "ood": False, "accuracy": 0.42, "count": 100},
            {"group": "aggregation", "ood": True, "accuracy": 0.57, "count": 100},
            {"group": "temporal_grounding", "ood": False, "accuracy": 1.00, "count": 1},
        ],
        columns=["group", "ood", "accuracy", "count"],
    )
    kept = m.assert_matches_sdk_pre_eval(r["bucket_mean"], sdk_buckets)
    assert abs(kept - 0.55) < 1e-9, f"vendor kept-mean {kept} != our bucket_mean 0.55"

    def _raises(fn, *a):
        try:
            fn(*a)
        except (AssertionError, ValueError):
            return True
        return False

    # a bucket_mean far from the vendor number must RAISE (guards the reimpl).
    assert _raises(m.assert_matches_sdk_pre_eval, 0.90, sdk_buckets), \
        "assert_matches_sdk_pre_eval must raise on a large divergence"
    print("  [A5] HYBRID cross-check: kept-mean == bucket_mean; RAISES on divergence  PASS")


def _synthetic_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Gold answers + a shared template for the synthetic df — enough to exercise the
    template-aware floor path deterministically (no pod, no parquet)."""
    g = df[["qID"]].copy()
    g["question"] = "How many objects at timepoint 00:00:01 ?"  # <TS> collapses the time
    g["answer"] = ["1" if i % 3 else "2" for i in range(len(g))]  # modal "1" → floor 2/3
    return g


def part_a_template_floor_margin() -> None:
    """WITHOUT gold every floor/margin is NaN (missing input, no crash); WITH gold the
    floors populate and margin == accuracy − floor everywhere, split ID/OOD."""
    df = _synthetic_df()

    r0 = stratified_report(df, n_boot=50)
    assert r0["by_format"]["floor"].isna().all(), "no-gold floors must be NaN"
    assert pd.isna(r0["margin_ID"]) and pd.isna(r0["floor_OOD"]), "no-gold ID/OOD floors NaN"
    m.assert_floors_vs_eval_set(r0)  # no-op, must not raise

    r = stratified_report(df, gold=_synthetic_gold(df), n_boot=50)
    for tbl in ("by_format", "by_bucket", "by_bucket_format"):
        t = r[tbl]
        assert t["floor"].notna().all(), f"{tbl} floors must populate with gold"
        assert (abs(t["margin"] - (t["accuracy"] - t["floor"])) < 1e-12).all(), \
            f"{tbl} margin != accuracy − floor"
    for k in ("floor_ID", "floor_OOD", "margin_ID", "margin_OOD"):
        assert pd.notna(r[k]), f"{k} must be populated with gold"
    assert abs(r["margin_ID"] - (r["acc_ID"] - r["floor_ID"])) < 1e-12
    assert abs(r["margin_OOD"] - (r["acc_OOD"] - r["floor_OOD"])) < 1e-12
    m.assert_floors_vs_eval_set(r)  # consistent → passes
    print("  [A6] template-aware floor/margin: NaN without gold, consistent with gold  PASS")


def part_a_rung05_self_consistency() -> None:
    if not _05_CSV.exists():
        print(f"  [A4] SKIP — {_05_CSV} not found")
        return
    df = pd.read_csv(_05_CSV)
    row = df[df["arm"] == "a0_real"].iloc[0]
    buckets = [
        row["acc_bucket_object_recognition_ID"],
        row["acc_bucket_object_recognition_OOD"],
        row["acc_bucket_aggregation_ID"],
        row["acc_bucket_aggregation_OOD"],
    ]
    recomputed = sum(buckets) / 4.0
    assert abs(recomputed - row["bucket_mean"]) < 1e-9, \
        f"05 a0_real: mean(4 buckets)={recomputed} != recorded bucket_mean={row['bucket_mean']}"
    assert abs(row["bucket_mean"] - 0.5503) < 1e-3, f"expected ~0.5503, got {row['bucket_mean']}"
    print(f"  [A4] rung-05 a0_real bucket_mean self-consistent: mean(4 buckets)="
          f"{recomputed:.4f} == recorded {row['bucket_mean']:.4f}  PASS")


# ── PART B — pod-gated ───────────────────────────────────────────────────────

def _find_rung02_results() -> Path | None:
    """The canonical rung-02 eval: ``eval_best/results.csv``, and ONLY that one.

    🔴 Fixed 2026-07-29. The previous body fell back to ``sorted(runs.glob("**/results.csv"))[0]``,
    and that recursive glob does not return the canonical file — it returns whichever
    sibling eval directory sorts first, which on a full clone is ``ep1_full`` (``"ep1" <
    "eval"``). So the assertions below were being checked against **epoch 1**
    (bucket_mean 0.5282) instead of the OOD-selected checkpoint-1720 (0.5486), and Part B
    failed on every clone that actually had the artifacts. It "passed" only where the
    artifacts were missing and the whole part SKIPped.

    This is the SAME root cause as the open tier-1 ledger defect (`context/NOW.md` §2:
    ``_discover_stratified`` recurses and resolves ``eval_best/stratified.json`` to the
    same (experiment, run) as the run-root file). Second instance of "a recursive glob
    picks up a sibling eval directory". Unlike the ledger case, there is no decision to
    make here: this test's docstring names the numbers it reproduces, and ``eval_best``
    is the only file that produces them.
    """
    runs = _ROOT / "experiments" / "02-lora-sft" / "runs"
    if not runs.exists():
        return None
    hits = sorted(runs.glob("*/eval_best/results.csv"))
    return hits[0] if hits else None


def _load_val_gold() -> "pd.DataFrame | None":
    """The val gold table (qID, answer, question) from the gitignored parquets, or
    None if they were not pulled (data card rung 08 §11)."""
    data_root = _ROOT / "external_data" / "orena-data"
    if not any(data_root.glob("*/data/frame/test.parquet")):
        return None
    from frame.ledger import gold_from_frame_parquets  # noqa: PLC0415
    return gold_from_frame_parquets(data_root)


def part_b_rung02_repro() -> None:
    path = _find_rung02_results()
    if path is None:
        print("  [B]  SKIP (PENDING-ON-POD) - no experiments/02-lora-sft/runs/**/results.csv "
              "(gitignored; present only on the pod that holds rung-02 predictions)")
        return
    df = pd.read_csv(path)
    m.assert_no_dup_qid(df)
    m.assert_all_rows_grouped(df)
    m.assert_ood_from_qid(df)
    gold = _load_val_gold()
    r = stratified_report(df, gold=gold)
    # REAL, S3-verified numbers (not arithmetic self-consistency): the full-val
    # eval of the OOD-selected checkpoint-1720. Tight tolerance — these are exact.
    assert abs(r["bucket_mean"] - 0.5486) < 5e-4, f"rung-02 bucket_mean {r['bucket_mean']} != 0.5486"
    assert abs(r["acc_OOD"] - 0.5918) < 5e-4, f"rung-02 acc_OOD {r['acc_OOD']} != 0.5918"
    assert abs(r["acc_ID"] - 0.5209) < 5e-4, f"rung-02 acc_ID {r['acc_ID']} != 0.5209"
    if gold is None:
        print(f"  [B]  rung-02 reproduced from {path.name}: bucket_mean={r['bucket_mean']:.4f}, "
              f"acc_OOD={r['acc_OOD']:.4f}, acc_ID={r['acc_ID']:.4f} (floors SKIPPED — "
              "val parquets not pulled)  PASS")
        return
    m.assert_floors_vs_eval_set(r)
    # The self-check that ties results/ to the data card §4b: the template-aware floor
    # is HIGHER on OOD, so acc_OOD > acc_ID reverses under margin.
    assert abs(r["margin_ID"] - 0.1838) < 5e-4, f"rung-02 margin_ID {r['margin_ID']} != +0.1838"
    assert abs(r["margin_OOD"] - 0.1320) < 5e-4, f"rung-02 margin_OOD {r['margin_OOD']} != +0.1320"
    assert r["floor_OOD"] > r["floor_ID"], "OOD floor must be the higher one (data card §4b)"
    assert r["margin_ID"] > r["margin_OOD"], "by margin the model adds LESS on OOD (§4b)"
    print(f"  [B]  rung-02 reproduced from {path.name}: bucket_mean={r['bucket_mean']:.4f}, "
          f"acc_OOD={r['acc_OOD']:.4f}, acc_ID={r['acc_ID']:.4f}, "
          f"margin_ID={r['margin_ID']:+.4f}, margin_OOD={r['margin_OOD']:+.4f}  PASS")


def part_a_equivalence_tost():
    """The four states of ``equivalence_verdict``, plus the two ways it must refuse.

    The point of the instrument is that INCONCLUSIVE and EQUIVALENT are different
    answers; a test that only checked "returns a string" would let the defect it exists
    to prevent walk straight back in.
    """
    ev = m.equivalence_verdict

    # EQUIVALENT — tight CI inside the margin, containing 0. This is a TIE.
    r = ev({"delta": 0.001, "ci_low": -0.008, "ci_high": 0.010}, epsilon=0.02)
    assert r["verdict"] == "EQUIVALENT", r
    assert r["within_margin"] and not r["excludes_zero"]

    # INCONCLUSIVE — same delta, wide CI. Contains 0 but is wider than the margin, so it
    # is NOT a tie. Under the old read this and the case above were both "null".
    r = ev({"delta": 0.001, "ci_low": -0.055, "ci_high": 0.058}, epsilon=0.02)
    assert r["verdict"] == "INCONCLUSIVE", r
    assert abs(r["epsilon_min"] - 0.058) < 1e-12, r
    assert "NOT a tie" in r["reason"]

    # TRIVIAL_DIFFERENCE — excludes 0 (a real effect) yet sits wholly inside the margin.
    r = ev({"delta": 0.011, "ci_low": 0.004, "ci_high": 0.018}, epsilon=0.02)
    assert r["verdict"] == "TRIVIAL_DIFFERENCE", r
    assert r["within_margin"] and r["excludes_zero"]

    # DIFFERENT — rung 21 arm A shaped: big, clears the margin, excludes 0.
    r = ev({"delta": 0.0584, "ci_low": 0.031, "ci_high": 0.086}, epsilon=0.02)
    assert r["verdict"] == "DIFFERENT", r

    # A boundary that must NOT be called equivalent: the CI touches ±epsilon exactly.
    # Strict inequality, so "as wide as the margin" is not "inside the margin".
    r = ev({"delta": 0.0, "ci_low": -0.02, "ci_high": 0.02}, epsilon=0.02)
    assert r["verdict"] == "INCONCLUSIVE", r

    # UNDEFINED — paired_delta_ci returns NaNs on an empty slice by design; that must
    # surface as its own state, never silently as a tie.
    r = ev(m.paired_delta_ci(pd.DataFrame()), epsilon=0.02)
    assert r["verdict"] == "UNDEFINED", r

    # REFUSALS. No default epsilon exists, and a non-positive one is a bug not a choice.
    for bad in (0, -0.01):
        try:
            ev({"ci_low": -0.01, "ci_high": 0.01}, epsilon=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"epsilon={bad} must raise")
    try:
        ev({"ci_low": 0.05, "ci_high": -0.05}, epsilon=0.02)  # reversed bounds
    except ValueError:
        pass
    else:
        raise AssertionError("reversed CI bounds must raise, not invert the verdict")

    # Tuple form, and the end-to-end wrapper on a real frame: two videos, arm B wins
    # every question, so the delta is +1.0 and nothing about it is a tie.
    assert ev((-0.005, 0.005), epsilon=0.02)["verdict"] == "EQUIVALENT"
    df = pd.DataFrame({
        "qID": [f"lapchole__{i}" for i in range(8)],
        "video": ["v1"] * 4 + ["v2"] * 4,
        "correct_a": [0.0] * 8,
        "correct_b": [1.0] * 8,
    })
    r = m.paired_equivalence(df, epsilon=0.02, n_boot=200)
    assert r["verdict"] == "DIFFERENT" and abs(r["delta"] - 1.0) < 1e-12, r
    assert r["n"] == 8 and r["n_videos"] == 2, r

    print("  [A]  equivalence_verdict: 4 states + boundary + UNDEFINED + refusals  PASS")


def main() -> int:
    print("PART A - OFFLINE (synthetic + committed CSVs):")
    part_a_synthetic_bucket_mean()
    part_a_leaf_vs_group_dropbug()
    part_a_gates()
    part_a_hybrid_crosscheck()
    part_a_template_floor_margin()
    part_a_equivalence_tost()
    part_a_rung05_self_consistency()
    print("PART B - POD-GATED (rung-02 saved predictions):")
    part_b_rung02_repro()
    print("\nOK — Part A passed offline; Part B passed or skipped cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
