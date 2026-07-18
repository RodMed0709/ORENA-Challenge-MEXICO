"""Reproduction test for the canonical scoring module (src/frame/metrics.py).

Lives INSIDE the owning experiment's ``_tools/`` (repo rule: never a top-level
``tests/`` folder). Two clearly-separated parts:

  PART A — OFFLINE (always runs, pure-python, no pod): synthetic bucket_mean==0.55,
           temporal_grounding n=1 dropped, the leaf-vs-group drop bug + the five
           gates, and a self-consistency check against rung-05's committed
           bucket_mean 0.5503 (needs NO saved predictions).

  PART B — POD-GATED (skips cleanly if artifacts absent): reproduce rung-02's
           known bucket_mean≈0.550 / acc_OOD≈0.592 from the gitignored saved
           results.csv under experiments/02-lora-sft/runs/ (present only on the pod).

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
    below = {"by_format": pd.DataFrame([{"answer_format": "number", "distribution": "ID",
                                         "accuracy": 0.10, "n": 10, "ci_low": 0.0, "ci_high": 0.2,
                                         "floor_majority": 0.50, "floor_train_prior": float("nan")}])}
    assert _raises(m.assert_floors_vs_eval_set, below), "floors gate must raise below floor"
    print("  [A3] all 5 gates pass on clean df and RAISE on crafted-bad df  PASS")


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
    runs = _ROOT / "experiments" / "02-lora-sft" / "runs"
    if not runs.exists():
        return None
    hits = sorted(runs.glob("*/results.csv")) + sorted(runs.glob("**/results.csv"))
    return hits[0] if hits else None


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
    r = stratified_report(df)
    assert abs(r["bucket_mean"] - 0.550) < 0.005, f"rung-02 bucket_mean {r['bucket_mean']} != ~0.550"
    assert abs(r["acc_OOD"] - 0.592) < 0.005, f"rung-02 acc_OOD {r['acc_OOD']} != ~0.592"
    print(f"  [B]  rung-02 reproduced from {path.name}: bucket_mean={r['bucket_mean']:.4f}, "
          f"acc_OOD={r['acc_OOD']:.4f}  PASS")


def main() -> int:
    print("PART A - OFFLINE (synthetic + committed CSVs):")
    part_a_synthetic_bucket_mean()
    part_a_leaf_vs_group_dropbug()
    part_a_gates()
    part_a_rung05_self_consistency()
    print("PART B - POD-GATED (rung-02 saved predictions):")
    part_b_rung02_repro()
    print("\nOK — Part A passed offline; Part B passed or skipped cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
