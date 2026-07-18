"""Offline test for the tiered results ledger (``src/frame/ledger.py``).

Lives INSIDE the owning experiment's ``_tools/`` (repo rule: never a top-level
``tests/`` folder), next to ``test_metrics_canonical.py`` — same eval-canonical family.

Builds a throwaway repo tree of synthetic ``stratified.json`` inputs (produced by the
REAL ``frame.metrics.stratified_report`` + ``frame.ledger.register_run``) plus one
fallback ``RESULTS.csv``, runs :func:`build_results_ledger`, and asserts the three tiers
have the right shape, columns, sorting, and ``needs_backfill`` semantics. Pure pandas —
no pod, no torch. Prints PASS/── and exits non-zero on failure.

Run:  python experiments/02-lora-sft/_tools/test_results_ledger.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# ── make `frame` (src/) and `focus` (vendored) importable without an install ──
_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT / "src", _ROOT / "vendor" / "orena-focus" / "src"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json  # noqa: E402

import pandas as pd  # noqa: E402

from frame import ledger  # noqa: E402
from frame.ledger import build_results_ledger, register_run  # noqa: E402
from frame.metrics import stratified_report  # noqa: E402


# ── synthetic fixtures ────────────────────────────────────────────────────────

def _rows(prefix, video, leaf, fmt, n, n_correct):
    return [
        {
            "qID": f"{prefix}__{leaf}_{fmt}_{i}",
            "video": video,
            "primary": leaf,
            "answer_format": fmt,
            "correctness": 1 if i < n_correct else 0,
            "ood": False,  # all-False on purpose — module must ignore it
        }
        for i in range(n)
    ]


def _synthetic_df(scale: float) -> pd.DataFrame:
    """2 groups × {ID,OOD} × 2 answer_formats — a non-trivial by_bucket_format cross.
    ``scale`` shifts every bucket's accuracy so two runs get different bucket_means."""
    def c(base):
        return max(0, min(100, int(round(base * scale))))
    rows: list[dict] = []
    for pref, vid in (("lapchole", "l_v1"), ("heico", "h_v1")):
        rows += _rows(pref, vid, "object_identification", "number", 100, c(60))
        rows += _rows(pref, vid, "object_identification", "binary", 100, c(70))
    for pref, vid in (("lapchole", "l_v2"), ("heico", "h_v2")):
        rows += _rows(pref, vid, "object_aggregation", "number", 100, c(42))
        rows += _rows(pref, vid, "object_aggregation", "binary", 100, c(57))
    return pd.DataFrame(rows)


def _make_repo(tmp: Path) -> tuple[str, str]:
    """Two rich runs (stratified.json) + one fallback RESULTS.csv. Returns the two
    rich bucket_means as (high, low) strings for reference (unused, kept explicit)."""
    # rich run A (higher bucket_mean) and rich run B (lower) in two experiments.
    strat_a = stratified_report(_synthetic_df(1.0), n_boot=50, seed=1)
    strat_b = stratified_report(_synthetic_df(0.5), n_boot=50, seed=1)

    run_a = tmp / "experiments" / "90-alpha" / "runs" / "runA"
    run_b = tmp / "experiments" / "91-beta" / "runs" / "runB"
    register_run(run_a, strat_a, model="Qwen3-VL-8B", date="2026-07-18")
    register_run(run_b, strat_b, model="Qwen3-VL-8B", date="2026-07-18")

    # a fallback experiment: RESULTS.csv only, NO stratified.json → needs_backfill.
    exp_c = tmp / "experiments" / "92-legacy"
    exp_c.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"arm": "legacyrun", "acc_ID": 0.40, "acc_OOD": 0.45, "acc_fmt_number": 0.33}]
    ).to_csv(exp_c / "RESULTS.csv", index=False)
    return f"{strat_a['bucket_mean']:.4f}", f"{strat_b['bucket_mean']:.4f}"


# ── assertions ────────────────────────────────────────────────────────────────

def test_ledger_shapes() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _make_repo(tmp)
        paths = build_results_ledger(tmp)

        # ── Tier 1 — summary.csv ────────────────────────────────────────────
        t1 = pd.read_csv(paths["summary"])
        assert list(t1.columns) == ledger._TIER1_COLS, list(t1.columns)
        assert len(t1) == 3, f"expected 3 Tier-1 rows, got {len(t1)}"
        # sorted by bucket_mean DESC, NaN last
        bm = t1["bucket_mean"].tolist()
        rich = [x for x in bm if pd.notna(x)]
        assert rich == sorted(rich, reverse=True), f"bucket_mean not sorted desc: {bm}"
        assert pd.isna(bm[-1]), "fallback (NaN bucket_mean) row must sink to the bottom"
        # needs_backfill semantics: 2 rich False, 1 fallback True
        nb = dict(zip(t1["run"], t1["needs_backfill"]))
        assert nb["runA"] == False and nb["runB"] == False, nb  # noqa: E712
        assert nb["legacyrun"] == True, nb  # noqa: E712
        # rich rows carry model + per-format overall accuracy
        rowA = t1[t1["run"] == "runA"].iloc[0]
        assert rowA["model"] == "Qwen3-VL-8B"
        assert pd.notna(rowA["acc_number"]) and pd.notna(rowA["acc_binary"])
        # fallback row surfaced acc_ID/acc_OOD but has no bucket_mean
        rowC = t1[t1["run"] == "legacyrun"].iloc[0]
        assert pd.isna(rowC["bucket_mean"]) and abs(rowC["acc_OOD"] - 0.45) < 1e-9
        print("  [1] summary.csv: 3 rows, sorted desc, needs_backfill 2xFalse+1xTrue  PASS")

        # ── Tier 2 — detailed.csv ───────────────────────────────────────────
        t2 = pd.read_csv(paths["detailed"])
        assert list(t2.columns) == ledger._TIER2_COLS, list(t2.columns)
        # one row per rich (run × capability_group × distribution × answer_format)
        expected = 0
        for rd in (tmp / "experiments").glob("*/runs/*/stratified.json"):
            expected += len(json.loads(rd.read_text())["by_bucket_format"])
        assert len(t2) == expected and expected > 0, f"{len(t2)} != {expected}"
        # only rich runs appear (fallback never in Tier 2)
        assert set(t2["run"]) == {"runA", "runB"}, set(t2["run"])
        # the join carried the by_format CI columns onto every cross row
        assert {"floor", "ci_low", "ci_high"} <= set(t2.columns)
        assert t2["ci_low"].notna().any(), "CI join produced no ci_low values"
        print(f"  [2] detailed.csv: {len(t2)} rows (group x dist x format), rich-only, CI joined  PASS")

        # ── Tier 3 — by_run/*.csv ───────────────────────────────────────────
        by_run = Path(paths["by_run"])
        files = sorted(p.name for p in by_run.glob("*.csv"))
        assert files == ["90-alpha__runA.csv", "91-beta__runB.csv"], files
        one = pd.read_csv(by_run / "90-alpha__runA.csv")
        assert list(one.columns) == ledger._TIER2_COLS
        assert set(one["run"]) == {"runA"}
        print(f"  [3] by_run/: one CSV per rich run {files}  PASS")

        # ── RESULTS.md — points to the tiers ────────────────────────────────
        md = (tmp / "RESULTS.md").read_text(encoding="utf-8")
        for token in ("results/summary.csv", "results/detailed.csv", "results/by_run", "needs_backfill"):
            assert token in md, f"RESULTS.md missing pointer: {token}"
        print("  [4] RESULTS.md written and points to results/ tiers  PASS")


def test_roundtrip_no_fabrication() -> None:
    """An empty repo yields empty tiers (headers only) — never fabricated rows."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "experiments").mkdir()
        paths = build_results_ledger(tmp)
        t1 = pd.read_csv(paths["summary"])
        t2 = pd.read_csv(paths["detailed"])
        assert len(t1) == 0 and list(t1.columns) == ledger._TIER1_COLS
        assert len(t2) == 0 and list(t2.columns) == ledger._TIER2_COLS
        print("  [5] empty repo -> empty tiers (headers only), no fabrication  PASS")


def main() -> int:
    print("results-ledger tiers (offline, synthetic stratified.json):")
    test_ledger_shapes()
    test_roundtrip_no_fabrication()
    print("\nOK — all ledger tier assertions passed offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
