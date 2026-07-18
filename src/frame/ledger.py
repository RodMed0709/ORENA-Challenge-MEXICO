"""Root results ledger — the committed, git-native, Claude-queryable results store.

Two layers, both AUTO-GENERATED from the canonical metrics so they cannot drift:

1. **Root ``RESULTS.md``** — the human-readable one-screen summary that POINTS to the
   ``results/`` CSV tiers.
2. **``results/`` tiers** (:func:`build_results_ledger`):
   - **Tier 1 ``results/summary.csv``** — one row per experiment/run (headline compare).
   - **Tier 2 ``results/detailed.csv``** — one row per run × capability_group ×
     distribution{ID,OOD} × answer_format (stratified per-question-type view).
   - **Tier 3 ``results/by_run/<experiment>__<run>.csv``** — the full per-run canonical
     breakdown.

The SINGLE source of truth is ``frame.metrics.stratified_report``. A run drops a
canonical ``stratified.json`` (the serialised ``stratified_report`` dict + a small
``_meta`` block) into its run dir via :func:`register_run`; the ledger aggregates those.
The ledger NEVER re-derives buckets / floors / ID-OOD — it only consumes the canonical
output (see ``context/RULES.md`` §EVAL and ``context/decisions/eval-canonical.md``).

**Run dirs are gitignored** (``experiments/*/runs/``), so ``stratified.json`` is the
pod-side source; the committed, git-native store is the ``results/`` folder + ``RESULTS.md``
that this module distils from it. A run WITHOUT a ``stratified.json`` (every experiment
predating this convention) still gets a Tier-1 row built from whatever its committed
``RESULTS.csv`` carries, flagged ``needs_backfill=true`` — meaning: to make it rich (populate
Tier 2/3) it needs a pod pass of ``stratified_report`` over its saved predictions. We never
fabricate stratified numbers we do not have.

Pure pandas + stdlib — imports and runs fully offline (no torch/GPU).
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── legacy narrow view (kept for back-compat with older callers of build_results_md) ──
_VIEW = ["experiment", "arm", "bucket_mean", "acc_ID", "acc_OOD", "verdict", "note"]

# ── Tier-1 schema (display / sort order) ──────────────────────────────────────
_TIER1_COLS = [
    "experiment", "run", "model",
    "bucket_mean", "acc_ID", "acc_OOD", "floor_ID", "floor_OOD", "margin_ID", "margin_OOD",
    "acc_fo_class", "acc_number", "acc_binary", "acc_open_ended", "acc_multiple_choice",
    "n_total", "date", "source_commit", "needs_backfill",
]
_FORMATS = ["fo_class", "number", "binary", "open_ended", "multiple_choice"]

# ── Tier-2 / Tier-3 schema ────────────────────────────────────────────────────
# ``floor``/``margin`` are the template-aware trivial floor and ``accuracy − floor``
# at the group×dist×format cell (from stratified_report's by_bucket_format); the
# bootstrap CIs stay at the coarser answer_format×distribution granularity.
_TIER2_COLS = [
    "experiment", "run", "capability_group", "distribution", "answer_format",
    "accuracy", "n", "floor", "margin", "ci_low", "ci_high",
]

# heterogeneous RESULTS.csv column aliases → the canonical field we coalesce to.
_ALIASES: dict[str, tuple[str, ...]] = {
    "run": ("run", "arm", "run_name", "rung", "manifest"),
    "model": ("model",),
    "bucket_mean": ("bucket_mean",),
    "acc_ID": ("acc_ID", "val_id_acc"),
    "acc_OOD": ("acc_OOD", "val_ood_acc"),
    "margin_ID": ("margin_ID",),
    "margin_OOD": ("margin_OOD",),
    "floor_ID": ("floor_ID",),
    "floor_OOD": ("floor_OOD",),
    "n_total": ("n_questions", "n_total", "n"),
    "date": ("date",),
    "acc_fo_class": ("acc_fo_class", "acc_fmt_fo_class"),
    "acc_number": ("acc_number", "acc_fmt_number"),
    "acc_binary": ("acc_binary", "acc_fmt_binary"),
    "acc_open_ended": ("acc_open_ended", "acc_fmt_open_ended"),
    "acc_multiple_choice": ("acc_multiple_choice", "acc_fmt_multiple_choice"),
}


# ── small helpers ─────────────────────────────────────────────────────────────

def _first(row: pd.Series, *names: str):
    """First present, non-null value among ``names`` (heterogeneous schemas)."""
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return None


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    try:
        if v is None or pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _num(v):
    """Coerce to float or NaN (never a string) for numeric columns."""
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _int_or_none(v):
    """Whole-number → int, else None (keeps n_total from rendering as ``6252.0``)."""
    f = _num(v)
    return None if math.isnan(f) else int(f)


def _clean_json(obj):
    """Recursively turn NaN/Inf → None so the JSON is valid + round-trips."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _df_records(obj) -> list[dict]:
    """A stratified sub-table (DataFrame or already-list-of-records) → records."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, list):
        return obj
    return []


def _git(repo_root: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 — offline / no git / not a repo
        return ""


def _source_commit(repo_root: Path, path: Path) -> str:
    """Short hash of the last commit touching ``path``; fall back to HEAD, else ''."""
    rel = path
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        pass
    last = _git(repo_root, "log", "-1", "--format=%h", "--", str(rel))
    return last or _git(repo_root, "rev-parse", "--short", "HEAD")


# ── the stratified.json convention (write side) ───────────────────────────────

def stratified_to_jsonable(
    strat: dict, *, experiment: str | None = None, run: str | None = None,
    model: str | None = None, date: str | None = None, extra: dict | None = None,
) -> dict:
    """Serialise a ``stratified_report`` dict (DataFrames → records) + a ``_meta`` block.

    The result is JSON-safe (NaN/Inf → null) and round-trips through the ledger.
    """
    by_bucket = _df_records(strat.get("by_bucket"))
    n_total = int(sum(int(r.get("n", 0) or 0) for r in by_bucket)) if by_bucket else None
    payload = {
        "_meta": {
            "experiment": experiment,
            "run": run,
            "model": model,
            "date": date,
            "n_total": n_total,
            "extra": extra or {},
        },
        "bucket_mean": strat.get("bucket_mean"),
        "acc_ID": strat.get("acc_ID"),
        "acc_OOD": strat.get("acc_OOD"),
        "acc_overall": strat.get("acc_overall"),
        "floor_ID": strat.get("floor_ID"),
        "floor_OOD": strat.get("floor_OOD"),
        "margin_ID": strat.get("margin_ID"),
        "margin_OOD": strat.get("margin_OOD"),
        "number_estimate": strat.get("number_estimate"),
        "by_bucket": by_bucket,
        "by_bucket_format": _df_records(strat.get("by_bucket_format")),
        "by_format": _df_records(strat.get("by_format")),
    }
    return _clean_json(payload)


def register_run(
    run_dir: str | Path, strat: dict, *,
    experiment: str | None = None, run: str | None = None,
    model: str | None = None, date: str | None = None, extra: dict | None = None,
) -> Path:
    """Drop the canonical ``stratified.json`` into ``run_dir`` (the tiny helper a
    notebook calls after scoring).

    ``experiment`` / ``run`` default from the path
    (``experiments/<experiment>/runs/<run>/`` → parents[1].name / name), so a
    conventionally-placed run needs only ``register_run(run_dir, strat)``. Returns the
    written path. The ledger then aggregates every such file via
    :func:`build_results_ledger`.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if run is None:
        run = run_dir.name
    if experiment is None:
        # experiments/<experiment>/runs/<run> → parents[1] is <experiment>
        parts = run_dir.parts
        if "runs" in parts:
            i = parts.index("runs")
            if i >= 1:
                experiment = parts[i - 1]
    payload = stratified_to_jsonable(
        strat, experiment=experiment, run=run, model=model, date=date, extra=extra
    )
    out = run_dir / "stratified.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("registered run → %s (experiment=%s run=%s)", out, experiment, run)
    return out


# ── ingest (read side) ────────────────────────────────────────────────────────

def _discover_stratified(root: Path) -> list[Path]:
    """Every committed-or-pod ``experiments/*/runs/**/stratified.json``."""
    return sorted(root.glob("experiments/*/runs/**/stratified.json"))


def _experiment_of(root: Path, strat_path: Path, meta: dict) -> str:
    if meta.get("experiment"):
        return str(meta["experiment"])
    try:
        rel = strat_path.relative_to(root)
        # experiments/<experiment>/runs/...
        if len(rel.parts) >= 2 and rel.parts[0] == "experiments":
            return rel.parts[1]
    except ValueError:
        pass
    return strat_path.parent.name


def _format_overall(by_format: list[dict], fmt: str):
    """n-weighted mean accuracy of ``fmt`` across the canonical ID/OOD cells.

    Arithmetic over canonical outputs (never a re-derivation of buckets/ID-OOD):
    combines the two canonical (fmt, distribution) accuracy cells by their canonical n.
    """
    num = den = 0.0
    for r in by_format:
        if str(r.get("answer_format")) == fmt:
            n = _num(r.get("n"))
            a = _num(r.get("accuracy"))
            if not math.isnan(n) and not math.isnan(a):
                num += a * n
                den += n
    return (num / den) if den > 0 else float("nan")


def _tier1_row_from_strat(root: Path, strat_path: Path) -> dict:
    """A rich Tier-1 row from a canonical ``stratified.json`` (needs_backfill=False)."""
    data = json.loads(strat_path.read_text(encoding="utf-8"))
    meta = data.get("_meta", {}) or {}
    by_format = _df_records(data.get("by_format"))
    experiment = _experiment_of(root, strat_path, meta)
    row = {
        "experiment": experiment,
        "run": meta.get("run") or strat_path.parent.name,
        "model": meta.get("model") or "",
        "bucket_mean": _num(data.get("bucket_mean")),
        "acc_ID": _num(data.get("acc_ID")),
        "acc_OOD": _num(data.get("acc_OOD")),
        "floor_ID": _num(data.get("floor_ID")),
        "floor_OOD": _num(data.get("floor_OOD")),
        "margin_ID": _num(data.get("margin_ID")),
        "margin_OOD": _num(data.get("margin_OOD")),
        "n_total": _int_or_none(meta.get("n_total")),
        "date": meta.get("date") or "",
        "source_commit": _source_commit(root, strat_path),
        "needs_backfill": False,
    }
    for f in _FORMATS:
        row[f"acc_{f}"] = _format_overall(by_format, f)
    return row


def _tier1_rows_from_csv(root: Path, csv: Path, seen: set[tuple[str, str]]) -> list[dict]:
    """Fallback Tier-1 rows from a committed RESULTS.csv (needs_backfill=True).

    Emits from whatever the heterogeneous CSV carries; never fabricates a missing
    canonical number. Skips any (experiment, run) already covered by a stratified.json.
    """
    experiment = csv.parent.name
    df = pd.read_csv(csv)
    commit = _source_commit(root, csv)
    rows: list[dict] = []
    for _, r in df.iterrows():
        run = _first(r, *_ALIASES["run"])
        run = "" if run is None else str(run)
        if (experiment, run) in seen:
            continue
        row = {
            "experiment": experiment,
            "run": run,
            "model": _first(r, *_ALIASES["model"]) or "",
            "bucket_mean": _num(_first(r, *_ALIASES["bucket_mean"])),
            "acc_ID": _num(_first(r, *_ALIASES["acc_ID"])),
            "acc_OOD": _num(_first(r, *_ALIASES["acc_OOD"])),
            # floors + margins need the gold answers a bare RESULTS.csv lacks → NaN
            # until the run is rescored with a stratified.json (needs_backfill).
            "floor_ID": _num(_first(r, *_ALIASES["floor_ID"])),
            "floor_OOD": _num(_first(r, *_ALIASES["floor_OOD"])),
            "margin_ID": _num(_first(r, *_ALIASES["margin_ID"])),
            "margin_OOD": _num(_first(r, *_ALIASES["margin_OOD"])),
            "n_total": _int_or_none(_first(r, *_ALIASES["n_total"])),
            "date": _first(r, *_ALIASES["date"]) or "",
            "source_commit": commit,
            "needs_backfill": True,
        }
        for f in _FORMATS:
            row[f"acc_{f}"] = _num(_first(r, *_ALIASES[f"acc_{f}"]))
        rows.append(row)
    return rows


def _tier2_rows_from_strat(root: Path, strat_path: Path) -> list[dict]:
    """Tier-2 (+ per-run Tier-3) rows for one rich run.

    ``by_bucket_format`` (group×dist×format accuracy, n, template-aware floor, margin)
    LEFT-joined to ``by_format`` for the bootstrap CIs (which stay at the coarser
    format×dist granularity). All numbers come straight from the canonical output —
    the ledger never re-derives a floor or a margin.
    """
    data = json.loads(strat_path.read_text(encoding="utf-8"))
    meta = data.get("_meta", {}) or {}
    experiment = _experiment_of(root, strat_path, meta)
    run = meta.get("run") or strat_path.parent.name
    bbf = _df_records(data.get("by_bucket_format"))
    bf = _df_records(data.get("by_format"))
    # index by_format on (answer_format, distribution) for the CI join.
    fkey = {
        (str(r.get("answer_format")), str(r.get("distribution"))): r for r in bf
    }
    rows: list[dict] = []
    for r in bbf:
        fmt, dist = str(r.get("answer_format")), str(r.get("distribution"))
        fr = fkey.get((fmt, dist), {})
        rows.append({
            "experiment": experiment,
            "run": run,
            "capability_group": r.get("capability_group"),
            "distribution": dist,
            "answer_format": fmt,
            "accuracy": _num(r.get("accuracy")),
            "n": int(_num(r.get("n"))) if not math.isnan(_num(r.get("n"))) else 0,
            "floor": _num(r.get("floor")),
            "margin": _num(r.get("margin")),
            "ci_low": _num(fr.get("ci_low")),
            "ci_high": _num(fr.get("ci_high")),
        })
    return rows


# ── gold answers for the template-aware floors (offline, reproducible) ────────

def gold_from_frame_parquets(data_root: str | Path, split: str = "test") -> pd.DataFrame:
    """Build the ``gold`` table ``stratified_report`` needs for template-aware floors.

    Reads every ``<data_root>/<ds>/data/frame/<split>.parquet`` (``test`` = val) and
    returns ``[qID, answer, question]`` with ``qID = "<ds>__<id>"`` — exactly the
    namespacing the scorer keys on. The QA parquets are ~410 KB, gitignored, pulled
    from the pod / RunPod S3 (data card rung 08 §11). Pure pandas — no torch, no SDK.
    ``stratified_report`` normalises ``question`` → template via ``metrics.template_of``.
    """
    import glob  # noqa: PLC0415 — local: keep the module's top imports minimal

    root = Path(data_root)
    files = sorted(glob.glob(str(root / "*" / "data" / "frame" / f"{split}.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no {split}.parquet under {root} — pull the QA parquets from the pod/S3 "
            "(gitignored, ~410 KB; data card rung 08 §11)."
        )
    frames = []
    for f in files:
        ds = Path(f).parents[2].name
        d = pd.read_parquet(f, columns=["id", "question", "answer"])
        d["qID"] = ds + "__" + d["id"].astype(str)
        frames.append(d[["qID", "answer", "question"]])
    return pd.concat(frames, ignore_index=True)


# ── public entry points ───────────────────────────────────────────────────────

def build_results_ledger(
    repo_root: str | Path = ".",
    results_dir: str | Path = "results",
    md_out: str | Path = "RESULTS.md",
) -> dict[str, Path]:
    """Regenerate all three ``results/`` tiers + root ``RESULTS.md`` in one call.

    Reads every ``experiments/*/runs/**/stratified.json`` (rich, canonical) and every
    ``experiments/*/RESULTS.csv`` (fallback). A run with a ``stratified.json`` is rich
    (``needs_backfill=False``, Tier 2/3 populated); any other run in a RESULTS.csv gets a
    fallback Tier-1 row (``needs_backfill=True``, logged). Tier 1 is sorted by
    ``bucket_mean`` DESC (blank sinks). Returns the written paths keyed
    ``summary``/``detailed``/``by_run``/``markdown``.
    """
    root = Path(repo_root).resolve()
    out_dir = (root / results_dir) if not Path(results_dir).is_absolute() else Path(results_dir)
    by_run_dir = out_dir / "by_run"
    by_run_dir.mkdir(parents=True, exist_ok=True)

    strat_paths = _discover_stratified(root)

    # ── Tier 1 rich rows + Tier 2/3 from stratified.json ─────────────────────
    tier1: list[dict] = []
    tier2: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for sp in strat_paths:
        try:
            r1 = _tier1_row_from_strat(root, sp)
            tier1.append(r1)
            seen.add((r1["experiment"], r1["run"]))
            run_rows = _tier2_rows_from_strat(root, sp)
            tier2.extend(run_rows)
            # Tier 3 — one CSV per rich run.
            t3 = pd.DataFrame(run_rows, columns=_TIER2_COLS)
            t3_path = by_run_dir / f"{r1['experiment']}__{r1['run']}.csv"
            t3.to_csv(t3_path, index=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping stratified.json %s: %s", sp, exc)

    # ── Tier 1 fallback rows from RESULTS.csv (needs_backfill) ────────────────
    backfill: list[tuple[str, str]] = []
    for csv in sorted(root.glob("experiments/*/RESULTS.csv")):
        try:
            for row in _tier1_rows_from_csv(root, csv, seen):
                tier1.append(row)
                backfill.append((row["experiment"], row["run"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping %s: %s", csv, exc)
    if backfill:
        logger.info(
            "needs_backfill=true for %d run(s) (no stratified.json; pod pass of "
            "stratified_report needed to populate Tier 2/3): %s",
            len(backfill), backfill,
        )

    # ── assemble + sort Tier 1 ───────────────────────────────────────────────
    t1 = pd.DataFrame(tier1, columns=_TIER1_COLS)
    if not t1.empty:
        t1["n_total"] = t1["n_total"].astype("Int64")  # int display, empty for missing
        t1["needs_backfill"] = t1["needs_backfill"].astype(bool)
        t1 = t1.sort_values(
            "bucket_mean", ascending=False, na_position="last"
        ).reset_index(drop=True)
    summary_path = out_dir / "summary.csv"
    t1.to_csv(summary_path, index=False)

    t2 = pd.DataFrame(tier2, columns=_TIER2_COLS)
    if not t2.empty:
        t2 = t2.sort_values(
            ["experiment", "run", "capability_group", "distribution", "answer_format"]
        ).reset_index(drop=True)
    detailed_path = out_dir / "detailed.csv"
    t2.to_csv(detailed_path, index=False)

    md_path = _write_results_md(root, t1, out_dir, md_out, n_rich=len(strat_paths))

    logger.info(
        "ledger built: %d Tier-1 row(s) (%d rich, %d needs_backfill), %d Tier-2 row(s), "
        "%d Tier-3 file(s)",
        len(t1), len(strat_paths), len(backfill), len(t2), len(strat_paths),
    )
    return {
        "summary": summary_path,
        "detailed": detailed_path,
        "by_run": by_run_dir,
        "markdown": md_path,
    }


def _write_results_md(
    root: Path, t1: pd.DataFrame, out_dir: Path, md_out: str | Path, *, n_rich: int
) -> Path:
    """Human-readable root ``RESULTS.md`` that POINTS to the ``results/`` tiers."""
    try:
        rel_dir = out_dir.relative_to(root).as_posix()
    except ValueError:
        rel_dir = str(out_dir)
    view = [
        "experiment", "run", "model", "bucket_mean",
        "acc_ID", "acc_OOD", "margin_ID", "margin_OOD",
        "n_total", "date", "needs_backfill",
    ]
    lines = [
        "# Results ledger",
        "",
        "Auto-generated by `frame.ledger.build_results_ledger` — do NOT hand-edit; regenerate.",
        "",
        "The committed, git-native results store. `bucket_mean` (unweighted mean over the 4 "
        "real capability_group × {ID,OOD} buckets, canonical headline from "
        "`frame.metrics.stratified_report`) is the number we track; rows sort by it.",
        "",
        "**Read `margin`, not raw accuracy.** `margin_ID`/`margin_OOD` = `acc − floor`, where "
        "the floor is the template-aware trivial constant (answer each template's modal answer; "
        "data card rung 08 §4b). It is the REAL skill the model adds. `acc_OOD > acc_ID` does NOT "
        "mean the model generalises better — the OOD floor is ~12 pts higher, so by margin the "
        "model usually adds *less* on OOD. A negative margin = below the trivial floor (a weak "
        "baseline legitimately is). Blank margins = `needs_backfill` (no gold answers joined yet).",
        "",
        "**Tiers (all in `" + rel_dir + "/`, auto-built from each run's canonical "
        "`stratified.json`):**",
        "",
        f"- `{rel_dir}/summary.csv` — Tier 1: one row per experiment/run (this table is its digest).",
        f"- `{rel_dir}/detailed.csv` — Tier 2: one row per run × capability_group × "
        "{ID,OOD} × answer_format (accuracy, n, floor, margin, CI).",
        f"- `{rel_dir}/by_run/<experiment>__<run>.csv` — Tier 3: the full per-run canonical breakdown.",
        "",
        "`needs_backfill=true` = no committed `stratified.json` for that run: the Tier-1 row is "
        "coalesced from its `RESULTS.csv`, and Tier 2/3 are empty for it until a pod pass of "
        "`stratified_report` over its saved predictions is registered via "
        "`frame.ledger.register_run`.",
        "",
        f"_{n_rich} rich run(s) with canonical stratified data; the rest are `needs_backfill`._",
        "",
        "| " + " | ".join(view) + " |",
        "| " + " | ".join("---" for _ in view) + " |",
    ]
    for _, r in t1.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in view) + " |")
    lines.append("")

    out_path = (root / md_out) if not Path(md_out).is_absolute() else Path(md_out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s (%d rows)", out_path, len(t1))
    return out_path


# ── legacy narrow builder (kept for back-compat; superseded by build_results_ledger) ──

def _coalesce_rows(csv: Path) -> list[dict]:
    experiment = csv.parent.name
    df = pd.read_csv(csv)
    has_bucket_mean = "bucket_mean" in df.columns
    rows: list[dict] = []
    for _, r in df.iterrows():
        bucket_mean = r["bucket_mean"] if has_bucket_mean else np.nan
        note_bits = []
        if not has_bucket_mean:
            note_bits.append("no bucket_mean col in RESULTS.csv")
        rows.append(
            {
                "experiment": experiment,
                "arm": _first(r, "arm", "run", "run_name") or "",
                "bucket_mean": float(bucket_mean) if pd.notna(bucket_mean) else np.nan,
                "acc_ID": _first(r, "acc_ID"),
                "acc_OOD": _first(r, "acc_OOD"),
                "verdict": _first(r, "verdict") or "",
                "note": "; ".join(note_bits),
            }
        )
    return rows


def build_results_md(repo_root: str | Path = ".", out: str | Path = "RESULTS.md") -> Path:
    """LEGACY narrow ``RESULTS.md`` builder (superseded by :func:`build_results_ledger`).

    Kept so older notebooks/callers keep working. Coalesces the heterogeneous
    per-experiment CSVs into ``[experiment, arm, bucket_mean, acc_ID, acc_OOD, verdict,
    note]``, sorts DESC by ``bucket_mean``, and writes a Markdown table. Prefer
    ``build_results_ledger`` for new work (it also emits the ``results/`` tiers).
    """
    root = Path(repo_root)
    csvs = sorted(root.glob("experiments/*/RESULTS.csv"))
    rows: list[dict] = []
    for csv in csvs:
        try:
            rows.extend(_coalesce_rows(csv))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping %s: %s", csv, exc)

    df = pd.DataFrame(rows, columns=_VIEW)
    if not df.empty:
        df = df.sort_values("bucket_mean", ascending=False, na_position="last").reset_index(drop=True)

    lines = [
        "# Results ledger",
        "",
        "Auto-generated by `frame.ledger.build_results_md` from every "
        "`experiments/*/RESULTS.csv`. Do NOT hand-edit — regenerate.",
        "",
        "`bucket_mean` (unweighted mean over the 4 real capability_group × {ID,OOD} "
        "buckets, canonical headline from `frame.metrics.stratified_report`) is the "
        "number we track; rows are sorted by it. A blank `bucket_mean` means that "
        "experiment's `RESULTS.csv` predates the canonical column (see `note`).",
        "",
        "| " + " | ".join(_VIEW) + " |",
        "| " + " | ".join("---" for _ in _VIEW) + " |",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in _VIEW) + " |")
    lines.append("")

    out_path = root / out if not Path(out).is_absolute() else Path(out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s (%d rows from %d experiments)", out_path, len(df), len(csvs))
    return out_path
