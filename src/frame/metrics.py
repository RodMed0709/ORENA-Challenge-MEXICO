"""Canonical FRAME evaluation metrics — the SINGLE source of truth for scoring.

Every experiment imports ``stratified_report`` (and the gates below) instead of
re-deriving metrics in prose. This kills a recurring defect class: the SDK
``results_df["primary"]`` column holds a taxonomy *leaf* (e.g.
``object_identification``, n=2457), but the scored buckets are *groups*
(``object_recognition``, n=3421). Filtering a group name against the leaf column
silently drops questions (964 dropped in rung 07, commit 8839ee5, caught by gate
G3, reverted 881d057). The correct leaf→group mapping already exists —
``split._group`` (split.py:76) → ``focus.taxonomy.Capability.group`` — but was
only ever used for split coverage, never for scoring. Here it becomes the ONLY
way scoring maps leaves to groups.

Pure pandas/numpy + ``focus.taxonomy`` only. No torch, no GPU, no judge — this
module is importable and verifiable fully offline.

──────────────────────────────────────────────────────────────────────────────
SDK API — confirmed (Task-0 spike, read from vendor/orena-focus/ source)
──────────────────────────────────────────────────────────────────────────────
1. "Hierarchical estimate for number":
   ``focus.data.formats.Number`` (formats.py:113-124) exposes NO dedicated
   hierarchical / interval / tolerance estimator — only ``verify``/``read``→int
   with the default exact ``compare``. Therefore the "hierarchical estimate for
   number" is interpretation (a): the ``answer_format == "number"`` row of
   ``Evaluator._hierarchical_summary`` — the two-level video→question bootstrap
   (evaluator.py:490-513, surfaced by the ``groupby("answer_format")`` loop at
   evaluator.py:546-557). We REPLICATE that bootstrap here (keyed on a
   dataset-namespaced video key) rather than import ``Evaluator``, because
   ``focus.evaluation`` eagerly imports ``transformers``/``torch`` via
   ``judges.py`` (judges.py:36) — which would break this module's "no torch,
   offline" contract. The replicated algorithm is deterministic for a fixed seed.

2. leaf→group mapping:
   ``Capability.from_any(leaf).group.value`` is the canonical leaf→group string
   (taxonomy.py:120 ``from_any``; :85 ``group`` property; :179 ``_PARENT_MAP``).
   ``from_any`` returns ``None`` (does NOT raise) on un-mappable input
   (taxonomy.py:130/167), so a gate can catch un-mappable leaves. This is the
   SAME ``.group`` that ``split._group`` uses on ``item.reference.primary``
   (split.py:76-80).

results_df schema (Evaluator._make_row, evaluator.py:373-392):
   [qID, video, ood, clinical, primary, answer_format, latency, timed_out, correctness]
   - primary = ref.primary.value  → a LEAF value string (the bug source)
   - ood     = ref.ood            → all-False on public data (data.py:87) — NEVER used here
   - qID     = f"{dataset}__{row_id}" → prefix is the REAL ID/OOD signal
                                        (heico=OOD, lapchole=ID; data.py:110)
   - video   = req.videoID (raw id; COLLIDES across datasets → namespace by dataset)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Capability import (lightweight, offline-safe) ────────────────────────────
# On the pod (full SDK env) ``from focus.taxonomy import Capability`` works. On a
# minimal/offline machine, ``focus/__init__.py`` eagerly imports transformers/
# datasets, which breaks even the stdlib-only ``taxonomy`` module. Since scoring
# must NOT require torch, fall back to loading the self-contained ``taxonomy.py``
# directly (it imports only stdlib). The fallback never runs where focus is
# fully installed.

def _load_capability():
    try:
        from focus.taxonomy import Capability  # noqa: PLC0415

        return Capability
    except Exception as exc:  # noqa: BLE001
        import importlib.util
        import sys

        for p in sys.path:
            cand = Path(p) / "focus" / "taxonomy.py"
            if cand.exists():
                spec = importlib.util.spec_from_file_location(
                    "focus._taxonomy_standalone", cand
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                logger.debug(
                    "focus package init unavailable (%s); loaded stdlib-only "
                    "taxonomy directly from %s",
                    exc,
                    cand,
                )
                return mod.Capability
        raise ImportError(
            "focus.taxonomy is required for scoring but could not be imported "
            "and no standalone taxonomy.py was found on sys.path."
        ) from exc


Capability = _load_capability()

_VALID_PREFIXES = ("heico", "lapchole")  # heico=OOD, lapchole=ID (data.py:110)


# ── private accessors (single place so a schema change is one edit) ──────────

def _leaf_to_group(leaf) -> str:
    """Canonical leaf→group *value* string via ``Capability.group``.

    RAISES ``ValueError`` on an un-mappable leaf so a group-vs-leaf filter can
    never silently drop a row again. This is the SAME mapping ``split._group``
    makes — never prose, never a hand-kept dict.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # from_any warns on junk; we raise instead
        cap = Capability.from_any(leaf)
    if cap is None:
        raise ValueError(
            f"un-mappable primary capability {leaf!r} — not a real Capability leaf/group; "
            "a group-name-vs-leaf filter would silently drop this row (ex-gate-G3 defect)."
        )
    return cap.group.value


def _dist_from_qid(qid) -> str:
    """ID/OOD from the qID prefix (heico=OOD, lapchole=ID). NEVER results_df['ood']."""
    prefix = str(qid).split("__", 1)[0]
    return "OOD" if prefix == "heico" else "ID"


def _prefix_from_qid(qid) -> str:
    return str(qid).split("__", 1)[0]


def _video_key(row) -> tuple:
    """Namespace ``video`` by dataset — req.videoID collides across datasets
    (data.py note), which would corrupt the video-level bootstrap by merging two
    surgeries' frames into one key."""
    return (_prefix_from_qid(row["qID"]), row["video"])


def _correct_series(df: pd.DataFrame) -> pd.Series:
    """Tolerant correctness column → float (mirrors delta.correctness_frame)."""
    ccol = next(
        (c for c in ("correctness", "correct", "is_correct") if c in df.columns), None
    )
    if ccol is None:
        raise KeyError(f"need a correctness column; got {list(df.columns)}")
    return df[ccol].astype(float)


def _distribution(df: pd.DataFrame, video_split: dict | None) -> pd.Series:
    """ID/OOD per row. Default = qID-prefix rule; ``video_split`` manifest is the
    OPTIONAL override (some experiments use a custom OOD procedure), never the
    reverse."""
    if video_split is None:
        return df["qID"].map(_dist_from_qid)

    def _from_manifest(row) -> str:
        key = (_prefix_from_qid(row["qID"]), str(row["video"]))
        # manifest keys on (dataset, video_id); val_ood = OOD, else ID.
        return "OOD" if video_split.get(key) == "val_ood" else "ID"

    return df.apply(_from_manifest, axis=1)


# ── two-level hierarchical bootstrap (replicates evaluator.py:490-513) ────────

def _hier_bootstrap(
    sub: pd.DataFrame, *, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Video→question two-level bootstrap over a slice.

    Point estimate = mean of per-video means; CI = resample videos with
    replacement, then questions within each sampled video with replacement.
    Keyed on the dataset-namespaced ``_vkey`` column. Deterministic for a fixed
    rng (algorithm mirrors Evaluator._hierarchical_summary._bootstrap).
    """
    if sub.empty:
        return float("nan"), float("nan"), float("nan")
    vid_groups = {
        v: sub.loc[sub["_vkey"] == v, "_correct"].to_numpy(dtype=float)
        for v in sub["_vkey"].dropna().unique()
    }
    video_keys = list(vid_groups.keys())
    n_videos = len(video_keys)
    if n_videos == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean([vid_groups[v].mean() for v in video_keys]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        # sample video POSITIONS (keys are tuples → cannot go through rng.choice).
        sampled_pos = rng.integers(0, n_videos, size=n_videos)
        boots[b] = float(
            np.mean(
                [
                    rng.choice(
                        vid_groups[video_keys[i]],
                        size=len(vid_groups[video_keys[i]]),
                        replace=True,
                    ).mean()
                    for i in sampled_pos
                ]
            )
        )
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ── trivial floors (majority-class + train-prior), computed vs the EVAL set ───

def _majority_floor(eval_answers: pd.Series | None) -> float:
    """Accuracy of always predicting the modal answer of THIS eval slice."""
    if eval_answers is None or len(eval_answers) == 0:
        return float("nan")
    counts = eval_answers.value_counts()
    return float(counts.iloc[0] / len(eval_answers))


def _train_prior_floor(
    eval_answers: pd.Series | None, train_answers: pd.Series | None
) -> float:
    """Accuracy of predicting the modal TRAIN answer, scored on the eval slice."""
    if eval_answers is None or train_answers is None:
        return float("nan")
    if len(eval_answers) == 0 or len(train_answers) == 0:
        return float("nan")
    modal_train = train_answers.value_counts().index[0]
    return float((eval_answers == modal_train).mean())


# ── public API ───────────────────────────────────────────────────────────────

def stratified_report(
    results_df: pd.DataFrame,
    video_split: dict | None = None,
    *,
    min_bucket_n: int = 2,
    n_boot: int = 1000,
    seed: int = 42,
    train_results_df: pd.DataFrame | None = None,
    eval_answers: pd.Series | None = None,
    train_answers: pd.Series | None = None,
) -> dict:
    """Canonical FRAME score for a per-question ``results_df``.

    leaf→group is ALWAYS ``Capability.group``; ID/OOD is ALWAYS the qID prefix
    (``video_split`` manifest is an optional override, never the default).

    Returns a dict with at least:
      - ``"bucket_mean"``  : unweighted mean over the populated group×{ID,OOD}
        buckets AFTER dropping any bucket with ``n < min_bucket_n`` (the headline).
      - ``"by_bucket"``    : DataFrame [capability_group, distribution, accuracy, n].
      - ``"by_bucket_format"`` : DataFrame [capability_group, distribution,
        answer_format, accuracy, n] — the group×{ID,OOD}×answer_format cross,
        using the SAME canonical leaf→group + qID ID/OOD. This is the
        results-ledger Tier-2 source; CIs/floors stay at the coarser
        ``by_format`` (answer_format×distribution) granularity.
      - ``"by_format"``    : DataFrame [answer_format, distribution, accuracy, n,
        ci_low, ci_high, floor_majority, floor_train_prior].
      - ``"acc_ID"``, ``"acc_OOD"`` : flat per-question mean of correctness over the
        ID / OOD halves (acc_OOD is the number the leaderboard weights ~half).
      - ``"acc_overall"``  : flat per-question mean over all rows.
      - ``"number_estimate"`` : the SDK per-format hierarchical estimate for
        ``number`` — {accuracy, ci_low, ci_high, n} from the replicated
        video→question bootstrap over all ``answer_format == "number"`` rows.

    ``eval_answers`` / ``train_answers`` (optional, indexed like ``results_df``)
    supply the ground-truth answers needed for the trivial floors; when absent the
    floor columns are NaN (results_df carries no answer column) and a note is logged.
    """
    cols = ["capability_group", "distribution", "accuracy", "n"]
    if results_df.empty:
        return {
            "bucket_mean": float("nan"),
            "by_bucket": pd.DataFrame(columns=cols),
            "by_bucket_format": pd.DataFrame(
                columns=["capability_group", "distribution", "answer_format", "accuracy", "n"]
            ),
            "by_format": pd.DataFrame(
                columns=[
                    "answer_format", "distribution", "accuracy", "n",
                    "ci_low", "ci_high", "floor_majority", "floor_train_prior",
                ]
            ),
            "acc_ID": float("nan"),
            "acc_OOD": float("nan"),
            "acc_overall": float("nan"),
            "number_estimate": {"accuracy": float("nan"), "ci_low": float("nan"),
                                "ci_high": float("nan"), "n": 0},
        }

    df = results_df.copy()
    df["_correct"] = _correct_series(df)
    # leaf→group — RAISES on an un-mappable leaf (no silent drop, ever).
    df["capability_group"] = df["primary"].map(_leaf_to_group)
    df["distribution"] = _distribution(df, video_split)
    df["_vkey"] = df.apply(_video_key, axis=1)

    # ── by_bucket + bucket_mean (the headline) ───────────────────────────────
    by_bucket = (
        df.groupby(["capability_group", "distribution"])["_correct"]
        .agg(accuracy="mean", n="size")
        .reset_index()
        .sort_values(["capability_group", "distribution"])
        .reset_index(drop=True)
    )
    kept = by_bucket[by_bucket["n"] >= min_bucket_n]
    dropped = by_bucket[by_bucket["n"] < min_bucket_n]
    if not dropped.empty:
        logger.info(
            "bucket_mean drops %d bucket(s) with n<%d: %s",
            len(dropped),
            min_bucket_n,
            list(zip(dropped["capability_group"], dropped["distribution"], dropped["n"])),
        )
    bucket_mean = float(kept["accuracy"].mean()) if len(kept) else float("nan")

    # ── by_bucket_format: group×{ID,OOD}×answer_format cross (Tier-2 source) ──
    # Same canonical leaf→group + qID ID/OOD already on ``df`` — a pure re-group,
    # never a re-derivation of buckets/ID-OOD. CIs/floors stay in ``by_format``.
    by_bucket_format = (
        df.groupby(["capability_group", "distribution", "answer_format"])["_correct"]
        .agg(accuracy="mean", n="size")
        .reset_index()
        .sort_values(["capability_group", "distribution", "answer_format"])
        .reset_index(drop=True)
    )

    # ── flat ID / OOD / overall means ────────────────────────────────────────
    dist_means = df.groupby("distribution")["_correct"].mean()
    acc_id = float(dist_means.get("ID", float("nan")))
    acc_ood = float(dist_means.get("OOD", float("nan")))
    acc_overall = float(df["_correct"].mean())

    # ── by_format (+ bootstrap CIs + trivial floors) ─────────────────────────
    rng = np.random.default_rng(seed)
    if eval_answers is None or train_answers is None:
        logger.info(
            "trivial floors: eval/train answers not supplied — floor columns are "
            "NaN (results_df carries no answer column)."
        )
    fmt_rows: list[dict] = []
    for (fmt, dist), sub in df.groupby(["answer_format", "distribution"], sort=True):
        mean, low, high = _hier_bootstrap(sub, n_boot=n_boot, rng=rng)
        ea = eval_answers.loc[sub.index] if eval_answers is not None else None
        fmt_rows.append(
            {
                "answer_format": str(fmt),
                "distribution": str(dist),
                "accuracy": float(sub["_correct"].mean()),
                "n": int(len(sub)),
                "ci_low": low,
                "ci_high": high,
                "floor_majority": _majority_floor(ea),
                "floor_train_prior": _train_prior_floor(ea, train_answers),
            }
        )
    by_format = pd.DataFrame(
        fmt_rows,
        columns=[
            "answer_format", "distribution", "accuracy", "n",
            "ci_low", "ci_high", "floor_majority", "floor_train_prior",
        ],
    )

    # ── number_estimate: the answer_format=="number" hierarchical estimate ────
    num = df[df["answer_format"].astype(str) == "number"]
    n_mean, n_low, n_high = _hier_bootstrap(num, n_boot=n_boot, rng=rng)
    number_estimate = {
        "accuracy": n_mean, "ci_low": n_low, "ci_high": n_high, "n": int(len(num)),
    }

    return {
        "bucket_mean": bucket_mean,
        "by_bucket": by_bucket,
        "by_bucket_format": by_bucket_format,
        "by_format": by_format,
        "acc_ID": acc_id,
        "acc_OOD": acc_ood,
        "acc_overall": acc_overall,
        "number_estimate": number_estimate,
    }


# ── gates — importable, unconditional, RAISE on malformed input ───────────────
# These encode the exact failure modes from the git archaeology (964 dropped in
# rung 07). They cannot be silently disabled: no flag, no env toggle. An
# experiment that imports them cannot pass a broken df past them.

def assert_all_rows_grouped(results_df: pd.DataFrame) -> None:
    """Every ``primary`` leaf must map to a real Capability group (ex-gate G3).

    RAISES ``ValueError`` listing any un-mappable leaves + example qIDs — this is
    the 964-drop guard: a group-name-vs-leaf filter would silently drop these.
    """
    if results_df.empty:
        return
    bad: dict[str, list] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for leaf in results_df["primary"].unique():
            if Capability.from_any(leaf) is None:
                qids = results_df.loc[results_df["primary"] == leaf, "qID"].tolist()
                bad[str(leaf)] = qids[:5]
    if bad:
        raise ValueError(
            f"{len(bad)} un-mappable primary leaf value(s) would be dropped by a "
            f"group filter: {bad}. Every scored row must map its leaf to a "
            "Capability group (Capability.from_any(leaf).group)."
        )


def assert_bucket_counts(
    results_df: pd.DataFrame, expected: dict[tuple[str, str], int]
) -> None:
    """Recompute {group}×{ID,OOD} counts and assert they equal ``expected``.

    Catches a group-name-vs-leaf filter: e.g. ``object_recognition`` totals 3421,
    NOT the ``object_identification`` leaf n=2457. RAISES on any mismatch.
    """
    assert_all_rows_grouped(results_df)
    df = results_df.copy()
    df["capability_group"] = df["primary"].map(_leaf_to_group)
    df["distribution"] = df["qID"].map(_dist_from_qid)
    got = {
        (g, d): int(n)
        for (g, d), n in df.groupby(["capability_group", "distribution"]).size().items()
    }
    diffs = {
        k: (got.get(k), expected.get(k))
        for k in set(got) | set(expected)
        if got.get(k) != expected.get(k)
    }
    if diffs:
        raise AssertionError(
            f"bucket count mismatch (got, expected) per (group, dist): {diffs}"
        )


def assert_floors_vs_eval_set(report: dict) -> None:
    """Every ``by_format`` accuracy must be >= its trivial floors, vs the EVAL set.

    Floors are computed vs the eval answers split by ID/OOD (never a global
    prior). NaN floors (answers not supplied) are skipped. RAISES listing any
    below-floor (format, distribution) rows.
    """
    bf = report.get("by_format")
    if bf is None or len(bf) == 0:
        return
    below: list[str] = []
    for _, r in bf.iterrows():
        for fcol in ("floor_majority", "floor_train_prior"):
            floor = r.get(fcol)
            if floor is not None and pd.notna(floor) and r["accuracy"] < floor:
                below.append(
                    f"{r['answer_format']}/{r['distribution']} acc={r['accuracy']:.4f} "
                    f"< {fcol}={floor:.4f}"
                )
    if below:
        raise AssertionError(
            "by_format accuracy below trivial floor (model no better than guessing): "
            + "; ".join(below)
        )


def assert_no_dup_qid(results_df: pd.DataFrame) -> None:
    """``qID`` must be unique (mirrors run.py:130 + evaluator.py duplicate-qID abort)."""
    if not results_df["qID"].is_unique:
        dups = results_df.loc[results_df["qID"].duplicated(keep=False), "qID"].unique()
        raise AssertionError(
            f"{len(dups)} duplicate qID(s), e.g. {list(dups[:5])} — Evaluator would abort "
            "and buckets would double-count."
        )


def assert_ood_from_qid(results_df: pd.DataFrame) -> None:
    """Every qID prefix must be a known dataset so ID/OOD comes from the qID.

    RAISES if any prefix ∉ {heico, lapchole} — a df whose distribution silently
    came from the all-False ``ood`` column would have unknown/foreign prefixes.
    """
    if results_df.empty:
        return
    prefixes = results_df["qID"].map(_prefix_from_qid)
    unknown = sorted(set(prefixes) - set(_VALID_PREFIXES))
    if unknown:
        raise AssertionError(
            f"qID prefixes {unknown} ∉ {_VALID_PREFIXES}: ID/OOD MUST be derived from the "
            "qID prefix (heico=OOD, lapchole=ID), never the all-False results_df['ood'] column."
        )
