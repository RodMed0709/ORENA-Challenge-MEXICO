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

import json
import logging
import math
import re
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


# ── template-aware trivial floor (THE canonical floor — single source) ────────
# The floor = accuracy of a dumb constant that answers each TEMPLATE's modal
# answer, exploiting the answer distribution alone (data card rung 08 §3–§4b). The
# template is the question text with its embedded HH:MM:SS timestamp normalised to
# ``<TS>`` (so each open_ended question does not become its own n=1 template).
# ``build_card`` (rung 08) imports BOTH functions below so there is exactly ONE
# implementation of the floor in the repo (context/RULES.md §EVAL rule 1 —
# "if the module lacks something, EXTEND it; never reimplement beside it").

_TS_RE = re.compile(r"\d{2}:\d{2}:\d{2}")


def template_of(question) -> str:
    """Normalise the embedded ``HH:MM:SS`` timestamp to ``<TS>`` → the canonical
    template key. On the raw string every open_ended question is unique (its own
    timestamp), inflating 188 real templates to 393 artefact ones (rung 08 §3)."""
    return _TS_RE.sub("<TS>", str(question))


def template_floor(
    df: pd.DataFrame | None, *, answer_col: str = "answer", template_col: str = "template"
) -> float:
    """Template-aware trivial floor over a slice: the fraction you would score by
    answering, for every question, its own template's modal answer.

    A STRICTLY smarter (and higher) baseline than the global slice-majority; the
    data card shows measuring against the dumber global constant inflated our margin
    (§3, Simpson's paradox). Returns NaN on an empty/absent slice. This is the SAME
    computation the data card's ``_floor_per_template`` performs — it now calls here."""
    if df is None or len(df) == 0:
        return float("nan")
    hits = sum(g[answer_col].value_counts().iloc[0] for _, g in df.groupby(template_col))
    return float(hits / len(df))


# ── public API ───────────────────────────────────────────────────────────────

def stratified_report(
    results_df: pd.DataFrame,
    video_split: dict | None = None,
    *,
    min_bucket_n: int = 2,
    n_boot: int = 1000,
    seed: int = 42,
    gold: pd.DataFrame | None = None,
) -> dict:
    """Canonical FRAME score for a per-question ``results_df``.

    leaf→group is ALWAYS ``Capability.group``; ID/OOD is ALWAYS the qID prefix
    (``video_split`` manifest is an optional override, never the default).

    Returns a dict with at least:
      - ``"bucket_mean"``  : unweighted mean over the populated group×{ID,OOD}
        buckets AFTER dropping any bucket with ``n < min_bucket_n`` (the headline).
      - ``"by_bucket"``    : DataFrame [capability_group, distribution, accuracy, n,
        floor, margin].
      - ``"by_bucket_format"`` : DataFrame [capability_group, distribution,
        answer_format, accuracy, n, floor, margin] — the group×{ID,OOD}×answer_format
        cross, using the SAME canonical leaf→group + qID ID/OOD. Tier-2 ledger source;
        bootstrap CIs stay at the coarser ``by_format`` granularity.
      - ``"by_format"``    : DataFrame [answer_format, distribution, accuracy, n,
        ci_low, ci_high, floor, margin].
      - ``"acc_ID"``, ``"acc_OOD"`` : flat per-question mean of correctness over the
        ID / OOD halves (acc_OOD is the number the leaderboard weights ~half).
      - ``"acc_overall"``  : flat per-question mean over all rows.
      - ``"floor_ID"``, ``"floor_OOD"`` : the template-aware trivial floor over the
        whole ID / OOD slice, and ``"margin_ID" = acc_ID − floor_ID`` /
        ``"margin_OOD" = acc_OOD − floor_OOD`` — the honest "real skill" numbers
        (data card §4b: rung-02 margin_ID ≈ +0.184, margin_OOD ≈ +0.132). ``acc_OOD >
        acc_ID`` reverses under margin because the OOD floor is ~12 pts higher.
      - ``"number_estimate"`` : the SDK per-format hierarchical estimate for
        ``number`` — {accuracy, ci_low, ci_high, n} from the replicated
        video→question bootstrap over all ``answer_format == "number"`` rows.

    ``gold`` (optional) supplies the ground-truth needed for the template-aware
    floors: a DataFrame keyed on ``qID`` carrying an ``answer`` column and either a
    ``template`` column or a ``question`` column (normalised here via ``template_of``).
    ``results_df`` itself carries only qID/format/primary/correctness — no gold — so
    WITHOUT ``gold`` every floor/margin is NaN (a MISSING INPUT, logged, not a bug).
    """
    _empty_bucket = ["capability_group", "distribution", "accuracy", "n", "floor", "margin"]
    if results_df.empty:
        return {
            "bucket_mean": float("nan"),
            "by_bucket": pd.DataFrame(columns=_empty_bucket),
            "by_bucket_format": pd.DataFrame(
                columns=["capability_group", "distribution", "answer_format",
                         "accuracy", "n", "floor", "margin"]
            ),
            "by_format": pd.DataFrame(
                columns=[
                    "answer_format", "distribution", "accuracy", "n",
                    "ci_low", "ci_high", "floor", "margin",
                ]
            ),
            "acc_ID": float("nan"),
            "acc_OOD": float("nan"),
            "acc_overall": float("nan"),
            "floor_ID": float("nan"),
            "floor_OOD": float("nan"),
            "margin_ID": float("nan"),
            "margin_OOD": float("nan"),
            "number_estimate": {"accuracy": float("nan"), "ci_low": float("nan"),
                                "ci_high": float("nan"), "n": 0},
        }

    df = results_df.copy()
    df["_correct"] = _correct_series(df)
    # leaf→group — RAISES on an un-mappable leaf (no silent drop, ever).
    df["capability_group"] = df["primary"].map(_leaf_to_group)
    df["distribution"] = _distribution(df, video_split)
    df["_vkey"] = df.apply(_video_key, axis=1)

    # ── attach gold answers + templates for template-aware floors (optional) ──
    has_gold = gold is not None and len(gold) > 0
    if has_gold:
        g = gold.copy()
        if "template" not in g.columns:
            if "question" not in g.columns:
                raise KeyError(
                    "gold must carry a 'template' or 'question' column for the "
                    f"template-aware floor; got {list(g.columns)}"
                )
            g["template"] = g["question"].map(template_of)
        g = g[["qID", "answer", "template"]].drop_duplicates("qID")
        df = df.merge(g, on="qID", how="left").rename(
            columns={"answer": "_answer", "template": "_template"}
        )
        missing = int(df["_answer"].isna().sum())
        if missing:
            logger.warning(
                "template-aware floor: %d/%d rows have no matching gold answer "
                "(qID not in the supplied gold set) — their templates are dropped "
                "from the floor.", missing, len(df),
            )
    else:
        logger.info(
            "template-aware floors: no gold answers supplied — floor/margin columns "
            "are NaN (results_df carries no answer column; this is a missing input)."
        )

    def _slice_floor(sub: pd.DataFrame) -> float:
        """Template-aware floor of a slice, or NaN when gold is absent."""
        if not has_gold:
            return float("nan")
        return template_floor(sub, answer_col="_answer", template_col="_template")

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
    # per-bucket template-aware floor + margin (NaN without gold).
    by_bucket["floor"] = [
        _slice_floor(df[(df["capability_group"] == r.capability_group)
                        & (df["distribution"] == r.distribution)])
        for r in by_bucket.itertuples(index=False)
    ]
    by_bucket["margin"] = by_bucket["accuracy"] - by_bucket["floor"]

    # ── by_bucket_format: group×{ID,OOD}×answer_format cross (Tier-2 source) ──
    # Same canonical leaf→group + qID ID/OOD already on ``df`` — a pure re-group,
    # never a re-derivation of buckets/ID-OOD. Bootstrap CIs stay in ``by_format``.
    by_bucket_format = (
        df.groupby(["capability_group", "distribution", "answer_format"])["_correct"]
        .agg(accuracy="mean", n="size")
        .reset_index()
        .sort_values(["capability_group", "distribution", "answer_format"])
        .reset_index(drop=True)
    )
    by_bucket_format["floor"] = [
        _slice_floor(df[(df["capability_group"] == r.capability_group)
                        & (df["distribution"] == r.distribution)
                        & (df["answer_format"] == r.answer_format)])
        for r in by_bucket_format.itertuples(index=False)
    ]
    by_bucket_format["margin"] = by_bucket_format["accuracy"] - by_bucket_format["floor"]

    # ── flat ID / OOD / overall means + template-aware floors/margins ─────────
    dist_means = df.groupby("distribution")["_correct"].mean()
    acc_id = float(dist_means.get("ID", float("nan")))
    acc_ood = float(dist_means.get("OOD", float("nan")))
    acc_overall = float(df["_correct"].mean())
    floor_id = _slice_floor(df[df["distribution"] == "ID"])
    floor_ood = _slice_floor(df[df["distribution"] == "OOD"])
    margin_id = acc_id - floor_id
    margin_ood = acc_ood - floor_ood

    # ── by_format (+ bootstrap CIs + template-aware floor + margin) ───────────
    rng = np.random.default_rng(seed)
    fmt_rows: list[dict] = []
    for (fmt, dist), sub in df.groupby(["answer_format", "distribution"], sort=True):
        mean, low, high = _hier_bootstrap(sub, n_boot=n_boot, rng=rng)
        acc = float(sub["_correct"].mean())
        floor = _slice_floor(sub)
        fmt_rows.append(
            {
                "answer_format": str(fmt),
                "distribution": str(dist),
                "accuracy": acc,
                "n": int(len(sub)),
                "ci_low": low,
                "ci_high": high,
                "floor": floor,
                "margin": acc - floor,
            }
        )
    by_format = pd.DataFrame(
        fmt_rows,
        columns=[
            "answer_format", "distribution", "accuracy", "n",
            "ci_low", "ci_high", "floor", "margin",
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
        "floor_ID": floor_id,
        "floor_OOD": floor_ood,
        "margin_ID": margin_id,
        "margin_OOD": margin_ood,
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
    """Verify the template-aware floor pipeline actually ran and is self-consistent.

    For every ``by_format`` / ``by_bucket`` row that carries a POPULATED (non-NaN)
    floor, assert the floor lies in [0, 1] and ``margin == accuracy − floor`` exactly.
    This is the real check that replaces the old NaN no-op: it catches a floor that was
    never wired (all-NaN when gold WAS supplied), a corrupted round-trip, or a margin
    that drifted from ``accuracy − floor``.

    It deliberately does NOT raise merely because a run sits BELOW its floor. Being
    below the template-aware floor is a documented FINDING, not malformed input: a weak
    baseline legitimately is (data card rung 08 §4b — the 00-baseline is below floor in
    every cell), and the honest signal is the NEGATIVE margin surfaced in ``results/``,
    never an aborted score. When NO floors are populated (``gold`` not supplied) this
    stays a logged no-op — a MISSING input, not a broken one.
    """
    problems: list[str] = []
    n_populated = 0
    for name in ("by_format", "by_bucket"):
        t = report.get(name)
        if t is None or len(t) == 0 or "floor" not in getattr(t, "columns", []):
            continue
        for _, r in t.iterrows():
            floor = r.get("floor")
            if floor is None or pd.isna(floor):
                continue
            n_populated += 1
            key = "/".join(str(r.get(k)) for k in t.columns if k in
                           ("answer_format", "capability_group", "distribution"))
            if not (0.0 <= float(floor) <= 1.0):
                problems.append(f"{name}[{key}] floor={floor} outside [0,1]")
            margin = r.get("margin")
            expected = float(r["accuracy"]) - float(floor)
            if margin is None or pd.isna(margin) or abs(float(margin) - expected) > 1e-9:
                problems.append(
                    f"{name}[{key}] margin={margin} != accuracy−floor={expected:.6f}"
                )
    if n_populated == 0:
        logger.info(
            "assert_floors_vs_eval_set: no template-aware floors populated (gold "
            "answers not supplied) — nothing to verify (missing input, not a bug)."
        )
        return
    if problems:
        raise AssertionError(
            "template-aware floor/margin pipeline inconsistent: " + "; ".join(problems)
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


def assert_matches_sdk_pre_eval(
    bucket_mean: float,
    sdk_buckets: pd.DataFrame,
    *,
    min_bucket_n: int = 2,
    tol: float = 0.02,
) -> float:
    """HYBRID cross-check: our ``bucket_mean`` must match the vendor SDK's own
    OOD-aware ``pre_evaluation_score`` restricted to the SAME populated buckets.

    ``sdk_buckets`` is the ``buckets_df`` returned by
    ``focus.evaluation.Evaluator.pre_evaluation_score`` — computed AFTER stamping
    ``ref.ood`` from the qID prefix (heico=OOD) so the vendor's ID/OOD split is
    meaningful (``ref.ood`` is all-False on public data — data_models.py:101,
    consumed at evaluator.py:440). Its columns are ``[group, ood, accuracy, count]``.

    The vendor number is an INDEPENDENT re-derivation (a different code path:
    ``Evaluator._to_group`` + its own group×ood loop, evaluator.py:394-462), so
    agreeing with it proves our reimplementation is faithful — the whole point of
    the hybrid. The one intended difference: our ``bucket_mean`` drops buckets with
    ``n < min_bucket_n`` (the n=1 temporal_grounding question) whereas the *raw*
    vendor pre_eval keeps them; we therefore compare against the vendor mean
    restricted to ``count >= min_bucket_n``. Both the restricted and the raw vendor
    means are logged. RAISES only on a large divergence (``> tol``), since the two
    methods can differ slightly. Returns the restricted vendor mean.
    """
    if sdk_buckets is None or len(sdk_buckets) == 0:
        logger.warning("HYBRID cross-check skipped: empty SDK pre_eval buckets_df.")
        return float("nan")
    kept = sdk_buckets[sdk_buckets["count"] >= min_bucket_n]
    sdk_mean = float(kept["accuracy"].mean()) if len(kept) else float("nan")
    sdk_raw = float(sdk_buckets["accuracy"].mean())
    logger.info(
        "HYBRID cross-check: our bucket_mean=%.4f | vendor pre_eval(count>=%d)=%.4f | "
        "vendor pre_eval(raw, incl n<%d)=%.4f",
        bucket_mean, min_bucket_n, sdk_mean, min_bucket_n, sdk_raw,
    )
    if not math.isnan(sdk_mean) and abs(bucket_mean - sdk_mean) > tol:
        raise AssertionError(
            f"HYBRID cross-check FAILED: our bucket_mean {bucket_mean:.4f} diverges from the "
            f"vendor SDK OOD-aware pre_evaluation_score {sdk_mean:.4f} (count>={min_bucket_n} "
            f"buckets) by >{tol}. Our reimplementation disagrees with "
            "focus.evaluation.Evaluator.pre_evaluation_score — investigate leaf->group / ID-OOD."
        )
    return sdk_mean


# ── paired delta CI (added for rung 10) ──────────────────────────────────────
# stratified_report and _hier_bootstrap each describe ONE arm. An A/B on the SAME
# questions (greedy vs voted) needs the bootstrap of the paired DIFFERENCE: two
# independent CIs would throw away the pairing and come out far too wide, because
# both arms share the questions, the frames and the model — nearly all of the
# variance is common and cancels in the difference. Added here rather than
# re-derived in a notebook (RULES §EVAL rule 1: extend the module, never
# reimplement beside it). Same video→question two-level scheme as
# ``_hier_bootstrap`` and the same deterministic-for-a-fixed-seed contract.


def paired_delta_ci(
    df: pd.DataFrame,
    *,
    correct_a: str = "correct_a",
    correct_b: str = "correct_b",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Video→question bootstrap of the paired difference ``b - a`` over a slice.

    ``df`` needs ``qID``, ``video`` and the two correctness columns, one row per
    question, both arms scored on the SAME questions. Returns the point estimate,
    its 95 % percentile CI, and how many questions each arm wins outright — the
    paired win counts are what a sign test reads, so they come back here instead
    of being recomputed by every caller.

    Returns NaNs on an empty slice instead of raising: callers slice by
    (format × distribution) and an empty cell is a legitimate reportable state.
    """
    empty = {
        "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
        "n": 0, "n_videos": 0, "wins_b": 0, "wins_a": 0,
    }
    if df is None or len(df) == 0:
        return empty

    work = df[["qID", "video", correct_a, correct_b]].copy()
    work["_vkey"] = work.apply(_video_key, axis=1)
    work["_diff"] = work[correct_b].astype(float) - work[correct_a].astype(float)

    vid_groups = {
        v: work.loc[work["_vkey"] == v, "_diff"].to_numpy(dtype=float)
        for v in work["_vkey"].dropna().unique()
    }
    keys = list(vid_groups)
    n_videos = len(keys)
    if n_videos == 0:
        return empty

    point = float(np.mean([vid_groups[v].mean() for v in keys]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        # sample video POSITIONS: keys are tuples and cannot go through rng.choice
        sampled_pos = rng.integers(0, n_videos, size=n_videos)
        boots[b] = float(
            np.mean(
                [
                    rng.choice(
                        vid_groups[keys[i]], size=len(vid_groups[keys[i]]), replace=True
                    ).mean()
                    for i in sampled_pos
                ]
            )
        )
    return {
        "delta": point,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n": int(len(work)),
        "n_videos": n_videos,
        "wins_b": int((work["_diff"] > 0).sum()),
        "wins_a": int((work["_diff"] < 0).sum()),
    }


# ── count DISCRIMINATION: rank correlation vs gold (added for rung 18) ───────
# Every `number` metric we have reported — accuracy, margin, the hierarchical estimate — is
# an EXACT-MATCH metric, and exact match cannot see the difference between a model that has
# learned the gold's marginal distribution and one that reads the frame. Rung 06 ep3 leaves
# `number_margin_OOD` at exactly 0.0 (the trivial floor) while its mean prediction sits 0.66
# below the mean gold: the scale is nearly right and the ordering is not.
#
# [[gold-is-signal-model-underuses-it]] made that the headline: on a stratified blind sample
# a HUMAN scores Spearman r = +0.7230 against the same gold and our model +0.43. So the gold
# carries frame-visible, ordered signal the model is not using, the target is DISCRIMINATION
# rather than calibration, and the metric that reads it has to be a RANK metric.
#
# It lives here, in the canonical module, for the reason rule §EVAL-1 gives: a notebook that
# re-derives it beside the module is exactly how the same defect came back in rung 07.
# Spearman is computed as Pearson over average ranks (its definition) using pandas' own
# tie-aware ranking — no scipy, so the module's "pure pandas/numpy, offline" contract holds.

_LEADING_INT_RE = re.compile(r"^\s*(\d+)")


def read_count(raw) -> float:
    """Lenient integer read of one model answer, or NaN.

    LENIENT ON PURPOSE, and it must not be confused with the SDK's gate. ``Number.verify``
    requires ``str.strip().isdigit()``, so ``"1."`` is auto-INCORRECT there — that is a
    scored FORMAT defect and rung 16d measures it separately. This function answers the
    different question *what number did the model mean*, because a rank correlation computed
    only over SDK-legal answers would silently drop exactly the rows the format defect hits
    and report a discrimination score for a biased subsample. Mirrors the lenient ``value``
    branch of probe 16a's ``read_answer``; the strict branch is not duplicated here.
    """
    m = _LEADING_INT_RE.match(str(raw))
    return float(m.group(1)) if m else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho = Pearson over average ranks. NaN when either side is constant."""
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def count_rank_report(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    template_pattern: str = r"^How many Clips appear in this frame\?",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Spearman r of the PREDICTED count vs the gold count, on one question template.

    ``predictions`` needs ``qID`` and a prediction column (``prediction`` / ``content`` /
    ``answer_pred``), and MAY carry ``video`` — without it the CI is skipped rather than
    computed wrong, since questions cluster on ~38 videos and an unclustered CI would be far
    too narrow (RULES §13).

    ``gold`` is ``[qID, answer, question]`` as ``ledger.gold_from_frame_parquets`` returns it.
    ``template_pattern`` is matched against the timestamp-normalised template, so the default
    selects the `Clips` per-class count template — the one the blind human adjudication was
    run on, and therefore the only cell where our r and the human's 0.7230 are comparable.

    Returns ``{"pooled": {...}, "ID": {...}, "OOD": {...}}``; each cell carries ``r`` with a
    video-clustered percentile CI, ``n``, ``n_videos``, ``n_unreadable`` (answers with no
    leading integer — reported, never silently dropped), ``mean_pred``, ``mean_gold``,
    ``bias`` and ``exact``.
    """
    pcol = next(
        (c for c in ("prediction", "content", "answer_pred", "pred") if c in predictions.columns),
        None,
    )
    if pcol is None:
        raise KeyError(
            f"predictions needs one of prediction/content/answer_pred/pred; "
            f"got {list(predictions.columns)}"
        )
    g = gold.copy()
    if "template" not in g.columns:
        if "question" not in g.columns:
            raise KeyError("gold must carry a 'template' or 'question' column")
        g["template"] = g["question"].map(template_of)
    g = g[g["template"].str.match(template_pattern, na=False)]

    keep = ["qID", pcol] + (["video"] if "video" in predictions.columns else [])
    df = g.merge(predictions[keep].drop_duplicates("qID"), on="qID", how="inner")
    df["_gold"] = df["answer"].map(read_count)
    df["_pred"] = df[pcol].map(read_count)
    df["distribution"] = df["qID"].map(_dist_from_qid)

    def _cell(sub: pd.DataFrame) -> dict:
        n_all = len(sub)
        ok = sub[sub["_pred"].notna() & sub["_gold"].notna()]
        out = {
            "n": int(n_all),
            "n_scored": int(len(ok)),
            "n_unreadable": int(n_all - len(ok)),
            "r": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "n_videos": 0,
            "mean_pred": float(ok["_pred"].mean()) if len(ok) else float("nan"),
            "mean_gold": float(ok["_gold"].mean()) if len(ok) else float("nan"),
            "exact": float((ok["_pred"] == ok["_gold"]).mean()) if len(ok) else float("nan"),
        }
        out["bias"] = out["mean_pred"] - out["mean_gold"]
        if len(ok) < 3:
            return out
        out["r"] = _spearman(ok["_pred"].to_numpy(), ok["_gold"].to_numpy())
        if "video" not in ok.columns:
            logger.info(
                "count_rank_report: no `video` column — r reported without a CI (an "
                "unclustered CI over ~38 videos would be far too narrow; RULES §13)."
            )
            return out
        work = ok.copy()
        work["_vkey"] = work.apply(_video_key, axis=1)
        # Held as plain numpy per video: the bootstrap resamples VIDEOS, and concatenating
        # arrays is orders of magnitude cheaper than concatenating 2000 DataFrames.
        groups = [
            (d["_pred"].to_numpy(dtype=float), d["_gold"].to_numpy(dtype=float))
            for _, d in work.groupby("_vkey")
        ]
        out["n_videos"] = len(groups)
        if len(groups) < 3:
            return out
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), size=len(groups))
            r = _spearman(
                np.concatenate([groups[i][0] for i in pick]),
                np.concatenate([groups[i][1] for i in pick]),
            )
            if not math.isnan(r):
                boots.append(r)
        if boots:
            out["ci_low"] = float(np.percentile(boots, 2.5))
            out["ci_high"] = float(np.percentile(boots, 97.5))
        return out

    report = {"pooled": _cell(df)}
    for dist in ("ID", "OOD"):
        report[dist] = _cell(df[df["distribution"] == dist])
    logger.info(
        "count_rank_report [%s]: pooled r=%.4f (n=%d, %d unreadable) | ID r=%.4f | OOD r=%.4f",
        template_pattern, report["pooled"]["r"], report["pooled"]["n"],
        report["pooled"]["n_unreadable"], report["ID"]["r"], report["OOD"]["r"],
    )
    return report


def predictions_frame(run_dir: str | Path, results_csv: str | Path | None = None) -> pd.DataFrame:
    """``[qID, prediction, video]`` from a run's ``predictions.json`` (+ ``results.csv``).

    ``run.run_baseline`` persists the raw model strings to ``predictions.json`` and the
    scored rows — which carry ``video`` — to ``results.csv``. ``count_rank_report`` needs
    both: the raw string, because correctness alone cannot express a rank; and the video,
    because the CI has to cluster on it.
    """
    run_dir = Path(run_dir)
    preds = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(
        [{"qID": p["qID"], "prediction": p.get("content", "")} for p in preds]
    )
    res_path = Path(results_csv) if results_csv else run_dir / "results.csv"
    if res_path.exists():
        res = pd.read_csv(res_path)
        if "video" in res.columns:
            df = df.merge(res[["qID", "video"]].drop_duplicates("qID"), on="qID", how="left")
    return df
