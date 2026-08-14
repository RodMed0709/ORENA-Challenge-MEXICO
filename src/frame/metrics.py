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


def assert_class_f1_reported(row, *, results_df: pd.DataFrame | None = None) -> None:
    """A published result that scored ``fo_class`` MUST carry a class-balanced macro-F1.

    🔴 **Why this is a gate and not advice.** ``fo_class`` accuracy is exact set equality,
    so it is dominated by the head of a long-tailed class distribution — and the failure it
    hides is not hypothetical, it is published on our exact backbone: CoA (FICHAS §v01)
    raises overall F1 on CholecT50 while **crushing class-balanced F1cls 20.7 → 15.3**.
    That is our own ``clip`` attractor, in print. Our own numbers say the same: rung 18 ep3
    scores 0.6478 on ``fo_class`` × ID while ``Gallstone`` recalls **0.036**. And rung 21's
    ``+0.174`` macro-F1 headline turned out to be **82% one `Needle` question** — the metric
    is only safe when its ``per_class`` table is read beside it.

    ``bucket_mean`` cannot see any of this, so a rung that reports only ``bucket_mean``
    cannot distinguish "the model got better" from "the model got better at `clip`". Every
    phase of the CoA/CoT roadmap changes the target, which is exactly the intervention that
    produces this failure — see ``context/decisions/class-balanced-f1-is-mandatory.md``
    (RULES §9b).

    ``row`` is the ledger-shaped result mapping (dict or Series). Pass ``results_df`` to
    scope the requirement to the distributions that actually carry ``fo_class`` rows: a run
    with no ``fo_class`` questions owes nothing. Compute the values with
    ``class_f1_report`` — never by hand (RULES §1).
    """
    keys = dict(row)
    have = {
        k: v for k, v in keys.items()
        if str(k).startswith("macro_f1") and v is not None and not pd.isna(v)
    }

    needed: list[str] = []
    if results_df is not None and len(results_df):
        if "answer_format" not in results_df.columns:
            raise KeyError("results_df has no `answer_format` column to scope fo_class rows")
        fo = results_df[results_df["answer_format"] == "fo_class"]
        if fo.empty:
            return  # no fo_class questions scored — nothing is owed
        for dist in sorted(fo["qID"].map(_dist_from_qid).dropna().unique()):
            needed.append(f"macro_f1_{dist}")
    elif have:
        return
    else:
        raise ValueError(
            "no `macro_f1*` value in this result row. A run that scored `fo_class` must "
            "publish a class-balanced F1 beside its accuracy — exact-set accuracy hides a "
            "long-tail collapse (RULES §9b; FICHAS §v01 measures SFT raising F1 while "
            "crushing F1cls 20.7 -> 15.3 on our exact backbone). Use "
            "`frame.metrics.class_f1_report`."
        )

    if missing := [k for k in needed if k not in have]:
        raise ValueError(
            f"missing {missing} — this run scored `fo_class` questions in "
            f"{[k.replace('macro_f1_', '') for k in needed]}, so each needs a "
            f"class-balanced F1 (RULES §9b). Present: {sorted(have)}. Use "
            "`frame.metrics.class_f1_report`; do not hand-roll it (RULES §1)."
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


# ── equivalence: reading a null as a TIE, not as "we saw nothing" (rung 24) ───
# ``paired_delta_ci`` answers *"is B different from A?"*. It cannot answer *"are they
# the SAME?"* — a CI containing 0 is equally consistent with a true tie and with an
# effect we lacked the power to see. We have read it as if it could, twice: rungs 14
# and 15 were written up as nulls, and rung 24's pre-registered rule says *"a tower
# that is INDIFFERENT ⇒ the branch is mapping"* while never defining indifferent. A
# decision rule whose most likely outcome has no numeric definition is not a rule.
#
# The fix is the standard two-one-sided-tests framing (MoE-Sieve, `literature/
# vlm-techniques/FICHAS.md` §v49, which declares ±2 pp *before* looking): equivalence
# at a margin ε is established when the WHOLE CI lies inside [−ε, +ε]. Crossed with
# the ordinary superiority read (does the CI exclude 0?) this gives FOUR states rather
# than two, and the two extra ones are the honest descriptions of where our results
# actually land — "real but smaller than we said we cared about", and "we could not
# tell". Collapsing those into "null" is how a faithful negative and an underpowered
# run came to look identical in our own records.
#
# This is not an imported convention. The challenge's FINAL ranking is Copeland over
# buckets with **pairwise significance tests**, where non-significant deltas collapse
# to the SAME rank (RULES §4c, `challenge_design.txt:1010-1040`) — "indistinguishable"
# is already a first-class outcome in how we will be scored, so we should be able to
# write it down.
#
# ε is REQUIRED and has no default, deliberately. A default is a margin nobody
# declared, and the entire value of the instrument is that the margin is fixed before
# the number is seen (RULES §7: a gate that fires is a finding, not an obstacle).

EQUIVALENCE_VERDICTS = (
    "EQUIVALENT",           # CI inside ±ε and contains 0 — a tie, established
    "TRIVIAL_DIFFERENCE",   # CI inside ±ε but excludes 0 — real, below the margin
    "DIFFERENT",            # CI excludes 0 and is not contained in ±ε
    "INCONCLUSIVE",         # CI wider than ±ε and contains 0 — underpowered
    "UNDEFINED",            # empty slice (paired_delta_ci returned NaNs)
)


def equivalence_verdict(ci, *, epsilon: float) -> dict:
    """TOST-style equivalence read of a paired delta, at a PRE-DECLARED margin ``epsilon``.

    ``ci`` is the dict ``paired_delta_ci`` returns, or any ``(ci_low, ci_high)`` pair.
    ``epsilon`` is the largest delta we agreed, in advance, to call "no practical
    difference" — in the same units as the delta (a proportion, so 0.02 is 2 pp).

    Returns the four-state ``verdict`` plus the two booleans it is built from, so a
    caller can report the reasoning rather than the label alone. ``epsilon_min`` is the
    smallest margin at which THIS CI could have established equivalence
    (``max(|ci_low|, |ci_high|)``) — read it as the honest power statement: declaring an
    ε below it can only ever return INCONCLUSIVE, no matter what the true effect is.
    v49's own failing row is the warning (OLMoE/Spider: ∆ +0.30 pp, CI [−2.04, +2.64],
    inconclusive at ±2 pp *with 8 seeds*).

    ⚠️ Equivalence is a statement about the CI, and the CI here is the video-clustered
    paired bootstrap (RULES §13 — effective n is ~38 videos, not 6252). It is NOT a
    variance estimate over re-runs: we hold zero seed repeats, and
    [[seed-variance-is-small-when-clean]] bounds that band from a published clean-data
    task, it does not measure ours.
    """
    if not (epsilon > 0):
        raise ValueError(
            f"epsilon must be > 0 (got {epsilon!r}). It is the pre-declared margin of "
            "practical indifference; there is no sensible default and no zero margin."
        )

    if isinstance(ci, dict):
        low, high = float(ci.get("ci_low", float("nan"))), float(ci.get("ci_high", float("nan")))
        delta = float(ci.get("delta", float("nan")))
        passthrough = {k: ci[k] for k in ("n", "n_videos") if k in ci}
    else:
        low, high = (float(v) for v in ci)
        delta, passthrough = float("nan"), {}

    out = {
        **passthrough,
        "delta": delta,
        "ci_low": low,
        "ci_high": high,
        "epsilon": float(epsilon),
    }

    if np.isnan(low) or np.isnan(high):
        return {**out, "verdict": "UNDEFINED", "within_margin": False,
                "excludes_zero": False, "epsilon_min": float("nan"),
                "reason": "empty slice — paired_delta_ci returned NaN bounds"}

    if low > high:  # a caller passing (high, low) would silently invert every verdict
        raise ValueError(f"ci_low ({low}) > ci_high ({high}) — bounds are reversed")

    within = bool(low > -epsilon and high < epsilon)
    excludes_zero = bool(low > 0 or high < 0)
    epsilon_min = float(max(abs(low), abs(high)))

    if within and not excludes_zero:
        verdict = "EQUIVALENT"
        reason = f"CI [{low:+.4f}, {high:+.4f}] lies inside ±{epsilon:g} and contains 0"
    elif within:
        verdict = "TRIVIAL_DIFFERENCE"
        reason = (f"CI [{low:+.4f}, {high:+.4f}] excludes 0 but lies inside ±{epsilon:g} — "
                  "a real effect smaller than the pre-declared margin")
    elif excludes_zero:
        verdict = "DIFFERENT"
        reason = f"CI [{low:+.4f}, {high:+.4f}] excludes 0 and is not contained in ±{epsilon:g}"
    else:
        verdict = "INCONCLUSIVE"
        reason = (f"CI [{low:+.4f}, {high:+.4f}] contains 0 AND exceeds ±{epsilon:g} — "
                  f"underpowered; the smallest establishable margin here is "
                  f"±{epsilon_min:.4f}. NOT a tie.")

    return {**out, "verdict": verdict, "within_margin": within,
            "excludes_zero": excludes_zero, "epsilon_min": epsilon_min, "reason": reason}


def paired_equivalence(df: pd.DataFrame, *, epsilon: float, **kwargs) -> dict:
    """``paired_delta_ci`` then ``equivalence_verdict`` — the one call a rung should make.

    Exists so the margin travels with the delta. Reporting a bare delta and deciding the
    margin afterwards is exactly the failure mode this instrument is for.
    """
    return equivalence_verdict(paired_delta_ci(df, **kwargs), epsilon=epsilon)


# ── flip ratio: what a delta FIXED vs what it BROKE (rung 24 companion) ───────
# A delta is a NET number, and a net number hides its own composition. +0.02 can be
# "fixed 40, broke 0" or "fixed 400, broke 360" — the same headline, two completely
# different models, and only the second one is fragile. We have been reading the first
# kind of statement and getting the second kind of surprise.
#
# The instrument is the flip ratio from FICHAS §v52: negative flips (A right → B wrong)
# over positive flips (A wrong → B right). The paper is weak for us — 32B text LLMs on
# AIME, budgets 400× ours — but the ratio is backend-free and computable over any two
# checkpoints we have already scored, at zero GPU.
#
# 🔴 This is the lens that would have caught the rung-21 `+0.174` macro-F1 artefact when
# it was written rather than six days later: 82% of it was ONE `Needle` question
# (n_gold = 1, 1/7 of an unweighted macro) flipping, while `Gallstone` did not move at
# all (recall still 0.036). A single flip cannot survive being counted.
#
# The counts already exist — ``paired_delta_ci`` returns them as ``wins_a``/``wins_b``,
# which ARE the negative and positive flips. What was missing is the ratio, the churn,
# and a report that reads them per stratum, because that is where a fragile win shows.


def flip_report(
    df: pd.DataFrame,
    *,
    correct_a: str = "correct_a",
    correct_b: str = "correct_b",
    by: str | list[str] | None = None,
) -> pd.DataFrame:
    """Decompose ``b - a`` into what it FIXED and what it BROKE, optionally per stratum.

    ``df`` is the same paired frame ``paired_delta_ci`` takes (one row per question, both
    arms scored on the SAME questions). ``by`` is a column or list of columns to group on
    — ``"answer_format"``, ``["distribution", "answer_format"]``, a template, whatever the
    rung is arguing about. ``None`` returns a single overall row.

    Columns: ``n``, ``fixed`` (A wrong → B right), ``broke`` (A right → B wrong),
    ``delta`` (= (fixed − broke)/n, identical to the unclustered mean difference),
    ``flip_ratio`` = broke/fixed, and ``churn`` = (fixed + broke)/n.

    Read them together:
      * ``flip_ratio`` **< 1** means the change fixes more than it breaks. **≥ 1** with a
        positive delta is impossible; **> 1** means the arm is net-negative here.
      * ``flip_ratio`` near 1 with a small delta is the fragile case — lots of movement,
        no direction. A "null" of that shape is NOT the same as a null with zero churn,
        and `bucket_mean` cannot tell them apart.
      * ``churn`` high with ``delta`` near 0 says the two arms disagree constantly and
        happen to tie; that is a warning about seed/order sensitivity, not a tie.
      * ``fixed`` or ``broke`` in the low single digits means the cell's delta rests on a
        handful of questions — quote the count, never the delta alone.

    ``flip_ratio`` is NaN when ``fixed == 0`` (division by zero is not "infinitely bad";
    it is "nothing was fixed", which the ``fixed`` column already says).

    ⚠️ This is a DESCRIPTIVE decomposition, not an inference. It carries no CI and is not
    clustered by video — for "will this hold on a new video?" the answer is still
    ``paired_delta_ci`` / ``equivalence_verdict`` (RULES §13). Use the two together: the
    CI says whether the delta is real, this says what it is made of.
    """
    need = [correct_a, correct_b]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"flip_report needs {missing} — both arms must be scored on the same rows")

    keys = [by] if isinstance(by, str) else list(by or [])
    for k in keys:
        if k not in df.columns:
            raise KeyError(f"flip_report: grouping column {k!r} not in df")

    work = df.copy()
    a = work[correct_a].astype(float)
    b = work[correct_b].astype(float)
    work["_fixed"] = ((a <= 0) & (b > 0)).astype(int)
    work["_broke"] = ((a > 0) & (b <= 0)).astype(int)

    def _row(sub: pd.DataFrame) -> dict:
        n = int(len(sub))
        fixed, broke = int(sub["_fixed"].sum()), int(sub["_broke"].sum())
        return {
            "n": n,
            "fixed": fixed,
            "broke": broke,
            "delta": (fixed - broke) / n if n else float("nan"),
            "flip_ratio": (broke / fixed) if fixed else float("nan"),
            "churn": (fixed + broke) / n if n else float("nan"),
        }

    if not keys:
        return pd.DataFrame([_row(work)])

    rows = []
    for gkey, sub in work.groupby(keys, dropna=False, sort=True):
        gvals = gkey if isinstance(gkey, tuple) else (gkey,)
        rows.append({**dict(zip(keys, gvals)), **_row(sub)})
    return pd.DataFrame(rows).sort_values("churn", ascending=False).reset_index(drop=True)


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


def _load_fotype():
    """``FOType``, imported the same defensive way as ``Capability``.

    ``focus/foreign_objects.py`` is stdlib-only, but ``focus/__init__.py`` eagerly
    imports transformers/datasets — so on a laptop the package import fails while the
    module itself is perfectly loadable. Mirrors ``_load_capability`` rather than
    duplicating a class list, because RULES §8b forbids hard-coding the accepted set:
    the prompt's 10-item list and the scorer's disagree on their 10th element, and an
    unrecognised token RAISES in ``verify()`` instead of scoring 0.
    """
    try:
        from focus.foreign_objects import FOType  # noqa: PLC0415

        return FOType
    except Exception as exc:  # noqa: BLE001
        import importlib.util  # noqa: PLC0415
        import sys  # noqa: PLC0415

        for p in sys.path:
            cand = Path(p) / "focus" / "foreign_objects.py"
            if cand.exists():
                spec = importlib.util.spec_from_file_location(
                    "focus._foreign_objects_standalone", cand
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                logger.debug("loaded stdlib-only foreign_objects directly from %s", cand)
                return mod.FOType
        raise ImportError(
            "focus.foreign_objects is required for fo_class scoring but could not be "
            "imported and no standalone foreign_objects.py was found on sys.path."
        ) from exc


_FO_NONE = "None"  # FOClass.NONE (formats.py:161) — absence, and it is a real answer state


def read_fo_class(raw, valid_lower: dict[str, str]) -> frozenset[str] | None:
    """One ``fo_class`` answer → the canonical set, or ``None`` if SDK-ILLEGAL.

    Byte-for-byte the semantics of ``FOClass.read``/``verify`` (formats.py:171-196):
    comma-split, strip, case-insensitive match against the registered names, ``"none"``
    only on its own, comparison by exact set equality. It differs in ONE deliberate way —
    the SDK RAISES on an unrecognised token and this returns ``None`` — because an illegal
    answer is a measurement here (rung 16d), not a crash. Illegal rows are counted and
    reported, never silently dropped.
    """
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return None
    out = set()
    for part in parts:
        low = part.lower()
        if low == "none":
            if len(parts) > 1:
                return None  # "'none' cannot be combined with other classes"
            out.add(_FO_NONE)
            continue
        if low not in valid_lower:
            return None
        out.add(valid_lower[low])
    return frozenset(out)


def _f1_counts(rows: list[tuple[frozenset[str], frozenset[str] | None]]) -> dict:
    """Per-class TP/FP/FN over (gold_set, pred_set) pairs; ``None`` pred = illegal.

    Multi-label by construction: one answer may name several classes, so a row
    contributes to as many classes as it mentions. An illegal prediction contributes FN
    to every gold class and FP to none — it is a miss, not a wrong guess.
    """
    stats: dict[str, dict[str, int]] = {}

    def _cell(name: str) -> dict[str, int]:
        return stats.setdefault(name, {"tp": 0, "fp": 0, "fn": 0, "n_gold": 0})

    for gold, pred in rows:
        p = pred if pred is not None else frozenset()
        for c in gold:
            _cell(c)["n_gold"] += 1
            if c in p:
                _cell(c)["tp"] += 1
            else:
                _cell(c)["fn"] += 1
        for c in p - gold:
            _cell(c)["fp"] += 1
    return stats


def _macro_f1(stats: dict) -> tuple[float, dict]:
    """Class-balanced F1 = unweighted mean of per-class F1 over SUPPORTED classes.

    Supported = ``n_gold > 0``. A class the eval set never asks about would otherwise
    enter the mean at F1 = 0 (or 1) and turn the headline into a property of the class
    registry rather than of the model. Classes emitted but never gold still show up in
    the per-class table with their FPs, where they belong.
    """
    per = {}
    for name, s in sorted(stats.items()):
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[name] = {
            "n_gold": s["n_gold"], "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": prec, "recall": rec, "f1": f1,
        }
    supported = [v["f1"] for v in per.values() if v["n_gold"] > 0]
    return (float(np.mean(supported)) if supported else float("nan")), per


def class_f1_report(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    results_df: pd.DataFrame | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Class-balanced (macro) F1 on the ``fo_class`` format — the metric exact-set
    accuracy hides.

    🔴 **Why this exists.** ``fo_class`` accuracy is exact set equality, so it is dominated
    by the head of a long-tailed class distribution: rung 18 ep3 scores 0.6478 accuracy on
    `fo_class` × ID while `Gallstone` (n=28) recalls 0.036 and `External Drain` (n=24)
    0.208 — a collapse the headline cannot see. [[coa-sft-published-null]] measures the
    same effect published on our exact backbone (SFT crushes F1cls 20.7 → 15.3), and
    [[loss-mass-is-token-weighted]] needs it because any reweighting that moves gradient
    *within* `fo_class` lands on class-name length, which is a function of class identity.

    ``predictions`` needs ``qID`` and a prediction column (``prediction``/``content``/
    ``answer_pred``/``pred``) and MAY carry ``video`` — without it the CI is skipped rather
    than computed wrong (questions cluster on ~38 videos; RULES §13).
    ``gold`` is ``[qID, answer, question]`` as ``ledger.gold_from_frame_parquets`` returns.

    Row selection: ``results_df['answer_format'] == 'fo_class'`` when a ``results_df`` is
    given (the authoritative label), otherwise the rows whose GOLD parses as a legal FO set
    — which is the format's own definition, and cannot admit a ``number``/``binary`` gold.

    Returns ``{"pooled": {...}, "ID": {...}, "OOD": {...}}``; each cell carries
    ``macro_f1`` (+ video-clustered percentile CI), ``exact_set_acc`` (the SDK's own
    correctness, recomputed here only so the two are read side by side), ``n``,
    ``n_illegal`` and a ``per_class`` table.
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
    valid_names = tuple(_load_fotype().names())
    valid_lower = {n.lower(): n for n in valid_names}

    keep = ["qID", pcol] + (["video"] if "video" in predictions.columns else [])
    df = gold.merge(predictions[keep].drop_duplicates("qID"), on="qID", how="inner")

    if results_df is not None and "answer_format" in results_df.columns:
        fo_ids = set(
            results_df.loc[results_df["answer_format"] == "fo_class", "qID"].astype(str)
        )
        df = df[df["qID"].astype(str).isin(fo_ids)]
        selection = "results_df.answer_format"
    else:
        selection = "gold parses as a legal FO set"

    df = df.copy()
    df["_gold_set"] = df["answer"].map(lambda a: read_fo_class(a, valid_lower))
    n_gold_illegal = int(df["_gold_set"].isna().sum())
    if results_df is None:
        df = df[df["_gold_set"].notna()]
    elif n_gold_illegal:
        # A gold the SDK's own reader rejects is a data defect, not a model result: scoring
        # it would price the model against a target it can never legally emit (RULES §8b).
        raise ValueError(
            f"{n_gold_illegal} rows labelled answer_format=='fo_class' carry a gold that "
            f"FOClass.read rejects against FOType.names()={valid_names}"
        )
    df["_pred_set"] = df[pcol].map(lambda a: read_fo_class(a, valid_lower))
    df["distribution"] = df["qID"].map(_dist_from_qid)

    def _cell(sub: pd.DataFrame) -> dict:
        rows = list(zip(sub["_gold_set"], sub["_pred_set"]))
        macro, per = _macro_f1(_f1_counts(rows))
        out = {
            "n": int(len(sub)),
            "n_illegal": int(sub["_pred_set"].isna().sum()),
            "n_classes_supported": int(sum(1 for v in per.values() if v["n_gold"] > 0)),
            "macro_f1": macro,
            "ci_low": float("nan"), "ci_high": float("nan"),
            "n_videos": 0,
            "exact_set_acc": (
                float(np.mean([g == (p if p is not None else frozenset()) for g, p in rows]))
                if rows else float("nan")
            ),
            "per_class": per,
        }
        if not rows or "video" not in sub.columns:
            if rows:
                logger.info(
                    "class_f1_report: no `video` column — macro_f1 reported without a CI "
                    "(an unclustered CI over ~38 videos would be far too narrow; RULES §13)."
                )
            return out
        work = sub.copy()
        work["_vkey"] = work.apply(_video_key, axis=1)
        groups = [
            list(zip(d["_gold_set"], d["_pred_set"])) for _, d in work.groupby("_vkey")
        ]
        out["n_videos"] = len(groups)
        if len(groups) < 3:
            return out
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), size=len(groups))
            resampled = [r for i in pick for r in groups[i]]
            m, _unused = _macro_f1(_f1_counts(resampled))
            if not math.isnan(m):
                boots.append(m)
        if boots:
            out["ci_low"] = float(np.percentile(boots, 2.5))
            out["ci_high"] = float(np.percentile(boots, 97.5))
        return out

    report = {"pooled": _cell(df)}
    for dist in ("ID", "OOD"):
        report[dist] = _cell(df[df["distribution"] == dist])
    report["_meta"] = {
        "selection": selection,
        "valid_names": list(valid_names),
        "n_gold_illegal": n_gold_illegal,
    }
    logger.info(
        "class_f1_report [%s]: pooled macro_f1=%.4f (n=%d, %d illegal, exact=%.4f) | "
        "ID %.4f | OOD %.4f",
        selection, report["pooled"]["macro_f1"], report["pooled"]["n"],
        report["pooled"]["n_illegal"], report["pooled"]["exact_set_acc"],
        report["ID"]["macro_f1"], report["OOD"]["macro_f1"],
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


# ── jackknife over videos — how much does a bucket estimate rest on one video? ────


def jackknife_by_video(
    results_df: pd.DataFrame, *, group: str | None = None, distribution: str | None = None
) -> dict:
    """Leave-one-VIDEO-out spread of a bucket's accuracy, from answers that already exist.

    🔴 **Why this measurement exists.** ``acc_OOD`` is half the challenge score and rests on
    **10 videos** against 28 for ID, and every rung selects its checkpoint by
    ``idxmax(acc_ood)`` over those same 10 and then reports on a set containing them
    ([[local-eval-vs-judge-calibration]]). If dropping ONE video moves the estimate by more
    than the deltas the ladder has been calling results, then no OOD comparison in the
    campaign meant anything. This answers that for the cost of re-reading a CSV.

    ⚠️ **This is NOT** ``split.kfold_lopo``, which holds out a whole procedure and costs one
    TRAINING RUN per fold. Nothing is retrained here; it is a spread of the estimator over
    the videos it happens to have.

    ``results_df`` is a scored ``results.csv``: needs ``video``, ``correctness`` and — when
    filtering — ``primary`` and ``ood``.

    Returns ``full``, ``lo``/``hi`` (min/max over the leave-one-out estimates), ``spread``
    (hi − lo), ``worst_video`` (the one whose removal moves it most) and ``n_videos``.
    """
    df = results_df
    if group is not None:
        # RULES EVAL §2: leaf→group ALWAYS through the canonical mapper, never a
        # hand-rolled prefix rule.
        assert "primary" in df.columns, "group= needs the `primary` column"
        df = df[df["primary"].map(_leaf_to_group) == group]
    if distribution is not None:
        want = distribution.upper() == "OOD"
        df = df[df["ood"].astype(bool) == want]
    if df.empty or "video" not in df.columns:
        return {"full": float("nan"), "n_videos": 0}

    corr = df["correctness"].astype(float)
    full = float(corr.mean())
    loo = {
        v: float(corr[df["video"] != v].mean())
        for v in df["video"].unique()
        if (df["video"] != v).any()
    }
    if not loo:
        return {"full": full, "n_videos": 1}
    lo, hi = min(loo.values()), max(loo.values())
    worst = max(loo, key=lambda v: abs(loo[v] - full))
    return {
        "full": full,
        "lo": lo,
        "hi": hi,
        "spread": hi - lo,
        "max_shift": abs(loo[worst] - full),
        "worst_video": worst,
        "n_videos": len(loo),
        "n": int(len(df)),
    }


# ── the leaderboard proxy (lifted from rung 21, 2026-08-14) ──────────────────
def leaderboard_proxy(report: dict) -> dict:
    """The ID-only leaderboard proxy: mean of the two ID buckets we are scored on.

    🔴 This is NOT a key of ``stratified_report`` and never has been. It was a
    notebook-local ``_proxy()`` helper in ``21b_epoch_eval.ipynb``, so every consumer
    that reached for ``report["proxy_leaderboard"]`` silently got ``None`` — rung 40's
    eval declared it as its PRIMARY_CELL and would have returned a NULL verdict on both
    arms after 16 h of training. Lifted here so it is computed once, canonically, per
    RULES §EVAL rule 1 (extend the module, never reimplement beside it).

    Purely additive: ``stratified_report``'s own output is untouched, so no archived
    number changes.

    ID-only on purpose — `decisions/local-eval-vs-judge-calibration.md`: the proxy is
    the least misleading local number we own *precisely because* it excludes OOD.

    Returns ``{"aggregation_ID", "object_recognition_ID", "proxy"}``. A missing bucket
    yields NaN rather than raising: that is expected in a smoke (a 40-row head-of-list
    sample need not contain a single aggregation×ID question) and is a FINDING in a
    full run — so the CALLER raises, with the count in hand.
    """
    bb = pd.DataFrame(report["by_bucket"])
    idc = bb[bb.distribution == "ID"].set_index("capability_group")
    out: dict = {}
    for key, group in (("aggregation_ID", "aggregation"),
                       ("object_recognition_ID", "object_recognition")):
        out[key] = float(idc.loc[group, "accuracy"]) if group in idc.index else float("nan")
    out["proxy"] = (out["aggregation_ID"] + out["object_recognition_ID"]) / 2
    return out
