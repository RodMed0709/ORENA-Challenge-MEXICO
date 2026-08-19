"""FRAME train / validation split + leak-guard.

**Only two roles: train + validation.** There is NO local test set — the real test
is the leaderboard submission, so reserving a local test would only waste data we
could train on. Validation exists solely to select checkpoints (epochs / recipe)
without fooling ourselves; keep it small to maximize training data. For the FINAL
submission model, fold validation back into train and retrain on everything.

The split key is ``(dataset, video_id)`` — NEVER a question or a frame. One video
yields many FRAME items (``frame_index = round(start_time * base_fps)``), so
splitting on rows leaks frames of the same surgery into both train and eval.

The two validation slices give the two numbers the challenge scores on (acc-ID vs
acc-OOD, OOD ≈ half the weight):

- ``val_id``  — held-out *videos* of procedure_types that ARE in train
                (new video, same domain) → in-distribution generalization gauge.
- ``val_ood`` — every video of one **whole held-out ``procedure_type``**
                (HeiCo Stage-3 domain gap) → OOD generalization gauge. NOTE: holding
                out ONE procedure_type (not a whole dataset) keeps the other surgery
                types IN train, so the model still learns cross-procedure — critical,
                since OOD is half the leaderboard score.

The written manifest CSV is the frozen **source of truth**: every downstream
experiment reloads it (``load_manifest``) and never re-derives the partition.

Grounding: reads only the public ``FrameItem`` fields produced by
``frame.data.load_frame_items`` — ``item.dataset``, ``item.video_id``,
``item.request.procedure_type``, ``item.reference.primary`` (a
``focus.taxonomy.Capability`` whose ``.group`` gives the scored capability group).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# (dataset, video_id) — the only valid split key. heico & lapchole raw video ids
# can collide, so the dataset must be part of the key.
VideoKey = tuple[str, str]

_SPLITS = ("train", "val_id", "val_ood", "test_id")
# v2 adds "test_id": a THIRD slice, touched once at the end, so "val_id" can be spent on
# checkpoint/epoch selection without selecting on the set we report. "val_ood" keeps its
# name on purpose — metrics._distribution, delta.py and ledger.py all hardwire the string,
# and renaming it would silently label every video "ID". It means UNSEEN PROCEDURE
# (Sigmoid), which is NOT the platform's OOD axis (centre); see decisions/split-v2-by-video.


@dataclass
class SplitConfig:
    """How to carve the FRAME items into train / val_id / val_ood (no local test)."""

    ood_procedure: str  # the WHOLE procedure_type held out as the OOD validation slice (required)
    ood_dataset: str | None = "heico"  # scope the holdout to one dataset (None = any)
    val_frac: float = 0.15  # fraction of the remaining VIDEOS held out as val_id
    seed: int = 42
    stratify_by: tuple[str, ...] = ("dataset", "procedure_type")
    manifest_path: Path = field(
        default_factory=lambda: Path("experiments/splits/frame_ood_v1.csv")
    )


# ── item field accessors (single place, so a schema change is one edit) ──────

def _key(item) -> VideoKey:
    return (item.dataset, item.video_id)


def _procedure(item) -> str:
    return item.request.procedure_type


def _group(item) -> str:
    """Scored capability GROUP name for this item (one of the 5)."""
    primary = item.reference.primary
    grp = getattr(primary, "group", primary)
    return getattr(grp, "value", str(grp))


def _answer_format(item) -> str:
    # ._format is a required field on Reference (data.py always populates it).
    return str(getattr(item.reference, "_format", "unknown"))


def _file(item) -> str:
    return getattr(item, "file", "test")


def manifest_hash(video_split: dict[VideoKey, str]) -> str:
    """Deterministic SHA-256 of the partition (order-independent)."""
    blob = "\n".join(f"{ds}\t{vid}\t{s}" for (ds, vid), s in sorted(video_split.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── inventory ────────────────────────────────────────────────────────────────

def videos_table(items: list) -> pd.DataFrame:
    """One row per ``(dataset, video_id)``: procedure_type + question counts.

    Call this FIRST in the notebook to see the available ``procedure_type`` values
    before choosing which one to hold out as OOD.
    """
    rows: dict[VideoKey, dict] = {}
    for it in items:
        k = _key(it)
        r = rows.setdefault(
            k,
            {"dataset": k[0], "video_id": k[1], "procedure_type": _procedure(it), "n_questions": 0},
        )
        r["n_questions"] += 1
    df = pd.DataFrame(rows.values())
    return df.sort_values(["dataset", "procedure_type", "video_id"]).reset_index(drop=True)


# ── build ────────────────────────────────────────────────────────────────────

def build_split(items: list, cfg: SplitConfig) -> dict[VideoKey, str]:
    """Deterministic ``(dataset, video_id) -> split`` map.

    All videos of ``cfg.ood_procedure`` (optionally scoped to ``cfg.ood_dataset``)
    go to ``val_ood``; the remaining videos are seeded-shuffled per stratum and
    ~``val_frac`` land in ``val_id``, the rest in ``train``.
    """
    vids = videos_table(items)

    # validate the requested OOD procedure exists, else fail loudly with options.
    procs = sorted(vids["procedure_type"].unique())
    if cfg.ood_procedure not in procs:
        raise ValueError(
            f"ood_procedure {cfg.ood_procedure!r} not found. Available procedure_types: {procs}"
        )

    rng = np.random.default_rng(cfg.seed)
    split: dict[VideoKey, str] = {}

    is_ood = vids["procedure_type"] == cfg.ood_procedure
    if cfg.ood_dataset is not None:
        is_ood &= vids["dataset"] == cfg.ood_dataset
    ood_vids = vids[is_ood]
    if ood_vids.empty:
        raise ValueError(
            f"No videos matched ood_procedure={cfg.ood_procedure!r} / ood_dataset={cfg.ood_dataset!r}."
        )
    for _, row in ood_vids.iterrows():
        split[(row["dataset"], row["video_id"])] = "val_ood"

    remaining = vids[~vids.index.isin(ood_vids.index)]
    # stratified per (dataset, procedure_type): every domain represented in val_id.
    for _, grp in remaining.groupby(list(cfg.stratify_by)):
        keys = [(r["dataset"], r["video_id"]) for _, r in grp.iterrows()]
        order = rng.permutation(len(keys))
        n_val = max(1, round(len(keys) * cfg.val_frac)) if len(keys) > 1 else 0
        for rank, idx in enumerate(order):
            split[keys[idx]] = "val_id" if rank < n_val else "train"

    assert_no_leak(split, cfg)
    counts = {s: sum(1 for v in split.values() if v == s) for s in _SPLITS}
    logger.info("Split built (videos): %s", counts)
    return split


def build_official_split(items: list, ood_dataset: str = "heico") -> dict[VideoKey, str]:
    """Use the organizers' OWN train/test partition as our train/validation.

    The FOCUS FRAME files already encode an OOD design: for ``heico`` the held-out
    procedure (Sigmoid Resection) lives ONLY in ``test.parquet`` (train has
    Proctocolectomy + Rectal), while ``lapchole`` is cholecystectomy in both. So:

    - ``train``   — every video from a ``train`` file (procto + rectal + chole-train).
    - ``val_ood`` — ``test``-file videos of ``ood_dataset`` (heico = the unseen Sigmoid
                    procedure) → the OOD gauge, faithful to the real leaderboard's OOD.
    - ``val_id``  — ``test``-file videos of the other datasets (chole, seen in train,
                    different videos) → the ID gauge.

    No local test set (real test = leaderboard). For the FINAL submission model, fold
    validation back into train and retrain on everything. Requires items loaded with
    ``load_frame_items(cfg, splits=("train", "test"))`` so ``item.file`` is populated.
    """
    split: dict[VideoKey, str] = {}
    for it in items:
        k = _key(it)
        if _file(it) == "train":
            split[k] = "train"
        elif it.dataset == ood_dataset:
            split[k] = "val_ood"
        else:
            split[k] = "val_id"
    assert_no_leak(split)
    counts = {s: sum(1 for v in split.values() if v == s) for s in _SPLITS}
    logger.info("Official split (videos): %s", counts)
    return split


# ── persistence (the manifest IS the source of truth) ────────────────────────

def build_split_v2(
    v1_manifest: Path | str,
    *,
    out_path: Path | str,
    seed: int = 42,
    ood_procedure: str = "Sigmoid Resection",
    fracs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build the v2 four-slice manifest FROM the v1 manifest — no dataset, no GPU.

    v1 already lists every video with its ``procedure_type`` and ``n_questions``, which
    is the whole input a video-level split needs. Building from it (rather than from
    ``items``) means the partition can be regenerated on any machine, including ones
    with no challenge data.

    Why the slices are what they are:

    - **Split by video count, not by question mass.** ``frame.metrics`` clusters its
      bootstrap on video, so the CI width is set by the number of CLUSTERS. heico videos
      carry 400 questions and lapchole ~80 — a 5:1 ratio — so a "60/20/20 of questions"
      would put 3 heico videos against 50 lapchole ones in the same slice. Question mass
      is used only to break ties.
    - **``test_id`` is new**, and exists so ``val_id`` can be spent on epoch/checkpoint
      selection without selecting on the set we report.
    - **``ood_procedure`` leaves whole**, before the stratified pass touches anything.
    - 🔴 **``val_ood`` keeps its name for compatibility, not for accuracy.**
      ``metrics._distribution``, ``delta.py`` and ``ledger.py`` all hardwire the literal
      string; renaming it would label every video "ID" in silence. It means UNSEEN
      PROCEDURE. The platform's OOD axis is CENTRE (">5 centres not represented"), and
      the two disagree — our Sigmoid cell read 0.4086 locally against 0.6064 on the
      platform. Never report this slice as a proxy for the platform's OOD half.
    """
    fracs = fracs or {"train": 0.60, "val_id": 0.20, "test_id": 0.20}
    v1 = pd.read_csv(v1_manifest, dtype={"dataset": str, "video_id": str})
    assign: dict[VideoKey, str] = {}

    ood = v1[v1.procedure_type == ood_procedure]
    if ood.empty:
        raise ValueError(f"no video has procedure_type={ood_procedure!r} in {v1_manifest}")
    for _, r in ood.iterrows():
        assign[(r.dataset, r.video_id)] = "val_ood"

    rest = v1[v1.procedure_type != ood_procedure]
    for _, g in rest.groupby("procedure_type", sort=True):
        g = g.sample(frac=1.0, random_state=seed)
        n = len(g)
        quota = {k: int(round(n * f)) for k, f in fracs.items()}
        quota["train"] += n - sum(quota.values())
        got = {k: 0 for k in quota}
        mass = {k: 0 for k in quota}
        for _, r in g.sort_values("n_questions", ascending=False).iterrows():
            cand = [k for k in quota if got[k] < quota[k]]
            tgt = {k: fracs[k] / sum(fracs[c] for c in cand) for k in cand}
            tot = sum(mass[k] for k in cand) + r.n_questions
            pick = min(cand, key=lambda k: (mass[k] / tot) - tgt[k])
            assign[(r.dataset, r.video_id)] = pick
            got[pick] += 1
            mass[pick] += r.n_questions

    # ── guards: these are the whole point of freezing a manifest ──────────────
    assert len(assign) == len(v1), f"{len(assign)} assigned vs {len(v1)} videos"
    train = {k for k, s in assign.items() if s == "train"}
    evals = {k for k, s in assign.items() if s != "train"}
    assert not (train & evals), f"VIDEO LEAK: {sorted(train & evals)}"
    ood_keys = {(r.dataset, r.video_id) for _, r in ood.iterrows()}
    assert all(assign[k] == "val_ood" for k in ood_keys)
    assert not any(s == "val_ood" for k, s in assign.items() if k not in ood_keys), \
        "val_ood must be EXACTLY the unseen procedure"

    out = pd.DataFrame([
        {
            "dataset": r.dataset, "video_id": r.video_id,
            "procedure_type": r.procedure_type,
            "split": assign[(r.dataset, r.video_id)],
            "n_questions": r.n_questions, "seed": seed,
            "ood_procedure": ood_procedure, "ood_dataset": "heico",
            "val_frac": fracs["val_id"],
            "ood_axis": "procedure",  # NOT the platform's axis, which is centre
        }
        for _, r in v1.iterrows()
    ]).sort_values(["dataset", "video_id"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    digest = manifest_hash(assign)
    Path(str(out_path) + ".sha256").write_text(digest + "\n")
    logger.info("Wrote v2 manifest: %s (%d videos, sha256=%s…)", out_path, len(out), digest[:12])
    return out


def write_manifest(split: dict[VideoKey, str], vids: pd.DataFrame, cfg: SplitConfig) -> Path:
    """Write the frozen split manifest CSV + a ``.sha256`` sidecar. One row per video.

    Every config field is persisted (full snapshot), and the sidecar hash lets
    ``load_manifest(..., verify=True)`` reject a hand-edited / corrupt manifest.
    """
    q_by_key = {(r["dataset"], r["video_id"]): r["n_questions"] for _, r in vids.iterrows()}
    proc_by_key = {(r["dataset"], r["video_id"]): r["procedure_type"] for _, r in vids.iterrows()}
    rows = [
        {
            "dataset": k[0],
            "video_id": k[1],
            "procedure_type": proc_by_key.get(k, ""),
            "split": s,
            "n_questions": q_by_key.get(k, 0),
            "seed": cfg.seed,
            "ood_procedure": cfg.ood_procedure,
            "ood_dataset": cfg.ood_dataset if cfg.ood_dataset is not None else "",
            "val_frac": cfg.val_frac,
        }
        for k, s in sorted(split.items())
    ]
    out = pd.DataFrame(rows)
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.manifest_path, index=False)
    digest = manifest_hash(split)
    Path(str(cfg.manifest_path) + ".sha256").write_text(digest + "\n")
    logger.info("Wrote manifest: %s (%d videos, sha256=%s…)", cfg.manifest_path, len(out), digest[:12])
    return cfg.manifest_path


def load_manifest(path: Path | str, verify: bool = True) -> dict[VideoKey, str]:
    """Reload the frozen split so every experiment reads the SAME partition.

    ``video_id`` is forced to ``str`` (numeric-looking ids must not become int64, or
    key lookups against ``FrameItem`` string ids would silently miss → empty splits).
    ``verify=True`` recomputes the SHA-256 and asserts it matches the ``.sha256``
    sidecar (rejects a hand-edited / Excel-mangled manifest).
    """
    df = pd.read_csv(path, dtype={"dataset": str, "video_id": str, "split": str})
    video_split = {(r["dataset"], r["video_id"]): r["split"] for _, r in df.iterrows()}
    if verify:
        sidecar = Path(str(path) + ".sha256")
        if sidecar.exists():
            want = sidecar.read_text().strip()
            got = manifest_hash(video_split)
            assert got == want, f"MANIFEST HASH MISMATCH: {path} edited/corrupt (got {got[:12]}…, want {want[:12]}…)."
        else:
            logger.warning("No .sha256 sidecar for %s — cannot verify integrity.", path)
    return video_split


def assert_all_matched(items: list, video_split: dict[VideoKey, str]) -> None:
    """Fail loudly if any item's (dataset, video_id) is absent from the split map.

    Guards the silent-drop failure mode: a key-type mismatch (see load_manifest) would
    otherwise just shrink every split with no error. Call after build/load, before use.
    """
    missing = {_key(it) for it in items} - set(video_split.keys())
    assert not missing, f"{len(missing)} item videos unmatched to any split, e.g. {sorted(missing)[:5]}"


# ── use ──────────────────────────────────────────────────────────────────────

def apply_split(items: list, video_split: dict[VideoKey, str], want: str) -> list:
    """Filter items to one split. ``want`` ∈ {'train', 'val_id', 'val_ood'}."""
    if want not in _SPLITS:
        raise ValueError(f"want must be one of {_SPLITS}, got {want!r}.")
    return [it for it in items if video_split.get(_key(it)) == want]


# ── guards ───────────────────────────────────────────────────────────────────

def assert_no_leak(video_split: dict[VideoKey, str], cfg: SplitConfig | None = None) -> None:
    """No video may sit in both train and any eval split (leak guard).

    ``cfg`` is accepted for call-site symmetry but not required: the whole-OOD-
    procedure invariant is enforced at build time (only ``ood_procedure`` videos
    are ever assigned ``val_ood``, before the stratified pass touches the rest).
    """
    train = {k for k, s in video_split.items() if s == "train"}
    evals = {k for k, s in video_split.items() if s != "train"}
    overlap = train & evals
    assert not overlap, f"VIDEO LEAK: {sorted(overlap)} in train AND an eval split."
    logger.info(
        "Leak check passed: train ∩ eval = ∅ (%d train, %d eval videos).",
        len(train), len(evals),
    )


def per_bucket_report(items: list, video_split: dict[VideoKey, str]) -> pd.DataFrame:
    """Coverage of the 10 scored buckets (capability_group × {ID, OOD}) per split.

    ID = train ∪ val_id (trained distribution); OOD = val_ood. Use this to CONFIRM
    the split populates all 5 groups on both sides — the baseline populated only 3.
    """
    recs = []
    for it in items:
        split = video_split.get(_key(it))
        if split is None:
            continue
        recs.append(
            {
                "capability_group": _group(it),
                "distribution": "OOD" if split == "val_ood" else "ID",
                "answer_format": _answer_format(it),
                "split": split,
            }
        )
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    cov = (
        df.groupby(["capability_group", "distribution"])
        .size()
        .reset_index(name="n_questions")
        .sort_values(["capability_group", "distribution"])
        .reset_index(drop=True)
    )
    return cov


def split_summary(items: list, video_split: dict[VideoKey, str]) -> pd.DataFrame:
    """The 80/20 table: videos + questions + % per split (train / val_id / val_ood)."""
    seen: dict[str, set] = {s: set() for s in _SPLITS}
    q: dict[str, int] = {s: 0 for s in _SPLITS}
    for it in items:
        s = video_split.get(_key(it))
        if s is None:
            continue
        seen[s].add(_key(it))
        q[s] += 1
    total_q = sum(q.values()) or 1
    rows = [
        {"split": s, "n_videos": len(seen[s]), "n_questions": q[s],
         "pct_questions": round(100 * q[s] / total_q, 1)}
        for s in _SPLITS
    ]
    return pd.DataFrame(rows)


# ── k-fold (Leave-One-Procedure-Out) — the FINAL certain OOD estimate ─────────

def kfold_lopo(items: list, ood_dataset: str = "heico", seed: int = 42):
    """Leave-One-Procedure-Out cross-validation over ``ood_dataset``'s procedures.

    Yields ``(procedure_name, video_split)``: each fold holds out ONE whole
    procedure_type as ``val_ood`` and trains on EVERYTHING else. No data is ever
    permanently wasted — each video is ``val_ood`` in exactly one fold and ``train``
    in the others — which is exactly why this fits our small (200-video) set.

    ``ood_dataset`` (default heico) is the domain we rotate; the dominant lapchole
    (single procedure, 170 videos) is NEVER held out whole (that would strand training
    on 30 heico videos). Reserve this for the final OOD readout — it costs one training
    run per fold, so it does not belong in the dev iteration loop.
    """
    vids = videos_table(items)
    procs = sorted(vids[vids["dataset"] == ood_dataset]["procedure_type"].unique())
    if not procs:
        raise ValueError(f"No procedures found for ood_dataset={ood_dataset!r}.")
    for proc in procs:
        split: dict[VideoKey, str] = {}
        for _, r in vids.iterrows():
            k = (r["dataset"], r["video_id"])
            split[k] = "val_ood" if (r["dataset"] == ood_dataset and r["procedure_type"] == proc) else "train"
        assert_no_leak(split)
        logger.info("LOPO fold: OOD=%s (%d val_ood videos)", proc,
                    sum(1 for v in split.values() if v == "val_ood"))
        yield proc, split
