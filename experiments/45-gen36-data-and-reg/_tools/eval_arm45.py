"""Rung 45 — score R00 / R0 / R1 on the held-out videos, in rung 42's shape.

Importable library. NEVER a launcher — `01_eval_heldout.ipynb` runs it.

Written from two files that already exist, deliberately, because both already paid
for their mistakes:

* **rung 40's `_tools/eval_arm.py`** — how to run a gen-3.6 27B through
  `frame.run.run_baseline` at all. Its four measured scars are carried, not
  re-discovered: `results.csv` lives under `out_dir/run_name/`; `BaselineConfig`
  wants `Path`, not `str`; `GenericVLMEngine` must RAISE on import rather than
  silently fall back to a `QwenFrameEngine` that cannot load this model class; and
  `proxy_leaderboard` is not a key `run_baseline` emits.
* **rung 42's `01_eval_heldout.ipynb`** — the CELL SHAPE. Rung 45 does not invent an
  eval; it re-runs rung 42's on new arms.

🔴 **RULES §EVAL is binding and this file does not relax it:** score ONLY through
`frame.metrics`; leaf→group ALWAYS via `Capability.group` (`metrics._leaf_to_group`);
ID/OOD ALWAYS from the qID prefix; the paired CI is **video-clustered**.


## The asymmetry this rung has and rung 42 did not

Rung 45 scores three arms trained on **two different corpora**, and they are not
contaminated the same way:

| arm | corpus | test videos in its training set | eval set that is CLEAN for it |
|---|---|---|---|
| R00 | rung 18, 14 415 | **0 of 38** | all 6 252 |
| R0  | rung 42 merged, 19 384 | **30 of 38** | the 1 283 only |
| R1  | rung 42 merged, 19 384 | **30 of 38** | the 1 283 only |

⇒ **Primary for all three: the 1 283** (`PRIMARY_EVAL`). It is the only set clean for
every arm, so it is the only set on which R0 − R00 is a fair paired comparison, and
R00 is the control (same backbone, same recipe, one variable: the corpus).

⇒ **The 6 252 is R00-only** (`BRIDGE_EVAL`), and it is a bridge, never a verdict: R00
shares rung 18's corpus with rung 40, so R00 − `40_B_connector_v1` prices NF4 against
bf16 — a number nobody in this campaign has. `assert_eval_set_is_legal` RAISES if it
is ever pointed at R0 or R1, because for those arms that number is leakage.

🔴 **`*_OOD` does not mean the same thing for R00 as for R0/R1, on the SAME questions.**
For R00, heico is a genuinely unseen procedure. For R0/R1, rung 42's merge promoted 8
of the 10 heico test videos into training, so heico is an unseen VIDEO of a SEEN
procedure. The two remaining heico videos are the whole OOD side. Consequence, stated
here so the write-up cannot quietly overclaim: an R0 − R00 win on an `*_OOD` cell is
partly the procedure moving from OOD to ID, and is NOT evidence that more data
generalises better. The unconfounded read of that claim is the `*_ID` cells alone.
`ood_procedure_in_train` is written per row so the table carries the distinction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# The 1 283 questions across rung 42's 8 held-out videos. Clean for all three arms:
# outside rung 18's corpus AND outside the merged corpus.
PRIMARY_EVAL = "8 held-out videos (1283 q)"
# The full test set. Legal for R00 alone — see the module docstring.
BRIDGE_EVAL = "full test set (6252 q)"

N_HELD_EXPECTED = 1283
N_HELD_VIDEOS = 8
N_HEICO_HELD_VIDEOS = 2

# Arms whose corpus promoted test videos into training. Scoring these on the 6 252
# reports leakage as a result. Keyed by arm name as `run_arm.py` writes it.
CONTAMINATED_ARMS = frozenset({"R0", "R1"})

# rung 40's winning arm, bf16, on the FULL 6 252 — the bridge R00 is read against.
#
# ⚠️ **The bridge carries one confound the primary does not, and it is ours, not NF4's.**
# Rung 40 was scored through `FrameProvider` (decord → raw array). R00 is scored through
# `CachedFrameProvider` (decord → JPEG q95 → decode). Same frame, same resolution — the
# cache stores the raw decord output with no resize (`qualitative.py`), and its varied
# sizes (960×540, 1280×720, 720×576 …) are the videos' native ones, verified 2026-08-17 —
# but one lossy round-trip apart. On the PRIMARY eval this cancels: R00, R0 and R1 all
# read the same cache. On the BRIDGE it does NOT, so `R00 − 40_B_connector_v1` prices
# NF4 **plus** a JPEG round-trip, and that is a ceiling on how finely it can be read.
# From experiments/40-gen36-recipe-connector/RESULTS.csv. Present so a transcription
# error is VISIBLE; the paired CI is always recomputed from the archived per-question
# answers, never read off this table.
BRIDGE_CONTROL = {
    "run": "40_B_connector_v1",
    "precision": "bf16",
    "epoch": 1,
    "proxy_leaderboard": 0.5260,
    "bucket_mean": 0.5763,
    "aggregation_ID": 0.4262,
    "object_recognition_ID": 0.6258,
    # 🔴 Its per-question archive is NOT on UNAM (`find ~/storage -name results.csv`
    # returns nothing, measured 2026-08-17). It is in S3, bucket `gf78k60nlt`, at
    # repo_leo/experiments/40-gen36-recipe-connector/runs/40_B_connector_v1/eval/
    # 40_B_connector_v1/results.csv (752 KB). Without it the bridge degrades to two
    # scalars with no CI, so fetch it BEFORE the eval, not after.
    "results_csv_s3": (
        "s3://gf78k60nlt/repo_leo/experiments/40-gen36-recipe-connector/runs/"
        "40_B_connector_v1/eval/40_B_connector_v1/results.csv"
    ),
}


class EvalFailure(AssertionError):
    """A guard that fires is a FINDING (RULES §7)."""


@dataclass
class EvalConfig:
    """Set inline in the notebook cell, never edited into this file."""

    arm: str = "R00"                       # R00 | R0 | R1
    merged_dir: str = ""                   # what `run_arm.py` wrote to runs/<run>/merged
    out_dir: str = ""
    run_name: str = "45_R00_v1_eval"
    eval_set: str = PRIMARY_EVAL           # PRIMARY_EVAL | BRIDGE_EVAL

    # 🔴 UNAM, not the pod. Rung 40's defaults are RunPod paths (`/workspace/...`) and
    # do not exist here. Measured 2026-08-17.
    data_root: str = "/home/uaq_user/storage/orena-data"
    # 🔴 MEASURED 2026-08-17: this is the ONLY checkout on UNAM that carries both
    # `vendor/orena-focus` (the SDK is NOT pip-installed in any env there —
    # `import focus` raises ModuleNotFoundError in orena-gen36) and rung 23's
    # `_tools/screen_engine.py`, which `score()` needs for GenericVLMEngine. It is a
    # symlink to /data/uaq_user/repo and it is SHARED — its `src/frame` predates
    # `CachedFrameProvider`, so it must be updated before this module can run.
    repo_root: str = "/home/uaq_user/storage/repo"
    split_json: str = ""                   # rung 42's RESULTS_split_42.json

    # 🔴 UNAM has the QA parquets and every test frame, and ZERO `.mp4`. Without this
    # the decord path raises on the first frame, after the model has loaded.
    frames_cache: str = "/home/uaq_user/storage/frames_cache"

    # 🔴 Identical to the eval that scored rungs 38, 40 and 42. The inference path is
    # not allowed to be a second variable: if it moves, the delta stops being the arm.
    max_pixels: int = 921600
    seed: int = 42
    n_boot: int = 4000

    # 🔴 vLLM, not HF, and it is forced rather than preferred. MEASURED on UNAM
    # 2026-08-17: HF materialises the FP8 build at 43.32 GiB of a 47.37 GiB card and
    # every generation OOMs. vLLM loads the same checkpoint at 33.46 GiB with ~14 GiB
    # spare, at 0.454 s/q against HF's 1.13, scoring an identical 30/50 (rung 44). See
    # `vllm_batch.py`. Held identical across all three arms, so it cancels in their
    # difference; against rung 40's HF control it is a declared bridge confound.
    use_vllm: bool = True

    smoke: bool = False
    smoke_n: int = 40


# ──────────────────────────────────────────────────────────────────────────────
# the split, and the gates that must fire before a GPU-second is spent
# ──────────────────────────────────────────────────────────────────────────────


def ensure_paths(repo_root: str) -> None:
    """Put the three source trees the eval needs on sys.path. Importable ON PURPOSE.

    🔴 `focus` is NOT installed in the gen-3.6 env — it is VENDORED in the repo. Rung
    40 lost a cycle "fixing" that by switching to an env that has focus installed;
    that env pins transformers 4.57, which cannot load
    `Qwen3_5ForConditionalGeneration` at all (`AutoConfig` raises `KeyError: 'qwen3_5'`)
    before a weight is read. The eval runs in the SAME env as training; the missing
    piece was never the environment, it was these three sys.path entries.
    """
    import sys

    for p in (f"{repo_root}/src",
              f"{repo_root}/vendor/orena-focus/src",                  # the vendored SDK
              f"{repo_root}/experiments/23-backbone-screen/_tools",   # GenericVLMEngine
              f"{repo_root}/experiments/43-thinking-at-inference/_tools",  # vllm_engine
              f"{repo_root}/experiments/45-gen36-data-and-reg/_tools"):    # vllm_batch
        if p not in sys.path:
            sys.path.insert(0, p)


def assert_eval_set_is_legal(cfg: EvalConfig) -> None:
    """R0 and R1 may not be scored on the 6 252. RAISES.

    Their corpus promoted 30 of the 38 test videos into training, so that number is
    leakage wearing a headline's clothes. This is the single most quotable wrong
    number this rung can produce, which is why it is a gate and not a footnote.
    """
    if cfg.eval_set == BRIDGE_EVAL and cfg.arm in CONTAMINATED_ARMS:
        raise EvalFailure(
            f"arm {cfg.arm!r} trained on 30 of the 38 test videos (rung 42's merge), so "
            f"scoring it on {BRIDGE_EVAL!r} reports LEAKAGE as a result. Its only legal "
            f"eval set is {PRIMARY_EVAL!r}."
        )
    if cfg.eval_set not in (PRIMARY_EVAL, BRIDGE_EVAL):
        raise EvalFailure(f"unknown eval_set {cfg.eval_set!r}")


def held_out_split(cfg: EvalConfig) -> dict:
    """Rung 42's 8 held-out videos, with its own gates re-run. RAISES.

    Every assert here is rung 42's, kept because each one catches a different way of
    silently scoring the wrong questions — a leak between the two sides, a video that
    is in neither, and a held-out count that has drifted from 1 283.
    """
    ensure_paths(cfg.repo_root)
    from frame.config import BaselineConfig
    from frame.data import load_frame_items

    split = json.loads(Path(cfg.split_json).read_text(encoding="utf-8"))
    held = {tuple(v.split("/", 1)) for v in split["videos_held_out"]}
    promoted = {tuple(v.split("/", 1)) for v in split["videos_promoted"]}
    if held & promoted:
        raise EvalFailure(f"LEAK: {held & promoted} is both held out and promoted")

    items = load_frame_items(BaselineConfig(data_root=Path(cfg.data_root)))
    allv = {(i.dataset, i.video_id) for i in items}
    if not (held <= allv and promoted <= allv):
        raise EvalFailure("a split video is not in the test set")
    if (held | promoted) != allv:
        raise EvalFailure("held + promoted does not cover the test set")

    held_items = [i for i in items if (i.dataset, i.video_id) in held]
    n_heico_v = len({v for v in held if v[0] == "heico"})
    if len(held_items) != N_HELD_EXPECTED or len(held) != N_HELD_VIDEOS:
        raise EvalFailure(
            f"expected {N_HELD_EXPECTED} questions across {N_HELD_VIDEOS} videos, got "
            f"{len(held_items)} across {len(held)}"
        )
    if n_heico_v != N_HEICO_HELD_VIDEOS:
        raise EvalFailure(f"expected {N_HEICO_HELD_VIDEOS} heico videos, got {n_heico_v}")

    return {
        "held_videos": held,
        "held_items": held_items,
        "held_qids": {i.request.qID for i in held_items},
        "all_items": items,
        "n_held": len(held_items),
        "n_heico_videos": n_heico_v,
    }


def assert_cache_covers(cfg: EvalConfig, items) -> dict:
    """Every frame this eval will ask for is already in frames_cache. RAISES.

    🔴 Runs BEFORE the GPU, on purpose. `CachedFrameProvider` raises on a miss rather
    than falling back, which is correct but expensive: the miss surfaces after the
    model has loaded and part of the eval is already paid for. This walks the same
    identity keys with `os.path.exists` and costs seconds.

    Measured 2026-08-17 on UNAM: 6252/6252 test frames and 1283/1283 held-out frames
    present, so a miss here means the cache moved, not that this is a new risk.
    """
    ensure_paths(cfg.repo_root)
    from frame.data import frame_cache_name

    root = Path(cfg.frames_cache)
    if not root.is_dir():
        raise EvalFailure(f"frames_cache is not a directory: {root}")

    missing = [frame_cache_name(it) for it in items if not (root / frame_cache_name(it)).exists()]
    if missing:
        raise EvalFailure(
            f"{len(missing)} of {len(items)} frames are NOT in {root} "
            f"(e.g. {missing[:3]}). The eval would raise mid-run with the model loaded. "
            "Populate the cache first — it is the single-source store (CLAUDE.md), so "
            "the fix is to add the frames there, never to copy them per experiment."
        )
    log.info("cache gate OK — %d/%d frames present in %s", len(items), len(items), root)
    return {"n_frames_checked": len(items), "frames_cache": str(root)}


# ──────────────────────────────────────────────────────────────────────────────
# scoring
# ──────────────────────────────────────────────────────────────────────────────


def build_baseline_config(cfg: EvalConfig):
    """Build the `BaselineConfig` the eval will actually run. Importable ON PURPOSE.

    🔴 `Path()`, not `str()`. `BaselineConfig` declares data_root/model_path/out_dir as
    `Path` and the code RELIES on it: `frame/data.py` does
    `cfg.data_root / ds / "data" / "frame" / ...`, which raises `TypeError` on a str.
    Rung 40 measured this on the pod, at the eval's very first data access, after
    training.
    """
    ensure_paths(cfg.repo_root)
    from frame.config import BaselineConfig

    bc = BaselineConfig(
        data_root=Path(cfg.data_root),
        model_path=Path(cfg.merged_dir),
        out_dir=Path(cfg.out_dir),
        run_name=cfg.run_name,
        max_pixels=cfg.max_pixels,
        seed=cfg.seed,
        n_eval=cfg.smoke_n if cfg.smoke else None,
        frames_cache=Path(cfg.frames_cache),
    )
    if cfg.use_vllm:
        from vllm_batch import make_batch_infer

        bc.batch_infer = make_batch_infer()
    return bc


def assert_inference_path_unmoved(cfg_eval) -> None:
    """The four asserts rungs 38/40 ran, kept verbatim. RAISES.

    The corpus and the regularisation are the subjects of this rung; the inference
    path may not be a second variable. These are the same four checks, not a
    paraphrase of them.
    """
    if cfg_eval.answer_postprocess is not None:
        raise EvalFailure("answer_postprocess must stay None — it would be a second variable")
    if cfg_eval.n_samples != 1:
        raise EvalFailure(f"n_samples is {cfg_eval.n_samples}, must be 1 (greedy, single sample)")
    if cfg_eval.enhance is not None:
        raise EvalFailure("enhance must stay None")
    if cfg_eval.aux_view is not None:
        raise EvalFailure("aux_view must stay None")
    log.info("inference path OK — unmoved from the eval that scored rungs 38, 40 and 42")


def arm_results_csv(cfg: EvalConfig) -> Path:
    """Where `run_baseline` ACTUALLY writes the per-question table. RAISES if absent.

    🔴 Rung 40 computed this inline as `out_dir/results.csv`, but `run_baseline` writes
    into `out_dir/run_name/`, so its paired CI died on FileNotFoundError AFTER 3.3 h of
    training and a 2 h eval that had already produced its scores. Derived here rather
    than in the notebook so it is importable, testable, and stated once.
    """
    p = Path(cfg.out_dir) / cfg.run_name / "results.csv"
    if p.exists():
        return p
    legacy = Path(cfg.out_dir) / "results.csv"
    if legacy.exists():
        log.warning("results.csv found at %s, not %s — using it", legacy, p)
        return legacy
    raise EvalFailure(
        f"no results.csv for this arm. Looked in {p} and {legacy}. The eval produced no "
        "per-question table, so the paired CI cannot be formed."
    )


def score(cfg: EvalConfig, split: dict) -> dict:
    """Run the eval on this arm's legal question set and return the canonical report.

    `enable_thinking=False` is handled inside the engine and is verified correct: the
    training format (`<think>\\n\\n</think>\\n\\n` + answer) is a byte-exact prefix of
    what inference hands the model with that flag. Without it the block stays OPEN and
    `max_new_tokens` truncates mid-reasoning — that is what scored rung 23 a 0.0000.
    """
    assert_eval_set_is_legal(cfg)
    ensure_paths(cfg.repo_root)
    from frame.run import run_baseline

    if not Path(cfg.merged_dir).exists():
        raise EvalFailure(f"no merged checkpoint at {cfg.merged_dir}")

    cfg_eval = build_baseline_config(cfg)
    assert_inference_path_unmoved(cfg_eval)

    # 🔴 The ONE deviation rung 38 had to make, and it is forced, not chosen:
    # `Qwen3_5ForConditionalGeneration` does not load under the transformers 4.57 pin,
    # so GenericVLMEngine is used. It keeps the SYSTEM_PROMPT (imported, never copied),
    # the message shape, greedy decoding, max_new_tokens and answer_char_cap identical.
    #
    # 🔴 It RAISES on a missing import and must keep raising. Rung 40 had this inside a
    # try/except that fell back to QwenFrameEngine — a class that cannot load this
    # model at all — so the fallback ALWAYS fired and the warning scrolled past in a
    # 16 h log. A forced deviation that silently un-forces itself is not a fallback, it
    # is a way to score nothing and call it a result.
    # On the vLLM path the engine is never constructed — `run_baseline` takes the
    # batch_infer branch — so importing GenericVLMEngine there would only assert that a
    # class we do not use is importable.
    if not cfg.use_vllm:
        from screen_engine import GenericVLMEngine

        cfg_eval.engine_factory = GenericVLMEngine

    video_filter = split["held_videos"] if cfg.eval_set == PRIMARY_EVAL else None
    report = run_baseline(cfg_eval, video_filter=video_filter)

    # 🔴 `proxy_leaderboard` is NOT a key `run_baseline` emits — it never was, at any n.
    # Derived canonically, never re-implemented in a notebook.
    from frame.metrics import leaderboard_proxy

    report.update(leaderboard_proxy(report))
    report["proxy_leaderboard"] = report.pop("proxy")
    log.info("eval done — %s", {k: report.get(k) for k in ("proxy_leaderboard", "bucket_mean")})
    return report


# ──────────────────────────────────────────────────────────────────────────────
# the paired CI — rung 42's cell shape, unchanged
# ──────────────────────────────────────────────────────────────────────────────


def paired_cells(arm_res, control_res, *, n_boot: int = 4000, seed: int = 42):
    """Video-clustered paired CI of (arm − control), in rung 42's 6-cell shape.

    Cells are `{aggregation, object_recognition, ALL} × {ID, OOD}`.

    🔴 **`ALL` is never pooled across the ID/OOD boundary**, and that is rung 42's
    choice, not a simplification of it. The OOD side of the 1 283 rests on **2 videos**;
    pooling it with the 6 ID videos drags one CI over two populations with an effective
    n of 8 and makes the readable half unreadable. Rung 42's own numbers are the
    evidence that the split shape works: on `ALL_ID` (n=6 videos) it got
    +0.0705 [+0.0206, +0.1246], excluding zero, while every `*_OOD` cell (n=2) included
    it. Expect the same here — an OOD cell failing to exclude zero is the design, not a
    finding.

    ⇒ **Only the `*_ID` cells can GRANT.** `*_OOD` cells are computed and printed so a
    VETO can still be seen (RULES §S8: any cell may veto, only a pre-declared cell may
    grant), never so a win can be claimed there. And see the module docstring: an OOD
    win for R0 over R00 is confounded with the procedure moving into its training set.

    Effective n is VIDEOS, not questions (RULES §13). An unclustered CI would be ~10×
    too narrow and would manufacture significance.
    """
    import pandas as pd

    from frame import metrics

    a = control_res[["qID", "video", "correctness", "primary"]].rename(
        columns={"correctness": "correct_a"})
    b = arm_res[["qID", "correctness"]].rename(columns={"correctness": "correct_b"})
    j = a.merge(b, on="qID", how="inner")
    if len(j) != len(arm_res) or len(j) != len(control_res):
        raise EvalFailure(
            f"arms are not scored on the same questions (joined {len(j)}, arm "
            f"{len(arm_res)}, control {len(control_res)}). An inner join here would "
            "shrink the denominator without saying so."
        )

    # RULES §2: leaf → group ALWAYS via Capability.group. RULES §3: ID/OOD from the qID
    # PREFIX, via the canonical helper — never from a results column, never re-derived.
    metrics.assert_ood_from_qid(j)
    j["group"] = j["primary"].map(metrics._leaf_to_group)
    j["dist"] = j["qID"].map(metrics.dist_from_qid)

    rows = []
    for grp in sorted(j["group"].dropna().unique()) + ["ALL"]:
        for dist in ("ID", "OOD"):
            sub = j[j["dist"] == dist]
            sub = sub if grp == "ALL" else sub[sub["group"] == grp]
            rows.append({"cell": f"{grp}_{dist}",
                         **metrics.paired_delta_ci(sub, n_boot=n_boot, seed=seed)})
    ci = pd.DataFrame(rows)
    ci["excludes_zero"] = (ci.ci_low > 0) | (ci.ci_high < 0)
    # `wins_a`/`wins_b` come back from paired_delta_ci — the paired win counts a sign
    # test reads. At 8 video clusters they are the honest secondary read, and they are
    # native output, not something this rung invented.
    return ci


def verdict(ci, *, granting_cells=("ALL_ID",)) -> dict:
    """Read the arms against the pre-registered condition. Never re-cut afterwards.

    Multiplicity is asymmetric: ANY cell may VETO, only a declared cell may GRANT.

    🔑 A positive point estimate whose CI includes zero is a **NULL**, not weak
    evidence. Below |Δ| = 0.01 nothing is readable at all (RULES §S4); acting needs
    |Δ| ≈ 0.03 and is a team call (RULES §S1).

    `granting_cells` defaults to `ALL_ID` because PLAN §6 asks for "ALL and ID", and in
    rung 42's shape there is no pooled `ALL` — `ALL_ID` is that bar. Stated as an
    argument so the choice is visible in the notebook rather than buried here.
    """
    won = ci[(ci.excludes_zero) & (ci.delta > 0)]["cell"].tolist()
    vetoed = ci[(ci.excludes_zero) & (ci.delta < 0)]["cell"].tolist()
    granted = [c for c in granting_cells if c in won]
    return {
        "granting_cells": list(granting_cells),
        "cells_won": won,
        "cells_vetoed": vetoed,
        "verdict": "HARM" if vetoed else ("WIN" if granted else "NULL"),
        "note": (
            "A CI that includes zero is a NULL, not weak evidence. |delta| < 0.01 is "
            "unreadable; acting needs |delta| >= 0.03 and is a team call. *_OOD cells "
            "rest on 2 videos and cannot grant."
        ),
    }


def write(cfg: EvalConfig, payload: dict) -> str:
    p = Path(cfg.out_dir) / "RESULTS_eval45.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("eval result -> %s", p)
    return str(p)
