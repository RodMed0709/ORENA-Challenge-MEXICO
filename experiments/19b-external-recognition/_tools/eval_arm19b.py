"""Rung 19b — score the external-data arm on the full 6,252, one row per epoch.

Importable library. NEVER a launcher — `01_eval_epochs_full6252.ipynb` runs it
(EXPERIMENT_REPO_STRUCTURE_SPEC; CLAUDE.md).

## What this file is, and what it deliberately is NOT

It is a **thin arm module**. Every piece of machinery that has already been paid for
lives in rung 45's `eval_arm45.py` (the eval, the cache gate, the paired CI) and rung
47's `eval_arm47.py` (the merge, the swift shim story, `cells`, `score_on_heldout`),
and is IMPORTED, never copied. What is redefined here is exactly what rung 47 pinned to
its own arm at module scope and cannot be parameterised from outside:

    RUN            "47_a2_ep5_v1"  ->  "19b_merged_ep5_v1"
    EPOCH_STEPS    901/epoch       ->  1,259/epoch   (20,133 rows, not 14,415)

`assert_checkpoint_complete` and `merge_checkpoint` are re-stated for that reason ONLY
— they close over `EPOCH_STEPS`. Their bodies are rung 47's, unchanged in behaviour.

## The sweep, and why all five epochs

Rung 47 scored ep3/ep4/ep5 because its question was `ep4 − ep3`. This arm's question is
the corpus, read at **ep4 against rung 47's ep4** (README), but the epoch curve itself
is now cheap and load-bearing: rung 42's whole advantage turned out to be epochs, not
corpus ([[rung42-gain-was-epochs-not-corpus]]), and rung 47's own curve is flat from
ep4 to ep5 (+0.0030, CI spans zero). A five-point curve says whether 5,718 external rows
move the *shape* of that curve or only its level. ep1/ep2 have no rung-47 control on the
6,252 — they are within-arm trajectory and are reported as such, never paired across arms.

## The comparison that is legal, and the one that is not

**Legal:** `19b_epN − 47_epN` on the 6,252, paired and video-clustered. Both arms trained
on ZERO of those 38 videos (gate 1 below re-measures it), both annealed a cosine over
five epochs, both were merged and scored on this box through this same code path.

🔴 **Not legal: reading the 6,252 as if it still meant what it means for rung 47.** It is
the same 38 videos for both, but this arm has now met Strasbourg. The *paired delta* is
valid; what a raw 19b number means on its own is not the same statement. Rung 48's
`bag_f1` over the 15 `hold` videos is the CENTRE axis and is reported separately and
**never averaged with `bucket_mean`** — that averaging is the dilution rung 48 v2 caught.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

from eval_arm45 import (  # noqa: E402
    BRIDGE_EVAL,
    PRIMARY_EVAL,
    EvalConfig,
    EvalFailure,
    arm_results_csv,
    assert_cache_covers,
    ensure_paths,
    score,
)
from eval_arm47 import (  # noqa: E402
    assert_run_moved_weights,
    cells,
    reclaim_merged,
    resolve_ckpt_root,
    score_on_heldout,
    swift_cmd,
    _merged_model_dir,
)

# ──────────────────────────────────────────────────────────────────────────────
# the arm
# ──────────────────────────────────────────────────────────────────────────────

RUN = "19b_merged_ep5_v1"

# 20,133 rows / (1 × 16) = 1,258.3 optimiser steps per epoch, which the trainer rounds
# UP to 1,259 and then accumulates: 6,295 total. Confirmed against the four checkpoint
# directories on disk 2026-08-20 (1259 / 2518 / 3777 / 5036) and `gates.json`'s
# `steps_per_epoch: 1258.3125`. Do NOT derive this from row counts in a notebook.
EPOCH_STEPS = {1: 1259, 2: 2518, 3: 3777, 4: 5036, 5: 6295}

ALL_EPOCHS = (1, 2, 3, 4, 5)

# The epoch the rung exists for. ep4 is what the README reads against rung 47's ep4;
# everything else is the curve around it.
DECISIVE_EPOCH = 4

# 🔴 The single variable, from the run's own `diff_vs_r47.json` (written by the launcher):
#     {"--dataset": [".../rung47/corpus/train_mntpaths.jsonl",
#                    ".../rung48/corpus/train_merged.jsonl"]}
SINGLE_VARIABLE = {"--dataset"}

# The corpus this arm trained on, and the frame roots the two halves live under. The
# challenge half is in the shared `frames_cache`; the external half is rung 48's own
# `train_frames`. Two roots, and the legality gate needs both patterns.
CORPUS = "/mnt/storage/uaq_user/rung48/corpus/train_merged.jsonl"
N_ROWS_EXPECTED = 20133
N_CHOLECT50_ROWS = 5718
N_CHOLECT50_VIDEOS = 35


# ──────────────────────────────────────────────────────────────────────────────
# the control: rung 47, same eval set, already on disk
# ──────────────────────────────────────────────────────────────────────────────

# Rung 47's re-score on the SAME 6,252, from `RESULTS_rescore_full6252.csv`. Present so
# a transcription error is VISIBLE; every CI is recomputed from the per-question
# `results.csv` under `CONTROL_47_DIR`, never read off this table.
CONTROL_47 = {
    3: {"bucket_mean": 0.6249, "acc_ID": 0.6066, "acc_OOD": 0.6647,
        "object_recognition_ID": 0.7045, "object_recognition_OOD": 0.7313},
    4: {"bucket_mean": 0.6468, "acc_ID": 0.6283, "acc_OOD": 0.6900,
        "object_recognition_ID": 0.7369, "object_recognition_OOD": 0.7741},
    5: {"bucket_mean": 0.6498, "acc_ID": 0.6292, "acc_OOD": 0.6940,
        "object_recognition_ID": 0.7346, "object_recognition_OOD": 0.7704},
}

CONTROL_47_DIR = "/mnt/storage/uaq_user/rung47/runs/47_a2_ep5_v1/eval_full6252"
CONTROL_47_EPOCHS = (3, 4, 5)   # ep1/ep2 were never re-scored on the 6,252


def control_47_csv(epoch: int, root: str = CONTROL_47_DIR) -> Path:
    """Rung 47's per-question answers for one epoch on the 6,252. RAISES.

    ⚠️ These are the ONLY control that makes this rung readable, and they are not
    reproducible in an evening: regenerating one costs a merge + a full 6,252 pass.
    Check them before the GPU, not after.
    """
    p = Path(root) / f"47_a2_ep5_v1_ep{epoch}_bridge" / "results.csv"
    if not p.exists():
        raise EvalFailure(
            f"no rung-47 control for ep{epoch} at {p}. Only {CONTROL_47_EPOCHS} were "
            "re-scored on the 6,252; ep1/ep2 are within-arm trajectory and have none."
        )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Rung19bConfig:
    """Set inline in the notebook cell, never edited into this file."""

    ckpt_root: str = ""          # runs/19b_merged_ep5_v1/ckpt/v0-<stamp> — resolve by LOOKING
    work_root: str = "/mnt/storage/uaq_user/rung19b"
    repo_root: str = "/mnt/storage/uaq_user/repo_leo"
    data_root: str = "/mnt/storage/uaq_user/orena-data"
    frames_cache: str = "/mnt/storage/uaq_user/frames_cache"

    # 🔴 TWO HF caches on this box with DIFFERENT models. The judge (`Qwen/Qwen3-4B`) and
    # the Qwen3-VL-8B base are in the 80 GB one. Pointing at the other fails ~30 min in,
    # with the model already loaded ([[pod-hf-home-is-not-set]]).
    hf_home: str = "/mnt/storage/uaq_user/hf_cache"

    epochs: tuple[int, ...] = ALL_EPOCHS

    # 🔴 Held identical to rung 47, which is the control. `use_vllm=False` is the
    # conservative choice for the same reason it was there: the merge needs ms-swift
    # (`orena-train`), vLLM is only in `orena-vllm`, and the control's answers were
    # produced on the HF path. Switching engines would put a second variable into a
    # comparison whose whole point is that `--dataset` is the only one.
    use_vllm: bool = False
    max_pixels: int = 1280 * 720     # 921600 — the value that scored 06/21/38/40/42/47
    seed: int = 42
    n_boot: int = 4000

    keep_merged: bool = False        # a merged 8B is ~16 GB; reclaim it immediately
    smoke: bool = False
    smoke_n: int = 40

    corpus: str = CORPUS
    split_json: str = field(default="")

    base_model: str = ""

    def __post_init__(self) -> None:
        if not self.split_json:
            self.split_json = (
                f"{self.repo_root}/experiments/42-merged-corpus/RESULTS_split_42.json"
            )

    # --- derived --------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.work_root) / "runs" / RUN

    def merged_dir(self, epoch: int) -> Path:
        return self.run_dir / "merged" / f"checkpoint-{EPOCH_STEPS[epoch]}"

    @property
    def eval_root(self) -> Path:
        """🔴 A dedicated directory. `score()` reuses an existing results.csv, which is
        what makes the sweep incremental and is exactly how a 40-row SMOKE table gets
        scored as if it were the 6,252 (rung 47's scar). Full and smoke never share."""
        return self.run_dir / ("eval_full6252_smoke" if self.smoke else "eval_full6252")

    def run_name(self, epoch: int) -> str:
        return f"{RUN}_ep{epoch}_bridge"

    @property
    def storage_root(self) -> Path:
        return Path(self.work_root).parent

    def base_model_path(self) -> Path:
        """The base model, re-prefixed onto THIS storage root. RAISES if absent.

        `swift export` resolves the base from the ADAPTER's `args.json`, which records an
        absolute path. Passing `--model` explicitly wins over the loaded value. Derived
        from `argv.json` rather than hardcoded: the base is part of the arm's identity.
        """
        if self.base_model:
            p = Path(self.base_model)
        else:
            argv = json.loads((self.run_dir / "argv.json").read_text(encoding="utf-8"))
            recorded = argv[argv.index("--model") + 1]
            tail = recorded.split("/hf_cache/", 1)
            if len(tail) != 2:
                raise EvalFailure(f"cannot re-prefix the recorded --model: {recorded}")
            p = self.storage_root / "hf_cache" / tail[1]
        if not p.exists():
            raise EvalFailure(
                f"base model not found at {p}. It is derived from the run's argv.json and "
                f"re-prefixed onto {self.storage_root}; if the volume moved again, that "
                "root is the one parameter to update."
            )
        return p

    def to_eval_config(self, epoch: int) -> EvalConfig:
        """Rung 45's `EvalConfig`, filled for one epoch of this arm, on the 6,252.

        `eval_set = BRIDGE_EVAL` is what makes `score()` pass `video_filter = None`.
        """
        return EvalConfig(
            arm=f"19b_external_ep{epoch}",
            merged_dir=str(self.merged_dir(epoch)),
            out_dir=str(self.eval_root),
            run_name=self.run_name(epoch),
            eval_set=BRIDGE_EVAL,
            data_root=self.data_root,
            repo_root=self.repo_root,
            split_json=self.split_json,
            frames_cache=self.frames_cache,
            max_pixels=self.max_pixels,
            seed=self.seed,
            n_boot=self.n_boot,
            use_vllm=self.use_vllm,
            smoke=self.smoke,
            smoke_n=self.smoke_n,
        )


# ──────────────────────────────────────────────────────────────────────────────
# gates that fire BEFORE a GPU-second is spent (all RAISE — RULES §7)
# ──────────────────────────────────────────────────────────────────────────────


def assert_checkpoint_complete(ckpt_root: Path, epoch: int) -> Path:
    """The epoch's adapter is fully written. RAISES.

    Scoring a checkpoint the trainer is still flushing is a silent corruption:
    `safetensors` will happily load a truncated file it can parse. This matters more
    here than it did for rung 47, because this sweep is CHAINED off the end of training
    — ep5 is written minutes before it is read.
    """
    d = ckpt_root / f"checkpoint-{EPOCH_STEPS[epoch]}"
    adapter = d / "adapter_model.safetensors"
    if not adapter.exists():
        raise EvalFailure(
            f"epoch {epoch} is not on disk yet: no {adapter}. "
            f"Steps per epoch = 1,259; this one lands at step {EPOCH_STEPS[epoch]}."
        )
    if adapter.stat().st_size == 0:
        raise EvalFailure(f"{adapter} is zero bytes — the write is still in flight")
    return d


def assert_single_variable(run_dir: Path) -> dict:
    """The run's own `diff_vs_r47.json` still says `--dataset` is the ONLY difference.

    🔴 This gate could NOT be `rst.diff_vs_control` (README): that helper builds the
    baseline with `control_cfg`, which shares `--dataset` by construction, so on this arm
    it returns an empty diff and certifies as single-variable a run whose only variable it
    is structurally unable to see. The launcher wrote this file against rung 47's own argv.
    """
    p = run_dir / "diff_vs_r47.json"
    if not p.exists():
        raise EvalFailure(f"no diff_vs_r47.json at {p} — the single-variable claim is unbacked")
    diff = json.loads(p.read_text(encoding="utf-8"))
    if set(diff) != SINGLE_VARIABLE:
        raise EvalFailure(
            f"this arm differs from rung 47 in {sorted(diff)}, not just "
            f"{sorted(SINGLE_VARIABLE)}. The delta below would not be the corpus."
        )
    log.info("single-variable gate OK — %s", diff)
    return diff


def corpus_videos(corpus: str = CORPUS) -> dict:
    """The videos this arm actually trained on, parsed from the corpus it was handed.

    Two frame roots, so two patterns — and that split is itself the check. The challenge
    half must resolve under the shared `frames_cache` with `<dataset>__<frame>.jpg`
    identity keys; the external half under rung 48's `train_frames` as
    `cholect50__VIDnn__nnnnnn.jpg`. A row matching NEITHER is not a row we can attribute
    to a video, and the caller must treat that as a failure, not a rounding error.
    """
    import re

    pat_cache = re.compile(r"/frames_cache/([A-Za-z0-9_]+)__(\d+)\.jpg")
    pat_ext = re.compile(r"/train_frames/cholect50__(VID\d+)__(\d+)\.jpg")

    challenge, external, n_rows, unattributed = set(), set(), 0, 0
    with open(corpus, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            n_rows += 1
            a = pat_cache.findall(ln)
            b = pat_ext.findall(ln)
            if not a and not b:
                unattributed += 1
            challenge.update(v for v, _ in a)
            external.update(v for v, _ in b)
    return {
        "n_rows": n_rows,
        "challenge_videos": challenge,
        "external_videos": external,
        "unattributed_rows": unattributed,
    }


def assert_corpus_legality(corpus: str, eval_keys: set, train_keys: set,
                           split_csv: str) -> dict:
    """The corpus touched NONE of the 38 eval videos, and NONE of the 15 `hold`. RAISES.

    Two separate promises, and they protect two different instruments:

    * **0 of the 38** protects `bucket_mean` on the 6,252. A leak here reports leakage as
      a result, and the paired delta against rung 47 becomes uninterpretable.
    * **0 of the 15 `hold`** protects rung 48's `bag_f1`. 🔴 The moment a `hold` video
      enters training, Strasbourg is inside the distribution and rung 48 stops measuring
      **centre** and starts measuring **unseen scene** (README, gate G3).

    The launcher measured both before training. They are re-measured here because the
    thing being scored is a different artifact from the thing that was gated, and a gate
    that only ever runs once is a claim.
    """
    import csv

    got = corpus_videos(corpus)
    if got["unattributed_rows"]:
        raise EvalFailure(
            f"{got['unattributed_rows']} corpus rows match neither frame root — their "
            "video cannot be attributed, so neither leak check covers them."
        )
    if got["n_rows"] != N_ROWS_EXPECTED:
        raise EvalFailure(f"corpus is {got['n_rows']} rows, expected {N_ROWS_EXPECTED}")

    leak = got["challenge_videos"] & eval_keys
    if leak:
        raise EvalFailure(
            f"CONTAMINATED: the corpus contains {len(leak)} of the 38 eval videos: "
            f"{sorted(leak)[:5]} — the 6,252 would report leakage as a result."
        )
    if got["challenge_videos"] != train_keys:
        raise EvalFailure(
            f"the challenge half is {len(got['challenge_videos'])} videos, the manifest's "
            f"`train` split is {len(train_keys)}. Provenance vs rung 47 is not established."
        )

    rows = list(csv.DictReader(open(split_csv, encoding="utf-8")))
    hold = {r["video_id"] for r in rows if r["split"] == "hold"}
    allowed = {r["video_id"] for r in rows if r["split"] == "train"}
    leaked_hold = got["external_videos"] & hold
    if leaked_hold:
        raise EvalFailure(
            f"CONTAMINATED: {len(leaked_hold)} held-out CholecT50 videos are in training: "
            f"{sorted(leaked_hold)} — rung 48's bag_f1 stops measuring centre."
        )
    stray = got["external_videos"] - allowed
    if stray:
        raise EvalFailure(f"external videos outside the frozen train split: {sorted(stray)}")
    if len(got["external_videos"]) != N_CHOLECT50_VIDEOS:
        raise EvalFailure(
            f"{len(got['external_videos'])} CholecT50 videos, expected {N_CHOLECT50_VIDEOS}"
        )

    out = {
        "n_rows": got["n_rows"],
        "challenge_videos": len(got["challenge_videos"]),
        "external_videos": len(got["external_videos"]),
        "eval_videos_touched": 0,
        "hold_videos_touched": 0,
    }
    log.info("corpus legality gate OK — %s", out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# the merge (rung 47's, re-stated because it closes over EPOCH_STEPS)
# ──────────────────────────────────────────────────────────────────────────────


def merge_checkpoint(cfg: Rung19bConfig, ckpt_root: Path, epoch: int) -> Path:
    """`swift export --merge_lora` one epoch's adapter into a loadable model.

    `swift_cmd()` is imported from rung 47 and goes through `sys.executable` — never the
    `swift` console script, whose baked shebang points at a path that has not existed
    since the 2026-08-18 reboot and reports as `FileNotFoundError: 'swift'`.
    """
    ckpt = assert_checkpoint_complete(ckpt_root, epoch)
    out = cfg.merged_dir(epoch)
    if out.is_dir() and any(out.iterdir()):
        log.info("epoch %d already merged -> %s", epoch, out)
        return _merged_model_dir(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    p = subprocess.run(
        swift_cmd() + ["export", "--adapters", str(ckpt), "--merge_lora", "true",
                       "--model", str(cfg.base_model_path()),
                       "--output_dir", str(out)],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        log.error("swift export stdout:\n%s", p.stdout[-3000:])
        log.error("swift export stderr:\n%s", p.stderr[-3000:])
        raise EvalFailure(f"swift export failed rc={p.returncode} for epoch {epoch}")
    log.info("merged epoch %d in %.0fs -> %s", epoch, time.perf_counter() - t0, out)
    return _merged_model_dir(out)


# ──────────────────────────────────────────────────────────────────────────────
# one epoch, end to end
# ──────────────────────────────────────────────────────────────────────────────


def answer_epoch(cfg: Rung19bConfig, ckpt_root: Path, epoch: int, split: dict,
                 n_expected: int) -> Path:
    """Merge -> answer the 6,252 -> reclaim the 16 GB. Returns the per-question CSV.

    Reuses an existing `results.csv` when one is on disk, so a sweep that dies at ep4
    resumes at ep4 and costs zero GPU for ep1–ep3. 🔴 That reuse is also how a SMOKE
    table gets scored as the full set, so the row count is CHECKED, not assumed — the
    separate `eval_root` per mode makes it hard, this makes it impossible.
    """
    ec = cfg.to_eval_config(epoch)
    existing = Path(ec.out_dir) / ec.run_name / "results.csv"
    if existing.exists():
        n_rows = sum(1 for _ in open(existing, encoding="utf-8")) - 1
        if not cfg.smoke and n_rows != n_expected:
            raise EvalFailure(
                f"{existing} has {n_rows} rows, not {n_expected}. It is not this eval's "
                f"table — delete {ec.out_dir} rather than scoring it."
            )
        log.info("ep%d already answered -> %s (%d rows, no GPU)", epoch, existing, n_rows)
        return existing

    ec.merged_dir = str(merge_checkpoint(cfg, ckpt_root, epoch))
    try:
        score(ec, split)
    finally:
        # Always, even on a failed eval: 16 GB per epoch on a SHARED box (CONSTITUTION
        # §IX). The adapter reproduces the merge in minutes; the answers are the artifact.
        reclaim_merged(cfg, epoch)
    return arm_results_csv(ec)
