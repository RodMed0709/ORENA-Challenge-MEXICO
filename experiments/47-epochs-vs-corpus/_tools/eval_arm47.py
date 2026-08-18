"""Rung 47 — score the `C_epochs` arm (A2's corpus, 5 epochs) on rung 42's held-out 1,283.

Importable library. NEVER a launcher — `01_eval_epochs.ipynb` runs it
(EXPERIMENT_REPO_STRUCTURE_SPEC; CLAUDE.md).

## The question, and why one epoch is not enough to answer it

Rung 42 beat A2 by **+0.0402** and that number bought a submission. It is a bundle:
its arm changed the corpus AND ran two epochs the control never ran. At the matched
epoch its corpus **loses 0.0079**; the whole gain sits at epoch 4. Rung 47 runs A2's
own corpus to 5 epochs, so `ep4 − ep4` has the corpus as its only difference.

⇒ **ep3 and ep4 are not two results, they are one experiment.** ep3 is the control
that says the measurement is readable at all; ep4 is the answer. Scoring ep3 alone
produces a number with nothing to compare it to, which is why the selection, the
paired CI and `RESULTS.csv` in the notebook are all gated on the full sweep.

## 🔴 The ep3 control carries THREE differences, not one

`AHORA.md` and `NOW.md` describe ep3 as "the stack control: must reproduce A2's
0.6342". That is right, but understating what "the stack" now means. A2's 0.6342 was
produced on a RunPod pod; rung 47 is trained and scored on UNAM. Between them:

| what moved | A2 (the 0.6342) | rung 47 |
|---|---|---|
| `transformers` | 4.57.x | **5.12.1** (`orena-train`) |
| frames | `FrameProvider` — decord straight off the `.mp4` | **`CachedFrameProvider`** — decord → JPEG q95 → decode |
| GPU | A100/L40S (pod) | **RTX 6000 Ada** (UNAM GPU 1) |

UNAM holds every test frame and **zero `.mp4`** (measured 2026-08-17), so the cache is
forced, not chosen — the same forced move rung 45 declared for its bridge. The three
confounds are therefore JOINT: a green ep3 bounds all of them at once and licenses
everything downstream; a red ep3 does **not** say which one moved.

## 🟢 The read that does not depend on any of that: difference-in-differences

`47_ep4 − 42_ep4` compares across the two stacks and inherits all three confounds.
But the **epoch effect inside each arm** is measured within one stack:

    rung 42, its own corpus:  ep4 − ep3  =  0.6744 − 0.6262  =  +0.0482
    rung 47, A2's corpus:     ep4 − ep3  =  (this rung)

Both differences are internal, so the stack, the JPEG round-trip and the GPU cancel in
each of them. If rung 47's ep3→ep4 rise is also ≈ +0.048, the epoch effect is
corpus-independent and rung 42's advantage at ep4 was **epochs**. This is the
primary read; `47_ep4 − 42_ep4` is reported beside it and is the one that needs ep3
to have come back green.

## What is on disk, and what had to be fetched

* The arm's checkpoints — UNAM, `~/storage/rung47/runs/47_a2_ep5_v1/ckpt/v0-*/`.
* **Rung 42's per-question answers, all five epochs — S3, `evidence_42/ep*_full/`.**
  Fetched 2026-08-18 and verified (1,283 rows, 800 heico) — without them there is no
  paired CI against the thing this rung exists to test, so they are checked BEFORE the
  GPU, not after. ⚠️ Two traps, both measured: **UNAM has no boto3 and no S3
  credentials**, so the fetch happens on the laptop and the files are rsynced to
  `runs/47_a2_ep5_v1/controls/`; and **`download_file` 403s** on this bucket because
  our credentials are denied `HeadObject` while `get_object` and `ListObjects` are
  allowed — the object can be seen and not downloaded by the obvious call.
* **A2's per-question answers do NOT exist any more.** They lived on the pod at
  `/workspace/repo_rodri/experiments/21-recipe-sweep/runs/21_lr_2e4_v1/ep3_full/`.
  Searched 2026-08-18: UNAM has no `results.csv` outside rung 45's, and the S3 prefixes
  `repo_rodri/ repo_leo/ repo_yyy/ repo_rung40/ tmp/ evidence_*/ submission/` carry zero
  keys matching `21_lr_2e4` across 14,000+. Consequence, stated here so
  the notebook cannot quietly overclaim: **the ep3 control is a SCALAR check against
  0.6342, with no paired CI.** `CONTROL_A2_EP3` carries the archived cells so a
  transcription error is visible.

🔴 **RULES §EVAL is binding and this file does not relax it:** score ONLY through
`frame.metrics`; leaf→group ALWAYS via `Capability.group` (`metrics._leaf_to_group`);
ID/OOD ALWAYS from the qID prefix; the paired CI is **video-clustered** (n = 8 videos,
not 1,283 questions — RULES §13).
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

# Rung 45's eval is imported, never copied. It already carries the four scars rung 40
# paid for (results.csv under out_dir/run_name; BaselineConfig wants Path; the engine
# must RAISE not fall back; proxy_leaderboard is derived) and rung 42's cell shape.
# `ensure_paths` must run before this import resolves — the notebook does it.
from eval_arm45 import (  # noqa: E402
    PRIMARY_EVAL,
    EvalConfig,
    EvalFailure,
    arm_results_csv,
    assert_cache_covers,
    ensure_paths,
    held_out_split,
    score,
)

# ──────────────────────────────────────────────────────────────────────────────
# the arm
# ──────────────────────────────────────────────────────────────────────────────

RUN = "47_a2_ep5_v1"

# `--save_strategy epoch` over 4,505 optimiser steps: 901 per epoch, exactly.
# ep1/ep2 confirmed on disk 2026-08-18; the rest are the same arithmetic.
EPOCH_STEPS = {1: 901, 2: 1802, 3: 2703, 4: 3604, 5: 4505}

# The two epochs the rung exists for. ep1/ep2/ep5 are trajectory and are scored only
# when they are free (their checkpoints are already merged) — never at the cost of
# delaying ep4.
DECISIVE_EPOCHS = (3, 4)

# 🔴 The single variable, from the run's own `diff_vs_A2.json` (read 2026-08-18):
#     {"--num_train_epochs": ["3", "5"]}
# Nothing else differs from A2 (`21_lr_2e4_v1`). If this file ever disagrees with that
# JSON, the JSON wins — it was written by the launcher, this is a transcription.
SINGLE_VARIABLE = {"--num_train_epochs": ("3", "5")}


# ──────────────────────────────────────────────────────────────────────────────
# the controls
# ──────────────────────────────────────────────────────────────────────────────

# Rung 42, merged corpus (19,384 rows), on the SAME 1,283. From
# `experiments/42-merged-corpus/RESULTS.csv` and its README. Present so a
# transcription error is VISIBLE; every CI is recomputed from the archived
# per-question answers below, never read off this table.
CONTROL_42 = {
    "run": "42_merged_v1",
    "corpus_rows": 19384,
    "bucket_mean": {1: 0.5649, 2: 0.6242, 3: 0.6262, 4: 0.6744, 5: 0.6592},
    "selected_epoch": 4,
    # 🔴 Its per-question archive is NOT on UNAM and NOT in either repo checkout in
    # S3 (`runs/` is gitignored, so it never left the pod that way). It IS at this
    # prefix, all five epochs, verified 2026-08-18 — 155 KB each.
    "archive_s3_bucket": "gf78k60nlt",
    "archive_s3_prefix": "evidence_42/{tag}/results.csv",   # tag = ep{N}_full
}

# A2 (`21_lr_2e4_v1`), 3 epochs, the recipe rung 47 re-runs unchanged. From
# `context/decisions/merged-corpus-buys-the-id-half.md` and rung 42's README.
#
# 🔴 SCALARS ONLY — the per-question archive is gone (see the module docstring), so
# the ep3 control is a scalar check and cannot produce a paired CI.
CONTROL_A2_EP3 = {
    "run": "21_lr_2e4_v1",
    "corpus_rows": 14415,
    "epoch": 3,
    "checkpoint": "checkpoint-2703",
    "bucket_mean": 0.6342,
    "acc_ID": 0.6646,
    "acc_OOD": 0.5988,
    "object_recognition_ID": 0.8670,
    "has_per_question_archive": False,
}

# RULES §S4: below |d| = 0.01 nothing is readable. The ep3 control is called GREEN
# inside this band and RED outside it — declared here, before the number exists.
CONTROL_TOLERANCE = 0.01

# Rung 42's own internal epoch effect, the right-hand side of the DiD. Derived from
# CONTROL_42, not typed twice.
D42_EP4_MINUS_EP3 = CONTROL_42["bucket_mean"][4] - CONTROL_42["bucket_mean"][3]


@dataclass
class Rung47Config:
    """Set inline in the notebook cell, never edited into this file."""

    # --- where the arm lives (UNAM, not the pod) -----------------------------
    ckpt_root: str = ""          # runs/47_a2_ep5_v1/ckpt/v0-<stamp>  — resolve by LOOKING
    work_root: str = "/home/uaq_user/storage/rung47"
    repo_root: str = "/home/uaq_user/storage/repo_leo"
    data_root: str = "/home/uaq_user/storage/orena-data"
    frames_cache: str = "/home/uaq_user/storage/frames_cache"

    # 🔴 TWO HF caches exist on this box with DIFFERENT models (12 GB under
    # `~/.cache/huggingface`, 80 GB under `~/storage/hf_cache`). The judge
    # (`Qwen/Qwen3-4B`) and the Qwen3-VL-8B base are in the 80 GB one; pointing at
    # the other fails ~30 min in, with the model already loaded. Verified offline
    # 2026-08-18: the judge gate passes from here.
    hf_home: str = "/home/uaq_user/storage/hf_cache"

    # --- the epochs to score, in order ---------------------------------------
    epochs: tuple[int, ...] = DECISIVE_EPOCHS

    # --- the inference path: HELD IDENTICAL TO RUNG 42 -----------------------
    # 🔴 `use_vllm=False` is a CHOICE and it is the conservative one. vLLM is only
    # in `orena-vllm` (0.27.1); the merge needs ms-swift, which is only in
    # `orena-train`. Splitting the pipeline across two envs would buy speed and add
    # an engine to a comparison whose control archive (rung 42) was produced on the
    # HF path. An 8B on 1,283 questions is not where this rung's time goes.
    use_vllm: bool = False
    max_pixels: int = 1280 * 720     # 921600 — the value that scored rungs 06/21/38/40/42
    seed: int = 42
    n_boot: int = 4000

    keep_merged: bool = False        # a merged 8B is ~16 GB; reclaim it immediately
    smoke: bool = False
    smoke_n: int = 40

    split_json: str = field(default="")   # rung 42's RESULTS_split_42.json

    # The Qwen3-VL-8B base the adapter was trained on. Left empty it is DERIVED from the
    # run's own `argv.json` and re-prefixed onto this storage root — see `base_model_path`.
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

    def eval_dir(self, epoch: int) -> Path:
        return self.run_dir / "eval"

    def run_name(self, epoch: int) -> str:
        return f"{RUN}_ep{epoch}" + ("_smoke" if self.smoke else "_full")

    @property
    def storage_root(self) -> Path:
        """The volume root, whatever it is mounted at today (`work_root`'s parent)."""
        return Path(self.work_root).parent

    def base_model_path(self) -> Path:
        """The base model, re-prefixed onto THIS storage root. RAISES if absent.

        🔴 `swift export` resolves the base model from the ADAPTER's `args.json`, which
        records an absolute path — `/data/uaq_user/hf_cache/...` for this arm. After the
        2026-08-18 reboot that path does not exist and the merge dies with
        `ValueError: path: '...' not found` *before* touching a weight. Passing `--model`
        explicitly wins over the loaded value (`model` is in swift's `load_keys`, which
        only apply when the current value is None), so the fix is to supply it.

        Derived from `argv.json` rather than hardcoded: the base is part of the arm's
        identity, and a merge against a different snapshot is not this arm's checkpoint.
        """
        if self.base_model:
            p = Path(self.base_model)
        else:
            argv = json.loads((self.run_dir / "argv.json").read_text(encoding="utf-8"))
            recorded = argv[argv.index("--model") + 1]
            # re-prefix: /data/uaq_user/hf_cache/... -> <storage_root>/hf_cache/...
            tail = recorded.split("/hf_cache/", 1)
            if len(tail) != 2:
                raise EvalFailure(f"cannot re-prefix the recorded --model: {recorded}")
            p = self.storage_root / "hf_cache" / tail[1]
        if not p.exists():
            raise EvalFailure(
                f"base model not found at {p}. It is derived from the run's argv.json and "
                f"re-prefixed onto {self.storage_root}; if the volume moved again, that root "
                "is what needs updating (one `STORAGE` parameter in the notebook)."
            )
        return p

    def to_eval_config(self, epoch: int) -> EvalConfig:
        """Rung 45's `EvalConfig`, filled for one epoch of this arm."""
        return EvalConfig(
            arm=f"C_epochs_ep{epoch}",
            merged_dir=str(self.merged_dir(epoch)),
            out_dir=str(self.eval_dir(epoch)),
            run_name=self.run_name(epoch),
            eval_set=PRIMARY_EVAL,
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


def resolve_ckpt_root(cfg: Rung47Config) -> Path:
    """Find the single `v0-*` training directory by LOOKING. RAISES on 0 or >1.

    Hardcoding the stamp is how a re-launched run gets scored from the abandoned
    directory. There is exactly one here; if there are two, a human decides which.
    """
    if cfg.ckpt_root:
        p = Path(cfg.ckpt_root)
        if not p.is_dir():
            raise EvalFailure(f"ckpt_root does not exist: {p}")
        return p
    hits = sorted((cfg.run_dir / "ckpt").glob("v0-*"))
    if len(hits) != 1:
        raise EvalFailure(
            f"expected exactly one v0-* under {cfg.run_dir / 'ckpt'}, found {len(hits)}: "
            f"{[h.name for h in hits]}. Set ckpt_root explicitly."
        )
    return hits[0]


def assert_run_moved_weights(ckpt_root: Path) -> dict:
    """`grad_norm` never hit 0.0 and the loss fell. RAISES.

    🔴 rc=0 is NOT evidence. AdamW's decoupled weight decay moves every tensor at zero
    gradient, so a checkpoint diff cannot separate a real run from a no-op — only the
    grad_norm log can ([[rc-zero-is-not-evidence]]). Rung 42's cell 5, kept verbatim.

    ⚠️ `full.log` on this run is block-buffered and has been frozen at 44,854 bytes
    since launch. It is NOT the progress signal and this function does not read it.
    """
    log_path = ckpt_root / "logging.jsonl"
    if not log_path.exists():
        raise EvalFailure(f"no logging.jsonl at {log_path}")
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    g = [r["grad_norm"] for r in rows if isinstance(r.get("grad_norm"), (int, float))]
    losses = [r["loss"] for r in rows if isinstance(r.get("loss"), (int, float))]
    zero = sum(1 for v in g if v == 0.0)
    if not g or zero:
        raise EvalFailure(f"{zero} of {len(g)} logged steps had grad_norm 0.0 — no weight moved")
    if losses[-1] >= losses[0]:
        raise EvalFailure(f"loss did not fall ({losses[0]:.4f} -> {losses[-1]:.4f})")
    ta = [r["token_acc"] for r in rows if isinstance(r.get("token_acc"), (int, float))]
    out = {
        "n_steps_logged": len(rows),
        "grad_norm_first": g[0], "grad_norm_last": g[-1],
        "loss_first": losses[0], "loss_last": losses[-1],
        # TRAIN token_acc is NOT a result. It is here for the memorisation read only:
        # rung 42 rose to 0.980 while its held-out curve turned over at ep4.
        "token_acc_first": ta[0] if ta else None, "token_acc_last": ta[-1] if ta else None,
        "last_epoch_logged": rows[-1].get("epoch"),
    }
    log.info("train gate OK — %s", out)
    return out


def assert_checkpoint_complete(ckpt_root: Path, epoch: int) -> Path:
    """The epoch's adapter is fully written. RAISES.

    Scoring a checkpoint that the trainer is still flushing is a silent corruption:
    `safetensors` will happily load a truncated file it can parse.
    """
    d = ckpt_root / f"checkpoint-{EPOCH_STEPS[epoch]}"
    adapter = d / "adapter_model.safetensors"
    if not adapter.exists():
        raise EvalFailure(
            f"epoch {epoch} is not on disk yet: no {adapter}. "
            f"Steps per epoch = 901; this one lands at step {EPOCH_STEPS[epoch]}."
        )
    if adapter.stat().st_size == 0:
        raise EvalFailure(f"{adapter} is zero bytes — the write is still in flight")
    return d


def assert_single_variable(run_dir: Path) -> dict:
    """The run's own `diff_vs_A2.json` still says epochs are the ONLY difference. RAISES.

    Written by the launcher, so it is evidence rather than a claim. A rung whose
    single-variable promise broke must not be scored as if it held.
    """
    p = run_dir / "diff_vs_A2.json"
    if not p.exists():
        raise EvalFailure(f"no diff_vs_A2.json at {p} — the single-variable claim is unbacked")
    diff = json.loads(p.read_text(encoding="utf-8"))
    if set(diff) != set(SINGLE_VARIABLE):
        raise EvalFailure(
            f"this arm differs from A2 in {sorted(diff)}, not just {sorted(SINGLE_VARIABLE)}. "
            "The delta below would not be the epochs."
        )
    log.info("single-variable gate OK — %s", diff)
    return diff


# ──────────────────────────────────────────────────────────────────────────────
# the control archive (fetched BEFORE the GPU)
# ──────────────────────────────────────────────────────────────────────────────


CONTROL_42_ROWS = 1283          # the whole held-out set, one row per question
CONTROL_42_HEICO_ROWS = 800
CONTROL_42_COLUMNS = ("qID", "video", "correctness", "primary")


def ensure_control_42(dest_root: Path, epochs=(3, 4, 5)) -> dict[int, Path]:
    """Rung 42's per-question answers, on local disk and shaped right. RAISES.

    🔴 Runs before anything expensive, on purpose. Rung 45 learned this the
    expensive way: without the archive the comparison degrades to two scalars with
    no CI, and that is discovered after the eval, not before it.

    🔴 **UNAM has no `boto3` and no S3 credentials** (measured 2026-08-18 — the smoke
    died here, correctly, before the GPU). S3 is reachable from the LAPTOP, not the
    box, so on UNAM this function only VERIFIES; the files are pushed with the rsync
    named in the error. Where boto3 does exist it fetches, so one function serves both.

    ⚠️ `s3.download_file` does NOT work on this bucket: our credentials are denied
    `HeadObject` (403) and `download_file` heads before it gets. `get_object` is
    allowed. Listing is allowed too, which is why the object can be *seen* and not
    *downloaded* by the obvious call.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    bucket = CONTROL_42["archive_s3_bucket"]
    out: dict[int, Path] = {}
    for e in epochs:
        key = CONTROL_42["archive_s3_prefix"].format(tag=f"ep{e}_full")
        dest = dest_root / f"42_ep{e}_results.csv"
        if not dest.exists():
            try:
                import boto3
            except ModuleNotFoundError:
                raise EvalFailure(
                    f"{dest} is missing and boto3 is not installed here (UNAM has no S3 "
                    f"access). Push it from the laptop, which does:\n"
                    f"    rsync -av <scratch>/controls42/ "
                    f"UNAM:{dest_root.relative_to(Path.home()) if dest_root.is_relative_to(Path.home()) else dest_root}/\n"
                    f"  (source: s3://{bucket}/{key}, via get_object — NOT download_file)"
                ) from None
            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
            dest.write_bytes(body)
            log.info("fetched s3://%s/%s -> %s (%d bytes)", bucket, key, dest, len(body))
        _assert_control_42_shape(dest, e)
        out[e] = dest
    return out


def _assert_control_42_shape(path: Path, epoch: int) -> None:
    """The archive is rung 42's whole held-out table, not a truncated copy. RAISES.

    A short or mis-columned control does not fail the paired CI — it silently makes
    it a comparison over whichever questions happened to survive the join.
    """
    import pandas as pd

    if path.stat().st_size == 0:
        raise EvalFailure(f"{path} is zero bytes")
    df = pd.read_csv(path)
    missing = [c for c in CONTROL_42_COLUMNS if c not in df.columns]
    if missing:
        raise EvalFailure(f"{path} lacks {missing} — the paired CI cannot be formed")
    if len(df) != CONTROL_42_ROWS:
        raise EvalFailure(
            f"{path} has {len(df)} rows, expected {CONTROL_42_ROWS} (rung 42 ep{epoch} "
            "answered the whole held-out set)"
        )
    n_heico = sum(1 for q in df["qID"] if str(q).startswith("heico"))
    if n_heico != CONTROL_42_HEICO_ROWS:
        raise EvalFailure(
            f"{path} has {n_heico} heico rows, expected {CONTROL_42_HEICO_ROWS} — this is "
            "not the 8-held-out-video table"
        )


# ──────────────────────────────────────────────────────────────────────────────
# merge
# ──────────────────────────────────────────────────────────────────────────────


def swift_cmd() -> list[str]:
    """The argv prefix that runs swift's CLI. NEVER the `swift` console script.

    🔴 **Do not call `swift` by name, and do not trust that it is on PATH.** pip's console
    scripts carry an absolute interpreter in their shebang, and that interpreter is not
    guaranteed to exist:

    * the env is a `conda create --clone` of `orena-gen36`, so the scripts first pointed at
      gen36's python and died on a missing `qwen_vl_utils` (repaired with `sed` 2026-08-18);
    * then the 2026-08-18 reboot moved the volume, and every shebang went back to pointing
      at `/data/uaq_user/envs/orena-train/bin/python`, which no longer exists.

    ⚠️ **The second failure reports as `FileNotFoundError: 'swift'`**, which reads as "swift
    is not installed". It is: the file is right there on PATH. `execve` returns ENOENT for a
    *missing interpreter* too, and Python attributes it to the command. That error cost one
    smoke run; do not re-diagnose it as a PATH problem.

    Going through `sys.executable` sidesteps all of it — the interpreter relocates fine
    because conda derives `sys.prefix` from the binary's real path — and needs no write to
    the env, so a later `mount --bind /mnt/storage /data` leaves nothing to undo.
    """
    return [sys.executable, "-c",
            "import sys; from swift.cli.main import cli_main; sys.exit(cli_main())"]


def merge_checkpoint(cfg: Rung47Config, ckpt_root: Path, epoch: int) -> Path:
    """`swift export --merge_lora` one epoch's adapter into a loadable model."""
    ckpt = assert_checkpoint_complete(ckpt_root, epoch)
    out = cfg.merged_dir(epoch)
    if out.is_dir() and any(out.iterdir()):
        log.info("epoch %d already merged -> %s", epoch, out)
        return _merged_model_dir(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    p = subprocess.run(
        swift_cmd() + ["export", "--adapters", str(ckpt), "--merge_lora", "true",
                       # 🔴 explicit, or swift reads the adapter's args.json and looks for
                       # the base under the OLD mountpoint. See `base_model_path`.
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


def _merged_model_dir(out_dir: Path) -> Path:
    """swift writes either INTO out_dir or into a `*-merged` child. Find config.json."""
    if (out_dir / "config.json").exists():
        return out_dir
    hits = sorted(out_dir.glob("*/config.json"))
    if not hits:
        raise EvalFailure(f"no config.json under {out_dir} — the merge produced nothing loadable")
    return hits[0].parent


def reclaim_merged(cfg: Rung47Config, epoch: int) -> None:
    """Delete the merged model once its answers are on disk (CONSTITUTION §IX).

    ~16 GB per epoch. Three epochs unreclaimed is ~48 GB, and this is a shared box.
    The per-question `results.csv` is the artifact worth keeping; the merge is
    reproducible from the adapter in minutes.
    """
    import shutil

    if cfg.keep_merged:
        return
    d = cfg.merged_dir(epoch)
    if d.is_dir():
        shutil.rmtree(d)
        log.info("reclaimed %s", d)


# ──────────────────────────────────────────────────────────────────────────────
# scoring + the reads
# ──────────────────────────────────────────────────────────────────────────────


def cells(report: dict) -> dict:
    """Rung 42's cell extraction, unchanged. Leaf→group is `metrics`' job, not ours."""
    import pandas as pd

    bb = pd.DataFrame(report["by_bucket"])
    out = {}
    for dist in ("ID", "OOD"):
        d = bb[bb.distribution == dist].set_index("capability_group")
        for name in ("aggregation", "object_recognition"):
            out[f"{name}_{dist}"] = float(d.loc[name, "accuracy"]) if name in d.index else float("nan")
    out["bucket_mean"] = float(report["bucket_mean"])
    out["acc_ID"] = float(report["acc_ID"])
    out["acc_OOD"] = float(report["acc_OOD"])
    out["margin_ID"] = float(report["margin_ID"])
    out["margin_OOD"] = float(report["margin_OOD"])
    return out


def score_epoch(cfg: Rung47Config, epoch: int, split: dict) -> dict:
    """Merge → eval on the 1,283 → reclaim. Returns the canonical report + cells.

    Reuses an existing `results.csv` when one is on disk: re-running ep3 to add ep4
    later must cost zero GPU. That reuse is why the sweep can be run incrementally.
    """
    eval_cfg = cfg.to_eval_config(epoch)
    existing = Path(eval_cfg.out_dir) / eval_cfg.run_name / "results.csv"
    if existing.exists():
        log.info("epoch %d already answered -> %s (no GPU)", epoch, existing)
        report = json.loads((existing.parent / "report.json").read_text(encoding="utf-8"))
    else:
        ckpt_root = resolve_ckpt_root(cfg)
        eval_cfg.merged_dir = str(merge_checkpoint(cfg, ckpt_root, epoch))
        try:
            report = score(eval_cfg, split)
        finally:
            reclaim_merged(cfg, epoch)
    return {
        "epoch": epoch,
        "step": EPOCH_STEPS[epoch],
        "report": report,
        "cells": cells(report),
        "results_csv": arm_results_csv(eval_cfg),
        "run_name": eval_cfg.run_name,
    }


def control_verdict(ep3_bucket_mean: float) -> dict:
    """🔑 Did the stack reproduce A2? Declared BEFORE the number (RULES §S4).

    GREEN inside ±0.01 of 0.6342 ⇒ the transformers-5.12.1 / JPEG-cache / RTX-6000
    stack is jointly bounded and ep4 reads clean against rung 42.
    RED outside it ⇒ we have measured a STACK effect. ep4 vs rung 42 is then
    uninterpretable as a corpus comparison, and the DiD below is what survives.
    """
    d = ep3_bucket_mean - CONTROL_A2_EP3["bucket_mean"]
    green = abs(d) <= CONTROL_TOLERANCE
    return {
        "ep3_bucket_mean": ep3_bucket_mean,
        "a2_ep3_bucket_mean": CONTROL_A2_EP3["bucket_mean"],
        "delta": d,
        "tolerance": CONTROL_TOLERANCE,
        "verdict": "GREEN" if green else "RED",
        "means": (
            "stack reproduces A2 — ep4 vs rung 42 is readable as a corpus comparison"
            if green else
            "a STACK effect is present; ep4 vs rung 42 confounds corpus with stack. "
            "Read the difference-in-differences instead, and do NOT quote 47_ep4 − 42_ep4."
        ),
    }


def did_verdict(ep3_bm: float, ep4_bm: float) -> dict:
    """🟢 The read that survives a red control: epoch effect, corpus vs corpus.

    Both differences are internal to one arm on one stack, so the stack, the JPEG
    round-trip and the GPU cancel inside each of them.
    """
    d47 = ep4_bm - ep3_bm
    return {
        "d47_ep4_minus_ep3": d47,
        "d42_ep4_minus_ep3": D42_EP4_MINUS_EP3,
        "did": d47 - D42_EP4_MINUS_EP3,
        "reads": (
            "the epoch effect is corpus-independent ⇒ rung 42's +0.0402 was EPOCHS"
            if abs(d47 - D42_EP4_MINUS_EP3) <= CONTROL_TOLERANCE else
            "the epoch effect differs by corpus ⇒ the two are not separable this cheaply; "
            "the paired CI against rung 42's ep4 is the next instrument"
        ),
    }


def paired_ci_vs_42(arm_csv: Path, ctrl_csv: Path, held_qids: set, n_boot: int = 4000):
    """Video-clustered paired CI, arm MINUS rung 42, per capability cell. RAISES.

    🔴 Effective n is 8 VIDEOS, not 1,283 questions (RULES §13). An unclustered CI
    here would be ~10x too narrow and would manufacture significance.
    🔴 Two videos carry the whole heico side, so no `*_OOD` CI is READABLE. Every cell
    is computed anyway because RULES §S8 lets ANY cell veto, while only a
    pre-declared cell may grant a win.
    """
    import pandas as pd
    from frame import metrics

    a = pd.read_csv(ctrl_csv)
    b = pd.read_csv(arm_csv)
    a = a[a["qID"].isin(held_qids)][["qID", "video", "correctness", "primary"]].rename(
        columns={"correctness": "correct_a"})
    b = b[b["qID"].isin(held_qids)][["qID", "correctness"]].rename(
        columns={"correctness": "correct_b"})
    j = a.merge(b, on="qID", how="inner")
    if len(j) != len(held_qids):
        raise EvalFailure(
            f"arms not scored on the same questions ({len(j)} of {len(held_qids)}) — "
            "a paired CI over a partial join silently drops the hard ones"
        )
    j["group"] = j["primary"].map(metrics._leaf_to_group)     # RULES §2
    j["dist"] = j["qID"].map(lambda q: "OOD" if str(q).split("__")[0] == "heico" else "ID")

    rows = []
    for grp in sorted(j["group"].unique()) + ["ALL"]:
        for dist in ("ID", "OOD"):
            sub = j[j["dist"] == dist]
            sub = sub if grp == "ALL" else sub[sub["group"] == grp]
            rows.append({"cell": f"{grp}_{dist}",
                         **metrics.paired_delta_ci(sub, n_boot=n_boot, seed=42)})
    ci = pd.DataFrame(rows)
    ci["excludes_zero"] = (ci.ci_low > 0) | (ci.ci_high < 0)
    return ci


__all__ = [
    "RUN", "EPOCH_STEPS", "DECISIVE_EPOCHS", "SINGLE_VARIABLE",
    "CONTROL_42", "CONTROL_A2_EP3", "CONTROL_TOLERANCE", "D42_EP4_MINUS_EP3",
    "Rung47Config", "EvalFailure",
    "resolve_ckpt_root", "assert_run_moved_weights", "assert_checkpoint_complete",
    "assert_single_variable", "assert_cache_covers", "held_out_split", "ensure_paths",
    "ensure_control_42", "merge_checkpoint", "reclaim_merged",
    "swift_cmd", "cells", "score_epoch", "control_verdict", "did_verdict", "paired_ci_vs_42",
]
