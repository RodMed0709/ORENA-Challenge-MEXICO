"""Rung 45 — the arm engine. ONE engine, three declared arms.

Importable library. NEVER a launcher — the notebooks run it.

## Provenance — this file is a COPY, and that is deliberate

Copied 2026-08-17 from `experiments/40-gen36-recipe-connector/_models/gen36_arm.py`,
source sha256 `047bc170d93a7056a09c135579d56fc6fbcf5d68d0094456752361604b13efc9`.

Importing rung 40's engine instead would mean that editing rung 40 silently moves rung
45's results. Each rung owns its artifacts; duplication is the price of that, and it is
the price this repo already pays elsewhere. What is NOT duplicated loosely: the five
guards below are carried over intact, because they are the reason to copy rather than
rewrite — they caught five bugs in rung 40, one of which would have wrecked every full
run of the pod.

## The three arms, and what each one's CONTROL is

🔴 **`BASELINE` here is R00, run in this rung, on this hardware, at this precision.**
Rung 40's `40_B_connector_v1` is CONTEXT, never a control: reading against it would move
four things at once (NF4 vs bf16, corpus, host, and three hyperparameters rung 40 never
archived). That is the confound rung 38's README called *"valid for a candidate search,
invalid for attribution"*. See PLAN §3.

    R00   the control itself — rung 40's B_connector recipe, re-anchored in NF4
    R0    corpus 14 415 -> 19 384             control: R00    -> does more data transfer?
    R1    lora_dropout 0.0 -> 0.1             control: R0     -> was the over-fit regularisation?

⚠️ R1's control is **R0, not R00**, so R1 legitimately diffs from `BASELINE` on TWO
fields. `ARMS` declares that explicitly; the point of `assert_single_variable` is that
every moved flag is declared, not that only one ever moves.

Everything this file guards against has already happened in this repo:

* rung 38 shipped **three undeclared deviations** => `assert_single_variable`.
* rung 38 recorded neither `max_grad_norm`, `weight_decay` nor `optim`, and A2 ran
  `weight_decay 0.1` against HuggingFace's 0.0 => `RECORDED_FIELDS`.
* `--vit_lr` was a **silent no-op** without `--optimizer multimodal`, and trl 0.24
  swallows `optimizers=` in `**kwargs` with no error (measured 2026-08-13, G4)
  => `assert_param_groups`, re-checked AFTER `.train()`.
* a `target_modules` entry that matches nothing fails **silently** in PEFT
  => `assert_coverage`.
* `save_pretrained_merged` keeps a `modules_to_save` weight and **drops its bias**
  (measured 2026-08-13, G3) => `assert_merge_carried`, per tensor.

SMOKE mode swaps the 27B for `Qwen3.5-4B` and the real data for generated noise, so
the whole chain is exercisable on UNAM — where the challenge data may not go — and on
a cheap pod. (The gates in 00/01 used the 2B on UNAM and stay as recorded; the pod's
cache has the 4B, and the 4B is the same `Qwen3_5ForConditionalGeneration` class.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# the control: R00, run in THIS rung. See the module docstring.
# ---------------------------------------------------------------------------

MERGER_MODULES = (
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
)

BASELINE = {
    "base_model": "Qwen/Qwen3.6-27B",
    "load_in_4bit": True,         # 🔴 forced by the hardware, not chosen — PLAN §2
    "corpus": "r18_14415",        # the label, not the path: the path differs per host
    "learning_rate": 2e-4,
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": "all-linear",
    "modules_to_save": MERGER_MODULES,           # R00 keeps rung 40 B_connector's connector
    "connector_lr": 4e-5,
    "num_train_epochs": 1,
    "effective_batch": 16,        # 🔴 NOT grad_accum — see `assert_effective_batch`
    "per_device_train_batch_size": 1,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_pixels": 921600,
    "max_seq_length": 2048,
    "seed": 42,
}

# Rung 40's B_connector, bf16 on one A100. CONTEXT for the report — never a control,
# and never subtracted from a rung-45 number to make a claim. See PLAN §3.
CONTEXT_RUN = "40_B_connector_v1"
CONTEXT_SCORES = {
    "precision": "bf16",
    "bucket_mean": 0.5763,
    "aggregation_ID": 0.4262,
    "object_recognition_ID": 0.6258,
    "paired_delta_ALL": 0.0540,
    "peak_vram_train_gib": 53.4,
    "eval_set": "6252 questions — CONTAMINATED for R0/R1, see PLAN §4",
}

# 🔴 Two corpora, two pins. Both were recovered from S3 on 2026-08-17 and their
# `images[]` roots rewritten from RunPod's `/workspace/frames_cache` to UNAM's store
# by `_tools/rehost_corpus.py`, which PROVES the rewrite is path-only. Rewriting a path
# changes the hash, so rung 40's pin (`180e28f0…`, the ORIGINAL bytes of r18) no longer
# matches the file a trainer opens here. Both hashes are kept so a reader can tell which
# file they hold and which guard applies to it.
CORPUS_SHA256 = {
    "r18_14415": {
        "original": "180e28f0325674197d52706beeabd846851bdd2875264b5c3505e0debfbd8e8b",
        "rehosted": "7c4abd84ab25d9cff1dea5d3afd4d31f06e3e3daf95c1c18251716f824a3fd50",
        "n_rows": 14415, "unique_frames": 10727,
    },
    "r42_19384": {
        "original": "71143c547ad444c5596fb8cfcebf6fa2030251226152327a5e25d4d862850e1a",
        "rehosted": "07977804893efe6bf44aaf39a745cdb5fbd5957c930e93a304b87a9c88b5e4aa",
        "n_rows": 19384, "unique_frames": 14181,
    },
}

# 🔴 Rung 38 recorded none of these and A2's weight_decay (0.1) differs from
# HuggingFace's default (0.0), so regularisation may have differed silently on the
# very run whose symptom was over-fitting. Unresolvable after the fact. Never again.
RECORDED_FIELDS = ("max_grad_norm", "weight_decay", "optim", "adam_beta1", "adam_beta2")

# Each arm declares the flags it is allowed to move AGAINST `BASELINE` (= R00), and
# names the arm that is its scientific control. The diff must equal the declared set.
#
# 🔴 The two are not the same thing. R1's single variable is `lora_dropout` against R0,
# but R0 already moved `corpus` against BASELINE, so R1's diff versus BASELINE is two
# fields. Declaring `control` separately is what keeps "one variable" meaningful once
# the arms form a chain instead of a star.
ARMS: dict[str, set[str]] = {
    "R00": set(),                              # the control itself: moves nothing
    "R0": {"corpus"},                          # vs R00
    "R1": {"corpus", "lora_dropout"},          # vs R0 — the only NEW field is lora_dropout
}

CONTROL_OF = {"R00": None, "R0": "R00", "R1": "R0"}


def world_size() -> int:
    """How many processes `torchrun` started. 1 when launched plainly.

    Read from the environment rather than from `torch.distributed`, because it has to be
    correct BEFORE the process group is initialised — the accumulation steps are computed
    while building `SFTConfig`, which happens first.
    """
    import os
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


@dataclass
class ArmConfig:
    arm: str = "R00"
    run_name: str = ""
    work_dir: str = "/home/uaq_user/storage/rung45/runs"
    corpus: str = BASELINE["corpus"]
    corpus_dir: str = "/home/uaq_user/storage/rung45/corpus"
    hf_home: str = ""

    load_in_4bit: bool = BASELINE["load_in_4bit"]
    # Effective batch is the INVARIANT; grad_accum is derived from it and the world size
    # so a 1-GPU and a 2-GPU run train the same model. See `assert_effective_batch`.
    effective_batch: int = BASELINE["effective_batch"]

    base_model: str = BASELINE["base_model"]
    learning_rate: float = BASELINE["learning_rate"]
    lora_rank: int = BASELINE["lora_rank"]
    lora_alpha: int = BASELINE["lora_alpha"]
    lora_dropout: float = BASELINE["lora_dropout"]
    target_modules: str = BASELINE["target_modules"]
    modules_to_save: tuple[str, ...] = MERGER_MODULES
    connector_lr: float | None = BASELINE["connector_lr"]
    num_train_epochs: int = BASELINE["num_train_epochs"]
    per_device_train_batch_size: int = BASELINE["per_device_train_batch_size"]
    warmup_ratio: float = BASELINE["warmup_ratio"]
    lr_scheduler_type: str = BASELINE["lr_scheduler_type"]
    max_pixels: int = BASELINE["max_pixels"]
    max_seq_length: int = BASELINE["max_seq_length"]
    seed: int = BASELINE["seed"]

    # 🔴 Declared by rung 45's PLAN §5, not inherited: rung 40 archived NONE of these, so
    # there is nothing to copy and "same as rung 40" is not a statement anyone can make.
    # These are the 8B ladder's values, chosen so the only deliberate departures from A2
    # stay the ones this rung is about.
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    optim: str = "adamw_8bit"

    smoke: bool = False
    # 🔻 **The 2B, and rung 40's reasoning for the 4B is host-specific, not wrong.** Rung 40
    # ran its arms on the pod, whose cache holds the 4B, and picked it so the rehearsal would
    # not begin with a download. On UNAM the sizes are inverted and measured 2026-08-17:
    # `Qwen3.5-4B` is **28 KB** (config.json only, weights never fetched) and `Qwen3.5-2B` is
    # **4.3 GB**, complete. Keeping the 4B default here would make the smoke start by pulling
    # 4 GB — or fail offline.
    # What actually has to hold is unchanged and holds for both: same
    # `Qwen3_5ForConditionalGeneration` class, same `vision_config`, same
    # `model.visual.merger.linear_fc{1,2}` names ⇒ the connector paths resolve for real
    # instead of passing vacuously. Rung 40's own gates used this same 2B on this same box.
    smoke_model: str = "Qwen/Qwen3.5-2B"
    smoke_steps: int = 5
    smoke_rows: int = 32

    # Seconds between `[pulse]` lines. This is the watchdog's liveness signal, so
    # it must stay well under the chain's `stale_seconds` (2700). See `_Pulse`.
    pulse_seconds: int = 60

    @property
    def run_dir(self) -> str:
        name = self.run_name or f"45_{self.arm}_v1{'_smoke' if self.smoke else ''}"
        return str(Path(self.work_dir) / name)

    @property
    def merged_dir(self) -> str:
        return str(Path(self.run_dir) / "merged")

    @property
    def effective_model(self) -> str:
        return self.smoke_model if self.smoke else self.base_model

    @property
    def train_jsonl(self) -> str:
        """Derived from the corpus LABEL, so the label is the only thing an arm sets.

        A free-text path is how two arms end up on the same file while their configs say
        otherwise — `assert_single_variable` compares labels, and only this property turns
        one into bytes on disk.
        """
        return str(Path(self.corpus_dir) / f"{'r00_train_14415' if self.corpus == 'r18_14415' else 'r0_train_19384'}.jsonl")

    @property
    def gradient_accumulation_steps(self) -> int:
        """DERIVED, never set. `effective_batch / (per_device x world_size)`.

        🔴 This property is the whole reason the engine had to be touched for DDP. Rung 40
        ran one GPU at `grad_accum 16`. Reusing that literal under `torchrun --nproc 2`
        trains at an effective batch of **32** — silently, with no error and no warning —
        and every comparison in this rung dies with it. The invariant is the effective
        batch; the accumulation is arithmetic.
        """
        denom = self.per_device_train_batch_size * world_size()
        if self.effective_batch % denom:
            raise ArmFailure(
                f"effective_batch {self.effective_batch} is not divisible by "
                f"per_device {self.per_device_train_batch_size} x world_size {world_size()}. "
                "Pick a world size that divides it rather than rounding the batch.")
        return self.effective_batch // denom


class ArmFailure(AssertionError):
    """A guard that fires is a FINDING (RULES §7). Raised so papermill exits non-zero."""


# ---------------------------------------------------------------------------
# liveness — the only thing that reaches the chain log
# ---------------------------------------------------------------------------
# 🔴 MEASURED 2026-08-14, rung 40 arm A. The chain runs these notebooks under
# `papermill --log-output`, which forwards *stream* outputs and silently drops
# `display_data`. HF's Trainer installs a notebook progress bar — a `display_data`
# HTML object updated in place — so from the instant `.train()` begins, papermill
# emits nothing at all. `leo_chain40.log` went silent at 06:52:46; the watchdog
# read 2912 s of that silence as a hang and stopped the pod on a perfectly healthy
# 27B ~48 min in. No checkpoint, no RESULTS_arm.json, both arms lost, ~$2.20.
#
# The premise in ChainConfig ("a live 27B writes a line every ~24 s") was false
# from the first training step onward — the one phase where the watchdog matters.
#
# Two layers, because one night was enough:
#   * `_Pulse` — a daemon thread printing for the WHOLE of main(). It covers the
#     phases no trainer callback can see, and the longest of those is
#     `save_pretrained_merged`: ~52 GB of merged 27B onto a network volume,
#     silent start to finish. That was the SECOND latent kill, still unfired.
#   * `_progress_callback` — real step/loss/ETA, so the log is worth reading and
#     not merely non-empty.
#
# Neither may raise. A liveness signal that kills the run it exists to protect is
# strictly worse than no signal, so both swallow their own errors.
class _Pulse(threading.Thread):
    """Prints `[pulse] <phase> | N min` every `every` seconds until stopped."""

    def __init__(self, every: int = 60) -> None:
        super().__init__(daemon=True, name="arm-pulse")
        self.every = max(1, int(every))
        self.phase = "starting"
        # 🔴 NOT `self._stop`: `threading.Thread._stop` is a METHOD, and shadowing it
        # with an Event makes CPython's `_after_fork` raise `'Event' object is not
        # callable` on every fork. Caught by the 2026-08-14 rehearsal — it printed
        # four tracebacks into the very log this thread exists to keep readable.
        self._stop_evt = threading.Event()
        self._t0 = time.perf_counter()

    def elapsed_min(self) -> float:
        return (time.perf_counter() - self._t0) / 60

    def mark(self, phase: str) -> None:
        """Name the current phase AND emit a line immediately, so every phase
        boundary is visible in the log even if it is over in under `every` s."""
        self.phase = phase
        self._emit()

    def _emit(self) -> None:
        try:
            print(f"[pulse] {self.phase} | {self.elapsed_min():6.1f} min elapsed", flush=True)
        except Exception:  # noqa: BLE001 -- never kill a run over a log line
            pass

    def run(self) -> None:
        while not self._stop_evt.wait(self.every):
            self._emit()

    def stop(self) -> None:
        self._stop_evt.set()


def _progress_callback(pulse: _Pulse):
    """Built lazily on purpose: a module-scope `from transformers import ...` would
    land before `import unsloth` in main() and blow up unsloth_zoo."""
    from transformers import TrainerCallback

    class _Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            try:
                logs = logs or {}
                step, total = state.global_step, (state.max_steps or 0)
                bits = [f"step {step}/{total}" if total else f"step {step}"]
                for key, fmt in (("loss", "{:.4f}"), ("grad_norm", "{:.3f}"),
                                 ("learning_rate", "{:.2e}")):
                    val = logs.get(key)
                    if isinstance(val, (int, float)):
                        bits.append(f"{key} {fmt.format(val)}")
                mins = pulse.elapsed_min()
                bits.append(f"{mins:.1f} min")
                if total and step:
                    bits.append(f"eta {mins * (total - step) / step:.1f} min")
                print("[step] " + " | ".join(bits), flush=True)
            except Exception:  # noqa: BLE001 -- see above
                pass

    return _Progress()


# ---------------------------------------------------------------------------
# guards — every one of these exists because its failure already happened here
# ---------------------------------------------------------------------------

def assert_single_variable(cfg: ArmConfig) -> dict:
    """The diff against rung 38's arm must equal the arm's DECLARED flag set. RAISES.

    Rung 38 shipped three undeclared deviations (framework, `lora_dropout`, LoRA
    coverage) and that is why its result took a day to interpret. A diff computed
    from the config object cannot drift the way a hand-written claim can.
    """
    if cfg.arm not in ARMS:
        raise ArmFailure(f"unknown arm {cfg.arm!r}; declared arms are {sorted(ARMS)}")

    diff = {}
    for k, base in BASELINE.items():
        got = getattr(cfg, k)
        if isinstance(base, tuple) or isinstance(got, tuple):
            base, got = tuple(base or ()), tuple(got or ())
        if got != base:
            diff[k] = {"baseline": base, "arm": got}

    declared = ARMS[cfg.arm]
    moved = set(diff)
    if moved != declared:
        raise ArmFailure(
            f"arm {cfg.arm!r} declares {sorted(declared)} but the config moves "
            f"{sorted(moved)}.\nDiff: {json.dumps(diff, indent=2, default=str)}\n"
            "Anything not on the declared list is a defect, not a detail."
        )
    log.info("single-variable OK — %s moves exactly %s", cfg.arm, sorted(declared))
    return diff


def assert_dataset(cfg: ArmConfig) -> dict:
    """The arm's corpus is the one its LABEL names, byte for byte. RAISES.

    In rung 40 this guard defended one pinned file. Here the corpus IS the variable, so
    the guard has to pin *each* label to its own digest — otherwise R0 and R00 could both
    open the same bytes while their configs claim they differ, and the rung's headline
    comparison would be a null by construction.
    """
    if cfg.smoke:
        return {"smoke": True, "sha256": None}
    if cfg.corpus not in CORPUS_SHA256:
        raise ArmFailure(f"unknown corpus label {cfg.corpus!r}; known: {sorted(CORPUS_SHA256)}")
    want = CORPUS_SHA256[cfg.corpus]
    p = Path(cfg.train_jsonl)
    if not p.exists():
        raise ArmFailure(f"train.jsonl not found at {p}")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != want["rehosted"]:
        raise ArmFailure(
            f"{p.name} sha256 {digest} != the pinned {want['rehosted']} for corpus "
            f"{cfg.corpus!r}. Either the file is not the rehosted one, or it was edited "
            "after `_tools/rehost_corpus.py` proved the rewrite was path-only."
        )
    n = sum(1 for _ in p.open())
    if n != want["n_rows"]:
        raise ArmFailure(f"{p.name} has {n} rows, pinned at {want['n_rows']}")
    log.info("dataset OK — %s, %d rows, sha %s", cfg.corpus, n, digest[:12])
    return {"corpus": cfg.corpus, "path": str(p), "sha256_rehosted": digest,
            "sha256_original": want["original"], "n_rows": n}


def assert_effective_batch(args, cfg: ArmConfig) -> dict:
    """The batch the trainer will ACTUALLY use is 16, read back from its own args. RAISES.

    🔴 Read from `args`, not from `cfg`. Rung 40's lesson, twice over: `--vit_lr` was a
    silent no-op and trl swallowed `optimizers=` in `**kwargs` with no error. A config
    object saying the right thing proves nothing about what the trainer does; both the
    accumulation and the per-device batch are therefore re-derived from the object the
    trainer holds, multiplied back out, and compared against the invariant.
    """
    got = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size()
    if got != cfg.effective_batch:
        raise ArmFailure(
            f"effective batch is {got} (per_device {args.per_device_train_batch_size} x "
            f"accum {args.gradient_accumulation_steps} x world {world_size()}), declared "
            f"{cfg.effective_batch}. Under DDP this does not raise on its own — it just "
            "trains a different model than every other arm.")
    log.info("effective batch OK — %d x %d x %d = %d",
             args.per_device_train_batch_size, args.gradient_accumulation_steps,
             world_size(), got)
    return {"per_device": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "world_size": world_size(), "effective_batch": got}


def assert_coverage(model, cfg: ArmConfig) -> dict:
    """LoRA reached what it should, and the connector is wrapped iff declared. RAISES.

    A `target_modules` entry that matches nothing fails SILENTLY in PEFT — `ValueError`
    fires only when NOTHING matched, so a partial miss reaches training with at most a
    `RuntimeWarning`. Measured 2026-08-13.
    """
    targeted = list(getattr(model, "targeted_module_names", []) or [])
    wrapped = [n for n, m in model.named_modules() if type(m).__name__ == "ModulesToSaveWrapper"]

    if not targeted:
        raise ArmFailure(
            "no LoRA target was matched at all — coverage was REPLACED, not extended."
        )
    want = len(cfg.modules_to_save)
    if len(wrapped) != want:
        raise ArmFailure(
            f"expected {want} wrapped connector module(s), got {len(wrapped)}: {wrapped}. "
            "A count that is neither the expected one nor zero is a finding to read, "
            "not something to accept silently."
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("coverage OK — %d LoRA targets, %d wrapped, %s trainable",
             len(targeted), len(wrapped), f"{trainable:,}")
    return {"n_targeted": len(targeted), "wrapped": wrapped, "trainable": trainable}


def build_param_groups(model, cfg: ArmConfig) -> list[dict] | None:
    """Connector on its own LR, everything else on the LoRA LR. None if not declared."""
    if not cfg.connector_lr:
        return None
    conn, rest = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (conn if "visual.merger" in n else rest).append(p)
    if not conn or not rest:
        raise ArmFailure(
            f"param-group split is degenerate: connector={len(conn)}, rest={len(rest)}"
        )
    return [
        {"params": conn, "lr": cfg.connector_lr, "name": "connector"},
        {"params": rest, "lr": cfg.learning_rate, "name": "lora"},
    ]


def assert_param_groups(optimizer, cfg: ArmConfig, when: str) -> dict:
    """The declared LRs are the ones the optimiser will step. RAISES.

    🔴 Called AFTER `.train()` as well as before. In trl 0.24 `SFTTrainer.__init__`
    no longer accepts `optimizers` and swallows it in `**kwargs` with no error, so a
    correctly built optimiser can be silently replaced by HuggingFace's default
    between construction and step 1. Measured on 2026-08-13 (G4): the build-time
    check PASSED and only the post-train check caught it.
    """
    # 🔴 `lr` is the LIVE value, which the scheduler has moved. Under `cosine` it is
    # 0.0 by the end of training BY DESIGN, so asserting `lr` post-train fires on every
    # completed run -- measured 2026-08-13 on the first B_connector smoke. What we mean
    # is the value the group STARTED at, and PyTorch stores that as `initial_lr` the
    # moment a scheduler is attached. G4 missed this because it used `constant`.
    groups = [
        {
            "name": g.get("name", f"g{i}"),
            "lr": g["lr"],
            "initial_lr": g.get("initial_lr"),
            "n_params": len(g["params"]),
        }
        for i, g in enumerate(optimizer.param_groups)
    ]
    if not cfg.connector_lr:
        return {"when": when, "groups": groups}

    conn = [g for g in groups if g["name"] == "connector"]
    if not conn:
        raise ArmFailure(
            f"[{when}] no 'connector' param group in the optimiser. Groups: {groups}. "
            "The trainer discarded ours — the silent no-op this guard exists for."
        )
    declared = conn[0]["initial_lr"] if conn[0]["initial_lr"] is not None else conn[0]["lr"]
    if abs(declared - cfg.connector_lr) > 1e-12:
        raise ArmFailure(
            f"[{when}] the connector group started at {declared}, declared "
            f"{cfg.connector_lr}. (live lr is {conn[0]['lr']}; under cosine that is "
            "expected to be 0 at the end and is NOT what this asserts.)"
        )
    log.info("param groups OK [%s] — %s", when, groups)
    return {"when": when, "groups": groups}


def assert_merge_carried(cfg: ArmConfig, base_dir: Path, trained: dict) -> dict:
    """Every connector tensor that moved in training moved in the merge. RAISES.

    🔴 PER TENSOR, never on the sum. Measured 2026-08-13: `save_pretrained_merged`
    carries a `modules_to_save` module's trained WEIGHT and silently drops its
    trained BIAS. The weights dominate the sum by ~4500x, so a summed criterion
    would have PASSED and the loss would have vanished. Same failure shape rung 39's
    `n_llm`/`n_vit` coverage criterion exists to prevent.

    Returns rather than raises when the ONLY thing dropped is a bias: that loss is a
    KNOWN, quantified, declared deviation of this rung (PLAN §3d), and path B does
    not fix it either since LoRA never adapts biases. It is recorded, not re-litigated.
    """
    from safetensors import safe_open

    def grab(d: Path) -> dict:
        out = {}
        for shard in sorted(d.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as f:
                for k in f.keys():
                    if "visual.merger" in k:
                        out[k] = f.get_tensor(k).float().cpu()
        return out

    merged, base = grab(Path(cfg.merged_dir)), grab(base_dir)
    if not merged or not base:
        raise ArmFailure(f"cannot read connector tensors (merged={len(merged)}, base={len(base)})")

    deltas = {
        k: float((merged[k] - base[k]).abs().sum())
        for k in sorted(set(merged) & set(base)) if merged[k].shape == base[k].shape
    }
    dropped = {k: v for k, v in trained.items() if deltas.get(k, 0.0) <= 1e-3 and v > 1e-3}
    non_bias_dropped = {k: v for k, v in dropped.items() if not k.endswith(".bias")}

    payload = {"per_tensor": deltas, "dropped_by_merge": dropped}
    if non_bias_dropped:
        raise ArmFailure(
            f"the merge dropped trained connector WEIGHTS: {non_bias_dropped}. That is "
            "beyond the known bias loss and the checkpoint cannot be shipped."
        )
    if dropped:
        log.warning(
            "merge dropped the trained BIAS as expected and declared (PLAN §3d): %s. "
            "The merged model is not the model that trained; path B would not fix it.",
            dropped,
        )
        payload["known_bias_loss"] = dropped
    return payload


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _resolve_base_dir(cfg: ArmConfig) -> Path:
    """Find the base snapshot by LOOKING, never by hardcoding.

    This project has twice been bitten by two HF caches holding different things;
    a hardcoded path that quietly points at the wrong one turns the merge check
    into nonsense.

    🔴 A LOCAL DIRECTORY is also a legal base_model, and this used to crash on one.
    A CONTINUATION trains from a previously merged checkpoint — `40_B_connector_ep23_v1`
    set `base_model` to `.../40_B_connector_v1/merged` — and the hub-cache branch below
    would then `split("/", 1)` a filesystem path into ('', 'workspace/...'), glob for a
    `models--` directory that cannot exist, and raise:

        ArmFailure: cannot locate the base snapshot for '/workspace/.../merged'
        under '/workspace/hf_cache'. The merge guard has nothing to compare against.

    It cost a completed run its exit code on 2026-08-16: 1802/1802 steps, a full
    51.7 GiB merge, both cosines annealed — and `rc=1` because the *verifier* could not
    find its reference. Nothing was wrong with the artifact.
    Worse than the crash is WHERE it happens: at the end, after ~6.7 h of training,
    when the same condition is knowable in preflight.
    """
    import glob
    import os

    local = Path(cfg.effective_model).expanduser()
    if local.is_dir():
        if not any(local.glob("*.safetensors")):
            raise ArmFailure(
                f"base_model {str(local)!r} is a directory but holds no *.safetensors. "
                "The merge guard would compare against nothing."
            )
        return local

    hf = cfg.hf_home or os.environ.get("HF_HOME", "")
    if "/" not in cfg.effective_model:
        raise ArmFailure(
            f"base_model {cfg.effective_model!r} is neither an existing directory nor a "
            "hub id of the form 'org/name'."
        )
    org, name = cfg.effective_model.split("/", 1)
    hits = sorted(glob.glob(str(Path(hf) / "hub" / f"models--{org}--{name}" / "snapshots" / "*")))
    if not hits:
        raise ArmFailure(
            f"cannot locate the base snapshot for {cfg.effective_model!r} under {hf!r}. "
            "The merge guard has nothing to compare against."
        )
    return Path(hits[-1])


def main(cfg: ArmConfig) -> dict:
    """Guards -> build -> train -> merge -> guards. Every failure raises."""
    import os
    import sys

    if cfg.hf_home:
        os.environ.setdefault("HF_HOME", cfg.hf_home)
    # 🔴 safe for eval, LETHAL for training: it kills Unsloth's merge.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ["MAX_PIXELS"] = str(cfg.max_pixels)

    import unsloth  # noqa: F401  MUST precede transformers
    import torch
    from unsloth import FastVisionModel

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_tools"))
    from merge_gate import build_synthetic_data, versions, _to_messages  # noqa: E402

    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Started BEFORE the guards: model loading and the merge are both long and
    # silent, and the watchdog cannot tell "still working" from "hung" without this.
    pulse = _Pulse(cfg.pulse_seconds)
    pulse.start()
    result: dict = {
        "config": asdict(cfg), "versions": versions(),
        "train_jsonl": cfg.train_jsonl, "world_size": world_size(),
        "control_arm": CONTROL_OF[cfg.arm],          # the arm this one is READ AGAINST
        "context_run": CONTEXT_RUN, "context_scores": CONTEXT_SCORES,
        "verdict": "INCOMPLETE",
    }

    try:
        # --- guards that cost nothing and must pass before a GPU is touched ---
        pulse.mark("guards")
        result["diff_vs_R00"] = assert_single_variable(cfg)
        result["dataset"] = assert_dataset(cfg)

        # --- data -------------------------------------------------------------
        pulse.mark("data")
        if cfg.smoke:
            from merge_gate import MergeGateConfig
            jsonl = build_synthetic_data(
                MergeGateConfig(work_dir=cfg.run_dir, n_rows=cfg.smoke_rows, seed=cfg.seed)
            )
        else:
            jsonl = cfg.train_jsonl
        rows = [_to_messages(json.loads(l)) for l in open(jsonl) if l.strip()]

        # --- build ------------------------------------------------------------
        pulse.mark("loading base weights")
        # 🔴 NF4, and not as a preference. The bf16 arm peaks at 52.64 GiB (rung 38) on
        # 48 GB cards; the 4-bit load measured 18.31 GiB reserved with 29.69 GiB spare
        # (`40-*/_tools/fit_gate_4bit.py`, 2026-08-17). This is what lets the rung run at
        # all, and it is a declared confound against rung 40's absolutes — PLAN §3, §7.
        model, tok = FastVisionModel.from_pretrained(
            cfg.effective_model,
            load_in_4bit=cfg.load_in_4bit, load_in_16bit=not cfg.load_in_4bit,
            full_finetuning=False, max_seq_length=cfg.max_seq_length,
        )
        result["model_class"] = type(model).__name__
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", random_state=cfg.seed, use_rslora=False, loftq_config=None,
            target_modules=cfg.target_modules,          # bare string -- a list would not expand
            modules_to_save=list(cfg.modules_to_save) or None,
        )
        result["coverage"] = assert_coverage(model, cfg)

        before = {
            n: p.detach().clone().float().cpu()
            for n, p in model.named_parameters()
            if p.requires_grad and "visual.merger" in n
        }

        # --- train ------------------------------------------------------------
        from trl import SFTConfig, SFTTrainer
        from unsloth.trainer import UnslothVisionDataCollator

        steps = cfg.smoke_steps if cfg.smoke else -1
        args = SFTConfig(
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.num_train_epochs, max_steps=steps,
            learning_rate=cfg.learning_rate, warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type, logging_steps=1,
            optim=cfg.optim, max_grad_norm=cfg.max_grad_norm,
            weight_decay=cfg.weight_decay, seed=cfg.seed, data_seed=cfg.seed,
            output_dir=str(Path(cfg.run_dir) / "ckpt"), save_strategy="epoch",
            report_to="none", remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=cfg.max_seq_length,
        )
        # 🔴 The three fields rung 38 never recorded, read back from the ARGS the
        # trainer will actually use -- not from our config object.
        result["recorded_fields"] = {f: getattr(args, f, None) for f in RECORDED_FIELDS}
        result["effective_batch"] = assert_effective_batch(args, cfg)

        FastVisionModel.for_training(model)
        tr = SFTTrainer(
            model=model, train_dataset=rows,
            data_collator=UnslothVisionDataCollator(model, tok), args=args,
            callbacks=[_progress_callback(pulse)],
        )

        groups = build_param_groups(model, cfg)
        if groups is not None:
            import torch as _t
            # `optimizers=` is gone in trl 0.24 and swallowed by **kwargs; assigning the
            # instance is what `create_optimizer`'s `if self.optimizer is None` guard honours.
            tr.optimizer = _t.optim.AdamW(groups, lr=cfg.learning_rate)
            result["param_groups_pre"] = assert_param_groups(tr.optimizer, cfg, "pre-train")

        pulse.mark("train")
        st = tr.train()
        result["train_loss"] = float(st.training_loss)
        result["grad_norms"] = [h["grad_norm"] for h in tr.state.log_history if "grad_norm" in h]
        result["train_secs"] = time.perf_counter() - t0
        result["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30

        if groups is not None:
            result["param_groups_post"] = assert_param_groups(tr.optimizer, cfg, "post-train")

        trained = {
            _canon(n): float((p.detach().float().cpu() - before[n]).abs().sum())
            for n, p in model.named_parameters() if n in before
        }
        result["connector_trained"] = trained

        # --- merge ------------------------------------------------------------
        # ~52 GB of merged 27B onto a network volume, and `save_pretrained_merged`
        # says nothing while it works. Without the pulse this alone can outlast
        # `stale_seconds` and get a finished, successful arm killed at the wire.
        pulse.mark("merging + writing ~52 GB")
        model.save_pretrained_merged(cfg.merged_dir, tok)
        result["merged_files"] = sorted(p.name for p in Path(cfg.merged_dir).iterdir())
        if cfg.modules_to_save:
            result["merge_check"] = assert_merge_carried(cfg, _resolve_base_dir(cfg), trained)

        result["verdict"] = "OK"
    except Exception as e:  # noqa: BLE001 -- a failure is still a result worth committing
        result["verdict"] = f"{type(e).__name__}: {e}"
        result["elapsed_s"] = time.perf_counter() - t0
        _write(cfg, result)
        raise
    finally:
        # Stopped on EVERY exit. A pulse still beating after main() returns would
        # tell the watchdog a dead chain is alive — the exact inverse of this bug.
        pulse.mark("done")
        pulse.stop()

    result["elapsed_s"] = time.perf_counter() - t0
    _write(cfg, result)
    return result


def _canon(peft_name: str) -> str:
    n = peft_name
    for junk in (".modules_to_save.default", ".original_module"):
        n = n.replace(junk, "")
    while n.startswith("base_model.model."):
        n = n[len("base_model.model."):]
    return n


def _write(cfg: ArmConfig, payload: dict) -> str:
    p = Path(cfg.run_dir) / "RESULTS_arm.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    log.info("result -> %s", p)
    return str(p)
