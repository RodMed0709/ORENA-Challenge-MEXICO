"""Rung 40 — the arm engine. ONE engine, two declared arms.

Importable library. NEVER a launcher — the notebooks run it.

Both arms are a single variable off **rung 38's own arm**, which is the control:
`38_qwen36_27b_v1` epoch 1. 🔴 NOT off A2 — A2 is a different backbone, and reading
against it would compare two variables labelled as one, which is exactly the
failure rung 38 committed and rung 39 was written to correct.

    A_alpha      lora_alpha 32 -> 16          (ratio 4 -> 2)
    B_connector  the connector trains, at 4e-5

⚠️ They are TWO variables and never share an arm.

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

SMOKE mode swaps the 27B for `Qwen3.5-2B` and the real data for generated noise, so
the whole chain is exercisable on UNAM — where the challenge data may not go.
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
# the control: rung 38's arm, verbatim. Numbers from its RESULTS.csv.
# ---------------------------------------------------------------------------

BASELINE = {
    "base_model": "Qwen/Qwen3.6-27B",
    "learning_rate": 2e-4,
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": "all-linear",
    "modules_to_save": (),        # rung 38 trained NO connector
    "connector_lr": None,         # ...so it had no connector LR either
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_pixels": 921600,
    "max_seq_length": 2048,
    "seed": 42,
}

CONTROL_RUN = "38_qwen36_27b_v1"
CONTROL_SCORES = {           # RESULTS.csv, epoch 1 — for the report, never re-typed into a claim
    "proxy_leaderboard": 0.4643,
    "bucket_mean": 0.5302,
    "object_recognition_ID": 0.5579,
    "aggregation_ID": 0.3707,
    "margin_OOD": 0.1388,
}

TRAIN_SHA256 = "180e28f0325674197d52706beeabd846851bdd2875264b5c3505e0debfbd8e8b"

MERGER_MODULES = (
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
)

# 🔴 Rung 38 recorded none of these and A2's weight_decay (0.1) differs from
# HuggingFace's default (0.0), so regularisation may have differed silently on the
# very run whose symptom was over-fitting. Unresolvable after the fact. Never again.
RECORDED_FIELDS = ("max_grad_norm", "weight_decay", "optim", "adam_beta1", "adam_beta2")

# Each arm declares the flags it is allowed to move. The diff must equal this set.
ARMS: dict[str, set[str]] = {
    "A_alpha": {"lora_alpha"},
    # ONE scientific variable -- "the connector trains" -- implemented by two flags,
    # exactly as rung 39 declared its own two-flag single variable. `connector_lr`
    # is meaningless without `modules_to_save`, so they move together or not at all.
    "B_connector": {"modules_to_save", "connector_lr"},
}


@dataclass
class ArmConfig:
    arm: str = "A_alpha"
    run_name: str = ""
    work_dir: str = "/workspace/repo_leo/experiments/40-gen36-recipe-connector/runs"
    train_jsonl: str = "/workspace/repo/experiments/18-count-aug/runs/18_count_aug_v1/train.jsonl"
    hf_home: str = ""

    base_model: str = BASELINE["base_model"]
    learning_rate: float = BASELINE["learning_rate"]
    lora_rank: int = BASELINE["lora_rank"]
    lora_alpha: int = BASELINE["lora_alpha"]
    lora_dropout: float = BASELINE["lora_dropout"]
    target_modules: str = BASELINE["target_modules"]
    modules_to_save: tuple[str, ...] = ()
    connector_lr: float | None = None
    num_train_epochs: int = BASELINE["num_train_epochs"]
    per_device_train_batch_size: int = BASELINE["per_device_train_batch_size"]
    gradient_accumulation_steps: int = BASELINE["gradient_accumulation_steps"]
    warmup_ratio: float = BASELINE["warmup_ratio"]
    lr_scheduler_type: str = BASELINE["lr_scheduler_type"]
    max_pixels: int = BASELINE["max_pixels"]
    max_seq_length: int = BASELINE["max_seq_length"]
    seed: int = BASELINE["seed"]

    # explicitly pinned rather than left to a framework default -- see RECORDED_FIELDS
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    optim: str = "adamw_8bit"

    smoke: bool = False
    smoke_model: str = "Qwen/Qwen3.5-2B"
    smoke_steps: int = 5
    smoke_rows: int = 32

    # Seconds between `[pulse]` lines. This is the watchdog's liveness signal, so
    # it must stay well under the chain's `stale_seconds` (2700). See `_Pulse`.
    pulse_seconds: int = 60

    @property
    def run_dir(self) -> str:
        name = self.run_name or f"40_{self.arm}_v1{'_smoke' if self.smoke else ''}"
        return str(Path(self.work_dir) / name)

    @property
    def merged_dir(self) -> str:
        return str(Path(self.run_dir) / "merged")

    @property
    def effective_model(self) -> str:
        return self.smoke_model if self.smoke else self.base_model


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
        self._stop = threading.Event()
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
        while not self._stop.wait(self.every):
            self._emit()

    def stop(self) -> None:
        self._stop.set()


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
    """The data is rung 18's, byte for byte. RAISES.

    A different dataset would silently turn a recipe result into a data result.
    """
    if cfg.smoke:
        return {"smoke": True, "sha256": None}
    p = Path(cfg.train_jsonl)
    if not p.exists():
        raise ArmFailure(f"train.jsonl not found at {p}")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != TRAIN_SHA256:
        raise ArmFailure(
            f"train.jsonl sha256 {digest} != the pinned {TRAIN_SHA256}. The control's "
            "data and this arm's data are not the same file."
        )
    n = sum(1 for _ in p.open())
    log.info("dataset OK — %d rows, sha %s", n, digest[:12])
    return {"sha256": digest, "n_rows": n}


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
    """
    import glob
    import os

    hf = cfg.hf_home or os.environ.get("HF_HOME", "")
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
        "config": asdict(cfg), "versions": versions(), "control_run": CONTROL_RUN,
        "control_scores": CONTROL_SCORES, "verdict": "INCOMPLETE",
    }

    try:
        # --- guards that cost nothing and must pass before a GPU is touched ---
        pulse.mark("guards")
        result["diff_vs_rung38"] = assert_single_variable(cfg)
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
        model, tok = FastVisionModel.from_pretrained(
            cfg.effective_model, load_in_4bit=False, load_in_16bit=True,
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
