"""Rung 40 — THE MERGE GATE. Does Unsloth's merge carry `modules_to_save`?

Importable library. NEVER a launcher — `00_merge_gate.ipynb` is what runs it
(EXPERIMENT_REPO_STRUCTURE_SPEC; CLAUDE.md).

The question is about a CODE PATH, not an architecture and not a metric:

    after training a LoRA whose config carries `modules_to_save`, does
    `save_pretrained_merged` write those trained weights into the merged
    checkpoint?

If it does not, the rung-40 connector arm would train ~5.9 h on a 27B and ship
the BASE model in exactly the two layers the arm is about. That failure is
invisible from the score, which is why it is a gate and not an observation.

Three assertions, each of which RAISES (RULES §7 — a gate that fires is a
finding, never an obstacle):

    G1  coverage      the 2 merger layers are wrapped, and the LoRA legs survive
    G2  it trained    the wrapped weights moved away from base
    G3  the merge     the MERGED checkpoint still carries the movement   <- the question

Order matters. G1 first because a `target_modules`/`modules_to_save` entry that
matches nothing fails **silently** in PEFT — only a `RuntimeWarning`, and
`ValueError` fires solely when *nothing at all* matched. Without G1 a green G3
could just mean "base == base".

G2 exists because `rc=0` is not evidence a run trained. AdamW's decoupled weight
decay moves every tensor at zero gradient, so a checkpoint diff cannot separate a
real run from a no-op; only the magnitude can, and `grad_norm` is the
corroborating instrument.

Proxy model: `Qwen3.5-2B`. Verified to be the right shape — same class
(`Qwen3_5ForConditionalGeneration`), same `model_type` (`qwen3_5`), same
connector names, and NO `deepstack_merger_list`, exactly like the 27B.
⚠️ A PASS here is strong evidence, not formal proof for the 27B: the save path
could branch on shard count. The same comparison is repeated on the real 27B
merge, which is free — two files already on disk.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

log = logging.getLogger(__name__)

# The connector, by FULL module path. Not a suffix.
#
# `_set_trainable` (peft/utils/other.py) matches with `key.endswith(target_key)`,
# so a suffix would work. The full path is used anyway because three independent
# research passes reported three different matchers for the RELATED exclusion
# check in `check_target_module_exists`, and the full path satisfies all of them
# at zero cost. Do not spend a run finding out which was right.
MERGER_MODULES = (
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
)

# What we compare, before and after. `norm` is included because it is part of the
# connector block and would expose a partial carry.
MERGER_TENSOR_PREFIX = "model.visual.merger."


@dataclass
class MergeGateConfig:
    """Everything the gate needs. Set inline in the notebook cell, never here."""

    # --- what we prove it on -------------------------------------------------
    base_model: str = "Qwen/Qwen3.5-2B"
    hf_home: str = "/data/uaq_user/hf_cache"

    # --- where it works (owner-tagged; see the pod convention) ---------------
    work_dir: str = "/data/uaq_user/tmp/leo_gate40"

    # --- the LoRA under test -------------------------------------------------
    # `target_modules` MUST be the bare string. A list containing "all-linear"
    # would not expand: `_maybe_include_all_linear_layers` early-returns unless
    # `target_modules` is a `str` equal to the shorthand, and "all-linear" would
    # then be matched as a literal suffix, hitting nothing. Verified against the
    # INSTALLED peft, not against GitHub main.
    target_modules: str = "all-linear"
    modules_to_save: tuple[str, ...] = MERGER_MODULES
    lora_rank: int = 8
    lora_alpha: int = 16          # the arm's ratio (4 -> 2); irrelevant to the gate
    lora_dropout: float = 0.0
    max_seq_length: int = 1024
    max_pixels: int = 256 * 256   # small on purpose: the gate is not about resolution

    # --- the smoke -----------------------------------------------------------
    n_rows: int = 32
    n_steps: int = 20
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    seed: int = 42

    # --- criteria ------------------------------------------------------------
    # sum|delta| over the connector weights must exceed this to count as "moved".
    # Deliberately not 0.0: bf16 round-trips are not bit-exact, so an exact-zero
    # threshold would make G2/G3 answer a numerics question instead of ours.
    min_abs_delta: float = 1e-3

    out_json: str = "RESULTS_merge_gate.json"

    @property
    def merged_dir(self) -> str:
        return str(Path(self.work_dir) / "merged")

    @property
    def data_dir(self) -> str:
        return str(Path(self.work_dir) / "data")


class GateFailure(AssertionError):
    """A gate that fires is a FINDING. Raised so papermill exits non-zero."""


# ---------------------------------------------------------------------------
# synthetic data — ours, generated fresh, no dataset of any kind
# ---------------------------------------------------------------------------

def build_synthetic_data(cfg: MergeGateConfig) -> str:
    """Noise images + invented Q/A. Returns the path to the jsonl.

    The gate measures whether a merge PRESERVES weights, not whether the model
    learns anything, so the content is irrelevant — all that is required is that
    gradient flows. Generating our own also keeps the gate free of any dataset
    licensing question by construction, which is what makes a shared machine a
    legitimate place to run it.
    """
    import numpy as np
    from PIL import Image

    data_dir = Path(cfg.data_dir)
    (data_dir / "img").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    rows = []
    for i in range(cfg.n_rows):
        p = data_dir / "img" / f"noise_{i:03d}.png"
        arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(p)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": "<image>How many objects?"},
                    {"role": "assistant", "content": str(i % 5)},
                ],
                "images": [str(p)],
            }
        )

    jsonl = data_dir / "gate_train.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    log.info("synthetic data: %d rows -> %s", len(rows), jsonl)
    return str(jsonl)


# ---------------------------------------------------------------------------
# the three assertions
# ---------------------------------------------------------------------------

def snapshot_merger_weights(model) -> dict[str, "object"]:
    """Clone the connector weights off the LIVE model, before training.

    Cloned rather than referenced: PEFT wraps these modules in place, and a
    reference would follow the training and silently compare a tensor to itself.
    """
    import torch

    out = {}
    for name, p in model.named_parameters():
        if MERGER_TENSOR_PREFIX in name and p.dim() > 0:
            out[name] = p.detach().clone().float().cpu()
    if not out:
        raise GateFailure(
            "no parameter matched the connector prefix "
            f"{MERGER_TENSOR_PREFIX!r} on the live model — the module census is "
            "wrong before anything was even trained, so nothing below can be read"
        )
    return out


def assert_g1_coverage(model, cfg: MergeGateConfig) -> dict:
    """G1 — the merger is WRAPPED and the LoRA legs survive. RAISES.

    Runs first because a non-matching entry is silent in PEFT: `ValueError` fires
    only when nothing at all matched, and a partial miss produces at most a
    `RuntimeWarning` nobody reads in a log. A green G3 on an unwrapped connector
    would just be "base == base" and would read as a PASS.
    """
    wrapped = [
        n for n, m in model.named_modules()
        if type(m).__name__ == "ModulesToSaveWrapper"
    ]
    targeted = list(getattr(model, "targeted_module_names", []) or [])

    missing = [
        m for m in cfg.modules_to_save
        if not any(w.endswith(m) or m.endswith(w) for w in wrapped)
    ]
    if missing:
        raise GateFailure(
            f"G1 FAIL — modules_to_save did not wrap {missing}. Wrapped: {wrapped}. "
            "This is the silent-failure mode: PEFT raises only if NOTHING matched, "
            "so a partial miss reaches training with no error at all."
        )
    if len(wrapped) != len(cfg.modules_to_save):
        raise GateFailure(
            f"G1 FAIL — expected exactly {len(cfg.modules_to_save)} wrapped modules, "
            f"got {len(wrapped)}: {wrapped}. Neither the expected count nor zero is a "
            "number to accept silently; read it before training."
        )
    if not targeted:
        raise GateFailure(
            "G1 FAIL — `targeted_module_names` is empty, so the LoRA legs were "
            "REPLACED rather than extended. Coverage must be extended, never replaced."
        )

    log.info("G1 PASS — wrapped=%s | LoRA targets=%d", wrapped, len(targeted))
    return {"wrapped": wrapped, "n_targeted": len(targeted)}


def assert_g2_trained(model, before: dict, cfg: MergeGateConfig) -> dict:
    """G2 — the connector actually MOVED. RAISES.

    `rc=0` is not evidence a run trained: AdamW's decoupled weight decay moves
    every tensor at zero gradient, so presence-of-change alone cannot separate a
    real run from a no-op. The magnitude is what separates them.
    """
    import torch

    deltas = {}
    for name, p in model.named_parameters():
        if name in before:
            deltas[name] = float((p.detach().float().cpu() - before[name]).abs().sum())

    # Names shift under PEFT wrapping (`...modules_to_save.default...`), so fall
    # back to matching on the connector prefix rather than on exact names.
    if not deltas:
        for name, p in model.named_parameters():
            if MERGER_TENSOR_PREFIX.rstrip(".") in name and "modules_to_save" in name:
                key = next((b for b in before if b.split(".")[-2:] == name.split(".")[-2:]), None)
                if key is not None:
                    deltas[name] = float((p.detach().float().cpu() - before[key]).abs().sum())

    total = sum(deltas.values())
    if total <= cfg.min_abs_delta:
        raise GateFailure(
            f"G2 FAIL — the connector did not move: sum|delta| = {total:.6g} "
            f"<= {cfg.min_abs_delta}. It was wrapped (G1 passed) but received no "
            "gradient, so G3 below would be meaningless. Read the grad_norm log."
        )
    log.info("G2 PASS — sum|delta| over the connector = %.6g", total)
    return {"sum_abs_delta_trained": total, "per_tensor": deltas}


def assert_g3_merge_preserved(cfg: MergeGateConfig, before: dict) -> dict:
    """G3 — 🎯 THE QUESTION. Did the MERGED checkpoint keep the movement? RAISES.

    Reads the merged checkpoint off disk, not the in-memory model: the failure we
    are hunting lives in the SERIALISER, so anything still in memory would answer
    a different question.
    """
    from safetensors import safe_open

    merged = Path(cfg.merged_dir)
    shards = sorted(merged.glob("*.safetensors"))
    if not shards:
        raise GateFailure(
            f"G3 FAIL — no .safetensors under {merged}. The merge produced no "
            "weights at all, which is a harder failure than the one being tested."
        )

    found, deltas = {}, {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith(MERGER_TENSOR_PREFIX):
                    found[k] = f.get_tensor(k).float().cpu()

    if not found:
        raise GateFailure(
            f"G3 FAIL — the merged checkpoint contains NO tensor under "
            f"{MERGER_TENSOR_PREFIX!r}. The merge dropped the connector entirely."
        )

    for k, t in found.items():
        base = next((v for kb, v in before.items() if kb.endswith(k.split(".", 1)[-1])), None)
        if base is None:
            base = before.get(k)
        if base is not None and base.shape == t.shape:
            deltas[k] = float((t - base).abs().sum())

    total = sum(deltas.values())
    if total <= cfg.min_abs_delta:
        raise GateFailure(
            f"G3 FAIL — 🎯 THE MERGE ATE THE CONNECTOR. The merged checkpoint's "
            f"{MERGER_TENSOR_PREFIX}* are byte-identical to base "
            f"(sum|delta| = {total:.6g} <= {cfg.min_abs_delta}) even though G2 proved "
            "they trained. `modules_to_save` is trained and then dropped at save "
            "time.\n\n=> Take PATH B (explicit suffix list, no modules_to_save). "
            "This is a publishable result, not an obstacle."
        )

    log.info("G3 PASS — merged connector differs from base, sum|delta| = %.6g", total)
    return {"sum_abs_delta_merged": total, "n_tensors": len(found), "per_tensor": deltas}


# ---------------------------------------------------------------------------
# provenance — a PASS that cannot be attributed is not transferable
# ---------------------------------------------------------------------------

def versions() -> dict:
    """The gate proves something about a code path, so the path must be recorded.

    A PASS under a different `unsloth`/`unsloth_zoo` than the arm will run does
    not transfer, and the only way to notice later is to have written it down.
    """
    import importlib.metadata as md

    out = {"host": platform.node(), "python": platform.python_version()}
    for pkg in ("unsloth", "unsloth_zoo", "peft", "transformers", "torch", "trl", "accelerate"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["capability"] = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return out


def write_result(cfg: MergeGateConfig, payload: dict) -> str:
    """Commit the result whatever the verdict. A FAIL is a finding worth keeping."""
    p = Path(cfg.work_dir) / cfg.out_json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    log.info("result -> %s", p)
    return str(p)


# ---------------------------------------------------------------------------
# the run — mirrors the API of the smoke that is KNOWN to work in this env
# (`~/storage/smoke/smoke_unsloth.py` on UNAM), so the gate is not also a test
# of whether we guessed Unsloth's signatures right
# ---------------------------------------------------------------------------

def _to_messages(rec: dict) -> dict:
    """`<image>` + `images` -> the parts form the collator expects.

    Reimplemented here rather than importing UNAM's `unsloth_data.py`: that file
    lives outside the repo, so depending on it would make the gate unreproducible
    from a checkout. It is six lines; the coupling is not worth it.
    """
    from PIL import Image

    out = []
    for m in rec["messages"]:
        if m["role"] == "user":
            parts = []
            for p in rec.get("images", []):
                parts.append({"type": "image", "image": Image.open(p).convert("RGB")})
            parts.append({"type": "text", "text": m["content"].replace("<image>", "")})
            out.append({"role": "user", "content": parts})
        else:
            out.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    return {"messages": out}


def run_gate(cfg: MergeGateConfig) -> dict:
    """Build -> G1 -> train -> G2 -> merge -> G3. Raises on the first failure.

    Every raise is a GateFailure, so papermill turns it into a non-zero exit and
    the chain stops before any 5.9 h arm is launched.
    """
    import os

    os.environ.setdefault("HF_HOME", cfg.hf_home)
    # 🔴 letal para entrenar: mata el merge de Unsloth. Explicitly OFF.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    import unsloth  # noqa: F401  MUST precede transformers or the patches are lost
    import torch
    from unsloth import FastVisionModel

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result: dict = {"config": asdict(cfg), "versions": versions(), "verdict": "INCOMPLETE"}

    try:
        # --- build ----------------------------------------------------------
        model, tok = FastVisionModel.from_pretrained(
            cfg.base_model, load_in_4bit=False, load_in_16bit=True,
            full_finetuning=False, max_seq_length=cfg.max_seq_length,
        )
        result["model_class"] = type(model).__name__

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", random_state=cfg.seed, use_rslora=False, loftq_config=None,
            target_modules=cfg.target_modules,          # bare string, see MergeGateConfig
            modules_to_save=list(cfg.modules_to_save),  # THE thing under test
        )

        # --- G1 -------------------------------------------------------------
        result["G1"] = assert_g1_coverage(model, cfg)
        before = snapshot_merger_weights(model)
        result["n_connector_tensors_tracked"] = len(before)

        # --- train ----------------------------------------------------------
        from trl import SFTConfig, SFTTrainer
        from unsloth.trainer import UnslothVisionDataCollator

        jsonl = build_synthetic_data(cfg)
        rows = [_to_messages(json.loads(l)) for l in open(jsonl) if l.strip()]

        FastVisionModel.for_training(model)
        tr = SFTTrainer(
            model=model, train_dataset=rows,
            data_collator=UnslothVisionDataCollator(model, tok),
            args=SFTConfig(
                per_device_train_batch_size=cfg.per_device_train_batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                max_steps=cfg.n_steps, learning_rate=cfg.learning_rate,
                logging_steps=1, optim="adamw_8bit", lr_scheduler_type="cosine",
                seed=cfg.seed, output_dir=str(Path(cfg.work_dir) / "trainer"),
                report_to="none", remove_unused_columns=False, dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True}, max_length=cfg.max_seq_length,
            ),
        )
        st = tr.train()
        result["train_loss"] = float(st.training_loss)
        # grad_norm is the corroborating instrument (rung 39's precedent)
        result["grad_norms"] = [
            h.get("grad_norm") for h in tr.state.log_history if "grad_norm" in h
        ]

        # --- G2 -------------------------------------------------------------
        result["G2"] = assert_g2_trained(model, before, cfg)

        # --- merge ----------------------------------------------------------
        model.save_pretrained_merged(cfg.merged_dir, tok)
        result["merged_files"] = sorted(p.name for p in Path(cfg.merged_dir).iterdir())

        # --- G3 -------------------------------------------------------------
        result["G3"] = assert_g3_merge_preserved(cfg, before)

        result["verdict"] = "PASS — path A is live (modules_to_save survives the merge)"
        result["path"] = "A"

    except GateFailure as e:
        result["verdict"] = f"FAIL — {e}"
        result["path"] = "B"
        result["elapsed_s"] = time.perf_counter() - t0
        write_result(cfg, result)
        raise
    except Exception as e:  # noqa: BLE001 — an unexpected error is still a finding
        result["verdict"] = f"ERROR — {type(e).__name__}: {e}"
        result["elapsed_s"] = time.perf_counter() - t0
        write_result(cfg, result)
        raise

    result["elapsed_s"] = time.perf_counter() - t0
    write_result(cfg, result)
    return result
