"""Rung 40 — FIT GATE: does the 27B arm train inside a 48 GB card in NF4?

## The one question

Rung 38 measured the bf16 LoRA arm at a **52.64 GiB** peak and concluded, in its own README:
*"52.64 GiB does not fit a 48 GB card. The arm needs >=80 GB (A100) or the 96 GB PRO 6000."*
Rung 40's own PLAN repeats it: the arms need *"a GPU with >=80 GB"*. So every real 27B arm has
run on rented A100s, and UNAM's 2x RTX 6000 Ada 48 GB have only ever run the small gates.

QLoRA NF4 loads the base in 4 bits (~54 GB of bf16 weights -> ~14 GB), which *should* clear the
48 GB bar. **Should is not a measurement**, and this repo has already been burnt once by
reasoning about VRAM instead of reading it (`training-peak-is-not-inference-footprint`). This
file reads it, for ~20 minutes and zero dollars, before anyone rents anything.

🔴 **DDP does not substitute for this.** Unsloth added multi-GPU, but DDP *"creates one copy of
the model on each GPU"* — it scales throughput, not memory per device. Two 53 GiB replicas on
two 48 GB cards fail exactly as one does. Only the 4-bit load changes the footprint.

## What it holds fixed

Everything from the `40_B_connector_v1` arm that touches memory, so the answer transfers:
`r=8` / `alpha=32` / `dropout=0.0` / `target_modules=all-linear`, the connector in
`modules_to_save` (full module paths, per PLAN §69 — suffixes match differently), effective
batch 16 as `per_device=1 x grad_accum=16`, and REAL frames at their real size.

⚠️ **Frames are 960x540 = 518 400 px, under the arm's `max_pixels` 921 600** — so the cap never
binds and image size here is the same image size the real arm sees. Feeding smaller images
would make the gate pass on a footprint the arm never has.

## What it deliberately does NOT answer

Whether NF4 costs quality. It does not, and cannot, be read from a VRAM number. NF4 is a
**confound** against rung 40's bf16 baseline; a screening sweep whose arms are all NF4 stays
internally comparable, and the winner's final run is a separate decision.

## Reading the verdict

`peak_reserved_gib` is the binding number, not `peak_allocated_gib`: the caching allocator
reserves more than it hands out, and it is the reservation that OOMs. Rung 38's 52.64 GiB was
an *allocated* figure (`smoke_unsloth.vram`), so `peak_allocated_gib` is the like-for-like
comparison and `peak_reserved_gib` is the one that decides whether it runs.
"""
from __future__ import annotations

import unsloth  # noqa: F401  ← MUST be first; importing transformers first loses the patches.

import json
import os
import random
import time
import traceback
from pathlib import Path

import torch
from unsloth import FastVisionModel

MODEL = os.environ.get("FIT_MODEL", "Qwen/Qwen3.6-27B")
FRAMES = Path(os.environ.get("FIT_FRAMES", "/home/uaq_user/storage/frames_cache"))
OUT = Path(os.environ.get("FIT_OUT", "/home/uaq_user/storage/tmp/fit_gate_4bit"))
N_ROWS = int(os.environ.get("FIT_ROWS", "32"))
MAX_STEPS = int(os.environ.get("FIT_STEPS", "2"))
GRAD_ACCUM = int(os.environ.get("FIT_ACCUM", "16"))
CARD_GIB = float(os.environ.get("FIT_CARD_GIB", "48"))

# The arm's memory-relevant recipe. Changing any of these invalidates the transfer.
LORA = dict(r=8, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules="all-linear", random_state=42)
CONNECTOR = ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]
MAX_SEQ = 2048
SYSTEM = ("You are an expert surgical assistant. Answer the question using ONLY the visual "
          "evidence in the frame.")

R: dict = {"model": MODEL, "card_gib": CARD_GIB, "recipe": {**LORA, "modules_to_save": CONNECTOR,
           "per_device_train_batch_size": 1, "gradient_accumulation_steps": GRAD_ACCUM}}


def vram(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    a = torch.cuda.max_memory_allocated() / 2**30
    r = torch.cuda.max_memory_reserved() / 2**30
    print(f"    VRAM [{tag}] peak allocated {a:.2f} GiB | peak reserved {r:.2f} GiB")
    R.setdefault("vram_gib", {})[tag] = {"peak_allocated": round(a, 2), "peak_reserved": round(r, 2)}


def build_rows() -> list[dict]:
    """A tiny dataset of REAL frames at REAL size. Never leaves the box."""
    frames = sorted(p for p in FRAMES.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    if len(frames) < N_ROWS:
        raise FileNotFoundError(f"only {len(frames)} frames under {FRAMES}")
    rng = random.Random(42)
    picked = rng.sample(frames, N_ROWS)
    from PIL import Image
    sizes = {Image.open(p).size for p in picked[:8]}
    R["frame_sizes_sampled"] = sorted(f"{w}x{h}" for w, h in sizes)
    print("    frame sizes:", R["frame_sizes_sampled"])
    return [{"messages": [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image", "image": Image.open(p).convert("RGB")},
                                     {"type": "text", "text": "How many instruments are visible?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "2"}]}]} for p in picked]


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    print("=== 1. load 4-bit")
    model, tok = FastVisionModel.from_pretrained(
        MODEL, load_in_4bit=True, full_finetuning=False, max_seq_length=MAX_SEQ)
    R["model_class"] = type(model).__name__
    vram("after_load")

    print("=== 2. LoRA + connector in modules_to_save")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        modules_to_save=CONNECTOR, use_rslora=False, loftq_config=None, **LORA)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    R["trainable_params"] = trainable
    print(f"    trainable: {trainable:,}")
    # The connector must actually be trainable, or the gate measures a cheaper model than the arm.
    hit = [n for n, p in model.named_parameters()
           if p.requires_grad and "merger.linear_fc" in n and "lora_" not in n]
    R["connector_trainable_tensors"] = len(hit)
    if not hit:
        raise AssertionError(
            "modules_to_save did not make the connector trainable — this gate would measure a "
            f"lighter model than {MODEL}'s real arm. Checked full paths: {CONNECTOR}")
    print(f"    connector trainable tensors: {len(hit)} e.g. {hit[0]}")
    vram("after_lora")

    print(f"=== 3. {MAX_STEPS} optimizer steps x grad_accum {GRAD_ACCUM}")
    from trl import SFTConfig, SFTTrainer
    from unsloth.trainer import UnslothVisionDataCollator
    FastVisionModel.for_training(model)
    tr = SFTTrainer(
        model=model, train_dataset=build_rows(),
        data_collator=UnslothVisionDataCollator(model, tok),
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
            max_steps=MAX_STEPS, learning_rate=2e-4, logging_steps=1,
            optim="adamw_8bit", lr_scheduler_type="cosine", warmup_ratio=0.03, seed=42,
            output_dir=str(OUT / "trainer"), report_to="none",
            remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=MAX_SEQ),
    )
    st = tr.train()
    R["train_loss"] = float(st.training_loss)
    vram("after_train")

    peak_res = R["vram_gib"]["after_train"]["peak_reserved"]
    peak_all = R["vram_gib"]["after_train"]["peak_allocated"]
    R.update(peak_reserved_gib=peak_res, peak_allocated_gib=peak_all,
             headroom_gib=round(CARD_GIB - peak_res, 2),
             fits=bool(peak_res < CARD_GIB),
             bf16_reference_gib=52.64, elapsed_min=round((time.time() - t0) / 60, 1))
    R["verdict"] = ("FIT — the NF4 arm trains on one 48 GB card"
                    if R["fits"] else
                    "NO-FIT — rent >=80 GB, or drop to the 8B")
    return R


if __name__ == "__main__":
    try:
        main()
    except Exception:
        R["error"] = traceback.format_exc()[-2000:]
        print(traceback.format_exc())
    finally:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "RESULTS_fit_gate_4bit.json").write_text(json.dumps(R, indent=2))
        print("\n" + json.dumps({k: v for k, v in R.items() if k != "error"}, indent=2))
        print(f"\n-> {OUT/'RESULTS_fit_gate_4bit.json'}")
