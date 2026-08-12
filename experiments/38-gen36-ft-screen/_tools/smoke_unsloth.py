"""SMOKE de tuberia Unsloth sobre un gen-3.5 pequeno. NO es un experimento.

Objetivo: que TODO el camino de codigo funcione antes de gastar pod con el 27B.
    cargar -> LoRA -> 2 pasos -> auditar modulos -> guardar -> merge

Sujeto: Qwen3.5-2B (~5 GB). Misma `model_type=qwen3_5` y misma clase
`Qwen3_5ForConditionalGeneration` que el 27B, luego mismo camino de codigo.

Datos: frames PUBLICOS de CholecT50 (colecistectomia laparoscopica), enrutados
en NUESTRO formato exacto. Cero datos del challenge.
"""
import unsloth  # noqa: F401  ← DEBE ir primero: la propia libreria avisa de que
                #                 importar transformers antes pierde los parches.
import json, os, sys, time, traceback
from pathlib import Path

import torch
from unsloth import FastVisionModel

sys.path.insert(0, str(Path(__file__).parent))
from unsloth_data import load_dataset  # noqa: E402

MODEL   = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3.5-2B")
JSONL   = os.environ.get("SMOKE_JSONL", "smoke_train.jsonl")
OUTDIR  = Path(os.environ.get("SMOKE_OUT", "smoke_out"))
RESULTS = {"model": MODEL, "stages": {}}


def stage(name):
    def deco(fn):
        def wrap(*a, **k):
            t0 = time.time()
            print(f"\n{'='*62}\n=== {name}\n{'='*62}")
            try:
                out = fn(*a, **k)
                RESULTS["stages"][name] = {"ok": True, "secs": round(time.time()-t0, 1)}
                return out
            except Exception:
                RESULTS["stages"][name] = {"ok": False, "secs": round(time.time()-t0, 1),
                                           "error": traceback.format_exc()[-1500:]}
                print(traceback.format_exc())
                json.dump(RESULTS, open("RESULTS_smoke_unsloth.json", "w"), indent=2)
                raise
        return wrap
    return deco


def vram(tag):
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated()/2**30
    p = torch.cuda.max_memory_allocated()/2**30
    print(f"    VRAM [{tag}] actual {a:.2f} GiB | pico {p:.2f} GiB")
    RESULTS.setdefault("vram_gib", {})[tag] = {"current": round(a,2), "peak": round(p,2)}


@stage("1_cargar_modelo")
def load():
    model, tok = FastVisionModel.from_pretrained(
        MODEL, load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
        max_seq_length=2048,
    )
    print("   clase:", type(model).__name__)
    RESULTS["model_class"] = type(model).__name__
    vram("tras_cargar")
    return model, tok


@stage("2_aplicar_lora")
def peft(model):
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=8, lora_alpha=32, lora_dropout=0.0, bias="none",
        random_state=42, use_rslora=False, loftq_config=None,
        target_modules="all-linear",
    )
    vram("tras_lora")
    return model


@stage("3_auditar_modulos")     # ← responde la pregunta del merger, gratis
def audit(model):
    hit = [n for n, m in model.named_modules() if "lora_A" in n]
    bases = sorted({n.rsplit(".lora_A", 1)[0] for n in hit})
    RESULTS["n_lora_modules"] = len(bases)
    def has(sub): return [b for b in bases if sub in b]
    for tag in ("visual", "merger", "deepstack", "language_model", "vision"):
        f = has(tag)
        RESULTS.setdefault("lora_by_part", {})[tag] = len(f)
        print(f"    {tag:16s}: {len(f):4d} modulos" + (f"   p.ej. {f[0]}" if f else ""))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    RESULTS["trainable"] = trainable; RESULTS["total_params"] = total
    print(f"    entrenables: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    return bases


@stage("4_dos_pasos")
def train(model, tok):
    from trl import SFTTrainer, SFTConfig
    from unsloth.trainer import UnslothVisionDataCollator
    ds = load_dataset(JSONL, lazy=False)
    print(f"    dataset: {len(ds)} filas")
    FastVisionModel.for_training(model)
    tr = SFTTrainer(
        model=model, train_dataset=ds,
        data_collator=UnslothVisionDataCollator(model, tok),
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            max_steps=2, learning_rate=2e-4, logging_steps=1,
            optim="adamw_8bit", lr_scheduler_type="cosine", seed=42,
            output_dir=str(OUTDIR/"trainer"), report_to="none",
            remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=2048,
        ),
    )
    st = tr.train()
    RESULTS["train_loss"] = float(st.training_loss)
    print("    loss:", st.training_loss)
    vram("tras_entrenar")


@stage("5_guardar_adapter")
def save(model, tok):
    p = OUTDIR/"adapter"
    model.save_pretrained(str(p)); tok.save_pretrained(str(p))
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())/2**20
    print(f"    adapter en {p} ({n:.1f} MB)")
    RESULTS["adapter_mib"] = round(n, 1)


@stage("6_merge_16bit")
def merge(model, tok):
    p = OUTDIR/"merged"
    model.save_pretrained_merged(str(p), tok)
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())/2**30
    print(f"    merged en {p} ({n:.2f} GiB)")
    RESULTS["merged_gib"] = round(n, 2)


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    m, t = load()
    m = peft(m)
    audit(m)
    train(m, t)
    save(m, t)
    merge(m, t)
    json.dump(RESULTS, open("RESULTS_smoke_unsloth.json", "w"), indent=2)
    print("\n✅ SMOKE COMPLETO — RESULTS_smoke_unsloth.json")
