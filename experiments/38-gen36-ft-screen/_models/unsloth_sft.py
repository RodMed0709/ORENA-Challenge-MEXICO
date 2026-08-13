"""Brazo real del rung 38 — SFT con Unsloth sobre un backbone gen-3.5/3.6.

CORRE EN EL POD (donde los datos del challenge viven legítimamente), NO en UNAM.
Toda la tubería que usa está validada en UNAM con cero datos del challenge:
  · RESULTS_smoke_unsloth.json  — cargar → LoRA → 2 pasos → merge, 6/6 etapas
  · RESULTS_eval_path.json      — el merge se lee con el camino de screen_engine
  · RESULTS_viability_v2.json   — por qué NO es ms-swift

USO (celda de notebook, según CONSTITUTION §VIII):
    from _models.unsloth_sft import Config, main
    main(Config(run_name="38_qwen36_27b_v1"))

⚠️ import unsloth VA PRIMERO. La librería avisa de que importar transformers antes
   pierde los parches y cambia el comportamiento de memoria.
"""
import unsloth  # noqa: F401  ← primero, a propósito
from unsloth import FastVisionModel

import json, os, subprocess, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_tools"))
from unsloth_data import load_dataset  # noqa: E402


@dataclass
class Config:
    # ── sujeto ────────────────────────────────────────────────────────────────
    model: str = "Qwen/Qwen3.6-27B"
    # 🔁 alternativa medida y NO elegida: "Qwen/Qwen3.5-9B" — 22 GB para entrenar,
    #    ~18 GB en bf16 al servir, luego SIN ruta FP8, y casi igualado en tamaño a
    #    nuestro 8B (lectura más limpia). Un solo cambio de línea.
    run_name: str = "38_qwen36_27b_v1"
    exp_dir: Path = Path(__file__).resolve().parents[1]

    # ── datos: el MISMO fichero de rung 18, con su sha256 aserido ─────────────
    # ⚠️ APUNTA A `/workspace/repo`, NO al checkout donde vive este codigo. No es
    #    un error: `experiments/*/runs/` esta GITIGNOREADO, asi que el train.jsonl
    #    no viaja con el repo -- existe solo en el checkout que corrio el rung 18,
    #    que es el compartido. Verificado 2026-08-13: 14.415 filas, 51 MB, y ni
    #    repo_leo ni repo_rodri lo tienen. Para nosotros es de SOLO LECTURA.
    #    Los frames que referencia viven en /workspace/frames_cache (15.213 ficheros).
    train_jsonl: Path = Path(
        "/workspace/repo/experiments/18-count-aug/runs/18_count_aug_v1/train.jsonl")
    train_sha256: str = "180e28f0325674197d52706beeabd846851bdd2875264b5c3505e0debfbd8e8b"

    # ── receta A2, trasladada (NO re-derivada) ────────────────────────────────
    learning_rate: float = 2e-4     # el óptimo del 8B; ver la advertencia de abajo
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    num_train_epochs: int = 1       # el screen es de 1 época, contra A2 ep1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16     # batch efectivo 16, como toda la escalera
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    seed: int = 42
    max_seq_length: int = 2048

    # ── superficie que LoRA toca ──────────────────────────────────────────────
    finetune_vision_layers: bool = True       # equivalente a --freeze_vit false
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    # ⚠️ NO alcanza el merger. Medido en RESULTS_smoke_unsloth.json (merger 0) y
    #    explicado en [[the-merger-is-unreachable-by-default]]. Es una limitación
    #    conocida y COMPARTIDA con ms-swift, no una regresión de este brazo.

    # ── entorno ───────────────────────────────────────────────────────────────
    max_pixels: int = 1280 * 720    # 🔴 Unsloth por defecto usa 512; hay que fijarlo
    gpu: int = 0
    require_free_gpu: bool = True
    smoke: bool = False
    smoke_rows: int = 32
    smoke_steps: int = 2

    @property
    def run_dir(self) -> Path:
        return self.exp_dir / "runs" / self.run_name


# ── guardas ──────────────────────────────────────────────────────────────────

def assert_dataset_is_the_controls(cfg: Config) -> str:
    """El brazo debe entrenar sobre el MISMO fichero que la escalera, byte a byte."""
    import hashlib
    h = hashlib.sha256(Path(cfg.train_jsonl).read_bytes()).hexdigest()
    if cfg.train_sha256 and h != cfg.train_sha256:
        raise AssertionError(
            f"train.jsonl NO es el de la escalera: {h} != {cfg.train_sha256}. "
            "Un dataset distinto convierte el brazo en dos variables."
        )
    return h


def assert_gpu_is_free(cfg: Config) -> dict:
    """Sin scheduler, nadie reserva nada. No lanzar sobre el trabajo de otro."""
    q = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    used = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    state = {ln.split(",")[0].strip(): int(ln.split(",")[1]) for ln in used if ln}
    mib = state.get(str(cfg.gpu), 0)
    if cfg.require_free_gpu and mib > 1024:
        raise AssertionError(
            f"GPU {cfg.gpu} tiene {mib} MiB en uso por otro proceso:\n{q}\n"
            "Reprograma o elige otra. (require_free_gpu=False para forzar.)"
        )
    return {"per_gpu_used_mib": state, "compute_apps": q}


def _heartbeat(run_dir: Path, **kw) -> None:
    """Para saber dónde iba un run de horas sin abrir un train.log gigante."""
    (run_dir / "HEARTBEAT.json").write_text(
        json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}, indent=2))


# ── el brazo ─────────────────────────────────────────────────────────────────

def main(cfg: Config) -> dict:
    import torch
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    R: dict = {"config": {k: str(v) for k, v in asdict(cfg).items()}}

    R["gpu_state"] = assert_gpu_is_free(cfg)
    R["train_sha256"] = assert_dataset_is_the_controls(cfg)
    print(f"✅ dataset verificado · GPU {cfg.gpu} libre")

    ds = load_dataset(cfg.train_jsonl, limit=cfg.smoke_rows if cfg.smoke else None)
    print(f"dataset: {len(ds)} filas")
    R["n_rows"] = len(ds)

    model, tok = FastVisionModel.from_pretrained(
        cfg.model, load_in_4bit=False, load_in_16bit=True,
        full_finetuning=False, max_seq_length=cfg.max_seq_length,
    )
    R["model_class"] = type(model).__name__

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision_layers,
        finetune_language_layers=cfg.finetune_language_layers,
        finetune_attention_modules=cfg.finetune_attention_modules,
        finetune_mlp_modules=cfg.finetune_mlp_modules,
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", random_state=cfg.seed, use_rslora=False, loftq_config=None,
        target_modules="all-linear",
    )

    # censo de lo que LoRA realmente tocó — se registra SIEMPRE, no se asume
    bases = sorted({n.rsplit(".lora_A", 1)[0] for n, _ in model.named_modules()
                    if "lora_A" in n})
    R["lora_by_part"] = {k: sum(k in b for b in bases)
                         for k in ("visual", "merger", "deepstack", "language_model")}
    R["n_lora_modules"] = len(bases)
    R["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("LoRA:", R["lora_by_part"], f"({R['trainable']:,} entrenables)")

    from trl import SFTTrainer, SFTConfig
    from unsloth.trainer import UnslothVisionDataCollator
    FastVisionModel.for_training(model)

    args = SFTConfig(
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        seed=cfg.seed,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",          # recuperable si muere a media noche
        output_dir=str(cfg.run_dir / "ckpt"),
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=cfg.max_seq_length,
        **({"max_steps": cfg.smoke_steps} if cfg.smoke else {}),
    )
    tr = SFTTrainer(model=model, train_dataset=ds,
                    data_collator=UnslothVisionDataCollator(model, tok), args=args)

    _heartbeat(cfg.run_dir, stage="train_start", rows=len(ds))
    t0 = time.time()
    stats = tr.train()
    R["train_secs"] = round(time.time() - t0, 1)
    R["train_loss"] = float(stats.training_loss)
    R["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(f"loss {R['train_loss']:.4f} · {R['train_secs']}s · pico {R['peak_vram_gib']} GiB")

    model.save_pretrained(str(cfg.run_dir / "adapter"))
    tok.save_pretrained(str(cfg.run_dir / "adapter"))
    _heartbeat(cfg.run_dir, stage="merging")
    model.save_pretrained_merged(str(cfg.run_dir / "merged"), tok)

    # 🟡 Unsloth escribe processor_config.json y NO preprocessor_config.json.
    #    AutoProcessor lo toleró en la verificación; el contenedor puede que no.
    merged = cfg.run_dir / "merged"
    R["merged_files"] = sorted(p.name for p in merged.iterdir())
    R["merged_missing_preprocessor"] = "preprocessor_config.json" not in R["merged_files"]

    (cfg.run_dir / "RESULTS_train.json").write_text(json.dumps(R, indent=2))
    _heartbeat(cfg.run_dir, stage="done", loss=R["train_loss"])
    print("✅ brazo completo →", cfg.run_dir)
    return R
