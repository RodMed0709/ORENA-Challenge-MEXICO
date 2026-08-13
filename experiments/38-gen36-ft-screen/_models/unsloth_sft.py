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

import json, os, shutil, subprocess, sys, time
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

    # ── supervivencia de un run de ~6 h ───────────────────────────────────────
    # 🔴 `save_strategy="epoch"` con 1 epoca guarda UNA sola vez, al final: morir
    #    en la hora 5 de 5,9 costaba las 5 horas enteras. 900,9 pasos / 100 => un
    #    checkpoint cada ~39 min. Ninguna de estas opciones toca la matematica del
    #    entrenamiento (misma receta, mismo seed): el brazo sigue siendo el mismo.
    save_steps: int = 100
    save_total_limit: int = 2       # ~400 MB cada uno (adapter + estado adamw_8bit)
    resume: bool = True             # reanuda solo si ya hay un checkpoint en ckpt/

    # ── cuota del volumen (ver disk_free_gib) ─────────────────────────────────
    volume_root: Path = Path("/workspace")
    volume_quota_gib: float = 640.0   # el tope de RunPod; `df` NO lo ve. None = no comprobar
    du_timeout_s: int = 300           # `du` sobre MooseFS no es gratis
    merge_need_gib: float = 60.0      # el merge escribe ~1x el modelo (27B bf16 ≈ 52 GB)

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


def disk_free_gib(cfg: "Config", need_gib: float, when: str) -> dict:
    """Espacio libre REAL en el volumen, que no es el que reporta el sistema.

    🔴 `shutil.disk_usage` es `statvfs`, o sea la MISMA mentira que `df`: en RunPod
       reporta el cluster MooseFS (~320 TB libres) y no la cuota del volumen
       (~640 GB), que es invisible para el kernel. Medido 2026-08-13: la guarda
       imprimio "319807.2 GiB libres" en un volumen con ~126 GB de margen.
       `mfsgetquota` no esta instalado, asi que no hay forma de preguntar por la
       cuota: la unica medida honesta es sumar lo ocupado y restarlo del tope.

    `du` sobre un volumen de red no es gratis, por eso va con timeout y por eso
    degrada a un aviso en vez de tumbar el run: preferimos entrenar con la duda a
    no entrenar por una comprobacion.
    """
    out = {"statvfs_free_gib_NO_FIABLE":
           round(shutil.disk_usage(cfg.run_dir).free / 2**30, 1)}
    if not cfg.volume_quota_gib:
        return out
    try:
        r = subprocess.run(["du", "-sx", "--block-size=1", str(cfg.volume_root)],
                           capture_output=True, text=True, timeout=cfg.du_timeout_s)
        used = int(r.stdout.split()[0]) / 2**30
    except Exception as e:                       # timeout, permisos, du ausente
        out["quota_check"] = f"NO CONCLUYENTE ({type(e).__name__}) — sigo sin garantia"
        print(f"⚠️  {when}: no pude medir la cuota ({type(e).__name__}); sigo a ciegas")
        return out
    free = cfg.volume_quota_gib - used
    out |= {"quota_used_gib": round(used, 1), "quota_free_gib": round(free, 1),
            "need_gib": need_gib}
    print(f"    {when}: {free:.1f} GiB libres de cuota (necesito ~{need_gib:.0f})")
    if free < need_gib:
        raise AssertionError(
            f"{when}: quedan {free:.1f} GiB de la cuota de {cfg.volume_quota_gib:.0f} "
            f"y hacen falta ~{need_gib:.0f}. Poda antes de seguir; morir a mitad "
            "deja basura que hay que barrer a mano. (volume_quota_gib=None lo salta.)"
        )
    return out


def _heartbeat(run_dir: Path, **kw) -> None:
    """Para saber dónde iba un run de horas sin abrir un train.log gigante."""
    (run_dir / "HEARTBEAT.json").write_text(
        json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}, indent=2))


def _progress_callback(run_dir: Path, t0: float, start_step: int = 0):
    """Latido POR PASO + `train.log` en disco.

    Antes el heartbeat solo marcaba train_start/merging/done: durante ~6 h de
    entrenamiento no decia nada, asi que no habia forma de saber por donde iba un
    run sin mirar el scrollback de tmux (que no es un fichero y se pierde).
    """
    from transformers import TrainerCallback

    class _P(TrainerCallback):
        # En un run reanudado `global_step` arranca alto pero el reloj arranca en
        # cero: dividir uno por otro daria un s/it inventado. El ritmo se mide
        # sobre los pasos de ESTA sesion, y el origen es el checkpoint del que se
        # reanudo (0 si el run es limpio).
        #
        # 🔴 Antes esto tomaba como origen el PRIMER on_log, y `logging_steps=10`
        #    hace que el primero sea el paso 10: en un run limpio salia un s/it
        #    inflado ~2x (tiempo de compilacion + 20 pasos dividido entre 10) y un
        #    ETA que mentia hacia arriba. Visto en vivo 2026-08-13, paso 10.
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            done, total = state.global_step, state.max_steps
            elapsed = time.time() - t0
            this_session = done - start_step
            s_it = elapsed / this_session if this_session > 0 else 0.0
            _heartbeat(
                run_dir, stage="training", step=done, max_steps=total,
                pct=round(100 * done / total, 1) if total else None,
                loss=logs.get("loss"),
                # `None` y no 0 en el primer latido tras reanudar: todavia no hay
                # ningun paso de esta sesion, y un ETA de 0,0 h es una mentira.
                s_per_step=round(s_it, 2) if s_it else None,
                eta_h=round((total - done) * s_it / 3600, 2) if (total and s_it) else None,
            )
            with (run_dir / "train.log").open("a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} step {done}/{total} "
                        f"{json.dumps(logs)}\n")

    return _P()


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
        # El smoke son 2 pasos: guardar solo ensucia runs/ sin comprar nada.
        **({"save_strategy": "no"} if cfg.smoke else {
            "save_strategy": "steps",
            "save_steps": cfg.save_steps,
            "save_total_limit": cfg.save_total_limit,
        }),
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

    # ¿Hay un run muerto que continuar? Si lo hay se reanuda desde su paso; si no,
    # `get_last_checkpoint` devuelve None y esto es un arranque limpio. Se registra
    # SIEMPRE en los resultados: un numero que sale de un run reanudado tiene que
    # poder identificarse como tal.
    from transformers.trainer_utils import get_last_checkpoint
    ckpt_dir = cfg.run_dir / "ckpt"
    last = get_last_checkpoint(str(ckpt_dir)) if (
        cfg.resume and not cfg.smoke and ckpt_dir.is_dir()) else None
    R["resumed_from"] = last
    if last:
        print(f"⏩ reanudando desde {last}")

    # El disco tambien se comprueba ANTES de entrenar, no solo antes del merge:
    # quedarse sin cuota a mitad de la epoca tira el run igual que tiro el merge.
    # Aqui hace falta poco (2 checkpoints x ~0,4 GB), pero si ya estamos al limite
    # antes de empezar, mejor saberlo ahora que en la hora 5.
    R["disk_before_train"] = disk_free_gib(cfg, need_gib=5.0, when="antes de entrenar")

    _heartbeat(cfg.run_dir, stage="train_start", rows=len(ds), resumed_from=last)
    t0 = time.time()
    # el origen del ritmo: el paso del checkpoint reanudado, 0 si el run es limpio
    start_step = int(Path(last).name.split("-")[-1]) if last else 0
    tr.add_callback(_progress_callback(cfg.run_dir, t0, start_step))
    stats = tr.train(resume_from_checkpoint=last)
    R["train_secs"] = round(time.time() - t0, 1)
    R["train_loss"] = float(stats.training_loss)
    R["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(f"loss {R['train_loss']:.4f} · {R['train_secs']}s · pico {R['peak_vram_gib']} GiB")

    # 🔴 Escribir los resultados ANTES del merge. Medido 2026-08-13: el merge
    #    murio por cuota de disco y se llevo por delante todos los numeros de
    #    entrenamiento, que ya estaban calculados. Un paso caro y fragil no debe
    #    poder borrar la evidencia de uno que ya salio bien.
    (cfg.run_dir / "RESULTS_train.json").write_text(json.dumps(R, indent=2))

    model.save_pretrained(str(cfg.run_dir / "adapter"))
    tok.save_pretrained(str(cfg.run_dir / "adapter"))

    # El merge necesita ~1x el tamano del modelo en disco. Comprobarlo antes de
    # empezar a copiar 15 shards: fallar a mitad deja basura que hay que barrer.
    R["disk_before_merge"] = disk_free_gib(
        cfg, need_gib=cfg.merge_need_gib, when="antes del merge")

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
