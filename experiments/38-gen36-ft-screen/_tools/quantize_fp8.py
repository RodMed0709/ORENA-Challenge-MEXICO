"""Cuantiza un merge gen-3.5/3.6 a FP8 con la receta OFICIAL de llm-compressor.

Por que existe: `FineGrainedFP8Config` de transformers FALLA sobre esta familia
(RESULTS_fp8_probe.json) -- cuantiza en bloques 128x128 y las proyecciones del
Gated DeltaNet (`linear_attn.in_proj_a/b`) son 16x2048. No es un fallo del merge:
es el esquema equivocado.

La receta correcta viene de la model card de `RedHatAI/Qwen3.5-4B-FP8-dynamic`, y
coincide con la lista de exclusion del `Qwen3.6-27B-FP8` oficial (871 modulos:
336 linear_attn, 246 visual, 289 embeddings/norms).

🔑 `FP8_DYNAMIC` es DATA-FREE -- no necesita calibracion. Luego este paso se puede
   correr en UNAM aunque el entrenamiento haya sido en el pod: NO toca datos del
   challenge.

USO:  python quantize_fp8.py <merge_bf16> <destino_fp8> [frames_para_verificar]
"""
import json, sys, time, traceback
from pathlib import Path

# Receta oficial, copiada literal. No inventar patrones: la torre visual y el
# DeltaNet se quedan en precision original a proposito.
IGNORE_LAYERS = [
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*visual.*",
    "re:.*model.visual.*",
    "re:.*linear_attn.*",
]

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "smoke_out/merged")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "smoke_out/merged_fp8")
FRM = Path(sys.argv[3] if len(sys.argv) > 3 else "frames")
SYSTEM = "You are a surgical assistant. Answer with a short phrase."
R = {"src": str(SRC), "dst": str(DST), "ignore": IGNORE_LAYERS, "steps": {}}


def gib(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30


try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    R["capability"] = list(torch.cuda.get_device_capability(0))

    print("=== 1. cargar el merge en bf16 (en CPU, a proposito) ===")
    # 🔴 `device_map="cpu"` EXPLICITO. Esta linea no lo decia y dependia del default de
    # transformers, que SE MOVIO: en 5.14.1 `from_pretrained` coloca en el acelerador, asi
    # que el merge bf16 de 52 GB intenta entrar en una tarjeta de 47.37 GiB y muere con
    # `torch.OutOfMemoryError` en `_materialize_copy` (medido en UNAM 2026-08-17, rung 45).
    # Rung 44 midio 33.4 s y 51.75 → 33.46 GiB con este mismo archivo, o sea que entonces
    # cargaba en CPU: el comportamiento que funciono es el que ahora queda escrito.
    # La caja tiene 502 GB de RAM; el paso 4 recarga el FP8 en `cuda:0`, donde ya cabe.
    model = AutoModelForImageTextToText.from_pretrained(
        str(SRC), dtype="auto", device_map="cpu")
    proc = AutoProcessor.from_pretrained(str(SRC))
    print("   ", type(model).__name__)

    print("\n=== 2. oneshot FP8_DYNAMIC (sin calibracion) ===")
    t0 = time.time()
    oneshot(model=model,
            recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC",
                                        ignore=IGNORE_LAYERS))
    R["steps"]["quantize_secs"] = round(time.time() - t0, 1)
    print(f"    {R['steps']['quantize_secs']}s")

    print("\n=== 3. guardar comprimido ===")
    model.save_pretrained(str(DST), save_compressed=True)
    proc.save_pretrained(str(DST))
    b, f = gib(SRC), gib(DST)
    R["steps"]["size_bf16_gib"], R["steps"]["size_fp8_gib"] = round(b, 2), round(f, 2)
    R["steps"]["size_ratio"] = round(f / b, 3)
    print(f"    {b:.2f} GiB → {f:.2f} GiB  ({100*f/b:.0f} %)")

    del model
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    print("\n=== 4. RECARGAR el FP8 y generar (lo que decide) ===")
    from PIL import Image
    proc2 = AutoProcessor.from_pretrained(str(DST), max_pixels=1280 * 720)
    m = AutoModelForImageTextToText.from_pretrained(
        str(DST), dtype="auto", device_map="cuda:0").eval()
    if hasattr(m, "generation_config"):
        m.generation_config.max_length = None
    R["steps"]["vram_loaded_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    print(f"    recargado, VRAM {R['steps']['vram_loaded_gib']} GiB")

    img = Image.open(sorted(FRM.glob('*.png'))[0]).convert("RGB")
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "How many foreign objects are visible?"}]}]
    with torch.no_grad():
        inp = proc2.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt",
                                        enable_thinking=False).to(m.device)
        t0 = time.time()
        g = m.generate(**inp, max_new_tokens=32, do_sample=False)
        R["steps"]["latency_s"] = round(time.time() - t0, 3)
        out = proc2.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    R["steps"]["answer"] = out
    print(f"    respuesta: {out!r}  ({R['steps']['latency_s']}s)")
    R["verdict"] = "PASS" if out else "FAIL:respuesta_vacia"

except Exception:
    R["error"] = traceback.format_exc()[-2500:]
    R["verdict"] = "FAIL:excepcion"
    print(traceback.format_exc())

Path("RESULTS_fp8_llmcompressor.json").write_text(json.dumps(R, indent=2))
print("\n" + "=" * 54)
print("VEREDICTO FP8 (llm-compressor):", R.get("verdict"))
print("=" * 54)
