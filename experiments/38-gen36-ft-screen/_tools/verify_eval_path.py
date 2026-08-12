"""¿Carga `screen_engine.py` (rung 23) un checkpoint MERGEADO POR UNSLOTH?

Es el riesgo #1 del rung 38: sin esto no hay veredicto por bien que entrene.

Replica FIELMENTE el camino de `experiments/23-backbone-screen/_tools/screen_engine.py:48-115`
-- AutoProcessor(max_pixels) + AutoModelForImageTextToText(dtype="auto") + apply_chat_template con
enable_thinking=False + generate(do_sample=False). No importa la clase real porque `frame.engine`
arrastra dependencias del SDK que no viven en el env de Unsloth; lo que se prueba es el MECANISMO,
que es identico linea por linea.

Sujeto: el merge que produjo el smoke (`smoke_out/merged`, Qwen3.5-2B + LoRA).
"""
import json, sys, traceback
from pathlib import Path

MERGED   = Path(sys.argv[1] if len(sys.argv) > 1 else "smoke_out/merged")
FRAME    = Path(sys.argv[2] if len(sys.argv) > 2 else "frames")
MAXPIX   = 1280 * 720          # el mismo de nuestra receta
MAXNEW   = 32                  # el mismo tope del contenedor
SYSTEM   = "You are a surgical assistant. Answer with a short phrase."
R = {"merged": str(MERGED), "checks": {}}

print("=== 1. ¿que dejo el merge en disco? ===")
files = sorted(p.name for p in MERGED.iterdir()) if MERGED.exists() else []
R["checks"]["files"] = files
print("   ", files)
need = ["config.json", "preprocessor_config.json", "tokenizer_config.json"]
missing = [f for f in need if f not in files]
R["checks"]["missing_expected"] = missing
print("    faltan de los esperados:", missing or "ninguno")

try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    print("\n=== 2. AutoProcessor ===")
    proc = AutoProcessor.from_pretrained(str(MERGED), max_pixels=MAXPIX)
    print("   ", type(proc).__name__)
    R["checks"]["processor"] = type(proc).__name__

    print("\n=== 3. AutoModelForImageTextToText(dtype='auto') ===")
    model = AutoModelForImageTextToText.from_pretrained(
        str(MERGED), dtype="auto", device_map="cuda:0"
    ).eval()
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
    n = sum(p.numel() for p in model.parameters())
    print(f"    {type(model).__name__}, {n/1e9:.2f}B params, dtype {next(model.parameters()).dtype}")
    R["checks"]["model_class"] = type(model).__name__
    R["checks"]["params_b"] = round(n/1e9, 2)

    print("\n=== 4. generate con enable_thinking=False ===")
    imgs = sorted(FRAME.glob("*.png")) + sorted(FRAME.glob("*.jpg"))
    img = Image.open(imgs[0]).convert("RGB")
    q = "How many foreign objects are visible?"
    # 🔴 transformers 5.x exige partes tipadas en TODOS los mensajes; screen_engine.py:69
    # pasa el system como string plano y revienta. Este es el arreglo que le falta.
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": q}]}]
    with torch.no_grad():
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        ).to(model.device)
        gen = model.generate(**inputs, max_new_tokens=MAXNEW, do_sample=False)
        plen = inputs["input_ids"].shape[1]
        out = proc.decode(gen[0][plen:], skip_special_tokens=True).strip()
    print(f"    prompt_tokens={plen}  respuesta cruda: {out!r}")
    R["checks"]["prompt_tokens"] = int(plen)
    R["checks"]["raw_answer"] = out

    # el fallo del rung 23: CoT que se come el presupuesto de tokens
    cot = any(k in out.lower() for k in ("<think", "the user wants", "let me", "step 1",
                                         "**analyze", "first,", "i need to"))
    R["checks"]["looks_like_cot"] = cot
    R["checks"]["answer_empty"] = (out == "")
    print("    parece CoT:", cot, "| vacia:", out == "")

    R["verdict"] = ("FAIL:respuesta_vacia" if out == "" else
                    "FAIL:emite_CoT" if cot else "PASS")
except Exception:
    R["checks"]["error"] = traceback.format_exc()[-2000:]
    R["verdict"] = "FAIL:excepcion"
    print(traceback.format_exc())

json.dump(R, open("RESULTS_eval_path.json", "w"), indent=2)
print("\n" + "="*54)
print("VEREDICTO RUTA DE EVAL:", R["verdict"])
print("="*54)
