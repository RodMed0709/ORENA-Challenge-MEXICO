"""¿EXISTE la ruta de entrega FP8? Sobre el merge pequeno, en Ada (sm_89).

Por que importa: un 27B solo cabe en la L40S de 48 GB cuantizado (30,9 GB vs 55,6 bf16,
challenge_design.txt:437), y NUNCA hemos cuantizado un merge nuestro. El pod viejo era
Blackwell sm_120 y los kernels `finegrained-fp8` no tenian build para esa capability
(backbone-generation-is-not-the-lever.md:87-89). UNAM es sm_89, igual que la L40S.

Se prueba lo minimo que demuestra que la ruta existe:
    merge bf16 -> cuantizar FP8 -> guardar -> RECARGAR -> generar
Si el recargado no genera, no hay ruta por muy bien que cuantice.
"""
import json, sys, time, traceback
from pathlib import Path

SRC  = Path(sys.argv[1] if len(sys.argv) > 1 else "smoke_out/merged")
DST  = Path(sys.argv[2] if len(sys.argv) > 2 else "smoke_out/merged_fp8")
FRM  = Path("frames")
SYSTEM = "You are a surgical assistant. Answer with a short phrase."
R = {"src": str(SRC), "dst": str(DST), "steps": {}}


def size_gib(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30


try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image

    cap = torch.cuda.get_device_capability(0)
    R["capability"] = list(cap)
    print(f"capability {cap} — FP8 nativo requiere >= (8,9)")
    R["steps"]["capability_ok"] = cap >= (8, 9)

    # --- 1. que API de cuantizacion FP8 ofrece esta transformers ---
    print("\n=== 1. API disponible ===")
    avail = {}
    for name in ("FineGrainedFP8Config", "FbgemmFp8Config", "QuantoConfig", "TorchAoConfig"):
        try:
            mod = __import__("transformers", fromlist=[name])
            avail[name] = hasattr(mod, name)
        except Exception:
            avail[name] = False
    print("   ", avail)
    R["steps"]["quant_api"] = avail
    if not avail.get("FineGrainedFP8Config"):
        raise RuntimeError("FineGrainedFP8Config no disponible en esta transformers")

    from transformers import FineGrainedFP8Config

    # --- 2. cuantizar ---
    print("\n=== 2. cuantizando a FP8 ===")
    t0 = time.time()
    qmodel = AutoModelForImageTextToText.from_pretrained(
        str(SRC), dtype="auto", device_map="cuda:0",
        quantization_config=FineGrainedFP8Config(),
    )
    dt = round(time.time() - t0, 1)
    print(f"    cuantizado en {dt}s — {type(qmodel).__name__}")
    R["steps"]["quantize_secs"] = dt
    mem = torch.cuda.memory_allocated() / 2**30
    print(f"    VRAM ocupada: {mem:.2f} GiB")
    R["steps"]["vram_fp8_gib"] = round(mem, 2)

    # --- 3. guardar ---
    print("\n=== 3. guardando ===")
    qmodel.save_pretrained(str(DST))
    AutoProcessor.from_pretrained(str(SRC)).save_pretrained(str(DST))
    b16, f8 = size_gib(SRC), size_gib(DST)
    print(f"    bf16 {b16:.2f} GiB → fp8 {f8:.2f} GiB  ({100*f8/b16:.0f}%)")
    R["steps"]["size_bf16_gib"] = round(b16, 2)
    R["steps"]["size_fp8_gib"] = round(f8, 2)
    del qmodel; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # --- 4. RECARGAR y generar: lo que de verdad decide ---
    print("\n=== 4. recargar el FP8 y generar ===")
    proc = AutoProcessor.from_pretrained(str(DST), max_pixels=1280*720)
    m = AutoModelForImageTextToText.from_pretrained(
        str(DST), dtype="auto", device_map="cuda:0").eval()
    if hasattr(m, "generation_config"):
        m.generation_config.max_length = None
    print(f"    recargado: {type(m).__name__}, VRAM {torch.cuda.memory_allocated()/2**30:.2f} GiB")

    img = Image.open(sorted(FRM.glob("*.png"))[0]).convert("RGB")
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "How many foreign objects are visible?"}]}]
    with torch.no_grad():
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt",
                                       enable_thinking=False).to(m.device)
        t0 = time.time()
        g = m.generate(**inp, max_new_tokens=32, do_sample=False)
        lat = round(time.time() - t0, 3)
        out = proc.decode(g[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"    respuesta: {out!r}  ({lat}s)")
    R["steps"]["fp8_answer"] = out
    R["steps"]["fp8_latency_s"] = lat
    R["verdict"] = "PASS" if out else "FAIL:respuesta_vacia"

except Exception:
    R["error"] = traceback.format_exc()[-2500:]
    R["verdict"] = "FAIL:excepcion"
    print(traceback.format_exc())

json.dump(R, open("RESULTS_fp8_probe.json", "w"), indent=2)
print("\n" + "="*54)
print("VEREDICTO RUTA FP8:", R.get("verdict"))
print("="*54)
