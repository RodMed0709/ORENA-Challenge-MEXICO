"""G-VIABILITY · V2 — ¿puede ms-swift apuntar a la torre visual de un gen-3.5/3.6?

NO entrena. NO descarga pesos. Solo lee configs (KB) y la tabla de arquitecturas
de la ms-swift INSTALADA.

Principio heredado de `experiments/27-vit-lr-decouple/_models/vit_lr_train.py:418`:
la tabla se IMPORTA de la version instalada; si no se puede leer, se LEVANTA
excepcion en vez de sustituir una copia a mano. Un gate que se inventa los
prefijos es peor que no tener gate.

Salida: RESULTS_viability_v2.json
"""
import json, sys, traceback

OUT = "RESULTS_viability_v2.json"
res = {"checks": {}, "verdict": None}


def _mapping():
    errors = []
    for mod in ("swift.model.model_arch", "swift.llm.model.model_arch"):
        try:
            return __import__(mod, fromlist=["MODEL_ARCH_MAPPING"]).MODEL_ARCH_MAPPING
        except Exception as exc:
            errors.append(f"{mod}: {type(exc).__name__}: {exc}")
    raise ImportError(f"no se pudo importar MODEL_ARCH_MAPPING: {errors}")


def _as_list(v):
    return [] if v is None else ([v] if isinstance(v, str) else list(v))


def prefixes(mapping, key):
    a = mapping[key]
    return {
        "language_model": _as_list(a.language_model),
        "aligner": _as_list(a.aligner),
        "vision_tower": _as_list(a.vision_tower),
    }


# ── 1. versiones ────────────────────────────────────────────────────────────
import torch, transformers, swift
res["versions"] = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "ms_swift": swift.__version__,
    "cuda": torch.version.cuda,
    "gpus": torch.cuda.device_count(),
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
}
print("versiones:", json.dumps(res["versions"]))

# ── 2. transformers PUEDE leer el config de un qwen3_5? (lado transformers) ──
from transformers import AutoConfig
for mid in ("Qwen/Qwen3.5-4B", "Qwen/Qwen3-VL-8B-Instruct"):
    try:
        c = AutoConfig.from_pretrained(mid, trust_remote_code=False)
        res["checks"][f"autoconfig::{mid}"] = {
            "ok": True, "model_type": c.model_type,
            "architectures": getattr(c, "architectures", None),
            "has_vision_config": hasattr(c, "vision_config"),
        }
    except Exception as exc:
        res["checks"][f"autoconfig::{mid}"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(f"AutoConfig {mid}: {res['checks'][f'autoconfig::{mid}']}")

# ── 3. LA PREGUNTA: la tabla de arquitecturas de ms-swift ───────────────────
try:
    mapping = _mapping()
    keys = sorted(mapping.keys())
    res["checks"]["arch_mapping_size"] = len(keys)

    # control: qwen3_vl DEBE estar y DEBE dar vision_tower
    ctrl = prefixes(mapping, "qwen3_vl")
    res["checks"]["control_qwen3_vl"] = ctrl
    print("CONTROL qwen3_vl ->", json.dumps(ctrl))
    assert ctrl["vision_tower"], "el control no da vision_tower: la sonda esta rota, no el modelo"

    # sujeto
    present = [k for k in keys if "qwen3_5" in k]
    res["checks"]["qwen3_5_keys_present"] = present
    if present:
        res["checks"]["subject_qwen3_5"] = {k: prefixes(mapping, k) for k in present}
        vt = any(prefixes(mapping, k)["vision_tower"] for k in present)
        res["verdict"] = "PASS" if vt else "FAIL:registrado_pero_sin_vision_tower"
    else:
        res["verdict"] = "FAIL:qwen3_5_no_registrado"
    print("claves qwen3_5 en la tabla:", present or "NINGUNA")
except Exception:
    res["checks"]["arch_mapping_error"] = traceback.format_exc()
    res["verdict"] = "ERROR"

# ── 4. y el registro de modelos? (distinto de la tabla de arquitecturas) ────
try:
    from swift.llm import MODEL_MAPPING
    mm = [k for k in MODEL_MAPPING if "qwen3_5" in k or "qwen3.5" in k.lower()]
    res["checks"]["model_mapping_qwen3_5"] = mm[:20]
    print("MODEL_MAPPING con qwen3_5:", mm[:10] or "NINGUNA")
except Exception as exc:
    res["checks"]["model_mapping_error"] = f"{type(exc).__name__}: {exc}"

json.dump(res, open(OUT, "w"), indent=2)
print("\n=========================================")
print("VEREDICTO V2:", res["verdict"])
print("=========================================")
print("escrito en", OUT)
sys.exit(0 if res["verdict"] == "PASS" else 1)
