"""Conversor train.jsonl (ms-swift) -> formato de conversacion de Unsloth.

Esta es la pieza que de otro modo se come tiempo de pod: los dos formatos NO son
compatibles y el fallo es silencioso (un `content` string donde se espera lista
de partes entrena sobre texto y tira la imagen).

NUESTRO formato (experiments/02-lora-sft/_models/lora_sft_train.py:174-181):
    {"messages": [{"role":"system","content": SYSTEM_PROMPT},
                  {"role":"user","content":"<image>{pregunta}"},
                  {"role":"assistant","content":"{respuesta}"}],
     "images": ["/ruta/frame.jpg"]}

UNSLOTH (doc de vision fine-tuning):
    [{"role":"user","content":[{"type":"text","text":...},{"type":"image","image":PIL}]},
     {"role":"assistant","content":[{"type":"text","text":...}]}]

Se conserva el mensaje `system` como parte de texto: las plantillas de chat de
Qwen lo admiten, y nuestro SYSTEM_PROMPT es parte de la receta medida -- tirarlo
seria cambiar una variable sin querer.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

IMAGE_TAG = "<image>"


def _parts_from_user(text: str, images: list[str], lazy: bool) -> list[dict]:
    """`<image>pregunta` -> [imagen, texto]. El orden importa: la etiqueta va delante."""
    n_tags = text.count(IMAGE_TAG)
    if n_tags != len(images):
        raise ValueError(
            f"desajuste imagen/etiqueta: {n_tags} '<image>' pero {len(images)} rutas. "
            "Un desajuste silencioso aqui entrena sobre la imagen equivocada."
        )
    clean = text.replace(IMAGE_TAG, "").strip()
    parts: list[dict] = []
    for p in images:
        parts.append({"type": "image", "image": p if lazy else _load(p)})
    parts.append({"type": "text", "text": clean})
    return parts


def _load(path: str):
    from PIL import Image
    im = Image.open(path)
    im.load()                      # fuerza la lectura: si no, el fichero queda abierto
    return im.convert("RGB")


def convert_record(rec: dict, lazy: bool = False) -> dict:
    images = [str(p) for p in rec.get("images", [])]
    out: list[dict] = []
    for m in rec["messages"]:
        role, content = m["role"], m["content"]
        if role == "user":
            out.append({"role": role, "content": _parts_from_user(content, images, lazy)})
        else:
            out.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
    return {"messages": out}


def load_dataset(train_jsonl: str | Path, limit: int | None = None,
                 lazy: bool = False) -> list[dict]:
    rows: list[dict] = []
    with open(train_jsonl, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            rows.append(convert_record(json.loads(line), lazy=lazy))
    if not rows:
        raise ValueError(f"{train_jsonl} no produjo filas")
    return rows


def iter_raw(train_jsonl: str | Path) -> Iterator[dict]:
    with open(train_jsonl, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


# ── constructor de datos de HUMO ────────────────────────────────────────────
# Frames PUBLICOS (CholecT50, colecistectomia laparoscopica -- mismo dominio que
# nuestro split `lapchole`). CERO datos del challenge, cero DUA.
# Se enrutan EXACTAMENTE como los reales para que el camino de codigo sea identico.

SMOKE_SYSTEM = "You are a surgical assistant. Answer with a short phrase."
SMOKE_QA = [
    ("How many foreign objects are visible?", "2"),
    ("What foreign object is present?", "Clip"),
    ("Is there a needle in the scene?", "no"),
    ("How many clips are visible?", "1"),
]


def build_smoke_jsonl(frames_dir: str | Path, out_path: str | Path) -> Path:
    """Genera un train.jsonl con NUESTRO formato exacto a partir de frames publicos."""
    frames = sorted(Path(frames_dir).glob("*.png")) + sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"sin frames en {frames_dir}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i, f in enumerate(frames):
            q, a = SMOKE_QA[i % len(SMOKE_QA)]
            rec = {
                "messages": [
                    {"role": "system", "content": SMOKE_SYSTEM},
                    {"role": "user", "content": f"{IMAGE_TAG}{q}"},
                    {"role": "assistant", "content": a},
                ],
                "images": [str(f.resolve())],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    import sys
    frames, out = sys.argv[1], sys.argv[2]
    p = build_smoke_jsonl(frames, out)
    rows = load_dataset(p, limit=2, lazy=True)
    print(f"escrito {p} ({sum(1 for _ in open(p))} filas)")
    print("muestra convertida:")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False)[:600])
