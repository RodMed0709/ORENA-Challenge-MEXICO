"""¿El fallo es de CONTEO, o el modelo no VE ciertas clases?

Recall/precision POR CLASE sobre "List all foreign objects that are visible".
Si el modelo ve bien unas clases y ciega a otras, el problema no es contar: es
percepción específica de clase — y eso cambia por completo qué hay que arreglar.

Cero GPU.
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path("/mnt/datos/code/ai/ORENA/proy/ORENA-Challenge-MEXICO")
Q_LST = "List all foreign objects that are visible in this video frame."


def split_classes(t):
    return {x.strip().lower() for x in str(t).split(",") if x.strip() and x.strip().lower() != "none"}


for name, run in [("rung 02 (LLM)", REPO / "experiments/02-lora-sft/runs/02_lora_sft_v1/eval_best"),
                  ("rung 06 (+ViT)", REPO / "experiments/06-vit-lora/runs/06_vit_lora_v1/eval_best")]:
    preds = pd.DataFrame(json.load(open(run / "predictions.json")))
    refs = pd.DataFrame(json.load(open(run / "references.json")))
    reqs = pd.DataFrame(json.load(open(run / "requests.json")))
    d = (refs[["qID", "answer"]].merge(reqs[["qID", "question", "videoID"]], on="qID")
         .merge(preds[["qID", "content"]], on="qID"))
    d = d[d["question"].str.startswith(Q_LST)].copy()
    d["gold"] = d["answer"].map(split_classes)
    d["pred"] = d["content"].map(split_classes)
    d["ds"] = d["qID"].str.split("__").str[0]

    classes = sorted({c for s in d["gold"] for c in s})
    rows = []
    for c in classes:
        in_gold = d["gold"].map(lambda s: c in s)
        in_pred = d["pred"].map(lambda s: c in s)
        tp = (in_gold & in_pred).sum()
        fn = (in_gold & ~in_pred).sum()
        fp = (~in_gold & in_pred).sum()
        rows.append({
            "clase": c, "n_gold": int(in_gold.sum()),
            "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "veces_predicha": int(in_pred.sum()),
        })
    t = pd.DataFrame(rows).sort_values("n_gold", ascending=False)
    print("=" * 78)
    print(f"{name} — 'List all foreign objects', n={len(d)} preguntas")
    print("=" * 78)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    # ¿cambia el recall cuando hay MÁS de un objeto?
    d["k"] = d["gold"].map(len)
    print()
    print("  recall medio por nº de objetos en el frame:")
    for k, g in d[d["k"] > 0].groupby("k"):
        rec = g.apply(lambda r: len(r["gold"] & r["pred"]) / len(r["gold"]), axis=1).mean()
        print(f"     k={k}: n={len(g):>4}  recall={rec:6.1%}")
    print()
