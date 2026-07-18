"""¿El ViT se degrada conforme entrena? — diagnóstico CERO GPU.

Compara la trayectoria por época de `number` en los DOS brazos. Los `sel/` evaluaron
cada checkpoint sobre los vídeos OOD al seleccionar, así que el dato ya existe.

Hipótesis: si vit_lr=2e-5 degrada el ViT pre-entrenado, `number` debe EMPEORAR con más
entrenamiento en el rung 06 y NO en el rung 02 (que tenía el ViT congelado).
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/mnt/datos/code/ai/ORENA/proy/ORENA-Challenge-MEXICO")
sys.path.insert(0, str(REPO / "src"))

from frame import ledger, metrics  # noqa: E402

gold = ledger.gold_from_frame_parquets(REPO / "external_data" / "orena-data")
gold = gold.assign(template=gold["question"].map(metrics.template_of))

ARMS = {
    "rung 02 (ViT CONGELADO)": REPO / "experiments/02-lora-sft/runs/02_lora_sft_v1/sel",
    "rung 06 (ViT con LoRA)": REPO / "experiments/06-vit-lora/runs/06_vit_lora_v1/sel",
}
EPOCH = {"ood_checkpoint-860": 1, "ood_checkpoint-1720": 2, "ood_checkpoint-2580": 3}


def slice_metrics(res: pd.DataFrame, fmt: str | None = None) -> tuple:
    """acc / floor / margin sobre un formato (o todo), con el suelo canónico."""
    df = res.merge(gold[["qID", "answer", "template"]], on="qID", how="left")
    assert df["answer"].isna().sum() == 0, "GATE: gold incompleto -> margen inflado"
    if fmt:
        df = df[df["answer_format"] == fmt]
    acc = float(df["correctness"].mean())
    floor = metrics.template_floor(df)
    return acc, floor, acc - floor, len(df)


for arm, d in ARMS.items():
    print("=" * 74)
    print(arm)
    print("=" * 74)
    print(f"{'época':<8}{'n':>6}{'':<3}{'acc_all':>9}{'margin_all':>12}{'':<3}"
          f"{'acc_number':>11}{'floor':>8}{'margin_number':>15}")
    for ck, ep in sorted(EPOCH.items(), key=lambda x: x[1]):
        f = d / ck / "results.csv"
        if not f.exists():
            print(f"  época {ep}: FALTA {f.name}")
            continue
        res = pd.read_csv(f)
        a, fl, m, n = slice_metrics(res)
        na, nf, nm, nn = slice_metrics(res, "number")
        print(f"{ep:<8}{n:>6}{'':<3}{a:>9.4f}{m:>+12.4f}{'':<3}"
              f"{na:>11.4f}{nf:>8.4f}{nm:>+15.4f}")
    print()
