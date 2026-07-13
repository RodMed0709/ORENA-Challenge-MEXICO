# Experiment 02 — LoRA instruction fine-tune (Qwen3-VL-8B)

> The PRIMARY lever. Fine-tune the open 8B on the FRAME `train` split and beat the
> zero-shot baseline (and the organizers' fine-tuned 4B) — especially on the weak
> formats **fo_class + number**, validated on the held-out **Sigmoid (OOD)** procedure.

## Ladder

| Notebook | Rung | Metric (pre_evaluation_score) | Verdict |
|---|---|---|---|
| `../00-baseline/00_zeroshot_qwen3vl.ipynb` | 00 | 0.174 (raw 0.262) | baseline |
| `02_lora_sft.ipynb` | 02 | _pending_ | _pending_ |

*Success = acc_OOD (val_ood/Sigmoid) up vs zero-shot, no format regressed, p99 < 5 s.*

## What this experiment is

- **ONE variable:** `LoRA ON`. Everything else (frame sampling, system prompt, judge,
  generation, `max_pixels`, `max_new_tokens=64`) byte-identical to rung 00.
- **Split:** the frozen `experiments/splits/frame_ood_v1.csv` (organizer partition).
  train **92 vid / 13,748 q** · val_id **28 / 2,252** (chole = ID) · val_ood **10 / 4,000**
  (Sigmoid = OOD). Never train on val.
- **Target:** FRAME's weak formats **fo_class (0.182) + number (0.127)**; groups
  object_recognition + aggregation. (No time / percentage / reasoning — not in FRAME.)
- **Recipe (S2Can):** r=8, α=32, dropout=0.1, lr=2e-5, frozen ViT+aligner, LoRA on all
  LLM linear layers, bf16, ≤5 epochs, checkpoint per epoch. ms-swift.
- **Selection:** by **acc_OOD (Sigmoid)**, never mean. Discard any checkpoint that wins
  ID but drops OOD (chole-overfit guard).
- **Δ report:** `src/frame/delta.py` → `RESULTS_delta.csv` (per answer_format × {ID,OOD}
  + per capability_group) + a Δ bar plot — exactly which question types improved vs 0-shot.
- **Final submission model:** retrain on ALL 20k (train + val) with the selected recipe.

## Layout
```
experiments/02-lora-sft/
├── README.md                 # this file (opens with the ladder)
├── 02_lora_sft.ipynb         # train → merge → eval → Δ report + plots
├── _models/
│   └── lora_sft_train.py      # ms-swift engine (importable; main(cfg, stage=...))
├── RESULTS.csv                # scorecard row
└── runs/                      # GITIGNORED — checkpoints, logs, predictions, plots
```
Context: `context/02-lora-sft/CONTEXT.md`. Shared split: `experiments/splits/`.
