# Experiment 02 — LoRA instruction fine-tune (Qwen3-VL-8B)

> The PRIMARY lever. Fine-tune the open 8B on the FRAME `train` split and beat the
> zero-shot baseline (and the organizers' fine-tuned 4B) — especially on the weak
> formats **fo_class + number**, validated on the held-out **Sigmoid (OOD)** procedure.

## Ladder

| Notebook | Rung | Metric (pre_evaluation_score) | Verdict |
|---|---|---|---|
| `../00-baseline/00_zeroshot_qwen3vl.ipynb` | 00 | 0.174 (raw 0.262) | baseline |
| `02_lora_sft.ipynb` | 02 | **0.708** (raw 0.566) | **PASS** — crushes zero-shot |

*Success = acc_OOD (val_ood/Sigmoid) up vs zero-shot, no format regressed, p99 < 5 s.*

## Result — PASS (crushes the zero-shot baseline)

Selected **checkpoint-1720 (epoch 2)** by acc_OOD. Per-epoch acc_OOD was **0.583 /
0.592 / 0.583** (epoch 1/2/3) — flat, **no OOD collapse** (PITFALLS #1 risk did not
materialize; epochs 1-3 all strong). Best full-val: **pre_evaluation_score 0.708
(vs 0.174), raw 0.566 (vs 0.262)**.

| slice | zero-shot | LoRA | Δ |
|-------|-----------|------|---|
| overall (6252) | 0.262 | **0.566** | **+0.305** |
| ID (2252) | 0.249 | 0.521 | +0.272 |
| **OOD (4000)** | 0.269 | **0.592** | **+0.323** |

Per answer_format (the two target weak buckets, bolded): **fo_class 0.168→0.588
(+0.42)**, **number 0.141→0.433 (+0.29)**, binary +0.15, multiple_choice +0.19,
open_ended +0.05. Every format up, ID and OOD; OOD gained more than ID (generalizes
to the held-out Sigmoid procedure, not memorizing chole). This settles the rung-03
finding: `fo_class` and `number` are **LoRA/data** wins, not prompt wins. Full
breakdown: `runs/RESULTS_delta.csv` + `runs/delta_by_format.png`.

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
└── runs/                      # GITIGNORED — small CSVs + inspect.csv only
                               #   (checkpoints/merged/train.jsonl live in /workspace/ckpt/<run>/;
                               #    frames in the shared /workspace/frames_cache/)
```
Context: `context/02-lora-sft/CONTEXT.md`. Shared split: `experiments/splits/`.
