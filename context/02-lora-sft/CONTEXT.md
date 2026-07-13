# CONTEXT — 02 LoRA instruction fine-tune

## Objective
Beat the zero-shot baseline (and the organizers' fine-tuned 4B) by LoRA-tuning
Qwen3-VL-8B on the FRAME train split — lifting the weak formats **fo_class + number**
while generalizing to the held-out **Sigmoid (OOD)** procedure (≈ half the score).

## Setup-config
- The ONE variable: `LoRA ON` vs rung 00 (zero-shot). Everything else identical.
- Split: frozen `experiments/splits/frame_ood_v1.csv` — train 92vid/13748q · val_id
  28/2252 (chole ID) · val_ood 10/4000 (Sigmoid OOD).
- Recipe (S2Can): r=8, α=32, dropout 0.1, lr 2e-5, frozen ViT+aligner, all LLM linears,
  bf16, ≤5 epochs, ckpt/epoch. `MAX_PIXELS`=921600 (train tokens == serve tokens). ms-swift.
- Engine: `experiments/02-lora-sft/_models/lora_sft_train.py` (`main(cfg, stage=...)`:
  export → train → merge → eval). Zero-shot arm reuses the 00-baseline predictions on the
  same test/val set (val == the 6252 test the baseline scored) → clean Δ, no re-run needed.
- Δ: `src/frame/delta.py` → per answer_format × {ID,OOD} + per group CSV + Δ plot.

## Decisions
- Select checkpoint by **acc_OOD (Sigmoid)**, never mean; discard ID-win/OOD-loss ckpts.
- Stay plain LoRA (Surgical-LVLM: instruction-FT +16, exotic adapters +2 → skip DoRA/
  vision-unfreeze/grounding). 8B not 4B.
- Notebook is the runnable surface (records how it went + saves plots). Detached full run
  = headless `nbconvert` (survives session death).

## Results
- PENDING: run on GPU pod (A100 80GB, EU-RO-1, volume gf78k60nlt). SMOKE → full → merge →
  eval on val → Δ CSV + plots. Fill RESULTS.csv + RESULTS_delta.csv.

## Next
1. Scaffold + independent review (GO) → GPU pod → SMOKE → full train.
2. Eval both arms on val → Δ report (which formats improved vs 0-shot, ID & OOD).
3. If OOD wins: sibling rungs (lr=1e-4, rank=16, fo_class/number oversampling).
4. Cross-corpus Cholec80 pHash scrub before trusting the number for the real leaderboard.
