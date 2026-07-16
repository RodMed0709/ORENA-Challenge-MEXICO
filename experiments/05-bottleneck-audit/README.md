# Rung 05 — Bottleneck Audit (Image Ablation vs Text Shortcut)

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 00-baseline | Qwen3-VL-8B zero-shot | — | done |
| 01-ood-split | frozen ID/OOD split (`frame_ood_v1`) | — | done |
| 02-lora-sft | LoRA instruction fine-tune | 00 | done (acc_OOD = 0.5918) |
| 03-prompt-variants | `SYSTEM_PROMPT` additions | 00 (a1_v0) | done — faithful negative |
| 04-vendor-baseline | External vendor baseline | 00 | done |
| **05-bottleneck-audit** | **The image passed to the model (Real vs Black vs Shuffled)** | **02-lora-sft (checkpoint-1720)** | **TBD** |

## Result (TBD)

(Awaiting full pod execution)
