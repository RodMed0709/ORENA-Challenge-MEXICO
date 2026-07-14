# Rung 03 — Prompt variants (single-variable, vs the shipped baseline)

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 00-baseline | Qwen3-VL-8B zero-shot | — | done (raw acc 0.262) |
| 01-ood-split | frozen ID/OOD split (`frame_ood_v1`) | — | done |
| 02-lora-sft | LoRA instruction fine-tune | 00 | training |
| **03-prompt-variants** | **`SYSTEM_PROMPT` additions** | **00 (a1_v0)** | **running** |

## Question

Can a prompt tweak beat the **already-strong** shipped prompt? The baseline
`SYSTEM_PROMPT` (`src/frame/engine.py`) already concatenates the full SDK
`FO_DEFINITIONS_FILE` (all 10 FO classes + the instrument/anvil/clip/needle
exclusions), so the FO-grounding lever shipped into "baseline" **un-A/B'd**.
This rung both (a) recovers that missing measurement and (b) tests 4 targeted
additions against the diagnosed failure mode (instrument↔FO confusion, miscounting).

## Arms (each changes ONE thing vs `a1_v0`)

| Arm | Addition |
|-----|----------|
| `a0_noFO` | **removes** the FO definitions → measures the FO-grounding delta |
| `a1_v0` | the shipped baseline (`== frame.engine.SYSTEM_PROMPT`) |
| `a2_decisive` | commit to one answer, never hedge |
| `a3_instrument` | explicit instrument-vs-FO tie-break (if unsure → instrument) |
| `a4_negexem` | worked negative disambiguation examples |
| `a5_baredigit` | silent-enumerate → bare integer (counting protocol) |

## Protocol (pre-registered — avoids test-set overfitting)

Split is **by dataset**: `val_id` = 2252 Q cholecystectomy, `val_ood` = 4000 Q
Sigmoid resection (heico). OOD = 50% of the challenge score and the closest proxy
to the hidden leaderboard test.

1. **Select** on `val_id`. The winner must show a **broad lift** (headline AND
   fo_class/number moving up together), not one chole-specific leaf.
2. **Confirm** the winner + `a1_v0` on the **held-out `val_ood`** (never used for
   selection).
3. **Ship rule:** ship the winner only if its `val_ood` bootstrap CI does **not
   overlap** `a1_v0`'s on the headline (and fo_class/number don't regress).
   Otherwise log a **faithful negative** and move to the LoRA / resolution / data
   rungs — where rung-00's diagnosis says the real signal lives.

CIs are the SDK's video-then-question hierarchical bootstrap (from each run's
`summary.csv`), which respects per-video correlation — point-estimate deltas at
this sample size sit inside the noise band, so the CI is the decision unit.

## Layout

- `_models/prompt_variants.py` — engine: the 6 prompt constants + `run_arms`.
- `_tools/run_select.py` — selection driver (val_id).
- `_tools/run_confirm.py` — confirmation driver (val_ood) + the ship rule.
- `03_prompt_variants.ipynb` — notebook (source of truth; mirrors the drivers).
- `runs/` — gitignored; `select_results.csv`, `confirm_results.csv`, per-arm dirs.

## Provenance

Design shaped by an adversarial review (findings: FO-grounding shipped un-A/B'd;
winner's-curse across arms; dataset-shift in the ID/OOD split; SDK CIs were being
dropped in `run.py`; better instrument/counting variants). See
`context/03-prompt-variants/CONTEXT.md`.
