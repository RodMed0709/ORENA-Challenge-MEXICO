# FRAME Attack Ladder — the master plan

> Synthesized by `vlm-strategist` from the 21-paper corpus fichas (`literature/FICHAS.md`),
> `CONSTITUTION.md`, and the rung-00 baseline (2026-07-13). Every rung is **single-variable vs the
> named prior rung**, flags default OFF = byte-identical. Sequenced by ROI × dependency.
> Selection is driven by **acc-OOD**, not acc-ID (OOD ≈ half the score).

## Baseline anchor (Rung 00)
`Qwen3-VL-8B-Instruct` zero-shot, 1 frame @ timestamp, greedy. Robust signals (not the fragile 0.174
macro-mean): **raw acc 0.262**; per-format binary 0.597 / open_ended 0.590 / MC 0.534 / **fo_class 0.182
/ number 0.127**. p99 latency **0.59 s** (≈8× headroom under 5 s). Diagnosis: format-clean, domain-blind
— confuses **instruments↔foreign objects** and **miscounts**. Weak cells: `fo_class`→object_recognition,
`number`→aggregation; OOD is 50% of weight and untested locally.

## THE LADDER

| # | ONE variable (vs prior rung) | Weak bucket / format | Expected Δ | Backing | Cost / risk | GO-gate |
|---|---|---|---|---|---|---|
| **01** | System-prompt FO taxonomy + "instruments are NOT foreign objects" exclusion (`USE_TAXONOMY_PROMPT`, default OFF) | fo_class (0.182) → object_recognition; #1 failure | +0.03–0.08 fo_class | SSG-VQA existence grounding, GP-VLS QA-as-taxonomy | Zero-train, minutes. Risk: over-constrain → false negatives | fo_class ↑ AND parse-fail <2% AND binary not regressed; no AdversarialDetector trip |
| **02** | Judge-robust answer canon + silent-gate guard (reference-anchored, dense, <300 char, no hedging) | open_ended/MC/matching (judge) | +0.00–0.03; prevents catastrophic 0s | Zheng LLM-judge biases; §I.4-bis gates | Zero-train. Pure insurance. Needs offline Qwen3-4B judge-mirror | No output >300 char, zero adversarial hits on our preds; mirror shows A ≥ B |
| **03** | OOD-safe leak-guarded split (hold out WHOLE procedure_type; split by videoID) | measurement harness (gates all LoRA claims) | enables true OOD readout | HeiCo 3-stage, ROBUST-MIS, Copeland worst-bucket | Low. Risk: mis-split = leak. **Prereq for 05+** | Zero video overlap train/val/OOD; report acc-ID AND acc-OOD separately |
| **04** | Synthetic QA data-factory (LLaVA-Surg two-stage extract→QA with LOCAL LLM; SSG-VQA scene-graph templates) | number (0.127), fo_class, aggregation, object_recognition | enables 05 | LLaVA-Surg, SSG-VQA, LLaVA-Med curriculum | Med. **Needs external public data we must fetch+release** (Cholec80/CholecT45, HeiCo). Prereq for 05 | Judge-mirror agreement ≥0.9 on generated pairs; per-format coverage balanced |
| **05** | **PRIMARY: LoRA instruction-tune Qwen3-VL-8B** — r=8, α=32, dropout 0.1, lr 2e-5, **frozen ViT**, LoRA on all LLM linear layers | ALL weak buckets, esp fo_class + number; OOD | **+0.10–0.16** (Surgical-LVLM IFT +16; S2Can +10.8 OOD) | Surgical-LVLM ablation "IFT +16, exotic +2 → skip"; LoRA; ms-swift `--freeze_vit` | Med train ($). Risk: OOD collapse / chole overfit — validate on rung-03 OOD, discard if OOD↓. bf16 no NF4 | acc-OOD ↑ vs 00 AND no format regressed; p99 <5 s after merge |
| **06** | Oversample weakest (group×format) cells in the 05 mix (counting + fo_class) | evens the WORST bucket → Copeland reward | +0.02–0.05 tail bucket | Vote'n'rank worst-bucket, challengeR stability | Low (data reweight). Risk: overfit weak cell | Worst-bucket ↑ with no strong-bucket drop >0.02 |
| **07** | Raise `max_pixels` (resolution) one step vs 05 | number (small clips/needles), fo_class fine ID | +0.02–0.05 | Qwen2.5-VL dynamic-resolution; banked headroom | Low. Risk: latency — **re-measure p99 on L40S** | fine-perception ↑ AND p99 <5 s |
| **08** *(MED, opt)* | Question-conditioned single-frame pick (CLIP/SigLIP one-shot) vs uniform timestamp | temporal_grounding; n=1 "end-of-procedure" case | +0.01–0.04 | Seenivasan/EndoNet/ROBUST-MIS (temporal stacking hurts); VideoAgent selector only | Med. Risk: retrieval latency + failure mode; single-shot only | temporal ↑, p99 <5 s, single forward pass |
| **09** *(WILDCARD, last)* | Backbone 8B → 32B FP8 | global lift if 8B plateaus | uncertain; only if 05–07 stall | Qwen scaling; QLoRA-NF4 to fit | High. Risk: **p99 may bust 5 s on L40S 48GB** — verify first | full FRAME p99 <5 s on L40S AND score > best 8B rung |

**Guardrails (do NOT ladder):** thinking-mode, agentic multi-round loops, vision-encoder swap
(SurgVLP/HecVL), memory multi-forward — all latency-fatal or infeasible.

## First two weeks
1. **Bank free wins now (01 + 02, d1–3):** prompt-only, zero-train, stackable, default-OFF flags. 01 hits the diagnosed #1 failure; 02 is submission insurance vs silent 300-char/adversarial/dup-qID zero-gates. Raises the floor before leaderboard opens **Jul 15**.
2. **Stand up harness in parallel (03, d2–5):** leak-guarded OOD split gates *every* LoRA claim; OOD = 50% of score, untested locally today. No LoRA number is trustworthy until this exists.
3. **Build data-factory (04, d4–10):** fetch + document + release Cholec80/CholecT45 + HeiCo, then generate QA with a **local** LLM (LLaVA-Surg two-stage + SSG-VQA templates), overweighting counting + fo_class. **Only rung needing data we don't have — start acquisition/licensing TODAY.**
4. **Fire primary lever (05, d8–14):** LoRA S2Can recipe (bf16). This is where +0.10–0.16 lives. Validate strictly on rung-03 OOD; discard any checkpoint that wins ID but loses OOD.
5. **Gate discipline throughout:** build → smoke → independent review (GO/NO-GO + file:line) → full; report Δ as acc-ID and acc-OOD separately; re-measure p99 on L40S after any merge/resolution change. Rungs 06–09 layer on 05 only after it clears GO; defer 08/09 unless 05–07 plateau.

## Critical path flag
Rung 04 is the sole dependency needing external assets not in-repo (Cholec80/CholecT45 + HeiCo downloads
+ public-release obligation per §II/§IV). It's on the critical path to the primary LoRA lever → **kick off
acquisition immediately.**
