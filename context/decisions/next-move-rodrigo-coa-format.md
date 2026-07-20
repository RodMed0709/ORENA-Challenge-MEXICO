---
question: Where should the second teammate attack without colliding with rung 06?
verdict: CoA-format SFT (R1); RL is deferred
status: ACTIVE_PLAN
date: 2026-07-17
measured_in: experiments/09-coa-sft/ (branch task/r1-coa-sft)
---
# Decision: Rodrigo owns "CoA-format SFT" (R1); RL is deferred

- **Status:** ACTIVE PLAN · 2026-07-17 · owner: **Rodrigo** · complementary to Leo's rung 06
- **Source:** vlm-strategist review, 2026-07-17.

## Question
Where should the second teammate attack — without colliding with Leo's rung 06 (ViT-LoRA) — and is
"RL + CoA from the start" the right first bet?

## What we sought
The highest-ROI single-variable move on the DATA/FORMAT front (the +16 lever) that starts cheap +
local, needs **no external-data DUA**, and does not depend on rung 06's outcome.

## What it gave us
- **RL first = NO.** CoA ablation: **SFT 65.7 → +RL 67.4 (+1.7) → +CoA-format 83.7 (+16.3)**. The
  value is the **FORMAT**, not the RL. GRPO on 1×L40S is unvalidated (official recipes assume ~6
  GPUs + server mode); the +1.7 exact-match RLVR variant was already killed.
- **Validated cheap lever = CoA structured-reasoning format via plain SFT.** Its OOD thesis IS our
  diagnosis: conventional SFT alters pretrained priors → less generalization; CoA preserves them.
  OOD = 50% of the score.

## The plan Rodrigo owns (single-variable vs rung-02: bucket_mean 0.550, acc_OOD 0.592)
- **R1 — own it.** Rebuild the 13.7k train QA into CoA-style scaffolds (reverse-generated **locally**
  from image+question+gold; filtered by the offline Qwen judge-mirror, ρ=0.94), re-run the rung-02
  LoRA recipe. **One variable = scaffold-data vs plain-QA.** Emit only `<answer>` at inference.
  ~$5 train + ~½-day local gen. **No DUA.** Targets object_recognition×OOD (50% wt); orthogonal to
  rung 06 (format/generalization vs perception).
  - **Gate with a cheap pilot:** scaffold a **~2k stratified subset**, train 2 epochs, read OOD
    bucket-mean. **Kill-criterion:** if it doesn't beat rung-02 on the pilot, stop before building
    the full 13.7k.
- **R2 — parallel, zero-train.** Guided/constrained decoding on ONLY the final answer span
  (`number`→`\d+`, `fo_class`→enum, `binary`→yes/no). Copeland insurance vs silent-0; becomes a
  regression test. No dependency on R1 or rung 06.
- **R3 — before reading any Δ.** Selection metric → mean-over-4-buckets; add `val_dataset` +
  eval-loss curve, eval every ½-epoch. Touches shared `run.py` → **coordinate with Leo**.
- **R4 — reserve.** RL/GRPO on top of R1 — only if R1 stalls AND GRPO-fit-on-L40S is proven first.

## Hard unknown gating R1
Unmeasured **L40S p99 with longer internal reasoning tokens**. The build→smoke step MUST measure
p99 on a scaffold-trained checkpoint before the full run (that same measurement also unblocks the
32B / resolution decisions — see [[qwen-size-ladder]]).

## Do-NOT (already adjudicated — don't waste a week)
Oversample `number` (not a quantity problem — already 31% of train, mean 2.74), prompt-enumeration
(rung 07 faithful negative), RLVR exact-match, encoder swap ([[viT-swap-nogo]]), thinking-mode
(latency-fatal).

## Sources
- vlm-strategist review, 2026-07-17; `THE_MAP.md` §0 + §A2; Bloque-A / Block-A §7 (CoA);
  `literature/FICHAS.md` tier2_01/08/09/11, tier1_04/07; `experiments/02-lora-sft`,
  `experiments/03-prompt-variants`, `experiments/05-bottleneck-audit`.
