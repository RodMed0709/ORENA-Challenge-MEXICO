---
question: Is the "untested cell" experiment 09 was designed on actually untested?
verdict: NO — scaffold-SFT without RL is a published wash on our exact backbone (62.0 vs bare-gold SFT's 65.7 on EndoVis2018). The +16.3 belongs to RLVR. Rung 09 is RE-SCOPED, not cancelled.
status: MEASURED
date: 2026-07-23
measured_in: arXiv:2603.20116 (our exact backbone) + Surgery-R1 replication — measured by someone else, in print
question_derived: false
---
# Finding: the "untested cell" R1 was built on is NOT untested — scaffold-SFT without RL is a published wash

- **Status:** MEASURED (by someone else, in print) · 2026-07-23 · owner: Rodrigo
- **Applies when:** anyone plans, budgets, or reads experiment 09 (CoA-format SFT), or cites the
  CoA "+16.3 is the format" line to justify a format-side rung.
- ⚠️ **This note FALSIFIES a premise, it does not cancel an experiment.** Rung 09 is **re-scoped**,
  not dead — the re-scope conditions are below and the decision is reserved to its owner.

## Question
Experiment 09 was designed on one explicit claim, stated in three places:

> *"The +16.3 is RL+format, not SFT+format — there is no SFT-only-CoA row in the source
> (`Bloque-A-Modelo.md:301-306`); R1 is that untested cell."*
> — `context/09-coa-sft/CONTEXT.md:12-13`, repeated verbatim in
> `experiments/09-coa-sft/README.md:9-11` and `context/decisions/coa-generator-qwen32b-onpod.md:40-41`.

The claim rests on a second-hand summary of the source, never on the source. We now have the
source. **Does the `SFT + CoA-format, no RL` row exist?**

## What we sought
The paper itself: Chain-of-Adaptation, Li, Xu, Liu & Xiong (University at Buffalo),
arXiv:**2603.20116v1**, 20 Mar 2026 — downloaded to
`literature/vlm-techniques/pdfs/v01_li_2026_chain-of-adaptation.pdf`, ficha at
`literature/vlm-techniques/FICHAS.md` §v01. Every number below was read off the PDF's Tables 1–3,
not off the ficha.

The paper is **not adjacent to rung 09 — it is rung 09, already run**: same backbone
(**Qwen3-VL-8B-Instruct**), same four-tag scaffold (`<general description>/<evidence>/<thought>/
<answer>`), same reverse-generation of the trace by a larger frame-seeing VLM, ms-swift, vLLM
greedy, and a cross-procedure OOD split (train EndoVis2018 + CholecT50 → test GraSP) that mirrors
ours.

## What it gave us
**The row exists. It is `+ Cold Start + SFT`.** (F1 / class-balanced F1cls, %; PDF Tables 1–3.)

| arm | EndoVis2018 | CholecT50 | GraSP (OOD) |
|---|---|---|---|
| Base Qwen3-VL-8B-Instruct | 48.5 / 42.0 | 35.3 / 20.7 | 16.4 / 13.5 |
| + SFT (bare gold) | **65.7** / 43.2 | 58.7 / **15.3** | 13.4 / 9.6 |
| + Cold Start (scaffold SFT only) | 59.3 / 45.5 | 36.2 / 20.4 | — |
| **+ Cold Start + SFT (scaffold, NO RL)** ← *our cell* | **62.0** / 45.6 | **62.4** / 15.4 | — |
| + Cold Start + RLVR = CoA | 83.7 / 58.0 | 64.4 / 20.2 | 18.3 / 13.3 |
| RLVR with **no thinking tags at all** | 67.4 / 46.3 | 61.5 / 14.0 | — |

Two sentences the paper writes itself:
*"Cold Start + SFT yields only marginal gains (62.0 vs. 65.7 on EndoVis2018)"* and
*"Even without the CoA reasoning format, RLVR achieves higher overall F1, i.e., 67.4 vs. 65.7."*

Two facts that bear on defects **we have already measured**:

1. **SFT crushes class-balanced F1cls: 20.7 → 15.3 on CholecT50** while overall F1 *rises*
   35.3 → 58.7. The paper names it (*"SFT overfits to dominant classes"*) and its per-class radar
   shows SFT collapsing on `clamp` to near zero. That is **our `clip` attractor, published** —
   875 `clip` emitted against 615 golds, needle/bag/sponge confused *into* it. We report no
   class-balanced metric at all, so this failure is currently invisible in `bucket_mean`.
2. **Their cold-start teacher was Gemini-Flash-2.5, an external API** (10,000 internet surgical
   frames, non-thinking mode). Our DUA forbids that route
   ([[no-external-api-for-challenge-data]]) → **their generation recipe is not reproducible by
   us**, and the on-pod generator decision ([[coa-generator-qwen32b-onpod]]) stands, now
   independently justified rather than merely permitted.

## Verdict
🔴 **The premise is FALSIFIED. Scaffold-SFT without RL is a published wash, and it LOSES on
EndoVis2018 (62.0 vs 65.7 for plain bare-gold SFT).** It wins on CholecT50 (62.4 vs 58.7); net
across the two datasets ≈ zero. **The gain is RLVR's, not the format's** — and RLVR with *no
reasoning tags whatsoever* already beats SFT (67.4 vs 65.7; 61.5 vs 58.7). The decomposition
"+16.3 = format, +1.7 = RL" is **backwards**: it was read off a table that never contained a
no-RL scaffold row, and the row that does exist points the other way. **Do not cite "+16.3 is the
format" anywhere again.**

**Corroboration, and the counterweight** (all from `literature/vlm-techniques/FICHAS.md`):
- **Surgery-R1 (§v10)** replicates the wash independently — different backbone (Qwen2.5-VL),
  different surgical dataset family, our data scale (9,014 QA / 1,560 frames). Its two SFT
  variants (CoT-in-target vs not) land within ~1 point of each other and in **opposite
  directions** on the two eval sets (EndoVis-18 0.6499 vs 0.6627; EndoVis-17 0.4125 vs 0.4021);
  adding RFT adds ~8 pts ID and ~10 pts cross-set.
- **General-domain: scaffolding degrades exactly our task.** Kancheti (§v06): CoT lowers accuracy
  ~3% across 13 spatial benchmarks, and 7 of 8 distilled reasoning models fail to beat their own
  backbone. Jin (§v07): CoT causes *"reduced performance in visual grounding and object counting"*
  on **perception** tasks — which is what FRAME is. Vo (§v08): counting accuracy peaks ~40% with
  thinking tokens then **declines with overthinking** — scaffold length is a dose, not a switch.
- **The honest counterweight is STaR (§v18):** **+12.5%** over a model fine-tuned to predict
  answers directly, at the same data scale, via exactly our rationalization move (regenerate the
  rationale *given* the gold). But it is **text-only, on reasoning tasks** (commonsense,
  arithmetic) — the category §v07 says CoT *helps*, not the perception category where it hurts —
  and its gain depends on **filtering to rationales that reach the gold**, which our pipeline does
  not yet do. Stated fairly, it is a real positive prior; stated honestly, most of its transfer is
  gutted by the modality and task mismatch.

🔴 **The genuinely open cell that remains.** No paper in the corpus trains **once** on a scaffold
target and then evaluates **the same checkpoint** under both (a) full-trace generation and (b)
answer-only generation, on a **perception** benchmark, reporting both. CoA generates the full
trace and scores only the `<answer>` span — it never tests suppression, and its gains are
confounded with the extra inference-time computation the trace buys. §v06/§v07/§v08 argue
emitting the trace hurts perception. SCALe (§v02) implies the emit/suppress answer is downstream
of a loss-weighting question and not independent of it. **That cell is empty, and it is the only
original contribution left in this direction.** We are unusually well placed to fill it: latency
is pooled (120 s + B×5 s) and our p99 is 0.352 s → ~14× headroom, so both inference modes cost
us essentially nothing.

**Re-scope, not cancellation (decision RESERVED to Rodrigo, who will run it personally).**
Rung 09 goes ahead only in a shape the literature has not already refuted:
1. **Answer-weighted loss** — segment-separated, scheduled reasoning→answer (SCALe, §v02). Uniform
   token CE on a ~95%-prose / ~5%-answer target reproduces v01's null at our expense.
2. **Correctness filter** — discard every generated trace whose `<answer>` ≠ gold and regenerate
   (STaR, §v18). Free for us: we already hold the gold.
3. **Both inference modes measured** on the same checkpoint, pre-registered two-arm, with scaffold
   **length** swept as a third axis (§v08's dose curve).
4. **Booked as a cold start for a later RLVR/GRPO rung**, not as a standalone win — that is where
   both surgical papers found the payoff.
5. **Report class-balanced F1cls alongside F1** from now on. Fact 1 above says our headline can
   rise while the thing we care about collapses, and today we would not see it.

**Honest prior for the standalone run: `bucket_mean` moves by roughly zero (−0.02 to +0.02), with
a real risk that `number` and `fo_class` get worse.** Budget it as an experiment that fills an
empty cell, not as a lever.

## Sources
- **PDF, read directly:** `literature/vlm-techniques/pdfs/v01_li_2026_chain-of-adaptation.pdf` —
  Table 1 (in-distribution, EndoVis2018 + CholecT50), Table 2 (OOD, GraSP), Table 3 (SFT vs RLVR
  w/o thinking), Remarks 2–5, §5.1 (Gemini-Flash-2.5 cold-start pseudo-labelling).
- Ficha + corroboration/counterweight: `literature/vlm-techniques/FICHAS.md` §v01, §v02, §v06,
  §v07, §v08, §v10, §v18, and the closing section "The honest prior on CoA-format SFT".
- Falsified statements corrected in place: `experiments/09-coa-sft/README.md`,
  `context/09-coa-sft/CONTEXT.md`, `context/decisions/coa-generator-qwen32b-onpod.md`,
  `context/decisions/next-move-rodrigo-coa-format.md`, `THE_MAP.md` §A2,
  `context/07-enumeration/CONTEXT.md`, `context/INDEX.md`.
- Our `clip` attractor + `number` erosion: `experiments/08-data-card/`,
  [[checkpoint-selection-vs-number]].
- Related: [[coa-generator-qwen32b-onpod]] (unaffected — now independently justified),
  [[next-move-rodrigo-coa-format]], [[no-external-api-for-challenge-data]], [[eval-canonical]].
