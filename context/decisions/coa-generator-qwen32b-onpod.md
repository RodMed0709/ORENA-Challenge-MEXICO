---
question: Which model generates R1's CoA scaffolds — a text-only reverse-generator, or a frame-seeing VLM?
verdict: Qwen3-VL-32B, VISION, zero-shot, ON-POD. Supersedes the text-only "no pixels" line; the on-pod constraint is forced by the DUA, not chosen.
status: SETTLED
date: 2026-07-18
measured_in: experiments/09-coa-sft/README.md + the adversarial GO-WITH-CHANGES gate of 2026-07-18
question_derived: false
---
# Decision: R1's scaffold generator = Qwen3-VL-32B (vision, ZERO-SHOT, ON-POD) — supersedes the text-only "no pixels" line

> ## 🔴 SUPERSEDED 2026-07-31 — §3's premise was measured and is false
>
> This note picked the 32B on the reasoning in **§3**: *"the generator's job is not perception, it
> is anchored writing … a strong general VLM does it zero-shot."* **Rung 17 measured it and it does
> not.** Asked the FRAME questions with no gold, the 32B scores **0.08** against a **0.295** floor —
> **margin OOD −0.2150**, worse than our own *untrained* 8B (−0.191).
>
> ⚠️ **The obvious confound is ruled out:** the probe uses `frame.engine.SYSTEM_PROMPT`, which
> embeds `FO_DEFINITIONS_FILE` — the model was given the role, the required output format **and the
> full 10-class definitions**, and still failed. This is not missing vocabulary.
>
> The rest of this note **stands**: the DUA reasoning (on-pod only), the model-availability search,
> and the licence analysis are all unaffected and still binding. Only §3 is retracted. Verdict and
> consequences: [[generator-32b-is-not-a-teacher]].
>
> 🔒 And the availability search in §2 was **re-verified on 2026-07-31**: nothing has changed —
> still no downloadable weights for SurgVLM / LLaVA-Surg / Surgical-LVLM, EndoChat still publishes
> no weights repository or licence, Gemma/MedGemma still excluded by the *Model Derivative* clause.
> **There is no eligible teacher to buy**, which is why the line closes rather than switching model.

- **Status:** 🔴 SUPERSEDED IN PART (§3) · SETTLED · 2026-07-18 · owner: Rodrigo · supersedes the text-only generator described
  in `experiments/09-coa-sft/README.md` (Stage-1 as first written) and `context/09-coa-sft/CONTEXT.md`.
- **Gated by:** the compliance rule [[no-external-api-for-challenge-data]] (on-pod only) and the
  adversarial GO-WITH-CHANGES gate (2026-07-18, below).

## Question
Which model generates R1's CoA scaffolds — the text-only reverse-generator first speced (deepseek/
Claude, no pixels), or a frame-seeing VLM? And does moving to a frame-seeing VLM re-open the
settled rejection of "generator sees the frame"?

## What changed since the first spec
1. **DUA compliance ([[no-external-api-for-challenge-data]]):** no challenge data (frames OR
   annotations) may go to an external API → the text-only step (deepseek/Claude via MCP) is
   DUA-invalid; any generator must run on-pod. The prior text-only ~50-sample is retained ONLY as
   a format/methodology proof, not a valid generation.
2. **Model-availability search (vlm-specialist, 2026-07-18):** the surgical-domain generative VLMs
   (SurgVLM, GP-VLS, LLaVA-Surg, Surgical-LVLM) have **no released weights** (verify-fail); the one
   released (EndoChat) carries a Llama-2 output-usage clause conflicting with training our Qwen +
   is weak at free-form structured generation. Gemma is ruled out (its Terms classify a model
   trained on Gemma outputs as a "Model Derivative" → propagates onto our released 8B).
3. **The generator's job is not perception, it is anchored writing:** the gold is GIVEN; the model
   must write coherent evidence→answer reasoning that lands on it. That is instruction-following +
   basic grounded perception — a strong general VLM does it zero-shot.

## Verdict
- **Generator = `Qwen/Qwen3-VL-32B-Instruct`, zero-shot (no finetune), run ON-POD (FP8).** Apache-2.0
  (outputs freely usable to train our 8B), same family as the Qwen3-VL-8B student (terminology
  aligned → judge-friendly), strongest instruction-follower of the released options. Optional
  higher-perception second pass: Qwen2.5-VL-72B (FP8) on hard frames only.
- **It SEES the frame.** This does re-open the earlier "generator does not see pixels" line — but
  that rejection was specifically about **our own below-floor 8B** self-distilling hallucinations.
  A strong 32B is a genuine perceiver; the rejection rationale (hallucination on multi-object OOD)
  applies **less severely**, and is now guarded by the eyeball-50 gate below.
- **Quality levers without training:** few-shot prompt (2-3 exemplars, varied across format strata)
  + **short scaffolds** (also protects latency + the `number` gradient).

## ⚠️ Required changes from the adversarial gate (2026-07-18) — before any GPU spend
🔴 ~~The +16.3 CoA figure is **RL+format, not SFT+format** (`Bloque-A-Modelo.md:301-306`); there is
no SFT-only-CoA row — R1 is that untested cell.~~ **CORRECTED 2026-07-23 — the row exists and R1's
cell is NOT untested: [[coa-sft-published-null]].** Read off the source PDF, `+ Cold Start + SFT`
(scaffold, no RL, Qwen3-VL-8B-Instruct) scores **62.0 F1 on EndoVis2018 against bare-gold SFT's
65.7** (62.4 vs 58.7 on CholecT50) — a published wash that *loses* on EndoVis. The +18.0 is RLVR's;
RLVR with no thinking tags at all already beats SFT (67.4 vs 65.7). The "+16.3 = format" split is
retired for good.

**This does NOT change the generator verdict above — it strengthens it.** The paper's cold-start
teacher was **Gemini-Flash-2.5, an external API** on 10,000 scraped surgical frames; our DUA
forbids that ([[no-external-api-for-challenge-data]]), so their recipe is **not reproducible by
us** and the on-pod Qwen3-VL-32B route is now independently justified rather than merely permitted.

What remains untested is the *other* half: the paper emitted the full CoA at inference, we emit
only `<answer>`, and **no paper evaluates the same scaffold-trained checkpoint under both modes on
a perception benchmark**. That is the only original cell left, and the pilot's value is now
measuring it — not measuring a "+16.3 that might survive". Prior: `bucket_mean` ≈ 0
(−0.02 to +0.02). Required design changes (unchanged, plus the re-scope conditions in
[[coa-sft-published-null]]):

1. **Eyeball ~50 on-pod 32B-vision scaffolds FIRST**, over-sampling multi-object-OOD + single-Q
   frames — the true first kill-gate (frame-contradicting evidence poisons the exact target
   bucket, and the Qwen judge-mirror, a below-floor perceiver, CANNOT catch evidence↔frame
   incoherence). Do not generate the 2k until these pass.
2. **Matched 2k-bare-gold control arm** (same subset/seed/recipe/epochs). Kill-gate reads
   `CoA-2k − bare-2k` (clean single variable), NOT CoA-2k vs full-13.7k rung-02 (confounds format
   with 6.9× data — Cholec80: data quantity > architecture).
3. **Evaluate the pilot in emit-only-`<answer>` deployment mode AND measure L40S p99 during that
   eval** (not after). Emit-full-reasoning eval is non-transferable → reject it.
4. **≤2 epochs, per-epoch save, `number`-margin-aware matched-epoch selection**
   ([[checkpoint-selection-vs-number]]); short scaffolds / consider up-weighting the `<answer>`
   span (CoA dilutes the answer gradient → may erode `number` faster).

NO-GO only if the eyeball-50 shows the 32B systematically contradicts multi-object-OOD frames and
no cheaper generator recovers coherence.

## Parallel insurance (CoA-independent, cheap)
- Leo's queued `checkpoint-860` epoch-1 eval (~24 min GPU; `number` best at epoch 1 — we may
  already own a better model).
- R2 constrained decoding on the final answer span only (zero-train Copeland silent-0 insurance).

## Sources
- vlm-specialist model-availability search + vlm-strategist adversarial gate, 2026-07-18.
- ~~`Bloque-A-Modelo.md` §6-7/§12 (CoA ablation is RL+format; SFT-only + emit-only-answer untested).~~
  **Superseded by the primary source:** `literature/vlm-techniques/pdfs/v01_li_2026_chain-of-adaptation.pdf`
  Tables 1–3 → [[coa-sft-published-null]]. Only the emit-vs-suppress half is untested.
- `experiments/08-data-card/README.md` §4/§6 (multi-object collapse; `number` CI contains floor).
- Related: [[next-move-rodrigo-coa-format]], [[checkpoint-selection-vs-number]],
  [[no-external-api-for-challenge-data]], [[eval-canonical]].
