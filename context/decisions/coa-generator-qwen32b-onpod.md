# Decision: R1's scaffold generator = Qwen3-VL-32B (vision, ZERO-SHOT, ON-POD) — supersedes the text-only "no pixels" line

- **Status:** SETTLED · 2026-07-18 · owner: Rodrigo · supersedes the text-only generator described
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
The +16.3 CoA figure is **RL+format, not SFT+format** (`Bloque-A-Modelo.md:301-306`); there is no
SFT-only-CoA row — R1 is that untested cell. And the paper emitted the full CoA at inference; we
emit only `<answer>`, betting structured SFT is a **weights-level regularizer**. Both are UNTESTED,
so **retire the +16.3 expectation**; the pilot's prior is "unknown, plausibly ~0" and its whole
value is measuring these two cells cheaply. Required design changes:

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
- `Bloque-A-Modelo.md` §6-7/§12 (CoA ablation is RL+format; SFT-only + emit-only-answer untested).
- `experiments/08-data-card/README.md` §4/§6 (multi-object collapse; `number` CI contains floor).
- Related: [[next-move-rodrigo-coa-format]], [[checkpoint-selection-vs-number]],
  [[no-external-api-for-challenge-data]], [[eval-canonical]].
