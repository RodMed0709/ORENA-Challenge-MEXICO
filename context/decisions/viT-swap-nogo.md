---
question: Can we swap Qwen3-VL-8B's ViT for a stronger open-source encoder?
verdict: NO-GO — licence plus a full ViT→LLM realignment we cannot afford
status: NO_GO
date: 2026-07-17
measured_in: context/decisions/viT-swap-nogo.md
---
# Decision: swapping Qwen3-VL-8B's vision encoder = NO-GO

- **Status:** SETTLED · NO-GO · 2026-07-17
- **Scope:** replacing the native ViT with a different (esp. surgical) encoder.
- **GO alternative:** LoRA-adapt the native encoder in place (`--freeze_vit false`) = **rung 06**.

## Question
Can we swap Qwen3-VL-8B's ViT for a better / surgically-stronger, publishable open-source
encoder, and re-run the same fine-tune to lift the vision path?

## What we sought
An open-source encoder that is (a) **commercially releasable** — the model MUST ship open for
the prize — and (b) droppable into Qwen3-VL **without a full realignment**, ideally surgical-domain.

## What it gave us — the set is EMPTY
1. **Architecture blocks a drop-in.** Qwen3-VL's ViT is a **SigLIP2-SO-400M** continually trained
   for native dynamic resolution, wired to the LLM via **DeepStack** — multi-level ViT features
   injected into the **first 3 LLM layers**, not a single input projector. A swap forces a
   from-scratch vision→language realignment (millions of pairs, multi-GPU-day) → out of our
   ~$60–120 / 1-GPU / 8-week budget. Precedent (NextFlow on Qwen2.5-VL) = full realign; official
   Qwen3-VL issue #114 (DINOv2 swap) closed with no drop-in path.
2. **License kills every surgical encoder.** All surgically-stronger towers are non-commercial →
   **disqualifying**: SurgVLP, HecVL, PeskaVLP, SurgeNet, SurgeNetDINO = CC-BY-NC(-SA). The only
   Apache-clean surgical towers (**EndoViT**, **EndoFM**) are **un-aligned to any LLM** → still pay
   the full realignment cost. (So the reason they can't enter is "unaligned bare encoder", NOT
   their license and NOT "not a VLM".)
3. **Permissive general encoders** (SigLIP2, DINOv2, InternViT) are either literally the native
   tower already, or general-domain + unaligned → no free surgical gain, realignment required.

## Verdict
**NO-GO on encoder replacement.** The only budget- and license-safe way to make the vision side
"more surgical" is **LoRA on the native SigLIP2 tower** — which preserves DeepStack/merger and
stays 100% Apache-2.0. That is exactly **rung 06** (already running). If rung 06 shows even LoRA on
the eyes doesn't help, the fix is still not a naive swap — it is domain-adaptation via data, or the
32B (bigger native ViT, same family) — see [[qwen-size-ladder]].

## Sources
- Research: vlm-specialist agent, 2026-07-17.
- Qwen3-VL = SigLIP2-SO-400M + DeepStack: Qwen3-VL technical-report review; DeepStack fusion writeup.
- License cards: SurgVLP / PeskaVLP / HecVL (CAMMA, CC-BY-NC-SA); SurgeNet (HF `cc-by-nc-4.0`);
  EndoViT (HF `apache-2.0`); EndoFM (Apache-2.0).
- Swap precedent = full realign: NextFlow (arXiv 2601.02204); Qwen3-VL GitHub issue #114.
