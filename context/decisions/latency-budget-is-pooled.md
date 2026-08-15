---
question: Is the 5 s FRAME cap per question or amortised across the batch?
verdict: POOLED — `120 s setup + B × 5 s` per run; self-consistency and higher resolution are affordable
status: MEASURED
date: 2026-07-19
measured_in: orena-focus-submission-template/README.md §"The latency budget"
---

# Decision: the FRAME latency budget is POOLED — the per-question ceiling was never real

> ## ⚠️ CONTESTED, 2026-08-15 — a second official source says PER-QUESTION
>
> The **challenge design document** (`documentacion/papers/332-ORena_-_SAVE_FOCUS_challenge…pdf`,
> dated **2026-04-22**, the BIAS/MICCAI structured description), p. 10 §"Submission method":
>
> > *"Inference will be limited to a single GPU and must be completed within **a maximum time
> > budget per question**. If the answering of a question takes longer than the budget, **the
> > respective question will be treated as answered incorrectly**."*
> > *— FRAME track: **5 seconds** on a 48GB VRAM GPU*
>
> That is per-question with a per-question penalty — flatly contradicting this note's verdict,
> which came from the submission template README.
>
> | source | date | says |
> |---|---|---|
> | submission template `README.md` | (undated in repo) | **pooled**, `120 s + B × 5 s`, graduated forfeit, `latency` field informational |
> | challenge design document | 2026-04-22 | **per question**, that question scored incorrect |
>
> **Which governs is UNRESOLVED.** The template describes the *implementation* ("the platform
> measures the execution time of each run itself"), which argues it is newer and operative; the
> design document is the *formal commitment*. Neither claim has been tested against the platform.
>
> ⇒ **Do not cite either reading as settled.** Cost levers against the STRICTER (per-question)
> reading until an organizer answers, because that is the one that can silently zero a
> submission. 🔴 The cheap resolution is to **ask the organizers**, not to keep inferring.
>
> 📌 What does NOT change: a lever measured at **1.33 s/question batched** (rung 43's thinking
> arm) fits comfortably under *both* readings. The discrepancy only bites for anything in the
> 5–11 s range, where the two answers differ.

- **Status:** MEASURED · 2026-07-19 · read from the **official submission template**, not inferred
- **Applies when:** costing anything that spends compute per question — self-consistency,
  higher `max_pixels`, a larger model. Several levers were closed against a ceiling that does
  not exist as we modelled it.
- **Origin:** the leaderboard's single row showed **18.44 s mean latency with `timed_out: 0`**
  and a 20.0× ratio against throughput, which looked like batching. It was.

## What the template says

```
allowed = 120 s  +  B × (5 s for FRAME)
          ^setup     ^per question, POOLED across the batch
```

Verbatim, from `README.md` §"The latency budget":

- *"The budget is **pooled across the batch**."*
- *"The `latency` field you write into a Response is informational and is **not** used for
  scoring."* — the platform times the process itself.
- *"The setup allowance covers everything you pay once per run — Python imports, and loading
  your model."*
- *"Going over does not fail the run outright. Instead, a share of your questions is
  **forfeited** … exceeding the budget by **20 %** forfeits the entire batch."*
- 🔴 *"The forfeited questions are **selected for you — not necessarily the slow ones**."*

And `inference.py` hands us the **whole batch up front** via `load_requests`, so nothing forces
one-question-at-a-time decoding: we control `run()` and can push the batch through vLLM's
continuous batching.

## What this unblocks

For a typical batch of `B = 20`: **220 s for 20 questions ≈ 11 s of effective budget each**,
against a measured p99 of **0.352 s** on dev hardware (n=6252, rung 06). Even assuming the L40S
is 4× slower than the dev GPU, k=5 self-consistency costs a small fraction of the pool.

- 🟢 **Self-consistency (k-voting) is affordable.** The vault had it marked
  *"blocked — if the cap is amortised this lever opens entirely; if per-question, it dies."*
  **It is amortised.**
- 🟢 **Higher `max_pixels` is affordable.** THE_MAP closed the resolution branch on the
  per-question ceiling. That premise was wrong — which matters because the OOD split is measured
  to receive **~56 % of the ID split's visual tokens** (960×540 vs 1280×720).
- 🟡 A larger model returns to the table on latency grounds, though not on VRAM or cost.

## 🔴 The risk inverted — it is STARTUP, not per-question

The clock starts when the process starts, so **imports + weight loading + CUDA-graph capture**
(`enforce_eager=false`) all consume the 120 s setup allowance. For an 8B in bf16 plus vLLM
warm-up this is the one number that could actually bite, and **it is measurable locally without
any jury hardware**.

This reverses the standing assumption: the danger was never `max_new_tokens` or a slow question.
It is a cold start that eats the pool before the first answer is produced.

## Consequences

- Levers previously costed against a per-question 5 s ceiling must be re-costed. In particular
  the "no beam search / `max_new_tokens ≤ 32`" guidance in `CLAUDE.md` is still *good practice*
  but is **no longer forced by the budget** — do not cite the cap as the reason.
- The `latency` field we emit is informational; **never optimise for it or read it as a score
  input**.
- Measure cold-start time locally before designing anything that raises k or resolution.

## Sources

- Submission template `README.md` §"The latency budget" + the track table footnote;
  `frame-algorithm/inference.py` docstring (batch handed over up front, model loaded once).
- Measured dev latency: `experiments/06-vit-lora/runs/06_vit_lora_v1/eval_best/predictions.json`
  (p50 0.170 s · p95 0.296 s · p99 0.352 s · max 1.169 s, n=6252).
- Related: [[the-gap-is-the-number-format]] (why `number` is the target) · `THE_MAP.md`
  (resolution branch, closed on the wrong premise).
