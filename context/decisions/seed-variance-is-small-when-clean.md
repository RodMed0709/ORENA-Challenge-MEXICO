---
question: How many seeds does a run need before its delta means anything — or do we declare every result inconclusive?
verdict: NONE, and the reason is now external and measured rather than an argument from cost. Seed variance is a property of training NEAR A NOISE-INDUCED CRITICAL THRESHOLD, not of fine-tuning. In the CLEAN condition three seeds span 0.5 pp; under 40% label-flip the same three span 68.5 pp. Our effects are read against the clean band, not the noisy one
status: LITERATURE (measured by others, on LLMs only — see the transfer limits)
date: 2026-07-29
measured_in: arXiv 2604.12469v1 Table 10 (p.21) + Appendix F.2 (p.17) — read end-to-end, not from a ficha
---

# Decision: seed repeats are not owed, because clean-condition seed variance is ~0.5 pp

- **Status:** LITERATURE · 2026-07-29 · zero GPU, zero cost.
- **Applies when:** anyone argues a rung is unreadable without seed replicates, or budgets a second
  seed. Also when quoting a delta near ±0.02.

## The question, and why it was open

This project has **never repeated a run with a different seed**, and several notes flag it as the
hole under every result we have: *"every delta sits inside a variance band we have never
measured."* The reading guide made #9 one of three must-reads on exactly this basis, describing it
as *"the most forceful warning against single-seed evaluation"*, citing a range of
**0% → −11% → collapse** driven by seed alone.

**That reading was built on a ficha, and the ficha inverted the paper's own conclusion.**

## What the source actually says

arXiv 2604.12469v1 is not a paper about seeds. It studies **three kinds of noise in the fine-tuning
data** — label flip (target-side), typographical and grammatical (input-side) — across GPT-2,
Qwen2-0.5B and Llama-2-7B on sentiment / QA / MT. Seeds appear once, as a robustness check in
**Appendix F**, on a single condition: **Llama-2, sentiment, label-flip 40%**.

**Table 10 (p.21), and the column that matters is the first one:**

| seed | **clean** | label-flip 40% |
|---|---|---|
| 1 | **94.5%** | 95.7% (+1.2%) |
| 22 | **94.5%** | 83.5% (−11.0%) |
| 7 | **94.0%** | **27.2% (collapse)** |
| 42 | *(default, used for the main experiments)* | |

- **Clean: 94.0 – 94.5. Total spread 0.5 pp.**
- **Noisy: 27.2 – 95.7. Total spread 68.5 pp.**

The authors attribute the second range explicitly, and it is not to fine-tuning being stochastic:

> *"This wide variance indicates that LLaMA-2 under 40% label-flip noise **operates near a critical
> threshold** — the noise level is severe enough that the random initialisation and data ordering
> determined by the seed can tip the optimisation trajectory toward either a robust or a collapsed
> solution."*

**Appendix F.2 corroborates at the representation level.** They compute CKA between *clean* models
trained with **different seeds** specifically to rule out seed variance as an explanation for their
main effect: the **clean-vs-clean CKA floor is 0.890**, against 0.11–0.42 under noise —
*"confirming that the representational changes … reflect genuine noise effects rather than
stochastic training variance."*

## What it licenses for us

Our effects, against a **0.5 pp** clean band:

| result | delta | vs the clean seed spread |
|---|---|---|
| rung 21 arm A (`bucket_mean`) | **+5.84 pp** | **≈ 11×** |
| rungs 14 / 15 | ±2 pp | ≈ 4× |

⇒ **Seed replicates are not owed.** Rung 21 clears the band by an order of magnitude, and rungs 14
and 15 were already killed by the paired video-clustered bootstrap (0 of 6 cells) — **two
independent instruments agreeing**, which is the closest thing to a replication this project can
afford.

## ⚠️ What this does NOT say — the transfer limits, stated so they are not skipped

- **It is a saturated binary task.** 94.5% accuracy is near ceiling, and variance compresses near a
  ceiling. Our `bucket_mean` is 0.63 — mid-range, where run-to-run spread is typically larger. The
  0.5 pp is a **lower bound on what we would see**, not an estimate of it.
- **n = 3 seeds.** A spread from three samples is a weak estimate of a standard deviation.
- **LLM only. Zero vision.** GPT-2 / Qwen2-0.5B / Llama-2-7B on text tasks. Our failure lives in the
  vision tower, and nothing here measures a ViT.
- **It does not make a paired CI into a variance estimate.** [[undertrained-was-real]] stands:
  a paired bootstrap says the difference between *these two models* on *these questions* is real;
  it still **does not say where a rerun lands**. What changed is that we now have an external reason
  to believe the rerun lands close, in the clean regime.
- 🔴 **And we are not perfectly "clean".** Our gold carries measured instability — the **±0.86** count
  drift between annotated frames <1 s apart, and the 8.4% under-naming behind rung 18's minted
  zeros. We are nowhere near 40% label-flip, but we are not at zero either, and this paper's whole
  point is that seed sensitivity is a **function of label-noise level**. If a future rung trains on
  pseudo-labels, this decision must be re-read. See [[target-noise-is-the-harmful-kind]].

## What it costs to be wrong

If clean seed variance on our task is really ~2 pp rather than 0.5, rung 21 (+5.8) survives
comfortably and every ±0.02 rung in the ladder was always unreadable — which is what the paired CI
already concluded for 13, 14 and 15. **No verdict in the repo flips under that assumption.** That
asymmetry is why this closes without a seed run.

## Sources

- arXiv 2604.12469v1 — Table 10 (p.21), Appendix F.1 (p.16), Appendix F.2 (p.17), Table 9 (p.17)
- [[undertrained-was-real]] · [[epoch-matched-control]] · [[target-noise-is-the-harmful-kind]]
