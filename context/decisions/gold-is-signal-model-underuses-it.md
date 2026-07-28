---
question: Is the counting gold recoverable from the frame — i.e. does `number` have an annotation ceiling?
verdict: NO CEILING IN THE "LABEL IS NOISE" SENSE — a blind human glance scores Spearman r = +0.72 against the gold across the full 1–12 range, far above our fine-tuned model's +0.43. The gold carries real, ordered, frame-visible signal that the model is NOT using. ⚠️ But the gold's SCALE is not what an eye perceives: the human sees ~1/3 as many (bias −3.76). The pessimistic reading is retired; the target moves from calibration to DISCRIMINATION
status: MEASURED
date: 2026-07-28
measured_in: experiments/16-count-probes/RESULTS_16b_human.csv (n=106 counted + 3 "cannot tell", 109 frames, 30 videos, gold stratified 1–12, label hidden until export)
---

# Decision: the gold is signal, and our model is leaving most of it on the table

> 🔴 **PARTLY RETRACTED (2026-07-28) — see [[model-out-ranks-the-blind-human]].** The
> "our model +0.43" in every table below is a **gold 3–6 range-restricted** number
> (`ERROR_ANATOMY.md:158`), placed next to a **full-range** human number. Measured on the
> human's OWN 109 frames the model scores **r = 0.8303** against the human's **0.7230**, and
> on the full `Clips` template **0.5866**. **The model out-ranks the human.** Everything
> below that reads "the model orders worse than a person", "the headroom is large", or
> "the model emits the prior rather than the frame-to-frame signal" is **withdrawn**. What
> stands: the frames are readable (2.8% "cannot tell"), so the annotation-ceiling reading
> stays retired; and the target is still discrimination, now for a different reason — the
> model DOES order (r ≈ 0.50 on OOD) and still scores exactly at the trivial floor.

- **Status:** MEASURED · 2026-07-28 · **zero GPU** · blind human adjudication
- **Applies when:** costing any `number` lever, or invoking "annotation ceiling" as a reason to
  abandon counting.

## Why this was measured

Rung 06 ep3 leaves **`number_margin_OOD` = 0.0** — exactly the trivial floor. Rung 05 had already
found that a **black image** returns that same floor **to 16 digits**, i.e. the counting output is
a prior, not perception. And `ERROR_ANATOMY.md` recorded a blind human at **r = −0.17**, which
read as *"not even a person can do this"* — an annotation ceiling that would kill the whole line.

Probe 16b tried to settle it with a frozen OWLv2 detector and **could not**: r = 0.046, but it
detected 0.67 clips where the gold says 3.52 and found nothing on 58.6% of frames. Its positive
control stayed alive, so the instrument was not dead — just far too weak to separate "the label is
invisible" from "this detector cannot see surgical clips". Recorded as **NOT MEASURABLE**.

So the question went to a human, blind, on a stratified sample.

## What was measured

109 frames, **stratified across gold 1–12** (12 each up to gold 4, tapering to 2 at gold 12), 30
videos, 44 ID / 65 OOD, **shuffled**, dataset label **hidden until export**, and *"no se puede
saber"* offered as a first-class answer.

| measure | value | reference |
|---|---|---|
| *"cannot tell"* | **2.8%** (3/109) | the frames ARE readable |
| **Spearman r vs gold** | **+0.7230** | our model **+0.43** · OWLv2 **+0.046** · earlier blind human **−0.17** |
| exact agreement | 4.7% | — |
| mean \|Δ\| | 3.80 | — |
| **bias (human − gold)** | **−3.76** | our model **−0.66** |
| Δ ≤ 0 | **99 of 106** | almost always sees fewer |

| gold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mean human count** | 0.50 | 0.58 | 0.82 | 0.91 | 1.10 | 1.67 | 2.50 | 2.60 | 2.60 | 3.17 | 4.67 | 3.50 |

Monotone in the gold — hence r = 0.72 — at roughly **one third of the slope**.

## 🔴 The reconciliation: −0.17 and +0.72 do not contradict

`ERROR_ANATOMY.md:158` measured **40 frames restricted to gold 3–6**. That is **range
restriction**, which attenuates rank correlation toward zero by construction: four adjacent values,
and within-band perceptual noise swamps them. This probe spans the **full 1–12 range**.

Both are true and together they say more than either:
- **Across the range**, the gold tracks what a person sees, strongly (**+0.72**).
- **Within a narrow band (3–6)**, it is unreadable (**−0.17**).

The old note's conclusion — *"the task needs clinical training to adjudicate"* — survives for
fine discrimination. Its implied conclusion, that the label carries no visible signal, does not.

## 🟢 The finding that changes the plan: scale vs ordering

| | scale | ordering |
|---|---|---|
| **our model** | ✅ nearly right (bias **−0.66**, mean pred 2.85 vs gold 3.52) | ❌ **r = 0.43** |
| **blind human** | ❌ wrong (bias **−3.76**) | ✅ **r = 0.72** |

**The model has learned the gold's marginal distribution and not its frame-to-frame signal.** It
emits the prior — precisely what rung 05's black-image result says, and why `number_margin_OOD`
sits at exactly 0.0.

⇒ **There is recoverable signal the model is not using, and the headroom is large: a human glance
orders better (0.72) than our fine-tuned model (0.43), under the gold's own scale.**

⇒ **The target moves.** Not calibration — the scale is already close, and
[[count-calibration-dead]] killed post-hoc calibration three ways. The target is
**discrimination**: telling a 3-frame from a 7-frame.

## A second, unplanned corroboration for minted zeros

At **gold = 1 the mean human count is 0.50** — half the time a person sees *no* clip where the
label asserts one. Zero-vs-nonzero is the coarsest and most learnable distinction on this axis,
and it is the one the model has **zero training examples** for ([[zero-is-format-localized]]).

## ⚠️ What this is NOT

- **n = 1 annotator, not a clinically trained one.** This cannot separate *"the gold counts things
  not visible in this frame"* from *"an untrained eye under-counts small, camouflaged, specular
  clips"*. Both remain live, and the earlier note's own observer reported clips being
  *"camouflaged against tissue"*. What IS established: the signal exists and the model under-uses it.
- **Not a licence to fit the human.** The human's scale is *further* from the gold than the
  model's. Training toward a human-perceived count would move the scale the wrong way.
- **Not a claim about fine discrimination.** Within gold 3–6 the earlier −0.17 stands.

## Consequence

The annotation-ceiling reading is **retired**. Counting is a live target, and the next `number`
lever should be judged on **rank/discrimination**, not on mean error. Recorded against
[[the-gap-is-the-number-format]], [[count-calibration-dead]] and [[epoch-matched-control]].

## Sources

- `experiments/16-count-probes/RESULTS_16b_human.csv` (the adjudication export)
- `experiments/16-count-probes/_tools/contar.html` + `README_adjudicate.md` (the instrument)
- `context/ERROR_ANATOMY.md:55-63` (the model's own bias-by-gold table), `:158-165` (the −0.17)
- Probe 16b (`RESULTS_16b.csv`) — the detector attempt, recorded NOT MEASURABLE
