# Context — 07-enumeration

> Curated notes. The report is `experiments/07-enumeration/README.md`; this file holds what the next
> person needs to know that the report does not say.

## What this probe cost, and why it was still cheap

Three designs, three deaths in the smoke, **zero hours of wasted GPU**. Every failure was caught at
n=20 in ~2 minutes. The full would have been ~436 inferences (v3) or 6252 (v2). **The smoke paid for
itself three times.**

The recurring defect was never a coding bug. It was always the same thing: **the intervention did not do
what its name said**, and the gate meant to catch that was too weak to notice.

## The landmine: `number` is not one question

Verified against the parquets (`/workspace/orena-data/{heico,lapchole}/data/frame/test.parquet`), not
assumed. Of the 2094 `number` questions in val:

| Template | n | Enumerable? |
|---|---|---|
| `How many different foreign object instances appear in this frame?` | 830 | ⚠️ An unprompted list names **classes**; 2 clips = 1 class, 2 instances |
| `How many different foreign object classes appear in this frame?` | **436** | ✅ The only clean target |
| `How many Clips / Sponges / Needles / … appear in this frame?` | 828 | ❌ Asks about **one class** |

**Anything that treats `number` as homogeneous is measuring a mixture.** This bit the probe's
pre-registered reference: the card's `1.24 / 1.69 / 2.01` is the *mixed pool*. On the 436 `classes`
questions the baseline is **1.158 / 1.400 / 1.778, r = 0.333** (not the 0.625 the card cited).

> **This landmine is not spent.** Any future work on `aggregation` — including reading rung 02's
> `acc_number = 0.4331` — is reading an average over three questions that measure different things.

## Baseline reference on the 436 `classes` questions (recomputed, zero GPU)

| | @1 | @2 | @3 | @4 | overall |
|---|---|---|---|---|---|
| n | 265 | 150 | 18 | 3 | 436 |
| acc | 0.8453 | 0.4000 | 0.0556 | 0.0000 | 0.6537 |
| `counted` mean | 1.158 | 1.400 | 1.778 | 1.667 | — |

Spearman(truth, counted) = **0.333**. Truth distribution `{1:265, 2:150, 3:18, 4:3}`; what it says:
`{1:320, 2:114, 3:2}`. **It says "1" more often than "1" is true, and never says 4.**

## The finding worth carrying forward

**The prefill does not unlock enumeration.** `ITEMS:` prefilled onto the assistant turn is read as a
**field label**, not a sentence to continue: the model answers `ITEMS: 1, CLASSES: 1.` — still a count.

This matters beyond the probe:

- 🔴 **A3 (constrained decoding) has its first negative datum.** The prefill is the *minimal* form of
  forcing this model's output. It did not take. Nothing says a full FSM would fail, but the cheap
  version of the mechanism did.
- **It agrees with rung 03** (0 of 6 prompts won; `a5_baredigit`, a counting protocol, *hurt*) and with
  v1 (ignored 20/20). **Three independent attempts to change this model's behaviour by prompt or prefill
  have failed.** After 13.7k bare-answer examples, the output format is not up for negotiation.
- ⇒ **If we want a reasoning format, the route is training (CoA via SFT), not decoding.** That is where
  the literature puts the value anyway: CoA's ablation gives **+16.3 for the format, +1.7 for the RL**.

## The paired analysis (independent of the probe, and it survives)

126 frames carry both `List all foreign objects…` and `How many different foreign object classes…`.
Same frame, same model, two ways of asking. Built by joining on `(dataset, video, timestamp_start)`.

At truth ≥2 (n=38): **names 1.447 · counts 1.526**, delta −0.079, p=0.45, 31/38 identical.

> **The model does not name more than it counts.** When it misses the second object, it misses it in
> both channels.

⚠️ **n=38. This does not carry a verdict** — it is the same bar (`n=20 is not a result`) applied to
myself. It is worth keeping because it is **independent of the probe** and free. If someone wants this
properly powered, the join is the cheap part; the sample is the limit.

⚠️ **The trap it dissolves:** pooled, `fo_class` names 1.76 while `number` counts 1.40 — which looks
like "sees but won't say". **It is an artifact of aggregating different frames.** Under pairing it
vanishes. *This nearly became a finding.*

## Process notes

- **`agy` was taken off this atomic** (HOME §10.3 fallback) after two rounds. Not for the code — his C1/C2
  fixes were correct and his commit was clean. **For the reports:** three times the numbers were right
  and the interpretation ran to the reassuring conclusion, the last one citing the taxonomy dump — the
  failure mode — as proof it "worked". **The `report numbers, not verdicts` guardrail earned its keep.**
- **The kernel is now pinned to `infer`** in this notebook (`chore(enumeration): declare the infer
  kernel`). Papermill aborts without a kernelspec, and Jupyter stamping it on first run makes the
  tracked file and the executed one differ for reasons nobody chose. **The other rungs still pin
  `python3` (`/usr/local/bin/python`), which has neither `transformers` nor `focus` — the open
  reproducibility hole behind "which env trained rung 02?". Not fixed here.**
- **Provenance is nailed for once:** the pod ran the tracked notebook at `1200f58`, verified byte-identical
  (`git diff --quiet HEAD`) before launching. No `.py` conversion, no hash to reconcile.
