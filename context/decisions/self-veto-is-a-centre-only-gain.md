---
question: The shipped checkpoint scores 0.0000 on every centre-probe item whose gold is `none` — it never says "none". [[zero-is-format-localized]] measured the same model saying "no" fluently in `binary` format. Can it delete its own false classes if asked in that format?
verdict: NO, NOT SHIPPABLE. It answers the yes/no form cleanly (parse rate 0.94-1.00) and never vetoes a class that is really present (0.000), so the intervention is one-directional — and it is worth +0.0380 exact-set on the centre probe, 15/15 folds. But on rung 42's own held-out videos it costs −0.0653, 0/8 folds. ID is half the platform score; the arm loses on the conjunction. Also: the 0.82 absence rate it was licensed on is rung 06's, measured on the corpus's CO-OCCURRENCE binaries; on r42, with a single-class presence question, the rate is 0.156
status: MEASURED
date: 2026-09-02
measured_in: experiments/58-self-veto-and-pair/RESULTS_{smoke_gate,veto,id_control}.json — 4,890 centre items over 15 held-out CholecT50 videos, 490 `fo_class` items over rung 42's 8 held-out videos, ~35 GPU-min on one RTX 6000 Ada, no training
---

# Decision: the self-veto pays on centre and ID pays for it

- **Applies when:** costing any lever that asks THIS checkpoint to re-answer in a different
  answer format and uses the second answer to edit the first.

## The failure it was aimed at

On rung 48's centre probe, `r42 ep4` is correct on **0 of 2,445** items whose gold is `none`.
Not rare — never. Half that probe is a population where the model scores zero, and the cause is
in the training data: **0 of 8,969 `fo_class` golds say `none`**, so the format has no way to
express absence and the model always fills the list.

## 🔻 The premise did not survive re-measurement, and that is the first finding

[[zero-is-format-localized]]'s `emit_absent 0.82` is **rung 06**, and its binary questions are the
corpus's own: *"Do Clips and Sponges co-occur in this frame?"* — a **pair** question. There is no
single-class presence question in the corpus at all. Asked one, `r42 ep4`:

| | cooccur-shaped | "is a X visible" |
|---|---|---|
| parses as yes/no | 0.939 | 1.000 |
| says "no" when the class is ABSENT | 0.156 | 0.125 |
| says "no" when the class is PRESENT | **0.000** | **0.000** |

The format rigidity [[debate-works-and-the-roles-are-backwards]] warned about did **not** appear —
it answers the new form cleanly. What did not transfer is the discrimination.

## The gate was relaxed after seeing that, and it was the right call to record

Pre-written gate: parse ≥ 0.90 **and** margin ≥ 0.20 → the run is worth 35 GPU-minutes. Observed
margin 0.156, so it aborted. It was then relaxed to ≥ 0.05, in the script, with the reason:
**this is a cost gate, not the test**, and a veto that removes 15.6 % of false classes and 0 % of
true ones cannot lose accuracy on its own axis. The pre-registered comparison — full-run delta
with a leave-one-video-out jackknife — was not touched. Anyone re-reading this should hold that
distinction; moving a *test* threshold after the fact would be the thing [[significance-rule]]
forbids.

## And the ID control killed it

| | centre (15 videos, 4,890) | ID (8 videos, 490 `fo_class`) |
|---|---|---|
| r42 ep4 | 0.3618 | 0.8408 |
| **+ self-veto** | **0.3998 · +0.0380 · 15/15** | **0.7755 · −0.0653 · 0/8** |

`bucket_mean` is half OOD and half ID. **The cost is 1.7× the gain and lands on the half our own
eval can actually see.** This is [[self-consistency-dead]]'s arm C mirrored — a large single-axis
gain that only the ID-AND-OOD conjunction catches. The conjunction stopped a slot being spent on
something measurably worse than what already ships.

## What survives

The *shape* of the finding does. The model's dominant centre error is naming a class that is not
there, and something that deletes those classes is worth a lot. What must not do the deleting is
the model's own second opinion in another format. A **differently-trained checkpoint** doing it
is [[checkpoint-pair-shorter-list-ships]], which pays on centre and is flat on ID.
