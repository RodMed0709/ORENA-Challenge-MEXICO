---
question: When may a local delta be called an improvement, and when does it justify spending one of the remaining submission slots?
verdict: Large local deltas (>= ~0.03) ship on sign alone — direction transferred in 8 of 8 measured bucket comparisons. Small deltas do not run automatically: there is no evidence they transfer, they are not separable from an unmeasured seed band, and the question stops being statistical and becomes whether they are worth one of 8 slots. That call belongs to the team, with the cost stated.
status: SETTLED
kind: process rule, zero GPU
date: 2026-08-05
basis: the local-vs-judge calibration over the two scored submissions ([[local-eval-vs-judge-calibration]]) — replaces the 2026-08-01 draft floor of eps=0.05, which was chosen for internal coherence and is larger than the entire competitive field
---

# Significance: large ships, small gets discussed

## Why this replaces the draft

The 2026-08-01 draft (`local/specs/regla-significancia/PROPUESTA.md`, never signed) set an
effect floor of **ε = 0.05** *"for coherence with the phase-0.4 equivalence test"*. Against
the leaderboard that number is absurd: **rank 1 to rank 11 spans 0.0365**, and adjacent ranks
near the top are separated by **0.002–0.009**. A 0.05 floor would call every move capable of
changing our rank — including winning the challenge — *trivial*.

The draft also asked the wrong question. It adjudicated *"is arm A better than arm B?"*, a
comparison against ourselves that pays nothing. What pays is distance to the baselines and to
the teams above.

And a rule cannot decide alone what an improvement is worth. **legokna's formulation, which is
the one adopted:** let the data decide the large cases, and let the team decide the small ones
with the price on the table.

## The rule

**R1 — Large ships.** A local delta of **≳ 0.03** on the primary cell is acted on without
further argument. Direction transferred in **8 of 8** bucket comparisons across both scored
submissions and never reversed ([[local-eval-vs-judge-calibration]]).
⚠️ **Evidence, not a guarantee** — two submissions, both large moves. Read the result on
return; never book the gain in advance.

**R2 — Small gets discussed, it does not run.** Below that, no automatic run. There is no
observation of a small delta transferring — **we have never shipped one** — and the question
becomes *is this worth 1 of 8 slots?*, which is the team's to answer, not a gate's.

**R3 — The threshold is derived, not chosen.** ~0.03 is what survives the measured deflation
(÷1 `agg_ID`, ÷1.5 `obj_ID`, ÷2 `obj_OOD`, ÷3.8 `agg_OOD`) while still moving rank. It moves
when the calibration moves — i.e. when a new submission adds a point.

**R4 — Declare the primary cell before the run.** One cell, named in the pre-registration.
🔴 It may **NOT** be local `bucket_mean`: that number overstates the judge by +0.12 and inverts
the bucket ordering. Use the ID cells (calibrated, 0.98–1.48×) or an explicitly deflated OOD cell.

**R5 — Below |Δ| = 0.01 nothing is readable.** [[seed-variance-is-small-when-clean]] is
borrowed literature (Llama-2, sentiment, one condition). **This project has never repeated a
run under a different seed.** A small delta has no sign to transfer if it is noise — this is
why R2 does not become "small deltas are fine, just cheaper". Repeating one seed on the
ladder's best arm retires this clause; nothing else does.

**R6 — Reserved vocabulary.** `WIN`/`NULL`/`LOSS` apply to the primary cell only. Every other
cell is exploratory and is reported as *"consistent with"*. ⚠️ The cells are **not
independent**: of the 10 per epoch, 3 (`ALL`, `ID`, `OOD`) are aggregates of the other 7, and
`ALL` is **64% OOD by question count** while the challenge weights ID and OOD equally.
Significance declared on `ALL` is not significance on the metric that pays.

**R7 — Retroactive in one direction only.** The ~150 cells already read are not relabelled.
But no NEW argument may lean on a secondary cell as if it were a win; a load-bearing old cell
is re-read under R4–R6 first.

**R8 — A faithful NULL is published.** No re-cutting or re-slicing a rung after the fact to
find a cell that clears the bar. Post-hoc slicing IS the multiplicity problem.

## The problem this still solves

The repo called WIN on **3 of 30 cells** in a single rung, where Binomial(30, 0.05) gives
**P(≥3 by noise) = 0.188**, and ~150 paired cells have been read across the campaign with no
multiplicity correction (~7.5 expected false positives). ⚠️ That 0.188 **assumes independence
the cells do not have** (see R6) — it is an order-of-magnitude argument, not an exact rate.

## The price, stated

`A2_lr` ep3 — the ladder's best point — is ALL **+0.0211**, CI [0.0037, 0.0391]. It clears R5
but **not R1's ~0.03**. Under this rule A2 was worth shipping (and was: submission 02, rank 11)
while its size may not be relied on. What actually happened on the judge: **+0.0521**, because
the comparison was against submission 01's checkpoint, not against arm A.

## What this does not fix

🔴 We still cannot test our **+0.0099 over the fine-tuned baseline** — that needs the
baseline's per-question answers, which only the organizers hold. **That, not this rule, is the
open risk to co-authorship.**

## Open, and cheap

- **A jackknife over the 10 OOD videos**, re-scoring predictions that already exist. If
  `acc_OOD` swings ±0.05 when one video is dropped, no OOD comparison in 27 rungs meant
  anything. Zero GPU. (**Not** `kfold_lopo` — that costs a training run per fold.)
- **One seed repeat** on the best arm — the only thing that retires R5.
