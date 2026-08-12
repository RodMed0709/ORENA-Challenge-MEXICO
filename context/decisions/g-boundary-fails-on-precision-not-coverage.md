---
question: Are SAM masks usable as a localization target on our footage — i.e. does rung 37 (step 7, attention vs SAM masks) get to run?
verdict: NO. G-BOUNDARY FAILS. B1 (covered) passes decisively at 0.9714, B2 (clean | covered) fails at 0.3529 with its ENTIRE CI below the 0.70 threshold. Per the pre-registration the rung dies here, with only the export spent. It failed where nobody was looking: SAM finds the objects and does not delimit them, and the metallic-clip objection that consumed 2026-08-11 decided nothing
status: SETTLED
date: 2026-08-12
measured_in: experiments/37-attention-vs-masks/RESULTS_gboundary.json (N=40 stratified 5×8 classes, random_state=42, single non-blind adjudicator)
---

# Decision: G-BOUNDARY fails, and it fails on sharpness — while the gate never asked the question that matters

- **Status:** SETTLED · 2026-08-12 · adjudicated by legokna in `runs/gboundary_viewer.html`.
- **Applies when:** anyone proposes SAM masks as a localization target, an attention target, or an
  input overlay on this footage; or reads rung 37's death as *"SAM cannot see foreign objects"*.

## The numbers, against thresholds pre-registered 2026-08-09 and never amended

| | quantity | value | threshold | |
|---|---|---|---|---|
| **B1** | fraction *covered* | **34/35 = 0.9714**, CI [0.9143, 1.0000] | ≥0.70 **and** CI-low >0.50 | ✅ PASS |
| **B2** | fraction *clean \| covered* | **12/34 = 0.3529**, CI [0.2059, **0.5294**] | ≥0.70 **and** CI-low >0.50 | 🔴 **FAIL** |

Both had to pass. B2 does not fail narrowly: **its entire interval sits below the 0.70 point
threshold.** The verdict is robust to the handling of `unadjudicable` — a mark added to the viewer
*after* pre-registration, whose treatment was therefore never fixed in advance. Excluded from the
denominator, B1 = 0.9714; counted as not-covered, B1 = 0.8500. Both pass; B2 is untouched either
way. The choice is not load-bearing and is recorded rather than argued.

`unadjudicable` ran **5/40 = 12.5 %**, above the 2.8 % "cannot tell" of the blind full-range
counting pass, and irrelevant to the outcome: **B2 fails on the frames the adjudicator did
resolve.** Worst class `silicone_loop` (clean 0/5), best `specimen_bag` (3/4).

## 🔴 It failed where nobody was looking

**The metallic-clip objection decided nothing. Clips scored `covered` 4/4.** The concern that
consumed all of 2026-08-11 — in its original form (`r = −0.17`, **RETIRED**,
[[gold-is-signal-model-underuses-it]]) *and* in the corrected form that replaced it — turned out
not to be the failure mode in either direction. 📌 The correction was right to make and it still
did not predict the result: *a retired number and its replacement can both be beside the point.*

**What died instead is this rung's enabling fact.** `PLAN.md` listed as settled-going-in that SAM
*"does not merge tissue regions with foreign objects even where it over-segments tissue"*, drawn
from an 8-frame eye pass. **B2 = 0.3529 is its direct refutation.** SAM *finds* the objects across
all eight classes and does not *delimit* them.

## 🔑 What the gate never measured — PRECISION

Raised by legokna on reading the adjudicated frames, and it is the finding worth keeping:

> B1 asks whether a mask exists over a foreign object. B2 asks whether it is sharp. **Neither asks
> how many OTHER masks are there.**

SAM emits **13–50 instances per frame** (`RESULTS_ab_overlay.json`) and masks the whole scene —
tissue, anatomy, instruments — not only the foreign objects. ⇒ **A VLM handed these masks receives
references without identity.** It can see that regions were marked, not which one is the foreign
object. That is consistent with the overlay's measured failure mode, which was **identity
substitution** (`'External drain'` → `'Needle'`, `'Needle'` → `'Clip'`), not blindness.

⚠️ This is a limitation of the **pre-registration**, not of the adjudication. **A future gate on
masks-as-input must carry a precision clause: B1/B2 alone cannot license the route even at 1.00.**

## What this does NOT show

- **Not** that SAM fails to find the objects. It finds them: B1 = 0.9714 across all eight classes.
- **Not** that masks are useless as input in general. One mask source, one rendering — see
  [[input-side-access-was-never-the-bottleneck]].
- **Not** that domain-adapting SAM would be wrong. It is deferred on a different argument entirely,
  [[sam-adaptation-has-no-route-to-points]], re-affirmed the same day.

## Limitations, recorded not fixed

**One adjudicator, not blind to the hypothesis** — as `PLAN.md` said going in. It is a **gate**: it
may only kill the rung, never grant it anything, so a non-blind adjudicator returning a kill is the
direction that costs least. The raw per-frame marks are embedded in
`RESULTS_gboundary.json` because `runs/` is gitignored (`.gitignore:34`) — that file is the durable
record, and the viewer export alone would have been the `RESULTS_controls.json` failure a third time.

📌 The publishable form, as pre-registered: **SAM masks are not a localization target on this
footage.**
