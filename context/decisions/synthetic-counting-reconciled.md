---
question: Is synthetic-counting SFT a live lever or a discarded one — and what exactly does the ±0.86 label-noise number measure?
verdict: DISCARDED stands. Calibration's death did not promote it, and rung 15 already ran the only half of it our labels can support — null.
status: SETTLED (reconciliation, zero GPU)
date: 2026-07-26
measured_in: context/ERROR_ANATOMY.md · context/decisions/epoch-matched-control.md · experiments/15-count-target/
---
# Decision: synthetic-counting SFT — the contradiction, reconciled

- **Status:** SETTLED · 2026-07-26 · zero GPU · **reconciliation, no new measurement**
- **Applies when:** anyone proposes counting-focused SFT, synthetic count QA, or a
  perception-grounded counting target (point-then-count, CoVT-style visual tokens).
- **Why this file exists:** two committed records said opposite things for a week, and a
  session proposed the lever a third time without noticing.

## The contradiction

| record | says |
|---|---|
| `context/CAMPAIGN_LOG.md:240` (§15 Discarded ideas) | **discarded** — *"poor prognosis: it teaches well-annotated visible objects; the real label moves ±0.86"* |
| `context/decisions/count-calibration-dead.md:62` | *"synthetic-counting SFT is now the **sole remaining lever** in the group that owns the gap"* |

Both are literally true and they are not in conflict once read carefully — but the second is
the one that gets quoted, and on its own it reads as an endorsement. It is not one. Its own
next sentence says so: **"This probe does not endorse it — it removes its competitor."**

Calibration died. That left synthetic counting standing **by elimination, not by merit**. The
CAMPAIGN_LOG entry is the merit verdict, and nothing has overturned it.

## What ±0.86 actually measures (it is not a tunable knob)

Asked directly, because the number carries the whole prognosis. From
`context/ERROR_ANATOMY.md` — computed from **the golds themselves**, 1,946 consecutive
annotated frame pairs inside the same video, clip counts:

| gap | n | mean \|Δcount\| | identical | jump ≥3 | max |
|---|---|---|---|---|---|
| ≤0.5 s | 1257 | 0.77 | 48.5 % | 6.0 % | **8** |
| 0.5–1 s | 242 | 1.33 | 16.9 % | 11.2 % | 7 |
| 1–3 s | 236 | 1.54 | 12.3 % | 14.8 % | 7 |

**±0.86 = the mean absolute change in the gold clip count between two annotated frames less
than one second apart**, across pairs where it changes at all (56.6 % of them). Observed
extremes: `8 → 14 → 6` in 690 ms; `7 → 7 → 1` in 270 ms. Set against the model's own mean
absolute counting error of **1.01** — the model is within ~1.2× of how far the *target itself*
moves between adjacent frames.

It is a **derived quantity with no free parameters**: no threshold, no ranking, no weighting.
The only choice in it is the time window, and the table shows all three windows agreeing in
direction. That is what distinguishes it from the constructed scores that have burned us
before (e.g. the transformation rankings, where the headline moved when the aggregation
changed — `CAMPAIGN_LOG.md` §9.2, "a maximum reads redistribution as loss", headline retracted).

🔴 **Its honest limit, and it is a real one:** frame-to-frame change mixes **genuine scene
change** (fast laparoscope motion, clips being placed, occlusion) with **annotation noise**.
So ±0.86 is an **upper bound** on the noise and **cannot separate the two**. It does not prove
any label is wrong. What it does establish — and this survives the caveat — is that **the
target is not stable at the timescale the model is asked to resolve.**

**Independently corroborated by a manual pass** (40 frames, gold 3–6, gold hidden): a motivated
non-clinical observer scored **r = −0.17** against the gold — no relationship at all — while the
model scores **r = +0.43**. The model already outperforms an untrained human here. The observer
reported clips are small and camouflaged against tissue, and that some frames give no clue where
"so many" come from. **The task needs clinical training to adjudicate**, which is why the
label-noise ceiling probe was postponed on 2026-07-25 rather than run by untrained eyes.

## 🔴 The thing that was missed: rung 15 already ran this

`experiments/15-count-target/` is the counting-target experiment. Its single variable is the
assistant target of `answer_format == "number"` rows: `2` → `{"label": "Clips", "counts": 2}`,
everything else byte-identical to rung 06 under three gates.

That is **the label-supported half of v05 (*Point, Detect, Count*)** — the paper that keeps
being proposed as the cheap route into perception-grounded counting. Its own CONTEXT says so
before the run (`context/15-count-target/CONTEXT.md`, Known limit #2):

> **No localization labels anywhere in the dataset**, so v05's pointing arm and v14's `[0,1000]`
> convention — **the two mechanisms both papers credit for the gain** — cannot be reproduced.
> This rung tests the weaker, label-supported half.

**Result** (`epoch-matched-control`): ep3 = 0.5699 vs rung 06's ep2 0.5667, i.e. **+0.0032
headline — and it does not survive the conjunction.** `margin_ID` +0.0169 but `margin_OOD`
**−0.0110**; the `number` margin on OOD is **−0.0053, below the template floor** — its own
pre-registered target; **0 of 10** paired cells exclude zero, all five OOD cells negative. And
the comparison is against an **unevaluated ep3 control** — rung 06 never scored its own epoch 3.

## Verdict

1. **DISCARDED stands.** `CAMPAIGN_LOG.md:240` is the operative verdict. `count-calibration-dead`
   is amended below so it can no longer be read as promotion.
2. **The half our labels can support has been run, and it is null** (rung 15). Proposing v05 or
   point-then-count as a *new* lever is a re-derivation — this is the fourth time the counting
   group has been re-proposed from a record that already answered it.
3. **The half our labels cannot support needs localization labels that do not exist in the
   dataset.** Any route that supplies them — SAM/DINO pseudo-points, CoVT-style visual tokens —
   inherits a harder version of the same problem: the pseudo-labels must agree with a gold count
   that moves ±0.86. You would be training a target that contradicts itself.
4. **Therefore the whole `aggregation` branch is gated on label trust, not on method.** Synthetic
   counting, point-then-count and CoVT all sit downstream of the same unmeasured quantity.
5. ⚠️ **And that quantity may not be cleanly measurable by us.** ±0.86 cannot separate scene
   change from annotation error; the blind human pass returned r = −0.17, i.e. an untrained
   observer cannot adjudicate. **A noise-ceiling probe designed without clinical adjudication
   would produce another interpretation-dependent number** — the failure mode this project has
   already paid for once. Do not build one on the assumption that it will settle anything.

**What this leaves.** The gap is real (`aggregation × ID` −0.0889 to 1st place) but every lever
inside it is either measured-null or gated on a label question we cannot currently answer.
The honest reading is that campaign effort belongs in `object_recognition`, latency/cold-start,
and seed variance — see [[epoch-matched-control]] §6b: **no run has ever been repeated with a
different seed, so we have no variance estimate at all**, and every effect discussed here is
smaller than the noise band we have never bounded.

## Sources

- `context/CAMPAIGN_LOG.md` §10 (the `number` ceiling), §15 (discarded ideas, line 240).
- `context/ERROR_ANATOMY.md` — the ±0.86 table, the caveat, the blind human pass.
- `context/decisions/count-calibration-dead.md` — the "sole remaining lever" sentence and its
  own non-endorsement.
- `context/15-count-target/CONTEXT.md` — Known limits #1 and #2 (pre-registered).
- `context/decisions/epoch-matched-control.md` — rung 15's numbers and the missing ep3 control.
- Related: [[the-gap-is-the-number-format]] · [[count-calibration-dead]] ·
  [[epoch-matched-control]] · [[class-imbalance-not-counting]]
