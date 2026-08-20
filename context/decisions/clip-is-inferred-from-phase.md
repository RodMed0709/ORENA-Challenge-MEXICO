---
question: Rung 48's probe v2 can finally measure precision. Which of its cells may be used to rank an arm, and what does the `Clip` cell say?
verdict: ONE cell ranks — `Specimen bag` F1, which reproduces the three platform anchors on 15 of 15 leave-one-video-out folds. The macro is a coin flip (7/15) and `Clip` INVERTS on 15 of 15. But the `Clip` cell is the best RESULT of the rung: on frames where the clip provably does not exist yet, all four models answer `Clip` in 85-95%, and the rate climbs monotonically with proximity to the clipping event (0.57 -> 1.00) ⇒ the models infer the clip from the surgical PHASE, not from the object
status: MEASURED
date: 2026-08-19
measured_in: experiments/48-centre-probe/RESULTS_probe_v2.csv · RESULTS_probe_v2_jackknife.csv · RESULTS_clip_fp_anatomy.csv (4 models, 4,890 items, 15 held-out CholecT50 videos)
---

# Decision: score rung 48 on the bag cell, and read the clip cell as anatomy

- **Status:** MEASURED · 2026-08-19 · ~1 h on two GPUs, no training
- **Applies when:** reading any arm through the rung-48 probe, or proposing a lever whose
  mechanism is *"make it stop over-calling `Clip`"*.

## The calibration question

Three of the four models carry a platform score: rung 06 **0.4767** < A2 **0.5288** < rung 42
**0.5809**. A probe cell is usable **iff** it reproduces that order — and reproduces it on every
leave-one-video-out fold, because 15 clusters is not many and one video should not decide.

| cell | orders 3/3 | r47 below r42 |
|---|---:|---:|
| **`Specimen bag` F1** | **15/15** | **15/15** |
| `set_size` | 9/15 | 15/15 |
| `macro` F1 | 7/15 | 15/15 |
| `Clip` F1 | **0/15** | 0/15 |

## 🔴 The macro is the trap, and it is the same trap as rung 47's

A2 **0.75028**, rung 42 **0.75030** — a **1.4e-5** tie where the platform separates them by
**0.052**. Averaging a cell that orders with a cell that inverts dilutes the real effect to
nothing. That is precisely the error [[split-v2-by-video]] caught one level up, where `ALL_ID`
averaged a live recognition effect with a null until it read as null. 📌 **Report the cell.**

## 🔴 v1's `set_size` verdict is downgraded, honestly

v1 read `set_size` 1.42 / 1.40 / 1.22 as 3/3 and called it the guard that beat the headline. On
v2's corpus it is 1.2207 / 1.2188 / 1.1202 and orders on **9 of 15** folds. The r06–A2 gap is
**0.0019**; it was **0.02** in v1 and should have been read then as *one usable separation, not
three*. The v1 note stands as written — this is what the jackknife it never had now says.
⚠️ Its direction is also inverted (lower is better): scored ascending like an F1 it reports 0/15
for a cell that is ordering. Ordering tests are per-metric, not per-table.

## 🟢 The finding: `Clip` is inferred from the phase, not seen

v2's negatives are provable — before a video's first `clipper,clip,*` triplet no placed clip
exists. In all 15 videos the last negative frame precedes the first positive one (gated in
`_tools/clip_fp_anatomy.py`, raises). On those **1,332** frames:

| | r06 | A2 | r42 | r47 |
|---|---:|---:|---:|---:|
| answers the bare string `'Clip'` | 0.950 | 0.853 | 0.909 | 0.893 |
| FP rate, `gap > 300` frames (n=808) | 0.954 | 0.813 | 0.870 | 0.861 |
| FP rate, `gap < 100` frames (n=166) | 0.988 | 0.994 | **1.000** | 0.982 |

**It is a ramp, not a constant** — 0.567 in the first decile of the negative window, 1.000 in
the last. The models are reading the dissection and the applier entering the field and inferring
a clip that has not been placed.

🔑 **This is a different defect from the one the campaign has been naming.**
[[frequency-prior-is-the-failure-shape]] describes a frequency prior — a constant. A monotone
function of time-to-event is not that. And it is consistent with
[[vcd-has-nothing-to-subtract]]: the model **does** look; the phase prior overrides what it sees.
⇒ a lever whose mechanism is *"make it stop answering blind"* still has nothing to subtract.

⚠️ **The honest limit.** The top of the ramp is where the label is weakest — a clip deposited
just before the first annotated triplet. That band is uninterpretable. But `gap > 300` is **61 %**
of the negatives, its label is not in doubt, and it runs **0.81–0.95**. The phenomenon survives
without the contaminated band.

## 🔴 What this does NOT license

**"Train on hard negatives so it stops over-calling `Clip`."** The corpus exists — 12,281
provable clip-negatives, and CholecT50 has 11,061 instrument-free frames besides. But in the
three anchors we have, **clip aggressiveness runs the wrong way**: A2 is the least aggressive
(FP 0.870) and rung 42 beats it by 0.052 on the platform. Every temporal cut inverts the pair.
n=3 concludes nothing, but it is enough to demand a cheap test — does fewer clip FPs buy
**exact-set-match**? — before spending a training run or an hour of human annotation on it.

## Sources

- `experiments/48-centre-probe/README.md` §v2 · `_tools/probe_v2_report.py` · `_tools/clip_fp_anatomy.py`
- Related: [[split-v2-by-video]] · [[vcd-has-nothing-to-subtract]] ·
  [[frequency-prior-is-the-failure-shape]] · [[the-podium-gap-is-object-recognition]]
