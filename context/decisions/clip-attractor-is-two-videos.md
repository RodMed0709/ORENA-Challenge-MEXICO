---
question: How big is the `fo_class` Clip front really, what shape is the defect, and does external recognition data move it?
verdict: The quoted +0.0702 is a LOOSE CEILING — it counts every error that *involves* Clip or Sponge, which is 90.7% of all `fo_class` errors, so it says little more than "fix fo_class". The attractor's own mass is +0.0387, still over the ship bar. But it is not a Clip-vs-Sponge pair (Needle loses 24.6% of its rows to Clip against Sponge's 13.3%) and it is not a global class prior — HALF of it sits in 2-3 of 38 videos, with `0024-Heico-Sigma-5` the worst offender in all six arms measured, so it cannot clear a video-clustered CI. External positives (rung 19b) have no attributable effect: a2, with zero CholecT50 rows, lands within 0.008 of the arm built to move the number
status: MEASURED
date: 2026-08-22
measured_in: experiments/51-clip-attractor/ — RESULTS_fo_class_headroom.json · RESULTS_fo_class_per_class.csv · RESULTS_clip_fp_by_video.csv · RESULTS_clip_fp_anatomy_49.csv · RESULTS_specular_summary.json
---

# Decision: the Clip front is half its advertised size, and it is two rooms

- **Status:** MEASURED · 2026-08-22 · 49a/49b zero GPU, 49c ~50 min on one card
- **Applies when:** costing anything on the `fo_class` front, quoting the `+0.0702`, proposing
  hard-negative or clip-supervision data, or reading rung 19b.

## 🔻 The headroom, priced five ways instead of one

Since 2026-08-10 the campaign has ranked this front on *"78.3 % of `fo_class` errors involve
Clip or Sponge ⇒ +0.0702 headline"*. Rung 36's README already recorded that **the figure had no
artifact** — prose in `context/NOW.md`, over a gitignored `inspect.csv`. Rebuilt on six arms
through `frame.metrics.stratified_report`, rung 19b ep4 (`bucket_mean` 0.6479):

| bound | rows | Δ headline |
|---|---:|---:|
| the error *involves* Clip or Sponge — **the quoted one** | 611 = **90.7 %** of errors | +0.0878 |
| **model named Clip, gold has none — the attractor** | 296 | **+0.0387** |
| only Clip/Sponge membership is wrong | 285 | +0.0397 |
| singleton `{Clip}`↔`{Sponge}` swaps | 110 | +0.0141 |

🔑 **The quoted bound is near-vacuous.** Clip and Sponge are so dominant that 90.7 % of *every*
`fo_class` error touches one, so *"fix the confusion"* ≈ *"fix `fo_class`"*. The honest size is
**+0.0387** — over the 0.03 ship bar and comparable to the +0.0426 gap to rank 1, so the front
survives; it is simply half what it was ranked on. Every bound is an ORACLE and assumes a fix
costs nothing elsewhere, which rung 48 already showed is false.

## 🔴 A sink, not a pair

Singleton golds answered with bare `Clip`: **Needle 49/199 = 24.6 %**, Sponge 64/481 = 13.3 %,
Specimen 20/160 = 12.5 %, Gallstone 4/5. **Naming this "Clip↔Sponge" names the second-worst
victim.** Any lever scoped to the pair is scoped to less than the defect.

It is a **calibration** defect and therefore attackable in principle: `Clip` is emitted 1,216
times against 1,089 golds (ratio **1.117**) with the **worst precision on the board, 0.757**;
`Needle` is the mirror at 0.686 and recall 0.608; every other class sits within ±5 %.
⇒ the older *"the marginal emission is calibrated"* reading does not hold for this arm.

📌 **Epoch 5 does part of the job by itself** — Clip ratio 1.117 → 0.974, precision → 0.810,
Needle recall → 0.707, and ep5 is the better arm (0.6557 vs 0.6479). **Measure any suppression
arm against ep5**, or it will claim the epoch — the mistake [[rung42-gain-was-epochs-not-corpus]]
records one rung earlier. And ep5 shows the price: Clip recall falls 0.845 → 0.789 as precision
rises. **The lever is calibration, not deletion.**

📌 `pred_illegal = 0` on all six arms, both halves ⇒ a recognition defect, not a parsing one.
No answer-canon work applies.

## 🔴 …and half of it is TWO VIDEOS. This is the part that governs the branch.

| arm | Clip FP | videos holding HALF | worst video |
|---|---:|---:|---|
| 19b ep3/ep4/ep5 | 335 / 296 / 202 | 3 / **2** / **2** of 38 | `0024 - Heico - Sigma - 5` |
| 47 ep3/ep4/ep5 | 253 / 268 / 230 | 2 / 3 / 3 of 38 | `0024 - Heico - Sigma - 5` |

**The same video is worst in every arm**, across two corpora and three epochs each: 91 of its
165 clip-free frames draw a Clip (**55.2 %**), and the video scores **0.416** against 0.748
overall. Every video in the top eight is `heico`/**Sigma** — the procedure with zero training
videos. ⇒ it is the Sigma domain shift seen from the emission side, not a new defect.

🔑 **Consequences that bind:**
1. A global Clip-suppression arm pays 36 videos to fix 2.
2. `RULES` §13 clusters on VIDEO and the final ranking is Copeland with pairwise significance.
   An effect carried by 2–3 clusters of 38 has effective n = 2–3. **+0.0387 is real as accuracy
   and fragile as a claim** — the shape rung 47's corpus effect had when it failed its CI.
3. The question worth answering is **not** "how do we suppress Clip" but **"what is different
   about `Sigma-5`"**, which costs nothing to look at.

📌 Also: the FP rate is **4× higher** on questions asking for a *combination* of classes (0.410)
than on ones presupposing a single object (0.108). Being asked for a set is part of the trigger.

## 🔴 External positives do not move it (rung 51c)

Rung 19b ep4 on rung 48's 1,332 provable negatives: `fp_rate` **0.8619** vs rung 47's 0.9024 —
**−0.0405, missing the pre-registered 0.05 line by 0.0095**; −0.0730 on the clean `gap>300`
band, which would clear it. **Both are recorded; the threshold was not moved to the statistic
that passes.** It is moot: **`a2`, with zero CholecT50 rows, reaches −0.0323 / −0.0483** and
lands within **0.008** of the arm built for this. Checkpoint-to-checkpoint variation in clip
aggressiveness is ±0.03–0.06 and swamps the variable — which is what
[[clip-is-inferred-from-phase]] warned from the other side.

📌 **`Clip` recall is 1.0000 on all six models**: a clip is named on 100 % of frames that have
one and 86 % of frames that provably cannot, at precision 0.4499. **Near-unconditional.**

⚠️ 19b tops the `Specimen bag` ruler (0.9043 / 0.9015 vs r42's 0.8958) **and the ruler is
contaminated for it** — it trained on the other 35 CholecT50 videos, so those 15 are *unseen
scene*, not *unseen centre*. Its headline 0.6557 is still under rung 42 ep4's 0.6744.

## 🔴 And the phase mechanism does not transfer at CholecT50's magnitude

The rung-48 ramp construction, run on **our own** eval, gives an FP rate of **0.14–0.25 and
flat**, against 0.85–0.95 climbing on CholecT50. **Underpowered — 190 rows over 16 videos — and
recorded as such, NOT as a refutation.** But the negative-supervision arm was sized against a
rate ~4× the one we are scored on. **Do not fund it on rung 48's number.**

## Closed cheaply: the specular / metallic hypothesis (51b)

Three `Sigma-5` false positives were opened by eye first and suggested brightness — specular
highlights, or the large metallic stapler shaft that a cholecystectomy-heavy training set never
shows. Measured within video on 2,676 frames, 13 videos, exact sign test: **specular mass 5/13
(p = 0.58) and low-saturation 5/13 (p = 0.58) are dead on the coin**; bright-blob count and
mean brightness reach 10/13 at p = 0.092, one correlated signal that does not clear.
⇒ **refuted**, and it closes desaturate/despeckle/highlight-suppression before it is built,
consistent with rung 12's transform bank. ⚠️ The **stapler** observation is untouched by this
probe and stays open — it is better shaped than brightness because it explains the per-video
concentration that brightness cannot.

Related: [[clip-is-inferred-from-phase]] · [[rung42-gain-was-epochs-not-corpus]] ·
[[pooled-screening-manufactures-winners]] · [[secondary-labels-are-fo-class]]
