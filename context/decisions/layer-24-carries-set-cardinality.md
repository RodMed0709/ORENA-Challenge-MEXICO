---
question: Does layer 24 carry the SIZE of the gold `fo_class` set better than the model's own emitted set size — i.e. does rung 34's `number` finding generalise to set cardinality?
verdict: YES, and it is the first positive on this front with a video-clustered CI that excludes zero. Probe 0.8849 vs the model's own 0.8504; cluster-balanced delta +0.0304, 95 % CI [+0.0046, +0.0569] over 10 videos, positive in 7 of 10 and never worse than -0.0286. The learning curve PLATEAUED, so it is not data-limited. But the probe is a DIAGNOSTIC, not a deployable, and the translation from cardinality accuracy to `bucket_mean` is UNMEASURED
status: MEASURED
date: 2026-08-24
measured_in: experiments/55-cardinality-probe/RESULTS_headline.csv
---

# Decision: the count is in the representation for SETS too, and the head loses it

- **Status:** MEASURED · 2026-08-24 · 1,313 enumeration `fo_class` questions over 38 videos,
  fit on 531 ID rows / read on 782 OOD rows, paired video-clustered bootstrap (n_boot 4000).
- **Checkpoint:** A2 ep3 `checkpoint-2703` — **the same one rung 34 probed**, deliberately not
  swapped for a newer arm, so the two results sit on one model.
- **Method:** rung 34's `hidden_probe.py` reused unchanged — last prompt token, layer 24,
  `float16`, StandardScaler → PCA-128 → logistic regression, `C=1e-4` frozen from rung 34.

## The result

| OOD, 782 questions, 10 videos | accuracy |
|---|---:|
| **probe on layer 24** | **0.8849** |
| the model's own emitted set size | 0.8504 |
| majority baseline — always answer "1" (505/782) | 0.6458 |

**Cluster-balanced delta +0.0304, 95 % CI [+0.0046, +0.0569], and it EXCLUDES ZERO.**

🟢 **And it is not one video.** 7 of 10 positive, one exact tie, and the two negatives are
−0.0286 and −0.0137. That distinction is what killed [[clip-attractor-is-two-videos]], whose
whole effect sat in 2 of 38 videos and could not clear a clustered CI. This one is spread.

🟢 **The learning curve PLATEAUED** — OOD 0.8747 / 0.8862 / 0.8887 / 0.8849 at 25/50/75/100 % of
the 531 fit rows. Flat from half the data on. Pre-registered as the control that separates *"the
count is not in the representation"* from *"we could not fit it"*; it says the former is not the
explanation, so a null would have been a real null and this positive is not an artefact of n.

## ⚠️ Two things this result is NOT, stated before anyone quotes it

**1. It is a diagnostic, not a deployable.** The probe predicts the SIZE of the set. Knowing there
are two objects does not tell you they are `Clip` and `Sponge`, and `fo_class` is scored by exact
set equality. Nothing here ships.

**2. +0.0304 in cardinality accuracy is NOT +0.0304 in `bucket_mean`.** Cardinality is necessary,
not sufficient — the classes still have to be named. The translation from "reads the size better"
to "scores better" is **unmeasured**, and could go either way: it is an upper bound on nothing and
a lower bound on nothing. Any downstream arm must re-measure its own headline.

⚠️ Scope: the 10 OOD videos are all `Sigma` (heico), and the OOD cell holds only gold sizes 1 and
2. The probe was fit on ID rows spanning 0/1/2/3+, which is why its train accuracy (0.7420) is
*lower* than its OOD accuracy — a harder problem fit, an easier one read. That is class
composition, not a broken probe, and the probe-vs-model comparison is unaffected because both are
scored on the identical rows.

## Why this matters now

[[enumeration-is-not-fixed-by-output-format]] closed the output side hours before this was
measured: rung 50's arm A learned a count-prefix target *perfectly* and bought +0.0012, arm B's
+0.0126 sat entirely at set size 1. Together with [[number-output-side-is-closed]], every lever
that changes what the head **says** has now failed on both halves of the front.

This is the first evidence on the other end. [[hidden-states-hold-the-count]] found it for
`number` (0.5264 vs 0.4680); rung 55 finds the same shape for **set cardinality**, on the same
model, with a clustered CI. ⇒ **One representation-level deficit governs both formats**, which is
what [[fo-class-and-number-are-one-front]] predicted from the error overlap (odds ratio 2.38,
z = 3.01) and could not otherwise explain.

## What it licenses — exactly one thing

An **auxiliary cardinality objective**: a small head off layer 24 trained jointly with the LM
loss, so the trunk is required to keep what it already carries and the decoder is required to use
it. Single variable against a named control, and it is the only arm on this front whose premise is
now measured rather than assumed.

🔴 **What it does NOT license:** shipping the probe, quoting +0.0304 as a score delta, or reviving
any output-side lever. The output side is closed and this result does not reopen it.
