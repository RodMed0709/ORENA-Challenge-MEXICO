---
question: Does the layer-24 probe advantage that [[hidden-states-hold-the-count]] and [[layer-24-carries-set-cardinality]] measured (probe reads the count/cardinality better than the model's own output, on an ID(lapchole)->OOD(heico) split) hold on the SHIPPED merged checkpoint (rung 42 ep4), at the FULL target grain (exact count / full set identity, not just coarse cardinality), against a genuinely fresh held-out population?
verdict: NO. On rung 42 ep4 merged, neither probe beats the model's own token output on the true 8-video held-out set — counting 0.446 vs 0.461, fo_class full-set-identity 0.814 vs 0.869. Both CV numbers on the 122-video fit pool were markedly higher (0.811, 0.922) and did not transfer; the counting probe's own winning layer moved from 24 to 34. A small aggregation hint (n=20 exact-frame pairs) points the same direction as rung 55's cardinality framing but is far too underpowered to confirm anything alone.
status: MEASURED
date: 2026-08-29
measured_in: experiments/56-enumeration-probe/RESULTS_final_read.json
---

# Decision: the layer-24 edge is checkpoint- and grain-specific, not a general property

- **Status:** MEASURED · 2026-08-29 · 8-video held-out set (rung 42's own declared partition),
  518 `number` rows + 490 `fo_class` rows, video-clustered bootstrap (`frame.metrics._hier_bootstrap`,
  n_boot 1000).
- **Checkpoint:** rung 42 ep4 **merged** — the actually-shipped model, not an unmerged adapter.
  [[hidden-states-hold-the-count]] and [[layer-24-carries-set-cardinality]] both probed earlier/
  different checkpoints (A2-line adapters).
- **Fit pool:** 122 videos (92 `train.parquet` + 30 promoted `test.parquet` videos), 5-fold
  video-grouped CV for layer/hyperparameter selection — NOT the ID(lapchole)/OOD(heico) dataset
  split rung 34 and rung 55 both used. This pool mixes both hospitals on both sides.

## What was asked

[[fo-class-and-number-are-one-front]] and [[enumeration-is-not-fixed-by-output-format]] closed
the output side of the enumeration deficit and pointed at the one surviving branch: the layer-24
probe reads more than the head emits, for `number` ([[hidden-states-hold-the-count]], 0.5264 vs
0.4680) and for `fo_class` cardinality ([[layer-24-carries-set-cardinality]], 0.8849 vs 0.8504,
CI excludes zero). This rung asks whether that edge is a property of the REPRESENTATION at large,
or a property of the specific checkpoint/split/grain it was measured on — by extending the same
family of probe to the model that ships, at the FULL target (not a coarse cardinality bucket),
read against a genuinely fresh 8-video slice rather than a cross-dataset split.

## What came back

| task | CV (122-video fit pool, in-fold) | held-out (8 videos) | held-out token-head | probe wins? |
|---|---:|---:|---:|---|
| counting (`number`, exact value) | 0.8109 (layer **34**, ordinal) | **0.4462** [0.318, 0.585] | 0.4613 [0.343, 0.584] | NO |
| `fo_class` (full set identity) | 0.9224 (layer **24**, C=0.01) | **0.8143** [0.759, 0.871] | 0.8690 [0.807, 0.926] | NO |

Both tasks: the probe does not clear the model's own token output on the held-out set, and both
CIs overlap heavily with n=8 videos — this is not a confident "probe is worse" so much as "the CV
number was not a reliable predictor of held-out performance," which is the more actionable half of
the finding.

## 🔴 The CV → held-out gap is large, and worse for the finer-grained task

Counting's CV accuracy (0.8109) collapsed by **36 points** on the true held-out set (0.4462).
`fo_class`'s dropped by 11 points (0.9224 → 0.8143) — real, but far less severe. Video-grouped CV
protects against frame-level leakage within a video; it does not protect against the 122-video
fit pool simply being an easier, more homogeneous population than genuinely fresh videos.
**Exact-value counting (13-way, ordinal) generalizes worse than coarse cardinality — the finer
the target grain, the more the in-fold number overstates true transfer.**

⚠️ **Even the winning layer moved.** Rung 34/55 both peaked at layer 24. Our counting task's
CV-winning layer is **34**, not 24 — on the SAME architecture, a different checkpoint (merged vs
adapter) and a larger, more diverse fit pool shifted which layer looks best. A probe's own layer
selection is not stable across these axes, which is one more reason not to treat "layer 24" as a
fixed architectural fact rather than a checkpoint-conditioned observation.

## What this is NOT

**Not a refutation of [[hidden-states-hold-the-count]] or [[layer-24-carries-set-cardinality]].**
Those measured a different checkpoint, a coarser target (cardinality bucket, not exact count or
full set), and a different split (cross-dataset, not cross-video-within-mixed-pool). All three
measurements can be true at once: the representation may carry MORE than the head emits in
general, while the specific probe-beats-head margin is small, checkpoint-conditioned, and easy to
lose once the target gets harder or the checkpoint changes.

**Not evidence the fit pool or CV pipeline is broken.** Video-grouped 5-fold CV, PCA fit on the
training fold only, zero leakage confirmed on both task splits (RULES-compliant). The gap is a
genuine generalization property of a 122-video-to-8-video transfer, not a methodology bug.

## A small, honest hint worth flagging, not leaning on

Of the 8 held-out videos, 20 `n_classes`-format questions share an EXACT frame (same video, same
timestamp) with an `fo_class` question. On just those 20: the dedicated counting probe scores
0.50, while `len(fo_class-probe's predicted set)` scores 0.55 and agrees with the counting probe
70% of the time. This points the same direction as [[layer-24-carries-set-cardinality]] — that
set SIZE may be more readable than set IDENTITY or exact count — but n=20 (under one held-out
video's worth of rows) is too small to be more than a note for whoever looks at this next.

## What this licenses — one thing, and one caution

🟢 **Any auxiliary-objective plan built on [[layer-24-carries-set-cardinality]]'s positive result
should re-measure its premise on rung 42 ep4 (or whichever checkpoint ships) before committing
GPU time** — the same probe family did not clear a fresh 8-video bar on the merged model at the
full target grain. The representation-level finding motivating that plan is real on the checkpoint
it was measured on; it is not yet shown to transfer to the checkpoint that would ship it.

🔴 **What it does NOT license:** treating "counting probe loses" or "`fo_class` probe loses" as a
closed, confident negative — n=8 videos with overlapping CIs cannot support that either. The
honest state is "unresolved on the shipped checkpoint," not "refuted."
