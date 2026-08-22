# Rung 48 — the centre probe: does recognition survive a change of hospital?

> **Status: v1 SCORED and its instrument FAILED calibration. v2 RUNNING (2026-08-19, 17:09).**
> Partition frozen (`cholect50_split_v1.csv`, sha256 `098b6c05`, 35 train / 15 hold).
> Probe v1 answered by four models. **The training corpus is built and validated.**

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 42-merged-corpus | promoted 30 of the 38 test videos into training | 21 A2 | shipped — `bucket_mean` 0.6744 on **8 videos** |
| 47-epochs-vs-corpus | `--num_train_epochs 3→5`, A2's own corpus | 21 A2 | done — does not ship |
| 47 re-score (`03_…`) | *nothing trained* — same checkpoints on the **full 6,252** | — | done — the epoch buys **recognition**, not counting |
| **48 (this)** | *nothing trained* — a **second, external** eval axis | — | partition frozen, probe not built |
| 19b | + external recognition data in the corpus | 47 re-scored | **the arm this exists to read** |

## Why this rung exists

The platform payload (2026-08-19) showed our cell-level diagnosis was **inverted**:
`object_recognition_OOD` reads 0.8285 locally and **0.4727** on the leaderboard. Two different
axes wear the same name — ours is **procedure** (Sigmoid), the platform's is **centre**
(">5 centres not represented"). We had no instrument for the axis that scores.

CholecT50 is the only one we have: **Strasbourg, cholecystectomy — our own procedure, another
hospital.** It isolates centre from procedure, which is exactly what the survey's filter 3
demanded and what GraSP and MISAW both fail ([[grasp-is-a-no-go]]).

## What this probe can and cannot do

| | |
|---|---|
| 🟢 measures | **recall** on `Clip` and `Specimen Bag` under a change of centre |
| 🔴 cannot measure | **precision** — CholecT50 never annotates gauze, needles or drains, so a missing label is not an absent object. The metric is **containment** (`gold ⊆ predicted`), and that is the only thing the data supports |
| 🔴 cannot measure | **counting** — `clipper` appears **exactly once** in every one of its 3,429 frames, never twice |
| ⚠️ needs | a **degenerate-answer guard**: a model answering *"clip, bag, gauze, needle"* to everything scores 100 %. Report mean predicted-set size beside recall, and veto any arm that inflates it |

## The label rule, and the adjudication behind it

CholecT50 annotates `clipper` — the **applier**, instrument id 4. Our class is `Clip`, and the
challenge's own definition says a clip counts *"only once placed"*. Thirty frames over 15 videos,
sampled from the start and end of each video's clipping sequence, adjudicated blind by legokna:

```
early frames:   0 yes · 14 no · 1 unsure
late  frames:  12 yes ·  3 no
```

Raw, that reads *"the applier does not imply a clip"* — 12 of 29, **41 %**. By TIME it reads
otherwise, and that is the rule: **`clipper` at or after the first `clipper,clip,*` triplet of
the same video.** Precision **~40 % → ~80 %**, discarding 122 of 3,429 frames.

📌 **The general lesson, which outlives this rung:** to translate someone else's labels into
ours, the unit is the **video**, not the frame. Ask *"has the event happened yet?"*, not *"is the
tool on screen?"*. PhaKIR ships the same `Clip-Applicator` and will need the same treatment.

⚠️ The residual ~20 % gold noise is a constant every arm meets equally: **paired comparisons
survive it, absolute recall does not.** Ordinal, not cardinal — as ever.

## The partition, and why 15 are held

| slice | videos | frames | clip | bag | positives | share |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 35 | 72,092 | 2,368 | 3,350 | 5,718 | 70.0 % |
| **`hold`** | **15** | 28,771 | **939** | **1,506** | **2,445** | **30.0 %** |

Balanced on **positive mass**, not video count alone — the videos carry 78 to 266 positives each,
so 15 drawn at random can be a third light. Per-class drift: clip 0.284, bag 0.310, both inside
tolerance.

🔴 **`hold` measures CENTRE SHIFT only for a model that trained on none of these 50.** Once an arm
trains on `train`, Strasbourg is in its distribution and `hold` measures **unseen scene** instead.
The paired comparison between such an arm and a model that never saw any of it stays valid — what
changes is what the arm's own number means. **Say which one you are reporting.**

## What is NOT here

- **The probe itself.** Question generation, the containment metric and the degenerate guard are
  not built. This rung so far is one frozen CSV.
- **Any score.** No model has answered a CholecT50 question.
- **SurgΣ-DB.** It supplies the *questions* for training on the 35; it is not part of the ruler.


---

# What v1 measured, and why the instrument was rebuilt the same day

Four models answered the 2,445 positive frames of the 15 held-out videos. Three of them carry a
platform score, which is the whole point: **if the probe ranks them the way the leaderboard did,
it has ordinal validity on the axis that scores.**

| model | platform | recall ALL | mean_set_size | Clip | Specimen bag |
|---|---:|---:|---:|---:|---:|
| rung 06 ep2 | **0.4767** | 0.8429 | **1.42** | 0.9989 | 0.7457 |
| A2 ep3 | **0.5288** | 0.8896 | **1.40** | 0.9936 | 0.8247 |
| rung 42 ep4 | **0.5809** | 0.8871 | **1.22** | 0.9968 | 0.8187 |
| rung 47 ep4 | — | 0.8748 | 1.33 | 1.0000 | 0.7968 |

## 🔴 As a recall instrument it FAILED, in two distinct ways

**1. It does not reproduce the platform order.** Platform is r06 < A2 < r42; recall gives
r06 < r42 < A2. But the A2–r42 gap on the probe is **0.0025** where the platform separates them
by **0.052** — this is a non-separation, not an inversion. With 15 clusters and ~20 % gold noise,
0.0025 is zero.

**2. The `Clip` cell is degenerate.** All four models score **0.9936–1.0000**. That is the `Clip`
attractor the campaign already documented, answering `Clip` to everything: on frames whose gold
*is* `Clip`, recall is 1 by construction. The cell measured the bias, not recognition. All of v1's
signal lived in `Specimen bag`.

## 🟢 But its GUARD ordered the anchors 3 of 3

```
platform    0.4767   0.5288   0.5809
set_size      1.42     1.40     1.22
```

Monotone, and with a mechanism: the challenge scores `fo_class` by **exact set equality**, so
listing extra classes is punished. The probe cannot measure precision, but `mean_set_size` is its
direct proxy — and the proxy is what tracked. **The metric added as an anti-gaming guard ordered
better than the metric added as the headline.**

⚠️ Three points have six orderings, so 3/3 is a 1-in-6 coincidence. This is a hypothesis with a
mechanism, not a validated ruler.

📌 It also places rung 47 (`set_size` 1.33) **between A2 and rung 42** — i.e. below what shipped,
which is what the local eval said when we decided not to send it.

# v2 — the fix, and why it was available all along

If `set_size` is what predicts, the probe should be scored on **precision** — which is exactly
what its data appeared not to support. It does, through the same temporal rule read backwards:

| | positives | negatives |
|---|---:|---:|
| `Clip` | 939 (post-event ∧ applier) | **12,281** (before the first clipping event) |
| `Specimen bag` | 1,506 (annotated) | **23,331** (before the bag first appears) |

🔑 **The negatives are the strong side, which is unusual and is the point.** A frame before the
first `clipper,clip,*` triplet cannot contain a placed clip — not because none is visible, but
because none exists yet. Absence is a fact about the timeline, not a judgement about visibility.
The positives are the ~80 %-precision side. Precision, which is what we need, is measured against
the clean one.

**What it fixes:** a `Clip`-attractor model answering `Clip` everywhere scored **1.0000** under
v1. Under v2 those same answers land on frames where no clip exists and become false positives —
validated on a synthetic attractor: **precision 0.2848, F1 0.4433** (an oracle scores 1.0).

Sampled 1:1 per video: **4,890 items** (2,445 pos + 2,445 neg), ~26 min per model.

⚠️ Still unmeasurable: whether `Sponge` on a CholecT50 frame is right. Scoring is **per class** —
"did it say Clip?" — never as a set.
⚠️ Still 15 clusters. No metric fixes that.

---

# v2 — SCORED 2026-08-19. One cell is a ruler; the `Clip` cell is a result.

Four models, 4,890 items, 15 videos. `_tools/probe_v2_report.py` → `RESULTS_probe_v2.csv`.

| model | platform | **bag F1** | clip F1 | macro F1 | set_size |
|---|---:|---:|---:|---:|---:|
| rung 06 ep2 | 0.4767 | **0.8453** | 0.5929 | 0.7191 | 1.2207 |
| A2 ep3 | 0.5288 | **0.8849** | 0.6156 | 0.7503 | 1.2188 |
| rung 42 ep4 | 0.5809 | **0.8958** | 0.6048 | 0.7503 | 1.1202 |
| rung 47 ep4 | — | 0.8785 | 0.6097 | 0.7441 | 1.1812 |

**Leave-one-video-out, 15 folds** (`RESULTS_probe_v2_jackknife.csv`) — the question is not what a
cell scores but whether it reproduces `r06 < A2 < r42`, and whether it does so on every fold:

```
bag_f1     orders 3/3 in 15/15 folds   r47 below r42 in 15/15
set_size   orders 3/3 in  9/15
macro_f1   orders 3/3 in  7/15         <- a coin flip
clip_f1    orders 3/3 in  0/15         <- it does not fail, it INVERTS
```

## 🟢 The ruler is `Specimen bag` F1, alone

Margins 0.040 and 0.011, never crossing on any fold. 📌 And it puts **rung 47 below rung 42 in
15/15**, a third independent agreement with the local eval and with v1's `set_size`.

## 🔴 The macro is not a ruler, and it is the number that looks like one

A2 0.75028 vs rung 42 0.75030 — a **1.4e-5** tie where the platform separates them by **0.052**.
Averaging a cell that orders with a cell that inverts reproduces exactly the error the rung-47
CI made at the corpus level: *a real effect diluted to nothing by the company it is averaged
with.* **Report the cell, never the macro.**

## 🔴 `set_size` lost its v1 margin

1.2207 / 1.2188 / 1.1202 — the r06–A2 gap is **0.0019**, and it orders on only 9 of 15 folds.
v1's 3/3 was on the same near-tie (1.42 / 1.40) and should have been read as one usable point,
not three. ⚠️ Its ordering test also runs the **other way** (lower is better); scored ascending
like an F1 it reports 0/15 for a cell that is in fact ordering. Direction is per-metric.

## The `Clip` cell: not an instrument, but the best result of the rung

`_tools/clip_fp_anatomy.py` → `RESULTS_clip_fp_anatomy.csv`, `RESULTS_clip_fp_ramp.csv`.

On the 1,332 frames where the clip provably does not exist yet, all four models answer the bare
string `'Clip'` in **85–95 %** of them. The rate is **not flat** — it climbs with proximity to
the clipping event (rung 42, and the other three have the same shape):

```
first decile of the negative window  0.567      gap > 300 frames   0.870  (n=808)
half way                             0.977      gap 100-300        0.980  (n=358)
last decile                          1.000      gap < 100          1.000  (n=166)
```

⇒ **the models are reading the surgical PHASE and inferring a clip that has not been placed.**
Not a constant prior — a ramp, with a mechanism. It is consistent with step 4
([[vcd-has-nothing-to-subtract]]): the model *does* look; the phase prior overrides what it sees.

⚠️ The top of the ramp is where the label is weakest (a clip deposited just before the first
annotated triplet), so that band is not interpretable. **`gap > 300` is 61 % of the negatives,
its label is not in doubt, and it still runs 0.81–0.95.** The finding survives without it.

🔴 **And this is why it stays a result.** Clip aggressiveness does not track the platform: A2 is
the *least* aggressive of the three anchors and rung 42 beats it by 0.052. Every temporal cut
inverts the pair. Anyone proposing "train on hard negatives so it stops over-calling `Clip`"
must first show that fewer clip FPs buy score — the three anchors we have say the opposite.

## What this licenses

🟢 **Read arms on `bag_f1`**, with the fold table beside it.
🔴 **Do not report `macro_f1`**, and do not rescue the `Clip` cell with a v3 — its problem is
the models, not the corpus.
📌 Rung 19b is the arm this exists to read: `experiments/19b-external-recognition/`.
