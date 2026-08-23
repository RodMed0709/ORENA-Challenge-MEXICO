---
question: Inside `fo_class`, which error class is the biggest and most robust lever — and is `fo_class` a separate front from `number` at all?
verdict: The biggest lever is NOT the Clip attractor but the MULTI-CLASS GOLD. It is worth more on all six arms (+0.0421 vs +0.0387 on rung 19b ep4) from fewer rows, and it is 3x less concentrated (32 of 38 videos, half across six, against 29 videos with half in two), so unlike the attractor it can survive a video-clustered CI. And it is not an `fo_class` problem: on the 364 frames carrying both question types, `fo_class` and `number` fail TOGETHER (odds ratio 2.38, z=3.01 against a within-video null), and scene multiplicity degrades both — direction replicates 6/6 arms, significance 5/6. They are one enumeration deficit in two output formats, 4,769 of 6,252 rows
status: MEASURED
date: 2026-08-22
measured_in: experiments/51-clip-attractor/ — RESULTS_fo_class_headroom.json (multiclass_gold bound) · RESULTS_clip_fp_by_video.csv · README.md §51d, §51e (six-arm replication)
---

# Decision: the lever is set enumeration, and it pays in two formats

- **Status:** MEASURED · 2026-08-22 · zero GPU, six arms, the eval already on disk
- **Applies when:** choosing what to attack inside `fo_class`, costing a `number` lever, or
  budgeting the two as separate fronts. They are not separate.

## 1. Exact-set scoring collapses on conjunctions

`fo_class` accuracy against the number of classes in the gold (rung 19b ep4, 38 videos):

| classes in gold | n | accuracy |
|---:|---:|---:|
| 1 | 2065 | **0.801** |
| 2 | 547 | 0.616 |
| 3 | 57 | **0.175** |
| 4 | 6 | **0.000** |

Across videos, accuracy correlates **−0.694** with the mean gold set size. Most of what makes a
video hard is how often its gold names more than one thing.

## 2. It beats the Clip attractor on size AND on robustness

| | Δ headline (19b ep4) | rows | videos touched | videos holding HALF |
|---|---:|---:|---:|---:|
| Clip attractor | +0.0387 | 296 | 29 / 38 | **2** |
| **multi-class gold** | **+0.0421** | **263** | **32 / 38** | **6** |

More headline from fewer rows, on **six of six arms**. And 3× less concentrated: present in both
`lapchole` and `heico`, so not a domain-shift artefact the way the attractor is
([[clip-attractor-is-two-videos]]). Under `RULES` §13's video clustering, six clusters is a
claim that can survive; two is the shape rung 47's corpus effect died in.

⇒ **The `fo_class` front was right and the target inside it was wrong.**

## 3. And it is not an `fo_class` problem — the two formats fail on the SAME FRAMES

364 frames carry both an `fo_class` and a `number` question:

```
P(number wrong | fo_class wrong) = 0.687      odds ratio 2.384
P(number wrong | fo_class right) = 0.479      z = 3.01 vs a WITHIN-VIDEO null
```

The null permutes `number` correctness inside each video, so video-level difficulty is
controlled. And the gradient is the mechanism rather than the correlation — by the number of
classes in the frame's `fo_class` gold, a property of the **scene** and not of either question,
both accuracies fall together:

| classes in gold | frames | `fo_class` | `number` |
|---:|---:|---:|---:|
| 1 | 263 | 0.852 | 0.580 |
| 2 | 88 | 0.578 | 0.449 |
| 3 | 11 | 0.091 | 0.182 |

**Replication, six arms:** odds ratio above 1 in **6/6** (1.60–3.48) and `number` easier on
single-class frames in **6/6** (gap 0.11–0.27). Significance **5/6** — `19b ep3` reaches z=1.05
and is reported as not clearing; it is also the weakest arm on the headline.

⇒ 🎯 **This is [[naming-equals-counting]] with a per-frame test behind it.** `fo_class` (2,675)
and `number` (2,094) are **not two fronts with two budgets** — they are one enumeration deficit
wearing two output formats, over **4,769 of 6,252** rows. The `fo_class` curve against gold size
(0.801 → 0.616 → 0.175 → 0.000) has the shape of the `number` curve against gold value that
[[number-is-an-annotation-ceiling]] re-measured on three models (0.886 → 0.507 → 0.200 → 0.161).

🔑 **A set-enumeration lever is the only one identified that pays in both.** Every lever costed
so far bought one format and was priced against one bucket. And it points where
rung 34's hidden-state probe already pointed: **before the output format splits** — the
count is present at layer 24 and lost by the head.


## 🟢 Independent corroboration, from a probe aimed somewhere else entirely

[[flip-equivariance-holds-small-residual-cost]] (Yingyu, 2026-08-21) asked whether the shipped
checkpoint tracks object POSITION under a horizontal flip. It is a spatial question on a
different eval set, and the answer to it is "mostly yes". But its failure analysis is about us:

> *"The other 8 changed, mostly by drifting in object COUNT or CLASS, not by getting the
> left/right axis wrong — e.g. `heico__2244558` (`"1. Sponge: bottom/left"` → `"1. Clip:
> bottom/left 2. Clip: bottom/left"`, a new class AND a duplicated item)."*

One row, both defects: a **Sponge → Clip** substitution ([[clip-attractor-is-two-videos]]) and a
**1 → 2 item inflation** (this note). ⇒ when a perturbation breaks this model's answer, what
breaks is **which things and how many**, not where they are — measured by someone who was not
looking for it, on a population this note never touched.

⚠️ It is corroboration, not a second measurement: n=8 changed rows, and her own rung retracted a
stronger enumeration claim over it (Fisher p = 0.088) before publishing. Cite the direction, not
a magnitude.

## What this does NOT establish

⚠️ **A shared cause is not proof the cause is multiplicity.** A cluttered, bloody or badly lit
frame is harder for both questions, and a within-video null removes the video, not the frame.
The gradient makes multiplicity the more parsimonious reading — gold set size is *itself* a
count of the scene — but frame difficulty is not excluded.

⚠️ **n = 364 frames**, and only 13 carry a gold of 3+ classes. Read the direction, not the
endpoints.

⚠️ **+0.0421 is an ORACLE.** It assumes fixing conjunctions costs nothing on the 2,065
single-class rows, which currently score 0.801.

⚠️ It names **no lever**. It says where one would have to act, and rules out the two that were
being proposed: a Clip-vs-Sponge intervention and external clip supervision
([[clip-attractor-is-two-videos]]).

Related: [[clip-attractor-is-two-videos]] · [[naming-equals-counting]] ·
[[number-is-an-annotation-ceiling]] ·
[[counting-is-a-mapping-failure]]
