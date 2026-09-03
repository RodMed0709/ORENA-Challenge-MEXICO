---
question: The shipped checkpoint's dominant centre error is naming a class that is not in the frame. Can a second, differently-trained checkpoint delete those classes without a training run?
verdict: YES, AND IT SHIPS AS SUBMISSION 06. Model A answers; where the answer parses entirely as legal class names, model B answers too and the SHORTER list wins. r42 ep4 + ep2: +0.0221 exact-set on the centre probe, 15/15 leave-one-video-out folds, never negative; −0.0041 on rung 42's own held-out `fo_class` and −0.0023 of `bucket_mean` through the canonical eval. Taking the UNION instead is catastrophic (−0.1562) — the direction is the whole lever. 🔴 `bag_f1` and the centre probe ORDER, they do not SCALE: there is no conversion from +0.0221 to a platform delta
status: MEASURED
status_note: SHIPPED as submission 06 on 2026-09-02
date: 2026-09-02
measured_in: experiments/58-self-veto-and-pair/RESULTS_{id_control,58C_headline,58D_pair}.json + experiments/48-centre-probe/runs/*/predictions_v2_full.csv; container in submissions/06-rung42-pair-ep4-ep2/
---

# Decision: two checkpoints, and the shorter class list wins

- **Applies when:** costing any inference-time lever that combines two of our own checkpoints,
  or choosing what a submission slot buys.

## The rule, and why it needs no `answer_format`

Model A answers. Where its answer parses **entirely** as legal class names, model B is asked the
same question and the shorter list wins; ties keep A.

`Request` carries no `answer_format`, so a container **cannot** gate on question type. It does
not need to: `"3"`, `"yes"` and a multiple-choice token do not parse as class names and fall
through untouched. The parse is the scope.

## Measured on both axes, with the function that ships

| arm | centre (15 videos, 4,890) | folds | ID (8 videos, 490 `fo_class`) | folds |
|---|---|---|---|---|
| r42 ep4 — submission 03 | 0.3618 | — | 0.8408 | — |
| **r42 ep4 + ep2** | **0.3838 · +0.0221** | **15/15** | 0.8367 · −0.0041 | 0/8, worst −0.0100 |
| + ep3 + ep5 + a2 (five) | 0.3894 · +0.0276 | 15/15 | 0.8367 · −0.0041 | 1/8 |
| r42 ep4 + a2 | 0.3716 · +0.0098 | 15/15 | 0.8367 · −0.0041 | 1/8 |
| r42 ep4 + ep5 | 0.3634 · +0.0016 | 15/15 | **0.8449 · +0.0041** | **8/8** |
| union of the sets instead of the shorter | 0.2056 · **−0.1562** | 0/15 | — | — |

Canonical eval, all 1,283 held-out questions, judge included:
**`bucket_mean` 0.6722 → 0.6698 (−0.0023)**.

## Two facts that decide how to read that

1. **The rule fired on 509 answers and changed 7.** On the hospital the checkpoints were trained
   on they agree almost everywhere. The arm's value lives where the model is unfamiliar, and
   [[local-eval-vs-judge-calibration]] is precise that our local set cannot see that axis. A flat
   or slightly negative local number is the EXPECTED reading, not a refutation.
2. **The mechanism is not "two heads are better".** It is that the two checkpoints disagree about
   *which* extra class to add, and the shorter answer is right more often. Union is −0.1562 on the
   same data. Anything built on this must preserve the direction.

## The bet, stated before the platform answers

`bucket_mean` is half OOD and the platform's OOD is **centre** (">5 centres not represented"),
which is what CholecT50 stands in for. The arm pays there and costs 0.0023 here. **If the centre
gain does not transfer at all, the cost is that 0.0023** — bounded and measured, which is the
argument for the slot. It does not reach the +0.0226 that the top-10 bar needs.

## Why the pair and not the five

+0.0276 vs +0.0221 is worth 0.0055 of an unscaled ruler. It costs three more merged 8B in the
image (85 GB of weights, ~120 GB image) which does not fit the build box, and it is more surface
for a silent failure — the thing that cost submission 04 twelve hours and a slot.

## What this does NOT establish

The centre probe is **one question template on one external dataset**, effective n = 15 videos.
It orders; it does not scale. `bag_f1`'s own anchors move by 3.6× for the same platform delta.
No expected score is derivable from +0.0221, and computing one would be inventing it.
