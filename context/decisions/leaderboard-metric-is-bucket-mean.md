---
question: What does the platform actually score — and is the metric we have been optimising against the right one?
verdict: THE HEADLINE IS THE UNWEIGHTED MEAN OVER ALL TEN BUCKETS, ID **AND** OOD — which is exactly our `bucket_mean`. The "ID-only mean over 2 populated buckets" reading is REFUTED by the platform's own documented payload, whose four populated buckets reproduce `pre_evaluation_score` exactly to 1e-15. RULES 4b is retracted. 🟢 No conclusion in the campaign changes ranking — rung 21's five arms rank identically under both instruments — but three readings built on the wrong comparator are withdrawn
status: MEASURED
date: 2026-07-30
measured_in: the full metrics payload the platform returned for submission 01, including its own `_docs` and `budget` blocks
---

# Decision: the leaderboard metric is `bucket_mean`, and it always was

- **Status:** MEASURED · 2026-07-30 · zero GPU, read from the platform's own payload.
- ⬆️ **Supersedes the central claim of [[leaderboard-metric-vs-our-headline]].** That note's
  batch-size arithmetic and its `aggregation`-is-the-gap trail survive; its metric definition
  does not.

## What the platform actually returns

The complete payload — the earlier reading was of a **partial** report that predated the
platform's own `_docs` and `budget` blocks.

```json
{"budget": {"batches": 100, "questions": 2000,
            "setup_allowance_s": 120.0, "latency_per_question_s": 5.0,
            "saturation_fraction": 0.2},
 "pre_evaluation_score":           0.47672219350591,
 "accuracy_aggregation_id":        0.45207956600361665,
 "accuracy_aggregation_ood":       0.5159574468085106,
 "accuracy_object_recognition_id": 0.5970548862115127,
 "accuracy_object_recognition_ood":0.341796875,
 "accuracy_temporal_grounding_*":  null,
 "accuracy_event_understanding_*": null,
 "accuracy_complex_reasoning_*":   null,
 "questions_forfeited": 0, "questions_unanswered": 0,
 "mean_batch_duration_s": 17.56287337,
 "mean_latency_per_question_s": 0.0,
 "throughput_questions_per_s": 1.1387658259930846}
```

Its own `_docs` defines the headline:

> *"Unweighted mean accuracy over the **10 buckets** (5 capability groups **× in-/out-of-distribution**),
> so every bucket counts equally regardless of how many questions it holds."*

**Check:** `(0.45207956600361665 + 0.5159574468085106 + 0.5970548862115127 + 0.341796875) / 4
= 0.47672219350591` — exact to 1e-15.

⇒ **The metric is the mean over POPULATED buckets, ID and OOD alike. That is `bucket_mean`.**

## 🔴 What is retracted

[[leaderboard-metric-vs-our-headline]] read a partial payload in which everything was reported
under two ID keys, and concluded the headline was ID-only. Consequences now withdrawn:

| withdrawn | replacement |
|---|---|
| "the right local comparator is mean-ID (0.5281)" | **`bucket_mean` (0.5667 for rung 06 ep2)** |
| "the local↔platform offset is −0.0571" | **−0.090** (0.5667 → 0.4767) |
| "`object_recognition` collapsed −0.150 on the platform" | **−0.040** (local 0.6373 → platform 0.5971) |
| RULES §4b ("score the ID-only proxy") | **retracted — score `bucket_mean`** |

⚠️ **[[wrong-judge-and-object-recognition-collapse]] needs re-reading.** It concluded the −0.150
collapse was real capability rather than a broken instrument, and that conclusion licensed the
backbone-generation branch. The collapse it was explaining is **−0.040**, not −0.150.

## The same 2000 questions, regrouped

Both payloads' accuracies are exact rationals, and both sets of denominators sum to 2000:

| bucket | old report | new report |
|---|---|---|
| `aggregation_ID` | 343/754 | **250/553** |
| `aggregation_OOD` | — | **97/188** |
| `object_recognition_ID` | 607/1246 | **446/747** |
| `object_recognition_OOD` | — | **175/512** |
| **questions** | 2000 | 2000 |
| **correct** | 950 | **968** |

So it is one batch, reported twice. Between the two reports **13 questions moved from
`aggregation` to `object_recognition`**, and **18 more answers count as correct**. The platform
changed its grouping and its scoring, not only its field names (`bucket_*` → `accuracy_*`).

⚠️ Consequence: **the 4B competitor's 0.5163 came from the OLD report and is no longer
comparable.** Where the field sits under the current scheme is unknown until their row is seen
again. Every "a 4B beats us by 4.5 points" statement is suspended, not refuted.

## 🟢 What does NOT change — and one thing that gets better

**No ranking in the campaign moves.** Rung 21's five arms, epoch 3, ranked under both
instruments:

| arm | proxy (wrong instrument) | `bucket_mean` (right one) | rank |
|---|---|---|---|
| `A2_lr` | 0.6104 | **0.6496** | 1 = 1 |
| `B_rank` | 0.6095 | 0.6478 | 2 = 2 |
| `D_clip` | 0.6053 | 0.6463 | 3 = 3 |
| `A_lr` | 0.5901 | 0.6305 | 4 = 4 |
| `A3_vitlr` | 0.5825 | 0.6185 | 5 = 5 |
| rung 18 | 0.5421 | 0.5721 | 6 = 6 |

**Identical, position for position.** The optimiser was reading a correlated proxy, not a
misleading one — every arm's verdict stands.

🟢 **And the audit's strongest objection to A2 inverts.** It held that A2's significance lives
in `ALL` and `OOD`, "two of three cells the proxy does not score". The metric **does** score
OOD, so A2's significant cells are exactly the ones that count. A2 is better supported now,
not worse.

## 🟢 The budget, settled with the real instrument

```
allowed per batch = 120 s setup + 20 questions × 5 s = 220 s
used              = 17.56 s          →  12.5× margin
mean_latency_per_question_s = 0.0    →  the whole job fit inside the SETUP allowance
questions_forfeited = 0   ·   questions_unanswered = 0
```

Confirms [[latency-budget-is-pooled]] with the platform's own numbers. ⇒ **Levers closed on
cost are open again**: self-consistency / k-voting, higher `max_pixels`, multi-crop.

⚠️ New rule, from the template and confirmed by `budget.saturation_fraction`: **overrunning by
20% forfeits the ENTIRE batch**; smaller overruns forfeit questions *proportionally*, selected
deterministically and stratified across buckets — not the individually slow ones. A single slow
question does not cost itself; it costs a share of the batch.

## ⚠️ The `null` buckets are correct, not an error

FRAME's ground truth holds no `temporal_grounding`, `event_understanding` or
`complex_reasoning` questions — those belong to the SEGMENT and PROCEDURE tracks, which get
video. `_docs`: *"null if the ground truth holds no such question."* And null buckets are
**excluded** from the mean rather than counted as zero, which is why four buckets divide by
four. Consistent with [[unused-metadata]]'s finding that `primary_capability` only ever takes
six values across all 20,000 public questions.

## What this costs us, honestly

Nineteen rungs were read against `bucket_mean`, then RULES §4b switched the headline to the
ID-only proxy, and rung 21 was designed and reported against that proxy. **The proxy was a
correlated instrument and no verdict flipped** — but three quantitative readings were wrong for
five days, and the largest of them (`object_recognition` −0.150) is load-bearing for a branch
that is still live.

**The lesson is not "read the payload more carefully".** It is that a metric definition
recovered by *inference* from a partial payload was written into RULES as if measured. The
`_docs` block that settles it was in the platform's own response all along.

## Sources

- The full submission-01 metrics payload (`_docs` + `budget` blocks)
- `experiments/21-recipe-sweep/RESULTS_*.csv` — the re-ranking above
- [[leaderboard-metric-vs-our-headline]] (superseded on the metric) ·
  [[wrong-judge-and-object-recognition-collapse]] (needs re-reading) ·
  [[latency-budget-is-pooled]] · [[recipe-axis-is-the-learning-rate]] ·
  [[submission-01-rung06]] · [[aggregation-is-the-gap]] · [[unused-metadata]]
