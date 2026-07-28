---
question: What does the leaderboard actually score, and where do we really stand?
verdict: `pre_evaluation_score` is a 2-bucket ID-only mean, NOT our `bucket_mean`; on identical questions a 4B beats us by 4.5 pts and the whole margin is `aggregation`
status: MEASURED
date: 2026-07-27
measured_in: submission 01's full metrics payload + external_data parquets + challenge_design.txt
question_derived: true
---
# Finding: we were reading a different number than the one that ranks us

- **Status:** MEASURED, arithmetic exact to 1e-17 · 2026-07-27
- **Applies when:** quoting any local number as "our score", choosing a checkpoint to submit, or
  comparing ourselves to anyone on the leaderboard.
- **Source:** the metrics payload the platform returned for submission 01 (rung 06 ep2).

## The payload

```json
{"pre_evaluation_score": 0.47103303515546835,
 "bucket_aggregation_id": 0.45490716180371354,
 "bucket_object_recognition_id": 0.4871589085072231,
 "bucket_*_ood": null, "bucket_temporal_grounding_id": null,
 "bucket_event_understanding_*": null, "bucket_complex_reasoning_*": null,
 "timed_out": 0, "no_response": 0,
 "mean_latency_s": 15.847790869999997, "throughput_samples_per_s": 1.2620055479063754,
 "mean_amortized_latency_s": null}
```

## 1. The metric is an unweighted mean over POPULATED buckets — and only two populate

`(0.45490716180371354 + 0.4871589085072231) / 2 = 0.47103303515546835` — **exact to 1e-17**.

So the pre-eval headline is **ID-only, two buckets**. No OOD, no `temporal_grounding`, no
`event_understanding`, no `complex_reasoning`. ⚠️ `null` means *empty bucket*, not *error*.

🔴 **Therefore our `bucket_mean` is NOT comparable to it.** `bucket_mean` averages **four**
buckets (ID+OOD). The right local comparator is **mean-ID**. For rung 06 ep2 that is **0.5281**,
not 0.5667 — so the real local↔leaderboard gap is **−0.057**, not the −0.096 a naive comparison
gives. See RULES §4b.

## 2. The batch size is recoverable exactly, and it proves the sets are identical

Both accuracies are exact rationals, and they share denominators with the 4B participant's row
recorded in [[aggregation-is-the-gap]]:

| | `aggregation_id` | `object_recognition_id` |
|---|---|---|
| **us** | **343/754** | **607/1246** |
| **4B** | 410/754 (= 205/377) | 609/1246 (= 87/178) |

**B = 754 + 1246 = 2000 questions over 20 videos** (100/video — consistent with the challenge's
"20 representative videos"). The shared denominators are the important part: **both submissions
were scored on the identical question set.** Every comparison below is therefore apples-to-apples.

## 3. 🔴 A 4B model beats us, and the entire margin is `aggregation`

| | ours | 4B | delta |
|---|---|---|---|
| `aggregation_id` | 343/754 = 0.4549 | 410/754 = **0.5438** | **−67 questions** |
| `object_recognition_id` | 607/1246 = 0.4872 | 609/1246 = 0.4888 | −2 questions (**tie**) |
| headline | **0.4710** | **0.5163** | **−0.045** |

We are **4.5 points behind a model less than half our size**, and it is **not** spread across the
exam — it is 67 questions in one bucket. `object_recognition` is a dead tie.

⚠️ **This falsifies the "+14.9 lead on `object_recognition`" in [[aggregation-is-the-gap]]**, which
compared our *local val* against their *platform* score. Retracted there.

⚠️ It also means **our local val overstates `object_recognition`** (0.6373 local vs 0.4872 on the
pre-eval set). Part of that is a harder question set — the 4B scores ~0.487 there too, so it is not
a defect unique to us — and part may be the selection bias in §6.

## 4. Latency is not a risk, and the alarm it caused was arithmetic

`15.847790869999997 × 1.2620055479063754 = 20.0` **exactly**. The 4B row gives `20 / 1.0843 =
18.4451 → "18.44"`. Two teams, two models, same product: the two fields carry **one** number, not
two, scaled by the 20-video case count.

The honest figure is `1/throughput = ` **0.79 s/question**, against a pooled ceiling of
`5 + 120/B =` **5.06 s/question** — **6.4× headroom** ([[latency-budget-is-pooled]]).

Corroborated independently: running this exact container on the organizers' fixture measured
**0.78 s/question warm** and **31.9 s setup** against a 120 s allowance.

Also ruled out: a CPU fallback. `timed_out: 0` with a non-zero score means every self-reported
latency was ≤ 5.0 s (the SDK gate at `evaluator.py:218`), and bf16 on CPU did not finish one
question in 35 minutes.

## 5. The FINAL ranking is not this metric

`challenge_design.txt:1010-1040`: the final ranking is **Copeland over the buckets with pairwise
significance tests** — non-significant deltas get the *same* rank — and *"we give equal weight to
ID and OOD questions"* (`:2001`). Mean accuracy applies **only** to *"outperform a baseline model
during the pre-evaluation phase"* (`:1039`).

⇒ Two consequences. **(a)** OOD is unscored today but worth half the final score, so work aimed at
robustness is not wasted even when the pre-eval cannot see it. **(b)** Chasing a +0.003 mean
improvement is worthless under a ranking that collapses non-significant deltas to the same rank.

## 6. Two defects this exposed, neither fixed here

🔴 **Selection bias.** `split.py:165-194` maps the organizers' `test.parquet` directly into
`val_id` (28 videos) + `val_ood` (10 videos) — their union IS the whole 6252-question set. Every
rung then selects its checkpoint by `idxmax(acc_ood)` over `val_ood` and reports the headline on a
set **containing those same questions**. Argmax over noisy estimates on a **10-video** pool biases
upward, always, and nobody has bounded it. Mitigating: every rung uses the identical procedure, so
rung-vs-rung *deltas* largely cancel it; what it costs is the credibility of *absolute* numbers.
The fix already exists unused — `kfold_lopo` (`split.py:342`).

🔴 **Two capability groups have zero training data.** Across all 20,000 public questions the
`primary_capability` values are `1a, 1c, 1d, 1e, 2a, 3a` only — **no 4x (`event_understanding`),
no 5x (`complex_reasoning`)**. The SDK defines five groups and the platform reports ten buckets.
If the hidden test populates them, a majority of the final headline comes from capabilities we
have never trained on and cannot measure locally.

## What this does NOT say

- It does **not** say submission 01 was broken. An adversarial audit cleared it: frames arrived,
  the prompt was byte-identical, generation matched, offline held, the Dockerfile is
  byte-identical to the template's. The gap is the model and the question set, not a bug.
- It does **not** identify where the baselines sit. `challenge_design.txt:453` gates the final
  stage on beating **both baselines**, and `:375` says they would be *"clearly identified"* on the
  leaderboard — but the visible leaderboard shows 13 rows, all participant teams. **We do not know
  the bar we must clear.** Worth asking the organizers.

**Sources.** Submission 01's metrics payload (user-supplied, leaderboard UI) ·
`external_data/orena-data/*/data/frame/*.parquet` · `context/challenge/challenge_design.txt`
:453, :1010-1040, :2001 · `src/frame/split.py:165-194,342` ·
`vendor/orena-focus/src/focus/evaluation/evaluator.py:218,437-461`.
