---
question: Does our local eval predict the challenge judge, and by how much? Which local number may be used to steer?
verdict: Direction transfers in 8 of 8 bucket comparisons; magnitude and absolute level do not. Local `bucket_mean` overstates the judge's score by +0.12, almost all of it in `object_recognition OOD` (+0.367), which is our BEST bucket locally and our WORST on the judge — the ordering is inverted at both extremes. Use the sign, deflate the magnitude, never quote the local OOD level.
status: SETTLED
date: 2026-08-05
basis: zero GPU — the two submissions' official per-bucket scores (the platform's returned payload) against the same two checkpoints' local stratified.json (rung 06 ep2, rung 21 A2 ep3 pulled from the volume over S3)
---

# The local instrument is calibrated on ID and inverted on OOD

## Why this note exists

The campaign has selected 27 rungs' worth of checkpoints against a local eval, on the
assumption that a local gain implies a leaderboard gain. That assumption had **never been
tested**, because until submission 02 was scored there was only one leaderboard point.
There are now two, and they are the only calibration data that will ever exist — the judge
returns per-bucket aggregates and nothing else, and its questions are not accessible in this
phase and are unlikely to become so.

## Where we actually stand (the number the plan lacked)

`pre_evaluation_score` is the unweighted mean of the **four** populated buckets
(`aggregation` × {ID, OOD}, `object_recognition` × {ID, OOD}); the other six are null. That
was already settled in [[leaderboard-metric-is-bucket-mean]]; **re-verified here across all
seven leaderboard entries** (not just ours), reproducing each reported score to 1e-9. New:
bucket sizes recovered from the rationals sum to exactly 2000 — `agg_ID` 553, `obj_ID` 747,
`agg_OOD` **188**, `obj_OOD` 512. ⚠️ `agg_OOD` carries **25% of the score on 188 questions**;
one question there is worth **4×** one in `obj_ID`.

| | rank | score |
|---|---|---|
| Qwen3.6 Finetuned | 1 | 0.5653 |
| QWen 3.5 V1 | 2 | 0.5635 |
| 9blora | 3 | 0.5546 |
| **ours — submission 02** | **11** | **0.5288** |
| **OFFICIAL Fine-tuned Baseline** | **13** | **0.5189** |
| ours — submission 01 | 23 | 0.4767 |
| **OFFICIAL Proprietary Baseline** | **30** | **0.3883** |

🟢 **Both official baselines are beaten** — the campaign's core value, nominally reached.
🔴 **The margin over the fine-tuned baseline is +0.0099**, and we cannot test it: the
significance test that adjudicates it needs the baseline's per-question answers, which only
the organizers hold. An unpaired estimate off the real bucket sizes gives se ≈ 0.0178,
ratio 0.56 against the ~1.96 needed. Paired will be tighter, but by the organizers' own rule
(`challenge_design.txt:1032-1033`, `:1061` — head-to-head win rate with bootstrapping,
merged by Copeland) **a non-significant delta collapses to the same rank**. Treat "we beat
the baseline" as unconfirmed, not as banked.

The proprietary baseline (+0.1406) is settled and does not come back.

## The calibration

Same checkpoint, local eval (6,252 questions, our split) against the judge (2,000, theirs).
Different question sets by construction, so some gap is expected — but not this shape.

| bucket | LOCAL (A2 ep3) | JUDGE (sub 02) | error 02 | error 01 |
|---|---|---|---|---|
| `agg_ID` | 0.4901 | 0.5244 | −0.034 | −0.033 |
| `obj_ID` | 0.7307 | 0.6600 | +0.071 | +0.040 |
| `agg_OOD` | 0.6064 | 0.5266 | +0.080 | +0.050 |
| **`obj_OOD`** | **0.7713** | **0.4043** | 🔴 **+0.367** | 🔴 **+0.302** |

`bucket_mean` local **0.6496** against a real score of **0.5288** — the local headline
overstates by **+0.121**, and `object_recognition OOD` carries +0.367 of it. The error
**grows** between submissions (+0.302 → +0.367): the more we train, the more that bucket lies.

## 🔑 The finding that matters: the ordering is inverted

```
local:  obj_OOD(0.771) > obj_ID(0.731) > agg_OOD(0.606) > agg_ID(0.490)
judge:  obj_ID(0.660)  > agg_OOD(0.527) > agg_ID(0.524) > obj_OOD(0.404)
```

**`obj_OOD` is our best bucket locally and our worst on the judge.** This is not a scale
error — the instrument names as a strength the exact bucket that costs us the competition.
Any prioritisation read off local buckets pointed at the wrong one, which is the cheapest
available explanation for why nineteen rungs of data work never moved the headline and one
learning-rate flag did.

⚠️ Corollary, and it reverses a criticism made the same day: the **ID-only
`proxy_leaderboard`** is the *least* misleading local number we own, precisely because it
excludes OOD. The defect was never the proxy; it is local `bucket_mean`.

## Direction survives — and that is what makes the campaign usable

| bucket | Δ local | Δ judge | ratio |
|---|---|---|---|
| `agg_ID` | +0.0712 | +0.0723 | **0.98×** |
| `obj_ID` | +0.0934 | +0.0629 | 1.48× |
| `obj_OOD` | +0.1271 | +0.0625 | 2.03× |
| `agg_OOD` | +0.0400 | +0.0106 | 3.76× |

**8 of 8 comparisons keep their sign; none reversed.** `agg_ID` is near-exact.
⚠️ **Both submissions were LARGE moves (+0.05 and +0.07 local).** Nothing here says a small
delta transfers — we have never shipped one. The sign evidence does not protect the bottom.

## What this is NOT

🔴 **The OOD split is not the defect.** An earlier reading in this session blamed `heico`=OOD
for not reproducing the organizers' procedure-type OOD. That is false and `RULES §3` already
said so: `heico`=OOD **is the organizers' own partition**, putting Sigmoid Resection — absent
from all training — in test. Redefining the split fixes nothing. Retracted before it was acted on.

🔴 **`kfold_lopo` is not the cheap tool for this.** Its own docstring (`split.py:342`) says it
costs **one training run per fold**. A jackknife over videos re-scoring existing predictions is
the cheap thing, and it is a different thing.

## What survives as explanation

1. **Only 10 OOD videos.** `acc_OOD` — half the challenge score — rests on **10 videos**
   against 28 for ID, and checkpoints are selected by `idxmax(acc_ood)` over those same 10 and
   then reported on a set containing them. No clean holdout exists in any current artifact.
   This was already on record as *selection bias, unmeasured*; it now has 0.30–0.37 of evidence.
2. **Frequency prior** ([[frequency-prior-is-the-failure-shape]]). Against our own locally
   measured trivial floors, the judge puts `agg_OOD` at **−0.018** and `obj_OOD` at **+0.019** —
   i.e. at the constant. Locally the same checkpoint shows +0.062 and +0.386.
   ⚠️ The floors are from our set, not theirs; suggestive, not an identity.
3. **Composition** — our `obj_OOD` holds 2,125 questions, theirs 512.

## How to use the local eval from now on

> **The sign is usable. The magnitude is not. The absolute level, never.**

Deflate local deltas before believing them: ÷1 on `agg_ID`, ÷1.5 on `obj_ID`, ÷2 on
`obj_OOD`, ÷3.8 on `agg_OOD`. ⚠️ Two points, one ratio each — provisional, not a calibration
curve. **Each future submission buys a score AND a calibration point**; with 8 slots left that
is a reason to spend them deliberately.

Feeds the significance rule → [[significance-rule]].

## Operational — how the artifacts were obtained, no pod

The 15 rung-21 `stratified.json` (all five arms × three epochs, ~113 KB total) came off the
volume over the S3 gateway with **zero GPU billed**. ⚠️ **`urllib` is blocked**: the endpoint
sits behind Cloudflare, which rejects its TLS fingerprint with `403 error code: 1010` — for
signed *and* unsigned requests alike, so it reads like a credentials failure and is not.
**`curl` passes.** SigV4 signed with stdlib `hmac`/`hashlib`, transport shelled out to `curl`.
Region `eu-ro-1`, bucket = volume id, `ListObjectsV2` works (`HeadObject` still 403s).
🔑 The two OOD buckets are **absent from `RESULTS_*.csv`**, which stores only `aggregation_ID`
and `object_recognition_ID` while `bucket_mean` silently averages all four — the desglose that
made this note possible exists **only** in `stratified.json`.
