---
question: Can majority voting over k sampled answers lift the `number` format?
verdict: NO — negative in all three pre-registered arms on the full 2094. k=16 SIGNIFICANTLY HARMS OOD (−0.043, CI excludes 0) and doubling k doubled the harm — the signature of a mode sitting on the wrong value. On OOD the voted answer falls BELOW the trivial floor
status: MEASURED
date: 2026-07-20
measured_in: experiments/10-self-consistency/RESULTS.csv
---

# Decision: self-consistency is dead on the gap — voting converges on the wrong mode

- **Status:** MEASURED · 2026-07-20 · full population (2094 `number`), paired video-level CIs,
  ~2.6 h GPU. Not "inconclusive" — measured negative.
- **Applies when:** costing any lever that aggregates the model's OWN output distribution —
  voting, k-sampling, reranking by self-agreement, ensembling one model with itself.

## The pre-registered rule, and what it returned

Three arms, one test, no extension after seeing the result. Bar: **≥ +0.02 with the CI excluding
0, in ID AND OOD**. Judged as margin over the template-aware floor.

| arm | k | T | ID | OOD | verdict |
|---|---|---|---|---|---|
| A | 8 | 1.0 | −0.0053 [−0.050, +0.041] | −0.0195 [−0.055, +0.012] | FAITHFUL NEGATIVE |
| B | 8 | 1.3 | −0.0172 [−0.067, +0.035] | −0.0074 [−0.048, +0.026] | FAITHFUL NEGATIVE |
| C | 16 | 1.0 | **+0.0284** [−0.009, +0.070] | **−0.0430 [−0.081, −0.008]** | FAITHFUL NEGATIVE |

Voting loses more often than it wins: `wins_greedy > wins_voted` in five of six arm × distribution
cells. Both scopes (`all` 2094 and `nondegenerate` 2030) return identical verdicts.

## 🔴 Arm C is a trap, and the ID-AND-OOD rule is what caught it

C rises **+0.0284 in ID** — within touching distance of the +0.02 bar, and promotable-looking if
read alone. Its **OOD is −0.0430 with the CI excluding zero**: a statistically significant harm.
Requiring the rise in BOTH distributions is the only thing that stopped a change that measurably
damages generalisation. **Do not relax that conjunction.**

## The mechanism — this is the part that generalises

**Doubling k doubled the OOD harm** (k=8 → −0.0195, k=16 → −0.0430). That is the precise signature
of a distribution whose mode sits on the wrong value: more samples estimate the mode *better*, and
a better estimate of a wrong mode is further from the truth. Sampling is working exactly as
designed; what it converges to is wrong.

**On OOD the voted answer falls below the trivial floor.** Floor 0.4448; greedy margin **+0.0142**
(already barely above it), voted margin **−0.0047** (A) and **−0.0189** (C). The vote is worse than
answering each template's modal answer.

This is the third independent measurement of one fact: [[count-calibration-dead]] (true 2, 3, 4
share modal prediction 1), [[naming-equals-counting]] (asked to name or to count, same
multiplicity), and now this. **Aggregating, re-encoding or re-sampling a distribution that lacks
the information creates none.** The deficit in [[the-gap-is-the-number-format]] is upstream of the
output — it will not be closed by anything that post-processes what the model already emits.

## What this does NOT say

- It does **not** say k-sampling is unaffordable. Measured cost is fine and is recorded in
  [[latency-budget-is-pooled]]: k=8 ≈ 1.15 s/q, k=16 ≈ 1.96 s/q against greedy 0.173 s — sublinear
  in k because the prefill is shared. **The lever died on quality, not on latency.**
- It does **not** condemn sampling for a *different* model whose errors are unbiased. It condemns
  it for THIS model, whose bias is measured.
- The pilot (T0, n=114) read **+0.026 at k=8/T=1.0 with p=0.296**; the same configuration at
  n=2030 returned **−0.0053**. A non-significant pilot signal inverted under power. Recorded as a
  caution: the correct response to an ambiguous screen was the full population, not a larger
  sample chosen after seeing the p-value.
