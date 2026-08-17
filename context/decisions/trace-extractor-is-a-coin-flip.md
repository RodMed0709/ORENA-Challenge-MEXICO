---
question: Can a second model read the gold answer out of the first model's reasoning trace — the "double model communicating by the trace" design?
verdict: NO — the "83 % of failed traces contain the gold" figure is an ORACLE RECALL with no denominator. On the same archived traces a median of 3 (strict) to 4 (loose) distinct plausible numbers coexist, the two trivial extractors score 0.1786 and 0.2233 against a 0.2189 random-pick control — i.e. exactly 1/k — and they BREAK 24.2 %/37.7 % of the questions the model already answered correctly. The extraction that reproduces the 83 % is the one with the LARGEST candidate set: recall and ambiguity are the same number seen from two sides.
status: MEASURED
date: 2026-08-17
measured_in: experiments/43-thinking-at-inference/RESULTS_trace_extractor_audit.json
question_derived: false
---

# The trace contains the answer and three of its rivals

## Why this was measured before anything was built

The proposal was a two-model pipeline — a 27B producing a reasoning trace, an 8B reading the
answer out of it — justified by *"the correct number appears in the trace of 83 % of the
failed `number` questions, against a 27.5 % chance control"*. An adversarial review objected
that **that is a recall, not an accuracy**: it counts traces where the gold is present and
never counts what it is present *alongside*. If several plausible numbers coexist, an
extractor choosing among them is a coin flip regardless of how high the recall is.

The objection was pre-registered with its own kill rule, written before the numbers were
looked at:

> median ≥ 2 candidates **and** trivial rules near 1/k ⇒ **DEAD BRANCH, close it**
> 1 dominant candidate **and** trivial rules high ⇒ proceed to build the pipeline

Both legs of the first row fired. **The branch closes.**

Cost: **zero GPU**, ~30 min, on already-archived generations. Nothing was built.

## The instrument

Rung 43's thinking arm was never reported as a verdict, but it left 4,000 real generations
on UNAM at `~/storage/rung43/r43_ep23_think_uncapped/predictions.json` — the ep2/3 27B
connector checkpoint, uncapped budget, judged by the canonical harness. Restricted to
`answer_format == number`: **1,326 questions, 963 failed, 363 correct.**

📌 That population is **heico test only** — Sigmoid Resection, the challenge's OOD procedure
(`RULES §3`). It is one distribution, not the headline's four cells, and this note claims
nothing about ID.

All arithmetic ran **on UNAM**; only aggregates were brought back (`RULES §14`, DUA).
Reproduce with `experiments/43-thinking-at-inference/_tools/trace_extractor_audit.py`.

## (a) How many numbers are in there with it

| | median | p75 | p90 | max |
|---|---|---|---|---|
| distinct plausible values, **failed** traces | **3** | 4 | 6 | 12 |
| distinct plausible values, **correct** traces | 1 | 2 | 3 | 8 |
| numeric tokens (not distinct), failed | 21 | 59 | 86 | 211 |

Only **82 of 963** failed traces contain exactly one candidate. **45 contain none at all.**

🔑 **And the rivals are not far away, they are adjacent: in 83.0 % of failed traces a
neighbour of the gold (`gold ± 1`) is also present.** That is the same population as
[[counting-has-two-failure-modes]] — 65.6 % of all counting errors are off by exactly one —
seen inside the trace. The extractor's job is therefore not "find the number in the text",
it is "decide between 2 and 3", which is the original task with the picture thrown away.

## (b) The trivial extractors are the chance control

Measured on the **same 963 failures** the 83 % was computed on:

| rule | accuracy |
|---|---|
| last number before `</think>` | **0.1786** |
| most frequent number | **0.2233** |
| **uniform random pick among the candidates** | **0.2189** |

Both rules land on the random-pick line. The mean candidate count is 3.29 ⇒ 1/k ≈ 0.30, and
the rules do not beat it. There is no signal in *position* and none in *frequency*.

⚠️ This does not prove no extractor can exist — it proves the cheap ones are chance, and that
a learned one would have to beat a 22 % baseline on a set where the right answer and its
off-by-one twin are both present in 83 % of cases. That is the counting problem again.

## (c) The half nobody measured: what it costs on the questions that were RIGHT

The 83 % was computed on failures only. Applying the same rules to the **363 questions the
thinking arm already answered correctly**:

| rule | breaks a correct answer |
|---|---|
| last number before `</think>` | 88/363 = **24.2 %** |
| most frequent number | 137/363 = **37.7 %** |

`RULES §S8` gives **any** cell a veto. This one vetoes.

## The arithmetic that closes it

| pipeline, all 1,326 `number` questions | accuracy |
|---|---|
| last-number extractor over traces | 0.3371 |
| most-frequent extractor over traces | 0.3326 |
| thinking arm as it stands | 0.2738 |
| **no-thinking control (what we would actually ship)** | **0.4811** |
| oracle extractor (an extractor that already knows the gold) | 0.7255 |

The extractor gains **+0.06** against the arm that was already losing and loses **−0.144**
against the arm we would ship. And the loose-extraction ceiling — every digit ≤ 12 anywhere
in the generation — is 0.8967, reachable only by something that knows the answer.

## 🔑 Where the 83 % came from, exactly

It reproduces, and the reproduction is the finding:

| extraction | oracle recall on failures | median distinct candidates |
|---|---|---|
| strict (pre-`</think>`, standalone digits + number words) | 0.6781 | 3 |
| **loose (any digit ≤ 12, whole generation)** | **0.8577** | **4** |

**The looser you read, the higher the recall and the more numbers you must choose between.**
The 83 % and the ~25 % coin flip are not two findings in tension; they are one quantity
reported without its denominator. Any future claim of the form *"the information is in
there"* owes this table: recall **and** the size of the set it was recalled from.

## What this does and does not close

- 🔴 **Closes** the two-model trace-extraction design. Do not build it.
- 🔴 It is the **fourth** independent measurement of one mechanism, and the strongest form of
  it yet: [[self-consistency-dead]] (voting converges on the wrong mode), [[count-calibration-dead]]
  (2, 3 and 4 share modal prediction 1), [[naming-equals-counting]], and now this.
  *Aggregating, re-encoding or re-sampling a distribution that lacks the information creates
  none* — and rung 30 already spent the strongest version, returning −0.0094.
- 🟢 **Does not touch** the hidden-state probe (`experiments/34-hidden-state-probe/`, which
  beats the model's own output 0.5264 vs 0.4680 at layer 24). That probe reads the internal
  representation, not its emitted text; a text trace is a lossy function of the same state and
  this result is consistent with the count being present internally and destroyed on the way
  out. The distinction is the whole reason the probe is still the live counting branch.
- ⚠️ **Scope:** one checkpoint (27B conn4e5 ep2/3), one format, one distribution (heico/OOD),
  greedy, uncapped budget. It does not say a *fine-tuned-to-reason* model would behave this
  way — rung 43's own pre-registered confound stands (901 steps trained with 0 % reasoning
  examples). It says the traces we can actually produce today do not carry an extractable
  answer.
