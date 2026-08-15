# Rung 43 — thinking at INFERENCE · PRE-REGISTRATION

**Written before a single number exists.** Nothing below is adjustable after the run.

## 🔴 PREREQUISITE — merge rung 40 into `main` before opening this rung

This rung reads rung 40's checkpoints and imports rung 40's `chain40`, and rung 40 is a
**closed, positive result**: `alpha16` is a WIN (`proxy_leaderboard` +0.0329, paired CI
`[+0.0030, +0.0498]`, no cell harmed), recorded in its `RESULTS.csv` and `CONTEXT.md`.
Leaving it on an unmerged branch means a rung that depends on it is built on something
`main` does not know about, and every teammate reading `main` re-derives a question that
is already answered — the failure [[survey-files-carry-retired-numbers]] records.

Merge first. `conn4e5`'s verdict lands the same night; if it changes the picture it
changes rung 40's own CONTEXT, which is another reason to close that rung before opening
this one rather than editing a merged rung later.

## Why this is its own rung and not a third arm of 40

Rung 40 pre-registered **two** arms, `A_alpha` and `B_connector`, and its own
`assert_single_variable` exists because rung 38 shipped three undeclared deviations.
Adding an inference-time question to it would be that same defect, committed inside the
rung written to prevent it. This moves a different variable, against a different control,
on a different eval scope — so it is a different rung.

Rung 40 owns the **checkpoints**; rung 43 owns the **question asked of them**.

## The question

> Given a checkpoint we have already trained, does turning `enable_thinking` ON at
> inference change its score?

**NOT** *"does reasoning help FRAME"*. That needs training data with traces and has a
published null on our exact backbone ([[coa-sft-published-null]], replicated by
Surgery-R1). This asks the version answerable in one GPU hour.

## THE ONE VARIABLE

`enable_thinking` **False → True**, which is a chat-template flag and nothing else:

```
False → …assistant\n<think>\n\n</think>\n\n     (block pre-closed and empty)
True  → …assistant\n<think>\n                  (block left OPEN)
```

Measured on the 27B's own template, 2026-08-15. Same weights, same data, same engine,
same 626 questions. `max_new_tokens` rises 64 → 512 **only** for the thinking arm — not a
second variable but the budget the mode needs to function at all: 64 cannot hold a trace,
which is exactly what scored the rung 23a smoke 0/24.

## Arms

| arm | GPU | why |
|---|---|---|
| `conn4e5` + thinking | ✅ | its merge exists, so it runs first and writes nothing |
| `alpha16` + thinking | ✅ | merge rebuilt from its 383 MB adapter after conn4e5's is deleted |
| `conn4e5`, `alpha16`, rung 38, A2 ep1/ep3 — **no-thinking** | ❌ | already judged on all 6252; subset to the same qIDs |

**Two requested modes are provably not arms**, and this is measured, not assumed:

* **`preserve_thinking`** emits a prompt **byte-identical** to `enable_thinking` on a
  single-turn conversation. FRAME has no previous turn whose trace could be preserved.
  Under greedy the outputs would be bit-identical — two arms tying perfectly would look
  like a measurement and would not be one.
* **A2 has no thinking mode**: zero mentions of `enable_thinking` in its chat template.
  Qwen3-VL is not a hybrid reasoning model; that starts at Qwen3.5.

## The subset — 626 questions, stratified

1/10 of the eval, sampled proportionally over (`answer_format` × ID/OOD), seed 42.
Verified locally: proportions match the full 6252 to three decimals on both axes, and
every qID is present in all five reference tables, so the comparison is **paired**.

🔴 **Never head-of-list.** `run.py:204` truncates with `items[:n_eval]` and the corpus is
dataset-ordered, so the first 625 are 100 % heico/OOD — measured 2026-08-14, when a
40-question smoke returned `acc_ID = nan` and an ID-only primary of NaN.

## The read

**Primary: `proxy_leaderboard`**, thinking vs the SAME arm's no-thinking run on the SAME
qIDs. Secondary and reported beside it, never quoted as the result: `bucket_mean`, the
two ID buckets, and `n_timed_out` (a trace that overruns 512 tokens is a truncation, not
an opinion — it must be counted, not averaged in).

### 🔴 Latency is a DECLARED read, not a footnote — added 2026-08-15, before any run

`n_timed_out` above counts **truncation at 512 tokens**. It does not count the thing that
actually disqualifies an answer in this challenge: **the 5.0 s/question wall clock**
(CONSTITUTION §II — a timeout IS a wrong answer). This arm raises `max_new_tokens`
**64 → 512**, and rung 40 measured p99 **1.515 s** (`alpha16`) and **1.755 s**
(`conn4e5`) **at 64 tokens**. An 8× token budget is the one change most likely to cross
5.0 s, so without this read the rung can return a WIN that cannot ship and we would not
know it.

Recorded per arm in `RESULTS.csv`, beside the accuracy columns, using the same names
rung 40 already writes so the two rungs stay comparable:

| column | meaning |
|---|---|
| `infer_latency_p99_s` | p99 wall clock per question. **The read** — §IV.7 says measure p99, not mean. |
| `infer_latency_max_s` | max. Context only: rung 38's 4.81 s `max` was an outlier among 6252 and was briefly misread as the budget. |
| `timed_out` | count over the 5.0 s cap. Distinct from `n_timed_out` (512-token truncation) — **both are reported, and they are not the same number.** |
| `mean_new_tokens` | the mechanism, so a latency result is explainable rather than just observed. |

🔴 **Pre-registered decision rule, so it is not argued after the numbers land:** a
thinking arm whose **`timed_out > 0` or `infer_latency_p99_s` ≥ 5.0** is reported as
**NOT DEPLOYABLE regardless of its accuracy delta**. Such an accuracy win is not void —
it is evidence the capability exists — but it may not be quoted as a submission
candidate without a second run that brings the trace inside budget. The eval pod is not
the L40S, so a p99 measured here is **indicative, not certifying**; the certifying number
is on the target hardware, and that gap is recorded now rather than discovered later.

⚠️ At n=626 the paired CI is **wider** than the 6252-question one. A |Δ| under ~0.05 will
not separate from zero here. This probe is powered to see a **large** effect or none; it
is not powered to certify a small one, and a null must be reported as *"no large effect"*.

## 🔴 Pre-registered confound

Both checkpoints were trained **901 steps on the no-thinking path and zero on the
thinking path** — the chat template inserts the closed empty block by default, so every
one of the 14,415 training examples taught "after `</think>`, emit the bare answer".
Unsloth's own guidance is to keep **≥75 % reasoning examples** to preserve the ability;
we have 0 %.

Therefore a null **cannot** separate *"reasoning does not help FRAME"* from *"our SFT
overwrote the reasoning ability"*. It remains actionable for the submission, which ships
one of these checkpoints — but it does not generalise, and that limit is recorded here
rather than argued afterwards.

## Cost and the disk constraint

Two GPU arms in series because the volume cannot hold two 52 GB merges: ~43 GB free on
2026-08-15 with a teammate writing frames to the same quota. Delete before rebuild, never
after. Both adapters (1.4 GB together) regenerate either merge in ~10 min — a 135×
saving over keeping them.

Estimated: 2 × (load ~5 min + 626 questions at 3–5× the no-thinking rate) + one re-merge
≈ **1 h 45 – 2 h 30**.

## 🔴 TWO LAUNCH BLOCKERS — check both, in this order, or do not start

Written 2026-08-15 while rung 40 arm B's continuation was still running. **Neither is a
preference; the first one this rung's own code enforces, and the second one this rung's
own code can destroy.**

### 1. Disk — the guard already says no

`remerge.py` sets `min_free_gib = 60` and RAISES before touching a GPU. The last measured
figure is **~43 GiB free** (2026-08-15), so **as written today this chain refuses to
start**, and correctly.

Free space will get **worse before better**: `40_B_connector_ep23_v1` writes ep2 and ep3
checkpoints to the same ~670 GB quota while it runs. So the check must happen **after**
that run finishes, not now.

Measure with `remerge.free_gib()` — `du -sx` against the quota. 🔴 **Never `df` or
`statvfs`**: they report the MooseFS cluster at **1.4 PB** and a chain already died
`Disk quota exceeded` trusting them.

### 2. 🔴 The chain deletes an artifact another run may be training FROM

`chain_probe.py` step 2 is `rm -rf "{B}/merged"` — `conn4e5`'s 52 GB merged model. Per
`1ab2509`, `40_B_connector_ep23_v1` uses **Leo's ep1 merged model as its `base_model`**,
and the MooseFS volume is shared across pods.

**If those are the same path, starting this rung pulls the base model out from under a
live ~11 h training run.** Not confirmed either way as of this writing — his run may read
a copy local to its pod.

**Verify before launch, do not assume:** resolve `base_model` in
`40_B_connector_ep23_v1`'s config and compare it to `{arms_exp_dir}/runs/40_B_connector_v1/merged`.
If they resolve to the same inode, this rung **waits** — the adapters are 383 MB and
regenerate either merge in ~10 min, so there is nothing to preserve and nothing to rush.

⚠️ Both blockers clear on the same event: **`40_B_connector_ep23_v1` finishing and its pod
stopping itself.** That is the trigger to re-check, not a clock.

## Before the full run

A ~20-question pass first, to see a real output and confirm the `</think>` split works.
The thinking arm has never executed, not even in smoke — and every defect that cost us a
night on 2026-08-14 was in a path nobody had run.
