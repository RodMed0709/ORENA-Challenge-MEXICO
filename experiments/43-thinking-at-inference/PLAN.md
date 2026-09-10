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

## 🔄 AMENDED 2026-08-15, BEFORE the run — the OOD split, on vLLM, both arms

Three changes to the pre-registration, all written before a number exists.

### 1. The subject is the 4000 HeiCo questions, and the primary read becomes OOD

`heico` is **4000 of the 6252** eval questions — measured, not estimated — and the challenge
design document (p.12) states the **30 HeiCo videos are already publicly released under
CC BY-NC-SA**. Only the 170 SAVE videos sit under the restrictive data usage agreement. So
this subset can be evaluated **off the rented pod, on UNAM, at zero cost and with no DUA
question to resolve**.

🔴 **But HeiCo is 100 % OOD, so `proxy_leaderboard` — this rung's declared primary — is
ID-only and would return NaN.** That is the exact defect this document already warns about
("a 40-question smoke returned `acc_ID = nan`"). Running on HeiCo without changing the read
would reproduce it.

⇒ **Primary read becomes the OOD buckets.** The justification is not convenience: the FINAL
ranking is Copeland with **equal weight to ID and OOD** (`challenge_design.txt:2001`), and
the **pre-evaluation cannot see OOD at all**. So this measures half the score that no
leaderboard submission can show us. What it does NOT do is tell us whether the leaderboard
proxy moves — that needs the 2252 `lapchole` questions and therefore the pod.

### 2. 🔴 The no-thinking control must be RE-RUN, not taken from the archive

The section below says to subset the archived `results.csv` for the control at "zero GPU".
**That is now wrong.** Those tables were produced by the **HF sequential** path; the thinking
arm will run on **vLLM batched**. Reusing them would put the RUNTIME inside the comparison as
a second variable — measured at 2 % of answers (47/50 agreement), which is the same order as
the effects this rung could find.

⇒ **Both arms run on vLLM, in the same session, on the same 4000 qIDs.** The archived HF
numbers stay valid for what they are — the campaign's historical comparison against A2 — and
are simply not this rung's control.

### 3. Sequenced: `conn4e5` ep1 now, ep2+3 when its chain lands

ep1's merge is already on UNAM (rebuilt from its 343 MB adapter). ep2+3 finishes tonight and
gets the identical treatment, so the two are comparable to each other as well.

📌 **This is a partial rung and must be reported as one.** A result here answers "does
thinking change this checkpoint's OOD score". Confirming it on ID needs the pod, and that is
deliberately the *second* step: if a 4000-question OOD arm shows nothing, spending pod time
on 2252 ID questions is hard to justify.

## The read

**Primary: `proxy_leaderboard`**, thinking vs the SAME arm's no-thinking run on the SAME
qIDs. Secondary and reported beside it, never quoted as the result: `bucket_mean`, the
two ID buckets, and `n_timed_out` (a trace that overruns 512 tokens is a truncation, not
an opinion — it must be counted, not averaged in).

### 🔻🔻 THE 5.0 s PER-QUESTION CEILING IS NOT REAL — this whole section was written against a retired number

**Corrected 2026-08-15, and the error is the same one twice in one day.**
`context/decisions/latency-budget-is-pooled.md` is **MEASURED and SETTLED since 2026-07-19**:

> **verdict: POOLED — `120 s setup + B × 5 s` per run; the per-question ceiling was never real**

For a typical batch `B = 20` that is **220 s for 20 questions ≈ 11 s of effective budget each**,
and the forfeit rule is **per BATCH** (exceeding by 20 % forfeits the whole batch), not per
question. The note also inverts the risk: the thing that can actually bite is **cold start** —
imports, weight loading and CUDA-graph capture all eat the 120 s setup allowance.

It says two more things that hit this rung directly:

- *"the `latency` field we emit is informational; **never optimise for it or read it as a score
  input**"*
- *"the `max_new_tokens ≤ 32` guidance in `CLAUDE.md` is still good practice but is **no longer
  forced by the budget** — do not cite the cap as the reason"*

⇒ Everything below that treats 5.0 s as a per-question wall is **wrong**, and I wrote it this
morning without checking `context/decisions/`. Same failure as the retired "does not fit the
L40S" claim earlier today. Keeping the original text below, struck, rather than deleting it.

### ⚠️ AND OUR LOCAL HARNESS IS STRICTER THAN THE CHALLENGE

`enforce_latency` defaults **True** (`src/frame/config.py:103`) and the SDK gate
(`evaluator.py:218`) marks **any single response over 5.0 s as INCORRECT**. The real evaluation
pools. ⇒ **our local eval can manufacture a loss for the thinking arm that the challenge would
not.** With the measured medians below (10.11 s greedy), `enforce_latency=True` would score
**most of the thinking arm as wrong on latency alone**, and the arm would post a catastrophic,
CI-excludes-zero NO-GO that means nothing about reasoning.

🔴 **Pre-registered consequence:** the thinking arm MUST be run and reported **both ways** —
with `enforce_latency=True` (comparable to the archived references) and with it **False** (what
the challenge actually scores). Reporting only the first would repeat, in latency, exactly the
`</think>`-parser defect this rung already had to fix.

### 📏 Measured on the REAL 27B, 2026-08-15 — the 512-token alarm is RETRACTED

`RESULTS_thinking_27b_base.json`. Qwen3.6-27B **base**, FP8, vLLM, one RTX 6000 Ada (CC 8.9),
6 synthetic FRAME-shaped questions, `max_tokens=2048` to see the natural length:

| | greedy | sampling (vendor's thinking settings) |
|---|---|---|
| closed `</think>` | **6 / 6** | 6 / 6 |
| trace tokens min/median/max | 40 / **177** / 439 | 280 / **319** / 1446 |
| **would fit 512** | **6 / 6** ✅ | 5 / 6 |
| latency median | **10.11 s** | 15.58 s |
| latency max | 20.77 s | 65.72 s |

🔑 **Two of my own findings are retracted by this:**

1. **`max_new_tokens = 512` is adequate.** The earlier "8/8 overran 1024" came from
   **Qwen3.5-2B**, a 2B base model, and was a size artifact — exactly the direction the caveat
   written beside it warned about. The real 27B closes the block in a median of 177 tokens.
2. **Greedy is not off-spec, it is BETTER.** Unsloth's doc prescribes `temperature=1.0` for
   thinking mode, and I raised the worry that greedy would make us measure "thinking
   misconfigured". Measured, greedy produces traces **half as long** (177 vs 319 median), closes
   just as reliably (6/6), and is **35 % faster**. ⇒ **no confound, and the constraint we are
   under happens to be the better setting.** That open question is closed.

⚠️ **What remains open, and it is now the only latency question that matters:** these are
**sequential, `max_num_seqs=1`** measurements. The pooled budget assumes the batch goes through
**vLLM continuous batching**, which the submission template hands us up front. A 10.11 s median
sequential could amortise to a fraction of that batched — the team's shipped container measured
**0.79 s/question**. **Batched throughput with thinking on is NOT measured**, and no verdict
about affordability should be issued from these sequential numbers.

### ✅ MEASURED 2026-08-15 — batched, thinking costs **1.33 s/question**. Latency is NOT a blocker.

`RESULTS_batched_thinking.json`. Same model and card, `B = 20` through vLLM continuous
batching — the path the submission template actually uses, since it hands the batch over up
front. Greedy both arms, synthetic images:

| B = 20, batched | wall for 20 | **s/question** | gen tokens | tok/s |
|---|---|---|---|---|
| no-thinking (`max_new_tokens=64`) | 5.49 s | **0.275** | 1,165 | 212 |
| **thinking** (`max_new_tokens=512`) | 26.63 s | **1.332** | 5,985 | 225 |

🔑 **The 10.11 s sequential median collapses to 1.332 s batched — 7.6×**, exactly the
amortisation `latency-budget-is-pooled.md` predicts. Against the warm pooled allowance of
`B × 5 = 100 s`, the thinking arm uses **26.6 s — 27 % of it**. Against the ~11 s/question
effective budget it uses **12 %**.

⇒ **Thinking is affordable and latency is not a reason to avoid it.** It costs **4.8×** a bare
answer, which is real and worth stating, but 4.8× of 0.275 s is still small. Every latency
alarm I raised about this rung — the "NOT DEPLOYABLE" rule, the 5 s wall, the worry that a
thinking loss would really be a clock loss — **dissolves at the batch size the challenge runs.**

🔴 **What replaces it as THE risk, exactly as the pooled note said it would: COLD START.**
`setup_secs_cold = 227.1 s` against a **120 s** allowance — over by itself, before a single
question. ⚠️ Not a verdict: that figure includes vLLM kernel compilation which a prepared
container caches, and this box is not the submission image. But it is now the only latency
number that can fail us, and it must be measured **in the container**, not here. The pooled
note said this a month ago: *"the danger was never `max_new_tokens` or a slow question. It is a
cold start that eats the pool before the first answer is produced."*

📌 Consequence for this rung's read: the `enforce_latency=True` / `False` double reporting
pre-registered above **stays**, because the local harness still gates per-response at 5.0 s and
our *sequential* eval path will trip it. But the expected size of that artifact is now known,
and it is a harness artifact, not a property of thinking.

<details><summary>Original section, written against the retired per-question ceiling — kept for the record</summary>

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

🔻 **Corrected 2026-08-15, before any run — the first draft of this section had it backwards
in both directions.** Traced through the code:

- **`n_timed_out` IS the 5.0 s wall-clock count**, not a truncation count. `frame.run:115`
  sums `results_df["timed_out"]`, which the focus Evaluator sets from
  `TRACK_MAX_LATENCY[Track.FRAME] = 5.0` (`vendor/.../focus/config.py:35`), and
  `enforce_latency` defaults **True** (`src/frame/config.py:103`).
- **The 512-token truncation is measured by nobody.** It is a separate quantity and needs
  adding if we want it.
- `latency_s` (`mean/p50/p95/p99/max`) is **already produced** by `frame.run:116`. Nothing
  new has to be computed — `thinking_probe.py` only has to stop dropping it.

| column | source | meaning |
|---|---|---|
| `infer_latency_p99_s` | `report["latency_s"]["p99"]` | p99 wall clock. **The read** — §IV.7 says p99, not mean. |
| `infer_latency_max_s` | `report["latency_s"]["max"]` | Context only: rung 38's 4.81 s `max` was an outlier among 6252 and was briefly misread as the budget. |
| `n_timed_out` | `report["n_timed_out"]` | Questions over the **5.0 s cap**. |
| `n_truncated` | **to add** | Responses that hit `max_new_tokens=512`. A trace that runs out of room is a truncation, not an opinion. |

### 🔴 The rule this replaces, and why the first one described an impossibility

The first draft said a thinking arm could be a **WIN on accuracy yet NOT DEPLOYABLE** on
latency. **That cannot happen here:** with `enforce_latency=True` the harness scores a
response slower than 5.0 s as **incorrect**. The accuracy number already has the latency
penalty folded into it.

⇒ **The real risk is the mirror image, and it is worse:** a thinking arm that is *slow*
would post a **loss**, and we would read it as *"reasoning does not help FRAME"* when it
actually means *"the trace did not fit in 5 s"*. Those are opposite conclusions from the
same number.

**Pre-registered rule:** if the thinking arm loses **and** `n_timed_out` is materially
above its no-thinking control, the result is reported as **CONFOUNDED BY LATENCY — not
evidence about reasoning**, and the honest follow-up is a shorter token budget, not a
verdict. A clean read requires `n_timed_out` at or near the control's. The eval pod is not
the L40S, so these seconds are **indicative, not certifying**.

</details>

📌 **The CONFOUNDED-BY-LATENCY rule above SURVIVES the correction — and matters more, not
less.** Its reasoning was that a slow arm posts a loss that reads as a claim about reasoning.
That is now *measured to be likely*: at a 10.11 s sequential median with `enforce_latency=True`,
most of the arm would score wrong on the clock. What changes is the remedy. It is **not** "cut
the token budget until the accuracy improves" — 512 tokens is already adequate and cutting it
would truncate real traces. It is **report both ways** (§ above) and **measure batched
throughput** before saying anything about affordability.

⚠️ At n=626 the paired CI is **wider** than the 6252-question one. A |Δ| under ~0.05 will
not separate from zero here. This probe is powered to see a **large** effect or none; it
is not powered to certify a small one, and a null must be reported as *"no large effect"*.

## 🔴 The arm needs an ANSWER EXTRACTOR, and both readings are reported

Found 2026-08-15 by reading the engine, before any GPU time. `screen_engine.predict_samples`
returns `out[: answer_char_cap]` — the raw decoded generation. **Nothing splits on
`</think>`.** With the block left open, the model emits `…trace…</think>\n\nanswer`, so what
reaches the judge is the *reasoning*, cut at **300 characters** (`frame/config.py:55`, ~75
tokens) — thousands of characters before the closing tag.

⇒ **As staged, this rung was guaranteed to report a confident NO-GO for a parser we never
wrote.** Precedent, in the engine's own docstring: rung 23a's first bf16 smoke scored
**0/24** exactly this way. Its fix was to suppress thinking; this rung turns it back on.

Two changes, both **constitutive of the arm rather than treatments of it** — same standing
as `max_new_tokens` 64 → 512, which this document already accepts on that ground:

| change | why it is not a second variable |
|---|---|
| split on the last `</think>` (`_tools/think_parse.py`) | without it, `enable_thinking=True` measures whether 300 chars of chain-of-thought collide with the gold string. That is ~0 and is not about reasoning. |
| `answer_char_cap` 300 → 4000, **thinking arm only** | 300 chars cannot hold a trace, so the tag never reaches the saved string and there is nothing to split. Raising tokens without this just guillotines later. References keep 300 and are not re-run. |

### Both readings are reported, and the unparsed one is the control for this defect

The run saves the **raw** generation to `predictions.json`, so both are derived offline from
one generation — no second GPU pass, and `eval_arm`'s ban on `answer_postprocess` (right for
rung 40, where it would be a second variable) is never fought.

| reading | what it is |
|---|---|
| **parsed** | the arm. The answer after `</think>`. |
| **raw** | the artifact. Expected **≈ 0** — and reported anyway. |

🔑 A near-zero raw reading beside a normal parsed one **demonstrates** the extraction is
load-bearing instead of asserting it. If raw is NOT ≈ 0, that is itself a finding: the trace
was collapsing into the answer and the whole framing needs re-reading.

### Two gates, because they fail differently

- `assert_tag_survived` **RAISES** if not one generation contains `</think>`.

  ✅ **The tokenizer half of this is CLOSED, 2026-08-15 — and it cost nothing.** The worry
  was that `</think>` might be a SPECIAL token, so `skip_special_tokens=True` in
  `screen_engine.predict_samples` would delete it and leave nothing to split on. Answered
  from `Qwen/Qwen3.6-27B`'s public `tokenizer_config.json` — **no GPU, no pod, no challenge
  data, just model metadata**:

  ```
  id=248068  content='<think>'   special=False
  id=248069  content='</think>'  special=False
  additional_special_tokens: nothing containing 'think'
  ```

  `decode(skip_special_tokens=True)` filters `all_special_ids`; a `special=False` added
  token is not in it. ⇒ **the tag survives the decode.** ⇒ the gate now has exactly ONE
  meaning if it fires: every trace overran the budget. Raise `max_new_tokens` /
  `answer_char_cap`; do not touch how the engine decodes.

  ✅ **Confirmed against a REAL tokenizer**, not just the JSON — `Qwen/Qwen3.5-2B` on the
  UNAM box (`university-gpu-box`), which carries **the same `<think>`/`</think>` ids
  (248068 / 248069) as Qwen3.6-27B**, so it is a faithful proxy for the template and decode
  path. Read-only, no GPU, nothing written to that shared machine:

  | check | result |
  |---|---|
  | `enable_thinking=False` | prompt ends `…assistant\n<think>\n\n</think>\n\n` — block **pre-closed** |
  | `enable_thinking=True` | prompt ends `…assistant\n<think>\n` — block **left OPEN** |
  | `decode(skip_special_tokens=True)` | returns `'razonando aqui</think>\n\n2'` — **identical** to `skip_special_tokens=False` |

  ⇒ THE ONE VARIABLE section above is verbatim correct, and the extractor's premise holds
  on real tokenizer behaviour rather than on a config file read.
- **`n_truncated`** = generations that never reached `</think>`. They are scored as wrong,
  which they are — nothing was answered — but counted separately so truncation is never
  read as a claim about reasoning.

## 📏 Measured 2026-08-15 — a Qwen3.5 trace does NOT fit in 512 tokens. 8/8 overran 1024.

Run on UNAM (`~/storage/envs/orena-gen36`, transformers 5.12.1, RTX 6000 Ada), on
**`Qwen/Qwen3.5-2B` with 8 synthetic FRAME-shaped questions written for the test**. No
challenge data, no annotations — the DUA is not in play, which is why the trace text could
be inspected at all.

| | |
|---|---|
| generations | 8 |
| **closed `</think>` within 1024 tokens** | **0** |
| trace tokens (min / median / max) | 1024 / 1024 / 1024 — all hit the ceiling |
| max trace chars | **4453** (above our 4000 cap, and not the end) |
| would fit rung 43's 512 tokens | **0 / 8** |

🔑 **It is not degenerate looping — it is deliberation.** 58 unique sentences out of 60; the
trace is structured (*"Thinking Process: 1. Analyze the Request…"*), reaches `Result: 2`
around the middle, and then keeps second-guessing itself instead of closing the block.

⇒ **`max_new_tokens = 512` and `answer_char_cap = 4000` are both too small**, and
`assert_tag_survived` would have RAISED — the gate works, and it earned its place before
any pod time was spent.

### 🔴 The implication is bigger than the budget, and it is decision-relevant

A trace of >1024 tokens **cannot** be generated inside FRAME's **5.0 s** cap at any
plausible decode rate. If our checkpoint behaves like this, thinking is not a tuning
problem — it is **structurally incompatible with the track's latency budget**, and the
right answer is to report that, not to keep raising the budget until the accuracy looks
better. This is exactly the failure the CONFOUNDED-BY-LATENCY rule above was written for.

### ⚠️ What this does NOT establish, and the direction it may be wrong in

`Qwen3.5-2B` is a **2B base model**. Rung 43's subject is a **27B fine-tuned for 901 steps
on the no-thinking path with 0 % reasoning examples** — an SFT that taught it *"after
`</think>`, emit the bare answer"*. That training pressure pushes toward closing the block
**immediately**, which is the opposite failure: near-empty traces rather than runaway ones.

⇒ This measurement bounds the **base** behaviour, not our checkpoint's. It says the budget
is unsafe, not that truncation is certain. The 20-question smoke on the real checkpoint is
still what settles it — but it now has a prior, and the gate that catches either outcome.

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

## 🔴 LAUNCH BLOCKERS — one real, one retired, one about whose pod this is

Written 2026-08-15 while rung 40 arm B's continuation was still running. Of the three
things checked, **only §2 actually blocks**; §1 turned out not to, and saying so matters
because waiting on a phantom costs the same as waiting on a real one.

### 1. Disk — NOT a launch blocker. The chain is self-clearing by design.

🔻 **Corrected 2026-08-15, same day, before any run.** An earlier draft of this section
said "~43 GiB free < `min_free_gib = 60`, so this chain refuses to start". **That was
wrong**, and it was wrong in the direction that would have made us wait for nothing.

The guard lives **inside `merge_arm()`** (`remerge.py:72`), which the chain calls only at
**step 3**, after step 2 has already deleted 52 GB:

| step | writes | passes the guard? |
|---|---|---|
| 1. `conn4e5` + thinking | nothing — its merge already exists | never reaches it |
| 2. `rm -rf {B}/merged` | frees **52 GB** → ~95 GiB | n/a |
| 3. re-merge `alpha16` | ~52 GB | 95 > 60 ✅ |

That is precisely the serial pattern `remerge.py` was written to encode: peak disk stays
at ONE merge. The chain needs only **~8 GiB free at launch** for step 3 to clear after the
delete — we have ~43.

⚠️ Still worth one look before launch, because `40_B_connector_ep23_v1` writes ep2 and ep3
checkpoints to the same ~670 GB quota while it runs: confirm free space is **> ~10 GiB**,
not > 60. Measure with `remerge.free_gib()` — `du -sx` against the quota.
🔴 **Never `df` or `statvfs`**: they report the MooseFS cluster at **1.4 PB** and a chain
already died `Disk quota exceeded` trusting them.

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

#### 🔴 VERIFIED 2026-08-15 14:0x UTC — it IS the same path. Resolved by `keep_conn_merge`.

`/workspace/tmp/armB_ep3.log`, the running job's own log:

```
continuing from /workspace/repo_leo/.../runs/40_B_connector_v1/merged (15 shards)
  base_model = /workspace/repo_leo/.../runs/40_B_connector_v1/merged
```

Same MooseFS volume (`networkVolumeId gf78k60nlt`), mid-run. ⇒ **step 2 as written would
have deleted a live training run's base model.**

**Resolution — and the deletion was never necessary.** `du -sx` measures **550.1 GiB used
of the 670 GB quota ⇒ ~120 GiB free**, not the ~43 GiB this document carried from the day
before. Both 52 GB merges fit at once. Set **`keep_conn_merge=True`**: step 2 becomes an
echo, `alpha16` is merged alongside, and the only surviving `rm -rf` targets
`40_A_alpha_v1/merged`, which is ours. Verified by rendering the script and grepping every
`rm -rf` line. The flag defaults **False**, so the script is unchanged for anyone who
renders it without asking for this.

### 3. Can this rung just be CHAINED onto arm B's pod? — **not unilaterally.** Three traps are armed.

Asked 2026-08-15: rather than wait for `5btpl229y7kuar` to stop and rent a new pod, append
rung 43 to that pod's existing chain. Checked against
`experiments_segment/NOW.md` §"THREE POD-STOP LAYERS ARE ARMED":

| layer | lives | fires on |
|---|---|---|
| chain `trap … EXIT` | on the pod | the chain ending — **stops the pod** |
| watchdog | on the pod | 14 h wall |
| **independent killer** | **on the operator's machine** | **15 h hard deadline, RunPod API only** |

Layers 1 and 2 are Rodrigo's to change. **Layer 3 is not on the pod at all**
(`scratchpad/killer_b200.py`, his machine) and is *deliberately* dumb — "a hard deadline
and a stop, no log parsing, no liveness heuristics — so it cannot kill a healthy run for a
clever reason". ⇒ **it will stop the pod at 15 h whether or not rung 43 is mid-run**, and
this rung needs **1 h 45 – 2 h 30**.

⇒ Chaining is **possible but not ours to arrange**: it needs all three layers moved, and
one of them only its owner can reach. 📌 Also recorded: `/workspace/tmp` is a shared
namespace and **a pod rented for someone else's rung is read-only for us** — appending to
his chain is a write to his run, not a neighbourly reuse of idle GPU.

📌 And the operational scar that makes this sharper than it looks (`1ab2509`): **`pkill` on
a chain fires its `EXIT` trap and stops the pod.** SIGTERM triggers bash EXIT traps. Any
attempt to swap or append a job by killing a process takes the pod down with it. It cost
one restart already.

⚠️ Both blockers clear on the same event: **`40_B_connector_ep23_v1` finishing and its pod
stopping itself.** That is the trigger to re-check, not a clock.

## Before the full run

A ~20-question pass first, to see a real output and confirm the `</think>` split works.
The thinking arm has never executed, not even in smoke — and every defect that cost us a
night on 2026-08-14 was in a path nobody had run.
