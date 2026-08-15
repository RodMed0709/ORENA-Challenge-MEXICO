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
- **`n_truncated`** = generations that never reached `</think>`. They are scored as wrong,
  which they are — nothing was answered — but counted separately so truncation is never
  read as a claim about reasoning.

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
