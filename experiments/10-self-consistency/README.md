# Rung 10 — Self-consistency (k-voting on `number`)

> 📌 **This directory also hosts step 5 of the August plan — the entropy gate.** Not a rung (the
> ladder is closed): it reuses this rung's `predict_samples()`/`vote.py` machinery to ask whether
> GRPO has a gradient. Ran twice — **SEED 42 (2026-08-05) and SEED 43 (2026-08-06)** — and both
> samples give the same verdict: **phase C lives, scoped to `number`.** Section 7 below; verdict
> in [`context/decisions/entropy-gate-scopes-phase-c-to-number.md`](../../context/decisions/entropy-gate-scopes-phase-c-to-number.md).

> **Status: CLOSED — FAITHFUL NEGATIVE (2026-07-20).** Ran in full on 1× RTX 5090: identity gate,
> T0 pilot, T1 over the whole 2094-question `number` population in three pre-registered arms, T2
> scoring. **All three arms negative; k=16 significantly HARMS OOD.** Verdict and mechanism in
> [`context/decisions/self-consistency-dead.md`](../../context/decisions/self-consistency-dead.md);
> numbers in `RESULTS.csv`. Spec: `local/specs/self-consistency/` (private vault).

## Ladder

| Rung | What changed (one variable) | Baseline | Status |
|------|-----------------------------|----------|--------|
| 00-baseline | Qwen3-VL-8B zero-shot | — | done |
| 01-ood-split | frozen ID/OOD split (`frame_ood_v1`) | — | done (infra) |
| 02-lora-sft | LoRA instruction fine-tune | 00 | done (`bucket_mean` 0.5486) |
| 03-prompt-variants | `SYSTEM_PROMPT` additions | 00 (a1_v0) | done — faithful negative |
| 04-vendor-baseline | External vendor baseline | 00 | done (tutorial, no results row by design) |
| 05-bottleneck-audit | the image passed to the model (real/black/shuffled) | 02 | done — NO SHORTCUT |
| 05b-number-probe | *(probe)* reads the predicted text rung 05 discarded | — | done — COUNTS BADLY |
| 05c-count-confusion | *(probe)* `P(pred\|true)` per template | — | done — CALIBRATION IS DEAD |
| 06-vit-lora | LoRA also on the ViT | 02 | done — PARTIAL (`bucket_mean` 0.5667) |
| 07-enumeration | *(probe)* can the model enumerate? | — | done — faithful negative *of the method* |
| 08-data-card | *(not an experiment)* what is in the data | — | done |
| 09-coa-sft | CoA-format SFT (Rodrigo) | 02 | in progress, `task/r1-coa-sft` |
| **10-self-consistency** | **greedy → majority vote over k samples, `number` only** | **06 (`ckpt-1720`)** | **closed — FAITHFUL NEGATIVE, k=16 harms OOD** |

## 1. Why this rung, and why now

**Where the gap is.** [[the-gap-is-the-number-format]] measured that `aggregation × ID` is
**80.4 % `number`** and that `fo_class` is not in the bucket at all. Closing the 12.5-pt deficit
requires lifting `number` from **0.327 → ~0.482**; answering 100 % of `binary` only reaches
0.4492. There is no other route.

**Why it is available now.** [[latency-budget-is-pooled]] read the official submission template:
the FRAME budget is **pooled** (`120 s setup + B × 5 s`), not per question, and the `latency`
field we emit is not a scoring input. Against a measured p99 of **0.352 s**, k candidates are
affordable. The idea had been parked with an explicit condition — *"if the cap is amortised this
lever opens entirely; if per-question, it dies"*. It is amortised.

**Why it is the right next move.** It is the only surviving lever in the group that owns the gap
that costs **no training GPU**. Its competitor, post-hoc calibration, is measured dead
([[count-calibration-dead]]).

## 2. 🔴 The risk that shapes the design

Self-consistency votes toward the mode of the model's own distribution. **Ours is measured to be
biased:**

- **05b:** 81.5 % of `number` errors are **under-counts**; the scale saturates at ~2.
- **05c:** true values **2, 3 and 4 all share the same modal prediction (1)**.

> If probability mass already sits on the wrong answer, voting **reinforces** the error instead of
> correcting it. This is the exact mechanism that killed calibration: aggregation without
> separation creates no information.

**Therefore this rung does not begin with a run.** It begins with T0, a cheap diversity probe that
can close it for ~20 min of GPU — the same discipline that let 05c kill calibration before any
training was spent.

## 3. The single variable

`do_sample=False` → sample `k` and vote. **Everything else is byte-identical** to rung 06:
checkpoint (`ckpt-1720`), prompt, `max_new_tokens`, `max_pixels`, data.

**Scope: `number` only** (2094 questions). Sampling where greedy is already right is a way to
*lose* points, and restricting scope keeps the A/B to one variable — the rest of the exam stays
literally the rung 06 output.

**Voting rule:** majority over the parsed integer (`frame.parsing.parse_number`, validated at 0
parse failures on all 2094). **Ties break to the LOWEST value** — deliberately the conservative
choice, since the model under-counts. Breaking ties upward would smuggle a bias correction in
under cover of the voting rule; any gain must come from the voting itself.

**The control is free and is not re-run:** rung 06's `eval_best/inspect.csv` already holds the
greedy answers. It is *validated* (its `acc` over the 2094 must reproduce **0.4250 ± 0.005**)
before being compared to anything — never assumed correct from its path.

## 4. Pre-registered rule

Judged on `number`, disaggregated ID/OOD, read as **margin** over the template-aware floor
(RULES §11 — `acc_OOD > acc_ID` is an artifact).

| Outcome | Verdict |
|---|---|
| T0: `mode_share ≈ 1.0` (no diversity) | **DEAD** — faithful negative, nothing further is run |
| T0: diverse, but the mode sits on the same biased value | **DEAD** — voting reinforces the error |
| `number` rises **≥ +0.02** with CI excluding 0, in ID **and** OOD | **PASS** → candidate |
| rises in one distribution only | **PARTIAL** — not promoted without understanding why |
| flat or down | **FAITHFUL NEGATIVE** — written up and closed |

🔴 **The threshold does not move afterwards.** And **`bucket_mean` does not decide this rung** —
it averages four buckets while this changes one format inside one of them. Reported, not used as
the verdict.

## 5. Files

| Path | What |
|---|---|
| `_models/vote.py` | engine — `vote_number`, `diversity`, `t0_diversity`. Library, not a launcher |
| `src/frame/engine.py` | `predict_samples()` added; `predict()` unchanged in signature and behaviour |
| `src/frame/config.py` | `n_samples` / `temperature` / `top_p`, **defaults reproduce greedy exactly** |
| `src/frame/parsing.py` | `parse_number` moved here (three rungs depend on it); 05b re-exports it verbatim |

**Flag-off guarantee:** with `n_samples = 1` the engine takes the original `do_sample=False`
branch. ✅ **VERIFIED 2026-07-20** — 50/50 strings byte-identical to rung 06's `predictions.json`,
reproduced across four independent model loads on the same GPU class (RTX 5090) that produced the
reference. The A/B is therefore single-variable.

`src/frame/metrics.py` also gained `paired_delta_ci`: the module had a video→question bootstrap for
ONE arm, but an A/B on the same questions needs the bootstrap of the paired DIFFERENCE — two
independent CIs discard the pairing and come out far too wide.

## 6. Result — FAITHFUL NEGATIVE

Full 2094 `number`, greedy control rescored by the same function and **gated to reproduce the SDK's
0.4250**. Margin over the template-aware floor, ID/OOD disaggregated, paired video-level CI.

| arm | k | T | ID | OOD | verdict |
|---|---|---|---|---|---|
| A | 8 | 1.0 | −0.0053 [−0.050, +0.041] | −0.0195 [−0.055, +0.012] | NEGATIVE |
| B | 8 | 1.3 | −0.0172 [−0.067, +0.035] | −0.0074 [−0.048, +0.026] | NEGATIVE |
| C | 16 | 1.0 | **+0.0284** [−0.009, +0.070] | **−0.0430 [−0.081, −0.008]** | NEGATIVE |

🔴 **Arm C is a trap the ID-AND-OOD conjunction caught.** Its ID rise is within reach of the +0.02
bar; its OOD harm is significant. Reading ID alone would have promoted a change that measurably
damages generalisation.

**Mechanism:** doubling k doubled the OOD harm — more samples estimate the mode better, and a
better estimate of a *wrong* mode is further from the truth. On OOD the vote falls **below the
trivial floor** (margin −0.0047 at k=8, −0.0189 at k=16). Full reasoning:
[`context/decisions/self-consistency-dead.md`](../../context/decisions/self-consistency-dead.md).

**Cost, for the record:** k=8 ≈ 1.15 s/q, k=16 ≈ 1.96 s/q vs greedy 0.173 s — sublinear in k
because the k candidates share one prefill. Cold start 8.7–23.9 s load + ≤1.3 s first inference
against the pooled 120 s setup allowance. **The lever died on quality, not on latency.**

## 7. Step 5 (August plan) — the entropy gate, run twice

**Not part of rung 10.** `10b_entropy_gate.ipynb`, 600 train questions, k=8, T=1.0, on **A2 ep3**
(`21_lr_2e4_v1/checkpoint-2703` — not rung 06, which we are about to replace). Scored by the SDK
verifier, 9 evaluator passes. Both thresholds pre-registered before the first run.

| format | n | `zero_adv` s42 | `zero_adv` s43 | headroom s42 | headroom s43 | verdict |
|---|---|---|---|---|---|---|
| `binary` | 200 | 0.720 | 0.735 | +0.050 | +0.040 | 🔴 DEAD (both) |
| `fo_class` | 200 | 0.660 | 0.705 | +0.090 | +0.045 | 🔴 DEAD (both) |
| **`number`** | 200 | **0.280** | **0.270** | **+0.240** | **+0.310** | 🟢 **ALIVE (both)** |

Kill lines: `zero_advantage >= 0.60`, `headroom < +0.05`. ⇒ **GRPO with a set-F1 reward is scoped
to `number`**; rung 22 (loss-mass) stays parked as the pre-decided Plan B.

🔴 **What the second seed changed.** `zero_advantage` is stable (±0.015) and carries the decision.
`headroom` is not: `fo_class` halved between samples and **would have flipped** had its verdict
rested on that leg — the conjunction saved it. And greedy on `number` moved **10.5 points**
(0.705 → 0.600) between two independent 200-question draws, so **a per-format delta below ~0.10
at n=200 is not separable from the draw.**

⚠️ **Two operational facts, paid for once.** The run is **~58 min**, not the 1h48 first recorded
(papermill's own cell timings; the generation cell is 56.7 min and all nine evaluator passes are
~38 s). And `FrameProvider` (`src/frame/data.py:154`) reads straight from the source video with
decord and **never touches `/workspace/frames_cache`** — during a run the GPU sits near 0% while
one CPU thread seeks inside multi-GB AVIs on the network volume. That profile looks hung and is
not; the frame cache failing to grow is not a symptom.

**Artifacts:** `RESULTS_step5_{gate,verdict,per_question}*.csv|json`, seed 42 unsuffixed and seed
43 with a `_seed43` suffix. ⚠️ The notebook writes both runs to the same `runs/.../full/`, so a
re-run overwrites — copy the artifacts out before launching another seed.
