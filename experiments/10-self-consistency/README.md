# Rung 10 — Self-consistency (k-voting on `number`)

> **Status: CODE READY, NOT YET RUN.** No `RESULTS.csv` exists and none should be invented.
> The local half (engine flag, voting library, unit checks) is done; T0 needs a pod.
> Spec: `local/specs/self-consistency/` (private vault).

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
| **10-self-consistency** | **greedy → majority vote over k samples, `number` only** | **06 (`ckpt-1720`)** | **code ready, T0 pending** |

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
branch. This must be verified bit-identically against rung 06's `predictions.json` on the pod
**before** T0 — there is no local GPU to check it here.
