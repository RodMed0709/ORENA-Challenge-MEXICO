# context — 10-self-consistency

> Curated context for rung 10. The artifact lives in `experiments/10-self-consistency/`;
> legokna's private spec. **Nothing has run yet.**

## State

**CODE READY · T0 PENDING (needs a pod).** No `RESULTS.csv` and none should be fabricated.
Local half done and unit-checked; the GPU half has not started.

| Step | Where | Status |
|---|---|---|
| Engine flag (`n_samples`/`temperature`/`top_p`) | `src/frame/config.py`, `src/frame/engine.py` | done, defaults OFF |
| Voting library | `experiments/10-self-consistency/_models/vote.py` | done, unit-checked |
| `parse_number` promoted to `src/frame/parsing.py` | 05b re-exports verbatim | done, **05c reproduces exactly** |
| Bit-identity check with the flag OFF | pod | **pending — no local GPU** |
| T0 diversity probe | pod, ~20 min | pending |
| T1 voting run + T2 scoring | pod, ~45 min + 0 GPU | gated on T0 |

## Why this rung exists

The gap is `number`: `aggregation × ID` is **80.4 % `number`**
([[the-gap-is-the-number-format]]), and no path to the target avoids lifting it 0.327 → ~0.482.
Self-consistency was parked as *"blocked — if the cap is amortised this lever opens entirely; if
per-question, it dies"*. [[latency-budget-is-pooled]] read the official template: the budget is
**pooled** (`120 s + B × 5 s`) and the emitted `latency` is not scored. Measured p99 is
**0.352 s**. It opens.

It is now the only lever in the gap's group that costs **no training GPU** — its competitor,
post-hoc calibration, is measured dead ([[count-calibration-dead]]).

## 🔴 The thing to hold on to

**Voting toward the mode is dangerous *for us specifically*.** 05b measured that **81.5 % of
`number` errors are under-counts** with the scale saturating at ~2; 05c measured that true values
**2, 3 and 4 all share the same modal prediction (1)**. Mass already sits on the wrong answer.

Aggregating a biased distribution reinforces the bias — the identical mechanism that killed
calibration. **That is why T0 exists and why it runs first.** Two of its three outcomes close the
rung for ~20 min of GPU.

The decisive T0 column is **`mode_closer_than_greedy`**, not entropy. Diversity alone proves
nothing: a model can be diverse and still centred on the wrong value, which is precisely what
05b/05c predict for us.

## Design decisions worth not re-litigating

- **`number` only.** Sampling where greedy is already right is a way to lose points, and it keeps
  the A/B to one variable — the rest of the exam stays literally rung 06's output.
- **Ties break to the LOWEST value.** The model under-counts; breaking ties upward would smuggle a
  bias correction in under the voting rule. Any gain must come from the voting.
- **One `generate` call with `num_return_sequences=k`**, never a loop of k calls — the prefill is
  shared, so k candidates cost far less than k forward passes.
- **`predict()` keeps its `-> str` signature.** `predict_samples()` is additive; existing
  consumers are untouched.
- **The control is not re-run** — but it *is* validated (acc over the 2094 must reproduce
  **0.4250 ± 0.005**) before any comparison. Never trusted from its path.
- **`bucket_mean` does not decide this rung.** It averages four buckets; this changes one format
  inside one of them.

## Landmines carried in from earlier rungs

- **Score only via `frame.metrics`** — never re-derive metrics in the notebook (RULES §EVAL).
- **Read margin, not raw accuracy**, and disaggregate ID/OOD (RULES §11).
- **Save the raw samples.** Rung 05 discarded the predicted text and 05b had to re-derive it all.
  `samples.json` is the expensive artifact; keeping it allows re-voting with a different rule
  without returning to the GPU.
- **Do not fall through to the expensive path silently.** Going from "T0 measured diversity" to
  "spend 45 min of GPU" is legokna's call, not an automatic fallback — the guardrail that worked
  in 05b.

## Open questions this rung does NOT answer

- **Cold start.** [[latency-budget-is-pooled]] inverts the risk: imports + weight load +
  CUDA-graph capture all eat the 120 s setup allowance. That is the number that could actually
  bite, and it is its own atomic.
- **Whether counting is teachable at all** where the model barely uses the image (rung 05 §5).
  Self-consistency sidesteps that question rather than answering it; the synthetic-SFT lever
  (idea 14) still faces it head-on.
