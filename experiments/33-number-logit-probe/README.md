# Rung 33 — the number-value distribution, read before the argmax collapses it

| rung | variable | verdict |
|---|---|---|
| 30 | GRPO, 0/1 exact-match reward on `number` | 🔴 NO-GO, `aggregation_ID` −0.0094 |
| — | global shift on the emitted integer | 🔴 0.4718 → 0.2197 |
| — | oracle value LUT | 🔴 ceiling +0.0148, `argmax_injective` False |
| **33** | **read P(value), not the emitted integer** | 🔴 **gate A FAILS**; the family is worth ≈ **+0.01 headline**, under the S1 bar |

Probe: `_tools/logit_dump.py`, A2 (`21_lr_2e4_v1/checkpoint-2703`), all **2,094** scored `number`
questions, RTX 5090, ~25 min, zero training.

## The pre-registered gate, and it failed

Declared **in code before the run** (`logit_dump.py`, `gate` block):

| clause | measured | line | |
|---|---|---|---|
| **A** — `P(gold in top-2 \| argmax wrong)` | **0.4560** | ≥ 0.55 | 🔴 **FAIL** |
| **D** — median `P(top-1)` | 0.5363 | ≤ 0.97 | 🟢 pass |

**D passing is the interesting half.** The distribution is **not collapsed** — the model's median
top-1 belief is only **54%**, so SFT left plenty of mass unspent. That specifically discharges the
gate `src/frame/loss.py:29` places on NTL (*"a proximity loss needs a distribution to widen"*):
there is a distribution to widen.

**A failing is the decisive half.** When greedy is wrong, the gold is the runner-up only **45.6%**
of the time. The `65.6% off-by-one` structure is therefore **substantially an argmax artefact**:
adjacent *in value* does **not** imply rank-2 *in probability*, and that assumption was the
unexamined premise under the whole ceiling argument.

## What every candidate rule is actually worth

Computed offline from the dumped distribution — no further GPU.

| rule | `number` accuracy | Δ |
|---|---|---|
| greedy argmax (baseline) | 0.4680 | — |
| **mask `0` from the support** | **0.4804** | **+0.0124** |
| `round(E[value])` | 0.4647 | **−0.0033** |
| oracle per-value bias `b_k`, fitted globally | 0.5053 | +0.0372 *(fitted on its own eval)* |
| **`b_k` fitted on ID → evaluated on OOD** | 0.5294 vs 0.5038 | **+0.0256** |

* **`round(E[y])` LOSES, exactly as decision theory predicts.** Under 0-1 loss the Bayes-optimal
  statistic is the **mode**; the mean is optimal for squared error. Every "expectation decoding"
  result in the literature reports correlation or MAE — and we are already at Spearman 0.735 with
  47% accuracy, so that axis was never ours to win.
* **The per-value bias `b_k` does transfer**: fitted on ID it is **+0.0256 on held-out OOD**,
  against the −0.016 negative transfer that killed the value LUT. It is the one member of the
  family that survives contact.
* **Masking `0` is free and principled**: `0` occurs **zero** times as a `number` gold (range
  1..12) while the model emits it 43 times. ⚠️ **But it is measured on OUR eval.** The hidden
  final test set is other data; if it contains a legal `0`, this converts free wins into
  guaranteed losses. Do not ship it without a stated assumption.

## Verdict

**The decoding family is real, and it is too small.** Its best honest number is `b_k` at
**+0.0256 on `number`**, plus at most `+0.0124` from masking `0`. `number` carries ~40% of the
headline, so the whole family is worth roughly **+0.010–0.015 on the headline** — under the S1
ship bar (0.03) and at the edge of what `significance-rule` calls readable.

🔴 **And the pre-registration governs.** Gate A was declared before the run and it failed. The
temptation is to note that `b_k` is not a top-2 rule and therefore "A was the wrong gate" — which
is true as a specification criticism and is **not** a licence to adopt the criterion that passed.
Recorded as: *gate A was mis-specified (it bounds a narrower rule than the one that pays), the
rung is reported as FAILED against its own declared line, and the `b_k` number stands as
measurement rather than as verdict.*

## What this buys the next rung, which is the real return

1. 🟢 **NTL / soft-label training is UNBLOCKED on evidence, not hope.** `loss.py:29` gated it on a
   distribution existing to widen; median `P(top-1)` = 0.536 says it exists. This probe was that
   gate.
2. 🔴 **The off-by-one ceiling of 0.8185 is retired as a target.** It assumed adjacency in value
   meant reachability in probability. It does not, and 0.456 is the number that says so.
3. **`round(E[y])` and every expectation-style decoder are closed** — refuted by measurement and
   by decision theory, in the same direction.

Ties: [[counting-has-two-failure-modes]] (the finding this tested) ·
[[count-calibration-dead]] (whose negative ID→OOD transfer `b_k` reverses) ·
[[rung30-grpo-number-state]] · [[significance-rule]].
