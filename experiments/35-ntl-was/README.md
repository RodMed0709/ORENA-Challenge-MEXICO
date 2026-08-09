# Rung 35 — NTL-WAS: teach the loss that numbers have an order

> **Pre-registered 2026-08-09, BEFORE the run**, per `RULES §S3`/`§S8`.

## The ladder

| rung | variable | verdict |
|---|---|---|
| 21 A2 | lr → 2e-4 | 🟢 the campaign's best: platform **0.5288**, rank 11, both baselines beaten |
| 30 | GRPO, 0/1 exact-match reward on `number` | 🔴 NO-GO, `aggregation_ID` −0.0094 |
| 33 | read P(value) before the argmax | 🔴 gate A failed; the decoding family is worth ~+0.01 headline |
| 34 | linear probe on the hidden states | 🟢 **the count IS there** — 0.5264 vs the head's 0.4680 at layer 24 |
| **35** | **+ λ·NTL-WAS on the number tokens** | ⏳ this file |

## The argument, and why it is not rung 30 again

Rung 34 measured that the count is **linearly decodable from the hidden state at layers 18–24 and
still present at layer 36** — the last state before the token head — while the head emits a worse
answer. The information survives to the end and is lost **in the projection into tokens**.

Cross-entropy is what trains that projection, and it treats numbers as a **nominal** scale:
predicting "3" when the gold is "4" costs exactly what predicting "9" costs. Nothing in the
objective knows 4 is near 5. NTL-WAS adds the Wasserstein-1 distance between the predicted
digit distribution and a point mass at the target value, as an auxiliary term on the existing
head — no architecture change, no runtime cost.

🔴 **Why this is a different intervention from rung 30, stated before the result so it cannot be
claimed afterwards.** GRPO gave a 0/1 reward: an answer one off scored identically to one five
off, so the objective carried **no ordinal information at all** and could only re-rank samples the
policy already produced. NTL changes the **gradient geometry of the loss**. Rung 30 is evidence
against re-ranking; it is not evidence against ordinal supervision.

## The gate that licensed this, discharged

`src/frame/loss.py` has carried NTL as lever **2b** since July, gated: *"a proximity loss needs a
distribution to widen."* Both halves now have numbers:

* rung 33 — median `P(top-1) = 0.536`. The distribution is **not** collapsed.
* rung 34 — the count is decodable from the hidden states at **0.5264** vs the head's 0.4680.

## Pre-registration

**Primary cell: `aggregation_ID`** (`S3` forbids local `bucket_mean` as the comparator; `§12`
forbids `acc_number` as a headline). **MDE ≈ 0.036.** Baseline: **A2 = `21_lr_2e4_v1`
checkpoint-2703**, its own recipe unchanged.

**Win requires all three (`S8`):**
1. `aggregation_ID` paired video-clustered CI excludes zero in the arm's favour;
2. it holds on ID and OOD jointly — `aggregation_OOD` may not move against us;
3. no cell anywhere shows significant harm.

🔴 **Declared veto cell: `object_recognition_{ID,OOD}`.** A term that fires only on digit
positions can pull the policy away from `fo_class`, which is 71% of that bucket.

**Verdict checkpoint declared in advance: the epoch-3 checkpoint of both arms**, matching A2's own
3-epoch recipe. Intermediate epochs are a collapse readout, never selection — the rung-30 rule,
which was added because the absence of one is six degrees of freedom.

**Single variable:** `λ·NTL-WAS` added to the identical A2 recipe. Everything else — lr 2e-4,
rank 8, alpha 32, all-linear, cosine, warmup 0.03, bf16, sdpa, per_device 1 × accum 16, seed 42,
the same `18_count_aug_v1/train.jsonl` — is unchanged. `λ = 0.3`, the paper's default.

## 🔴 The three things most likely to make this fail, recorded before the result

1. **The confound we cannot separate with the controls we own.** An auxiliary term that fires only
   on digit positions also raises `number`'s share of the gradient
   ([[loss-mass-is-token-weighted]]: 20.8% of gradient off 34.2% of rows). So the arm is
   *"ordinal supervision AND more number gradient"*. Rung 22 was meant to be the
   format-reweight-only control and was NO-GO for an unrelated reason (its hook was an arithmetic
   identity). **A win here does not attribute cleanly, and that is stated now, not after.**
2. **NTL is validated on text math up to 3B, with no vision anywhere.** Its exact-match gains
   (T5-base 64% → 75%) are on a different modality and a different scale of model.
3. **Our answers are 1–2 tokens.** The Wasserstein term's support is a 10-way digit distribution
   at one or two positions; there is far less for it to shape than in a multi-digit math answer.
   It may be a numerical no-op.

## Implementation

`src/frame/loss.py::ntl_was` — pure torch, no tokenizer import, unit-testable offline. Composed
through the existing `make_compute_loss_func` hook, which returns `None` when disabled so **OFF is
byte-identical**.

⚠️ **WAS, not MSE, and the reason is load-bearing.** The paper documents that the MSE variant has
**non-unique minima** — 50% mass on "0" and 50% on "8" scores zero against a target of 4. Under
exact-match scoring that bimodal escape hatch would train the model into the shape that hurts us
most.

⚠️ **The adapter is the unverified seam.** `make_compute_loss_func`'s docstring says it: ms-swift
is not installed locally, so the call signature and the shapes it hands over **must be confirmed
on the pod with a 3-step smoke that prints the received args** before any full run — the same
precondition rung 22 discharged.
