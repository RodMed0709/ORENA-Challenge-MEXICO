---
question: Is the Qwen3-VL-32B a usable teacher for the CoA/CoT knowledge-transfer line?
verdict: NO — and the reason is narrower than "CoT does not work". Asked the FRAME questions with no gold, the 32B scores 0.08 where the trivial constant scores 0.295 (margin −0.215 OOD) — worse than our own UNTRAINED 8B (−0.191). A bigger general model is not better than the base model we would be teaching, so there is nothing to transfer. This does NOT invalidate CoT/CoA as a technique; it invalidates this teacher
status: MEASURED
date: 2026-07-31
measured_in: experiments/17-generator-probe/runs/17_generator_probe_v1/blind_probe.csv
---

# Decision: the 32B is not a teacher — but CoT is not what died here

- **Status:** MEASURED · 2026-07-31 · 200 questions, inference only, RTX PRO 6000 96 GB, bf16
- **Applies when:** anyone proposes generating reasoning traces with a larger general VLM to
  train the 8B — or cites this rung as evidence that CoT/CoA "does not work".

## The pre-registered rule, and what fired

Rung 17 was written by Rodrigo on 2026-07-23 (`7982730`) and had never been run. Its rule was
fixed before any number: **margin > 0 in BOTH ID and OOD ⇒ GO; margin < 0 on either ⇒ STOP.**

| | |
|---|---|
| accuracy over 200 questions | **0.08** |
| template-aware floor (OOD) | **0.2950** |
| **margin OOD** | **−0.2150** |
| our own 8B, zero-shot, for scale | −0.191 OOD |

🔴 **STOP.** And the comparison that makes it legible: **the 32B is worse than our untrained 8B.**

## 🔑 What this does and does not settle

**It does NOT invalidate CoT/CoA.** The technique is untouched. What the probe measures is
narrower and harder to argue with: **a bigger general-purpose model is not better at this task
than the base model we would be teaching**, so there is no knowledge to transfer. Distillation
moves a capability from a model that has it to one that does not; this teacher does not have it.

**It could change if we fine-tuned the 32B instead of the 8B.** A fine-tuned 32B would very
likely be a capable teacher — the escape hatch is real and is recorded here so nobody has to
rediscover it. 🔴 **But it is double the work with no guarantee of a better cost/efficiency
result than continuing to fine-tune the 8B directly**, which is the line that has actually been
moving our score. That is the reason we are not taking it, and it is a judgement about cost, not
a measurement.

**Corroborating evidence that the task is not in any general model.** Three independent points,
all off-the-shelf:

| model, no fine-tuning | `bucket_mean` |
|---|---|
| Qwen3-VL-8B (ours) | 0.2557 |
| **Qwen3.6-27B** — three generations newer | **0.2913** |
| **our fine-tuned 8B** | **0.5724** → **0.6496** at rung 21's A2 |

[[backbone-generation-is-not-the-lever]] puts it in one line: three generations of backbone buy
**+0.036**; our own fine-tuning buys **+0.317** — nearly 9×. The capability is not on the shelf.
It is in the 14,415 annotations.

## Why the probe had to exist at all

The rung-09 generator receives the gold and writes reasoning that *derives* it, so **a perception
failure cannot appear as a wrong answer** — it appears as `<evidence>` describing a scene that is
not there, with a correct `<answer>` by construction. Nothing downstream catches it: the answer
matches by construction and the judge-mirror is a text model that cannot see the frame either.

⚠️ That mechanism is **not specific to this teacher**. Any gold-anchored trace hides a perception
failure inside a correct-looking answer, including a STaR-style self-generated one filtered on
correctness — correctness is exactly what is guaranteed. Worth knowing before the next variant is
proposed.

## The result is not an engine failure — checked, because it looks like one

rung 23a once "completed" at `bucket_mean` 0.0000 because every call hit a missing kernel, and a
silent total failure is indistinguishable from incapacity. So the raw outputs were read before the
verdict was accepted. They are **well-formed**: `Specimen`, `Clip`, `0`, `1` — correct format, no
errors, no prose. The 32B answered, and was wrong. On `number` it repeatedly answers **0**; on
`fo_class` it collapses toward **`Specimen`**.

## ⚠️ A sampling defect, recorded because it is real

The pre-registration says *"stratified over `answer_format` × dataset"*. It was not: all **200/200
questions are `heico`**, and only two formats (`number` 101, `fo_class` 99). `margin_ID` is
therefore `NaN` — **the ID half was never measured**.

🔑 **The verdict survives it structurally.** The rule is a conjunction — *< 0 on either side ⇒
STOP* — and OOD fails by 21.5 points. No ID value could rescue it. The defect bounds what else we
can say (nothing about ID), not whether the rule fired.

Two bugs are open and neither touches the number: `select_blind_probe` does not stratify by
dataset, and `register_run()` is called with a duplicate `run_dir` argument, which is why the
notebook exits 1 *after* printing the verdict.

## What it costs and what it saves

Roadmap **phase 1 is answered and phase 3 is closed**: the 2,000 cold-start scaffolds, the
suppress arm, and the SCALe loss that only existed to serve them. Bought for ~30 minutes of
inference on one pod.

**The perceptual route is the only live branch** — consistent with [[roadmap-fork-points-at-phase4]]
(the vision tower is not saturated) and [[margin-is-vision-not-phrasing]] (the model invents
classes when allowed to enumerate). ⚠️ Being the only branch left does not make it good: phase 4
still owes an answer to v05's 5.8× negative before it is worth building.
