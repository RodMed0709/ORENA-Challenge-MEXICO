---
question: Can the model emit 0, is the capability missing or did our SFT destroy it, and should we mint synthetic zeros?
verdict: FORMAT-LOCALIZED — the fine-tuned model says "no" fluently (binary emit_absent 0.82) but NEVER says "0" (number 0.008/0.017). The base is NOT a counter-example: it is a zero ATTRACTOR that negates 83% of PRESENT cases, so the capability never existed in this domain and there is nothing to "restore". Minting is licensed, but only for `number`, and it is no longer the most valuable finding this probe produced
status: MEASURED
date: 2026-07-27
measured_in: experiments/16-count-probes/RESULTS_16a.csv (n=960, 240 frames, balanced ID/OOD x absent/present x number/binary)
---

# Decision: the missing zero is a FORMAT deficit, not a missing capability — and the probe found a bigger problem

- **Status:** MEASURED · 2026-07-27 · ~2 GPU-minutes of inference, no training
- **Applies when:** costing any lever whose theory is *"teach the model that zero/absence is an
  emittable answer"* — minted zero-counts, absence binaries, `fo_class → none`, abstention training.

## The question

Our training set contains **no negative example in any numeric or class format**: 0 of 2,495
per-class count golds are `0`, 0 of 2,628 total-instance golds are `0`, 0 of 8,969 `fo_class`
golds are `none`. Only `binary` carries negatives (1,008 `no` / 946 `yes`), and those are
co-occurrence questions about class *pairs*.

The proposed remedy — mint zero-count questions from the per-frame inventory — costs a training
run and carries ~8.4% one-directional label noise pointed *toward* zero, on a model already at a
−0.66 count bias. HoloCount (arXiv:2607.06420, Table 3) measures **Qwen3-VL-8B at 96.4%** on
absent-object questions, which raised a cheaper hypothesis: the capability ships in the backbone
and *our own SFT destroyed it*, in which case the cure is regularisation, not data.

## What was measured

Paired probe, same frames and same questions for both models. `base` =
`/workspace/models/qwen3-vl-8b`; `ft` = rung 06 **ep3** (`checkpoint-2580`, the current best,
`bucket_mean` 0.5724). Two arms — ABSENT / PRESENT — × two formats — `number` / `binary` —
balanced across ID and OOD. **240 frames, 960 questions.**

`emit_rate` = how often the model emits `0`/`no`. It is a property of the model's OUTPUT and is
therefore **noise-free**. `acc` compares against a closure-derived label that is ~8.4% wrong
(see below) and is read for direction only.

| model | cell | emit_absent | emit_present | hmean | sdk_invalid |
|---|---|---|---|---|---|
| base | ID `number` | 0.9417 | **0.7667** | 0.374 | 0.0000 |
| base | ID `binary` | 0.9500 | **0.7750** | 0.364 | 0.0000 |
| base | OOD `number` | 0.9167 | **0.8833** | 0.207 | 0.0000 |
| base | OOD `binary` | 0.9417 | **0.9000** | 0.181 | 0.0000 |
| **ft** | **ID `number`** | **0.0083** | 0.0000 | 0.017 | **0.8667** |
| ft | ID `binary` | **0.8167** | 0.1000 | **0.856** | 0.0583 |
| **ft** | **OOD `number`** | **0.0167** | 0.0000 | 0.033 | **0.8750** |
| ft | OOD `binary` | **0.8250** | 0.1167 | **0.853** | 0.1458 |

## What it gives us

### 1. 🔴 The base model is a ZERO ATTRACTOR, not a counter-example

Pooled, `base` emits `0`/`no` on **83.1% of PRESENT cases** — cases where the object *is* there.
Its 0.9375 on ABSENT is worthless: it is not discriminating, it is negating. Its harmonic mean
runs **0.18–0.37** against `ft`'s 0.85 on binary.

**HoloCount's 96.4% does not transfer to laparoscopic frames.** The capability never existed in
this domain, so there is nothing for regularisation to restore, and the "our SFT broke it"
hypothesis is dead. ⚠️ This is exactly what the PRESENT control arm was built to catch — without
it the ABSENT column alone would have read as *"base has it, we destroyed it"* and sent rung 16
to repair something that never existed.

### 2. The deficit is FORMAT-LOCALIZED, and the training data explains it exactly

`ft` says **"no"** fluently — 0.82 on ABSENT with only 0.10–0.12 false-negatives on PRESENT — and
**never says "0"**: 1 case in 120 (ID), 2 in 120 (OOD). It answers `1.` instead.

That maps one-to-one onto the supervision: `binary` has **1,008 `no` golds** to learn from;
`number` has **zero zeros in 2,495 examples**. The model learned absence in the format where
absence was taught, and only there.

⇒ **Minting is licensed for `number` only.** The `binary`/`fo_class` half of the original plan is
unnecessary — the model already has that concept. This halves the scope, the cost and the label
noise of the minting proposal. Dose and sampling per [[the-negatives-dosing]] remain: 3:1–2:1,
adversarially sampled toward `Clip` and the never-seen classes, verified against the 1,008
co-occurrence binaries, scored on both axes.

### 3. 🔴 The bigger finding: fine-tuning introduced a FORMAT FRAGILITY the base does not have

On question phrasings outside the corpus templates, `ft` emits **`"1."`** — with a trailing
period — on **86.7% (ID) / 87.5% (OOD)** of `number` questions. `Number.verify` gates on
`str.strip().isdigit()`, so **`"1."` is auto-incorrect**. The base model's `sdk_invalid` rate is
**0.0000**.

This is not a counting error. It is a **scored formatting error that our own fine-tuning created**,
and it is invisible to every number we have reported, because the scored eval uses the corpus
templates where the model emits bare integers.

⚠️ **Scope, stated honestly:** this is measured on *our probe's* phrasing, not on the organizers'
hidden test. It does not touch rung 06's 0.5724. But `open-class-vocabulary.md` already records
that 70% of our questions lack the class list the spec says they carry — a possible train/test
regime mismatch — and the hidden test is generated by the organizers, not by us. If their
`number` phrasing differs from ours at all, this costs points outright, for a trailing period.

**Cheapest mitigation, no training:** the `answer_postprocess` hook already exists in
`frame.config` (rung 15) and already runs between generation and SDK verification. Stripping a
trailing `.` from a `number` answer is a one-line, flag-off-byte-identical change. It should be
measured as its own arm before anything else in rung 16.

## ⚠️ What this is NOT

- **Not a measurement of counting accuracy.** It measures whether `0` is emittable at all.
- **The ABSENT label is closure-derived**, from the assumption that the `fo_class` gold names
  every class present. Measured non-circularly against the co-occurrence binaries: **0.00% false
  positives**, **8.4% under-naming** (n=333; a second cross-check agrees at 10.7%). So ~1 in 12
  ABSENT labels is wrong, always in the same direction. `emit_rate` is immune to this; `acc` is not.
- **Not a statement about the hidden test's zero-rate.** [[the-negatives-dosing]]'s crossover
  arithmetic still applies: minting only beats not-minting if the true zero-rate exceeds ~4–6%,
  and we have no estimate of it. This probe says the model *cannot* answer such a question; it
  does not say the question will be asked.

## Sources

- `experiments/16-count-probes/RESULTS_16a.csv` + `runs/16a_zero_probe_v1/` (rows, paired, score)
- HoloCount, arXiv:2607.06420 Table 3 (the 96.4% reference this probe refutes for our domain)
- LRV-Instruction, arXiv:2306.14565 Table 7 (the dosing curve, if minting proceeds)
- POROver, arXiv:2410.12999 (the two-axis scoring discipline the PRESENT arm implements)
