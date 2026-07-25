# CONTEXT — 17 generator perception probe

> **Status: PRE-REGISTERED, UNRUN.** Everything below is fixed before any number exists.
> Owner: Rodrigo · branch `task/r2-lit-levers` · zero training, inference only.

## Objective

One question, and it gates the whole CoA line:

> **Does `Qwen3-VL-32B-Instruct` actually perceive these surgical scenes, or is it at the
> noise floor like our own 8B zero-shot?**

The generator's job in rung 09 is to write `<description>/<evidence>/<thought>` that *derive*
a gold it is handed. The gold is given, never guessed — so **a perception failure cannot show
up as a wrong answer.** It shows up as evidence that describes a scene which is not there,
attached to a correct answer. Training the 8B on that teaches it to invent plausible
justifications for right answers: a failure mode that is invisible in every automatic check we
have, because `<answer> == gold` by construction.

⚠️ The Qwen judge-mirror **cannot** catch it either — it is a text model and itself a
below-floor perceiver of these frames. Nothing downstream catches it. That is why this probe
exists and why it runs before the 2k generation, not after.

## Method

The generator answers the FRAME question **directly**: no gold, no scaffold, no sibling
fact-sheet. Pure perception, scored exactly like any other model in this repo.

- Engine: `experiments/09-coa-sft/_tools/gen_onpod.py`, stage `blind` — already written and
  already used; this rung adds the canonical scoring it lacked, not a second generator.
- Sample: stratified over `answer_format` × dataset, frozen with a sha256 sidecar via
  `frame.subsample` so the set cannot drift between attempts.
- Scoring: **`frame.metrics.stratified_report` only** (RULES §1). Read **margin over the
  template-aware floor**, never raw accuracy — `acc_OOD` flatters because the OOD floor sits
  ~12 points higher, and a generator that beats raw accuracy while sitting at the floor has
  told us nothing.

## The comparisons that make the number readable

A raw accuracy for the 32B alone means nothing. It is read against three fixed references:

| reference | `bucket_mean` | what it means |
|---|---|---|
| trivial floor | — | answer each template's modal answer |
| **00-baseline** — our 8B, zero-shot | **0.2557** | margin −0.088 ID / −0.191 OOD: **below the floor everywhere** |
| **06-vit-lora** — our fine-tuned 8B | **0.5667** | margin +0.207 ID / +0.148 OOD |

## Decision rule (pre-registered)

| outcome | reading | consequence |
|---|---|---|
| margin **> 0** in BOTH ID and OOD | the 32B genuinely perceives | 🟢 GO — generate the 2k cold-start scaffolds |
| margin **≈ 0** (CI crosses 0) either side | indistinguishable from the trivial constant | 🟡 The scaffolds are gold-anchored prose, not evidence. Generate only with the human eyeball gate, and never trust `<evidence>` as a training signal on its own |
| margin **< 0** on either side | below floor, like our own zero-shot | 🔴 **STOP.** A below-floor teacher cannot supervise perception. Switch generator or drop the vision-scaffold line |

⚠️ **Zero-shot is a poor instrument and this cuts both ways.** A general VLM answering a
domain-specific closed-format question fails on *task format* as well as on perception, so a
weak score under-states perception. Read a NEGATIVE as decisive (below floor means it cannot
teach) and a POSITIVE as directional (it perceives *something*). This asymmetry is
pre-registered so a mediocre number cannot be argued either way after the fact.

## Cost

Inference only. ~200 questions, one model load. No training, no checkpoint, nothing merged.
🔴 The 32B is **63 GB in bf16** — it does NOT co-reside with an 8B LoRA training run on a
96 GB card (63 + ~30 + activations > 96). This rung is queued after the wave-R2 rungs finish,
never run beside them.

## Scope deliberately excluded (v2)

- **The sibling fact-sheet.** Measured on the 13,748 train rows: only **2,210 frames (38 % of
  rows) have a sibling**; **62 % of questions are alone on their frame.** It is the most complex
  part of the generator and it reaches the minority of rows. Deferred to v2 on purpose.
- Prompt tuning. The prompt is **frozen at v4**. Four versions were written without a
  measurement between them; the adversarial review said stop, and it was right. This probe is
  the measurement that must come before a v5 exists.

## What this rung does NOT decide

It does not decide whether CoA-format SFT helps. That is already answered in the literature and
recorded in [[coa-sft-published-null]]: scaffold-SFT **without** RL is a wash (62.0 vs 65.7 on
EndoVis2018) and the +18 belongs to RLVR. This probe decides only whether the teacher can see —
i.e. whether the cold-start data is worth generating at all, for the RLVR rung that would follow.
