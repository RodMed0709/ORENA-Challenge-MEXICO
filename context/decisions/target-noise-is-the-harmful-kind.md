---
question: Why did our input-side data lever return a null and our label-side data lever return a null too — and what does that predict about the levers we have not run yet?
verdict: THEY ARE NOT THE SAME NULL. Noise on the INPUT is near-harmless and sometimes regularising (+2 pts measured on Qwen2-QA); noise on the TARGET is the single most damaging thing you can do to a fine-tune (21% average degradation). Rung 14 perturbed inputs and rung 18's L1 minted imperfect labels — the literature predicts exactly the two outcomes we got. Every pseudo-label route we have proposed is target-side
status: LITERATURE (external), explaining OUR OWN measured results retrospectively
date: 2026-07-29
measured_in: arXiv 2604.12469v1 Tables 2, 5, 6 — cross-read against experiments/14-appearance-aug/ and 18-count-aug/
---

# Decision: the damage is on the TARGET side, and every pseudo-label plan we have is target-side

- **Status:** LITERATURE · 2026-07-29 · zero GPU.
- **Applies when:** designing any lever that **creates or derives labels** — minted zeros,
  pseudo-masks, SAM 2 centroids, generated traces, synthetic counts. Also when reading rungs 14
  and 15 as "data doesn't help".

## The distinction we never made

We have been classifying our data levers by *what they add* (rows, wordings, augmentations). The
literature classifies them by *which side of the pair is corrupted*, and that turns out to be the
axis that predicts the result.

arXiv 2604.12469v1, three noise types across three models and three tasks:

| noise | what it corrupts | measured effect |
|---|---|---|
| **Label flip (LF)** | the **target** only; input untouched | **the most harmful, always.** ~21% average degradation |
| **Typographical (TN)** | the **input** (character perturbations, 10% of words) | near-neutral, **sometimes improves** |
| **Grammatical (GN)** | the **input** (agreement, articles, ~15% of words) | near-neutral, **sometimes improves** |

The cleanest cell is Qwen2 on QA — the closest of their three tasks to ours (Table 2, p.5):

| condition | F1 |
|---|---|
| clean fine-tune | 68.00 |
| **label flip 40%** | **49.72** |
| typo 20 / 30 / 40% | **70.01 / 69.63 / 69.62** |
| grammatical 20 / 30 / 40% | **70.71 / 70.51 / 70.63** |

**Corrupting 40% of the inputs beat the clean run by ~2 points. Corrupting 40% of the targets cost
18.** They read the input-side gain as a regularisation effect. The paper's cited precedent agrees
(Zhu et al. 2024): in MT fine-tuning, **target-side noise is substantially more damaging than
source-side**.

## 🔑 This retro-predicts two of our own rungs, and neither was bad luck

| our rung | what it really was | the prediction | what we measured |
|---|---|---|---|
| **14 — appearance-aug** | colour / white-balance perturbation of the **image** = **input-side** | neutral, possibly mild gain | **statistical null** ✓ |
| **18 — L1 minted zeros** | rows whose **`number` target** is derived under a closure assumption with a **measured 8.4% under-naming error** = **target-side** | the damaging kind | **null, and ID ALL significantly worse in its own paired CI** ✓ |

**We were adding the harmless kind of noise and the harmful kind, and we got exactly what the
literature says to expect from each.** [[epoch-matched-control]] closed those rungs on the
instrument; this explains *why* they landed where they did, which the instrument cannot.

⚠️ Rung 18 was also run at `lr 2e-5` ([[undertrained-was-real]]), so its null is doubly confounded.
This note does not rescue it — it says the label-quality half of the problem would still be there
at 1e-4.

## 🔴 What this constrains going forward

**Every generative-label route currently on the table is target-side**, i.e. in the damaging
category:

- **minted zeros** (rung 18 L1) — 8.4% of the ABSENT labels are wrong, always in the direction of
  zero, on a model already carrying a −2.07 count bias;
- **SAM 2 pseudo-masks / centroids** ([[covt-reduced-sam-route]], rung 25) — a mask that is wrong is
  a wrong *target*;
- **CoA reverse-generated traces** ([[coa-sft-published-null]]) — the trace IS the target;
- **synthetic counting rows** ([[synthetic-counting-reconciled]] — already DISCARDED).

⇒ **Rule adopted: a lever that derives targets must carry a measured label-error rate before it is
allowed to train, and that rate is budgeted as damage, not as an acceptable rounding.** Rung 18's
L1 is the model to copy in *method* — it measured its own 8.4% against an independent channel (the
1,230 co-occurrence binaries) rather than assuming it — and the model to avoid in *conclusion*.

🟢 **And the mirror of the rule: input-side transforms are cheap to try and hard to hurt with.**
That is not a licence to re-open rung 12 (dominated, and the effect was −0.056 monotone at
inference), but it does mean an input-side arm carries far less risk than a label-side one of the
same size — the opposite of how we have been pricing them.

## ⚠️ Transfer limits

- **Text only.** GPT-2 / Qwen2-0.5B / Llama-2-7B; sentiment, SQuAD QA, EN-FR MT. **No VLM, no
  vision encoder.** "Input-side" for them is characters; for us it is pixels, and a colour
  transform is not a typo. The *direction* transfers; the magnitude does not.
- Their noise floor is **20%** — they never test the 5–10% band our derived labels live in.
- The regularisation gain is not free money: it appears at 20–40% input corruption on tasks where
  the input is highly redundant.

## A third finding worth carrying

**Section 6 (robust vs vulnerable):** samples whose predictions do **not** change still show
internal representational distortion, and the authors conclude that *"task-level performance alone
underestimates the true extent of noise-induced representational change."*

That is direct published support for the caveat written into rung 24's plan — a component can
degrade while the headline looks healthy because the rest of the model compensates. It is also a
reason to keep `class_f1_report` and the per-stratum reads: a stable `bucket_mean` is not evidence
that nothing broke.

## Sources

- arXiv 2604.12469v1 — Table 2 (p.5), Tables 5–6 (p.15), §5.1, §6; Zhu et al. 2024 as cited p.3
- [[seed-variance-is-small-when-clean]] (same paper, the other finding) ·
  [[epoch-matched-control]] · [[undertrained-was-real]] · [[covt-reduced-sam-route]] ·
  [[coa-sft-published-null]] · [[synthetic-counting-reconciled]]
