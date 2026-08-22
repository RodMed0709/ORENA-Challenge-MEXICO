---
question: Rung 45 asked whether the 27B gains from rung 42's merged corpus. It does not — it is harmed. So does the 27B line continue as a model we might SHIP?
verdict: NO as a shipped candidate, YES as a teacher. The merged corpus HARMS the 27B (paired ALL_ID -0.1366 [-0.1906, -0.0788], four cells vetoed including the pre-declared veto cell), while the SAME corpus gained the 8B +0.0402. And on the clean 1,283 the 8B leads every 27B arm we have by a margin larger than every confound combined. 🔴 The closure is on TIME AND RESOURCES, not on proof — the 27B has NEVER been tuned for its own size, so "the 27B backbone loses" is STILL NOT ESTABLISHED
status: MEASURED
status_note: 'was `MEASURED (data axis) + TEAM CALL (the stop)`, which the indexer rejects — it accepts one bare value. Normalised 2026-08-22 so `MEASURED.md` can be rebuilt; the note''s content and its two-part reading are unchanged, and the TEAM CALL half is stated in the verdict above.'
caveat: the backbone question is UNANSWERED and this note must not be cited as answering it — see [[gen36-fails-the-8b-recipe-not-the-backbone-test]]. Every 27B arm ever run inherited hyperparameters swept on an 8B
date: 2026-08-17
measured_in: experiments/45-gen36-data-and-reg/ — RESULTS.csv, RESULTS_paired_ci.csv, RESULTS_class_f1.csv, RESULTS_bridge_ci.csv, RESULTS_bridge_flips.csv · experiments/19-external-count/RESULTS_19a_*.json
question_derived: false
---

# Decision: the 27B stops being a candidate and becomes a teacher

- **Status:** the data axis is MEASURED and closed. The stop is a **team call under a deadline**.
- **Subject:** `Qwen/Qwen3.6-27B` + NF4, arms `45_R00_v1` (rung 18's 14,415) and `45_R0_v1`
  (rung 42's merged 19,384). One variable — `diff_vs_R00` lists exactly two fields, `arm` and
  `corpus`. R1 (`lora_dropout` 0.1) was **never run**.
- **Control:** R00, run here, at this precision, on this hardware. Rung 40 is context, never control.

## 1. The measured half — the merged corpus harms this backbone

Scored on rung 42's 8 held-out videos, 1,283 questions, the only set clean for both arms.

| | R00 · 14,415 | R0 · merged | Δ |
|---|---|---|---|
| `bucket_mean` | 0.4315 | 0.3170 | **−0.1145** |
| `object_recognition_ID` | 0.5680 | 0.3120 | −0.2560 |
| `macro_f1_ID` | 0.6876 | 0.3465 | −0.3411 |
| train loss ep1 | 0.0694 | **0.0589** | — |

Paired, video-clustered: `ALL_ID` **−0.1366 [−0.1906, −0.0788]**, `object_recognition_ID`
−0.2439, `object_recognition_OOD` −0.2047, `ALL_OOD` −0.0913 — **four cells veto**. Both
`aggregation` cells are null. **`object_recognition` is the cell PLAN §6 named in advance as this
rung's modal failure**, before a number existed.

**Three things make the negative harder, not softer:**

- **R0's train loss was LOWER.** Selecting on loss would have selected the worse arm. PLAN §6
  ring-fenced this in advance: the screen can kill an arm, never crown one.
- **The OOD cells structurally FAVOUR R0** — rung 42's merge put 8 of the 10 heico test videos in
  its training set, so that procedure is *seen* for R0 and unseen for R00 — and R0 loses them anyway.
- **The failure has a shape.** R0 emits **zero** illegal `fo_class` tokens against R00's 31 and its
  macro-F1 still halves: it stopped guessing outside the vocabulary and collapsed onto the head
  class, answering `Clip` to almost everything. It bet the mode.

⇒ **The +0.0521 rung 42 bought the 8B was not its data half**, for this backbone. PLAN §6's
pre-registered NO-GO was "R0 ≈ R00, it does not transfer"; the result is stronger than the null it
prepared for. ⇒ supersedes, for the 27B only, any read of [[merged-corpus-buys-the-id-half]] as a
corpus that generalises across backbones. **It gained the 8B +0.0402 and cost the 27B −0.1145 —
same rows, opposite sign.**

## 2. Why the line stops — the arithmetic on the only clean common ground

All on the same 1,283, same 14,415-row corpus, nobody with test videos in training:

| | precision | 1,283 held-out |
|---|---|---|
| **A2 · 8B** | bf16 | **0.6342** |
| **`conn4e5` · 27B** | bf16 | **~0.558** |
| **R00 · 27B** | NF4 | **0.4315** |

*(`conn4e5`'s figure is raw accuracy on that subset. On this eval set raw accuracy and
`bucket_mean` coincide almost exactly — R00 is 0.4310 vs 0.4315 — so the approximation holds.
A2's 0.6342 is derived: rung 42's ep4 0.6744 minus its archived `d_bucket_mean_vs_A2_ep3` +0.0402.)*

**Both halves of the "keep going" bet die separately.**

- **"bf16 would beat it."** Quantization costs more than the headline: **−0.127** on this clean
  subset against the 6,252's −0.091. That part of the intuition was right. **But handing the full
  amount back leaves the 27B at ~0.558, still −0.076 behind the 8B on identical footing.**
  Quantization did not open the gap; it widened one that already existed.
- **"more data would beat it."** That is precisely what this rung measured: **−0.1145, HARM.**

And the ceiling: rung 42's 8B reaches **0.6744**. The best 27B imaginable — bf16, 14,415 — lands
**0.116 below it**, and the route to close that (more data) was just measured going the wrong way.

## 3. 🔴 What this does NOT establish, and must never be cited as establishing

**The 27B has NEVER been compared to the 8B on equal footing, and was never tuned for its own size.**

- Every 27B arm inherited hyperparameters from a sweep run **on an 8B**: lr 2e-4, r8, α32,
  dropout 0.0 all come off the 8B ladder. [[recipe-axis-is-the-learning-rate]] found 2e-4 optimal
  **for the 8B** and it was never re-searched at 27B.
- The 8B trains on **ms-swift** and the 27B on **Unsloth** — a framework confound rung 38's README
  calls *"fine for a candidate search, invalid for attribution"*.
- The only recipe rung ever done **for** the 27B (rung 40) swept two knobs: `alpha` and the
  connector LR. Two arms.
- `weight_decay` is **0.0 in `conn4e5` and 0.1 in R00** — found 2026-08-17, and it means even the
  bridge is not precision-only. See §5 of the rung's PLAN, corrected in the same commit.
- The bridge flips show **23.7 % of answers changed** with the loss spread across every format and
  35 of 38 videos, and rung 44 measured vLLM-vs-HF disagreeing on 6 % with *identical* weights — so
  even the decode path is a real, if bounded, term. NF4, `weight_decay` and the JPEG round-trip all
  degrade broadly and **none of them betrays itself in the pattern**.

⇒ **"The 27B loses" is measured. "The 27B backbone loses" is not, and this note does not claim
it.** This closes the line for **time and resources** — 15 days to pre-eval, ~12 h per arm, and
bf16 needs a rented pod — not because the backbone was fairly beaten.
⇒ extends [[gen36-fails-the-8b-recipe-not-the-backbone-test]], which already scoped its own
verdict to the recipe transplant. Nothing here narrows that scope further.

## 4. What the 27B becomes instead

**It counts better than the 8B**, on data outside the challenge, measured 2026-08-16
(`experiments/19-external-count/`):

| | 27B `conn4e5` | 8B `r42` |
|---|---|---|
| MISAW, n=2,080 | **0.3913** | 0.2048 |
| SurgSigma, n=1,105 | **0.4905** | 0.4027 |
| Spearman, SurgSigma | **0.6921** | 0.4965 |
| MAE, SurgSigma | **0.5692** | 0.7529 |

It wins on both datasets and on all three metrics — and the Spearman is the interesting one: it
does not merely get more right, it **orders quantities better**. `number` is 71 % of what we get
wrong.

⇒ That is an argument for **teaching**, not for shipping. The one demonstrated 27B advantage is a
capability that can be distilled into the model we actually send. The distillation bet now has two
teachers: SurgΣ's 309,123 human-annotated traces and a 27B that provably ranks counts better.
⇒ **Ship the 8B, distil the 27B into it.**

## What would reopen this

1. A recipe sweep run **at 27B scale** — LR and rank re-searched for the size, same framework as
   the 8B. That is the experiment nobody has run, and it is the only thing that would make
   "the 27B backbone loses" a measured claim.
2. The 8B line stalling below the podium with time left to spend.
3. A 27B-taught 8B beating an untaught one — which would prove the transfer works and might make
   a better teacher worth paying for.
