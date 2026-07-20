---
question: Does acc_OOD checkpoint selection choose against `number`?
verdict: NO — on the full 6252 epoch 2 wins; RULES §6 is CONFIRMED, not qualified
status: MEASURED
date: 2026-07-18
measured_in: experiments/06-vit-lora/06b_epoch1_eval.ipynb
withdrawn: the selection claim — it was measured on the OOD slice only
---
# Finding: training erodes `number` on OOD — in BOTH arms, so it is not the ViT

*(Original title claimed acc_OOD selection picks the worse checkpoint. **Retracted** — see bottom.)*

- **Status:** ⚠️ **PARTIALLY RETRACTED 2026-07-19** — the core measurement stands, the
  selection claim does NOT. See "Retraction" at the bottom **before** citing this file.
- **Status (original):** MEASURED (zero GPU, from artifacts we already had) · 2026-07-18
- **Applies when:** choosing a checkpoint, choosing how many epochs to train, or reading
  `acc_OOD` as if it tracked the thing we are trying to fix.
- ⚠️ **The claim that this QUALIFIES `context/RULES.md` §6 is RETRACTED.** Measured on the
  full 6252 (T7, 2026-07-19), §6 held up: acc_OOD selection picked the better checkpoint
  on every headline metric. **§6 needs no change. Do not cite this file against it.**

## Question
Rung 06's `number` came out worse than rung 02's, and the hypothesis on the table was
"`vit_lr` at the LLM's 2e-5 is degrading the pre-trained vision tower" — a hypothesis
worth ~7.5 h of GPU to test. Before spending it: does `number` actually degrade *as the
ViT trains*?

## What we sought
Both rungs saved a checkpoint per epoch and evaluated each one on the OOD videos during
selection (`runs/*/sel/ood_checkpoint-{860,1720,2580}/`). That is a per-epoch trajectory
for **both arms**, already computed. If the ViT LR is the culprit, `number` must decay in
the ViT-LoRA arm and NOT in the frozen-ViT arm.

Reproduce: `experiments/06-vit-lora/_tools/epoch_trajectory.py` (pure pandas, no GPU).

## What it gave us
`number` margin over the template-aware trivial floor, OOD slice (n=4000, floor 0.4691):

| epoch | rung 02 — ViT **FROZEN** | rung 06 — ViT **LoRA** |
|---|---|---|
| 1 | **+0.0317** | **+0.0151** |
| 2 | +0.0226 | +0.0128 |
| 3 | +0.0038 | **+0.0000** |

**Both arms decay, and the frozen-ViT arm decays MORE in absolute terms** (−0.0279 vs
−0.0151 from epoch 1 to 3).

## Verdict
**The `vit_lr` hypothesis is DISCONFIRMED as the cause of the decay.** `number` degrades
with training time while the vision tower is completely frozen, so the decay is not a
vision-tower effect. **Do not spend 7.5 h re-running with a lower `vit_lr` on this
rationale.** (What remains true, and is a different claim: ViT-LoRA sits *below* the
frozen arm at **every** epoch — a level penalty present from epoch 1, not a progressive
degradation.)

**Two larger findings fall out of it:**

1. **Training drives `number` to the trivial constant.** By epoch 3 rung 06 sits at
   **+0.0000** — literally indistinguishable from answering each template's modal answer.
   Rung 02 reaches +0.0038. More fine-tuning does not merely fail to help `number`; it
   actively removes what little skill is there. The bucket THE_MAP set as the target is
   being erased by the process meant to improve it.

2. ~~**`acc_OOD` selection systematically picks against `number`.**~~ 🔴 **RETRACTED — see
   bottom.** Measured on OOD only; on the full 6252 epoch 2 wins. Kept as written so the
   error is legible. In BOTH arms the best checkpoint for `number` **on the OOD slice** is
   epoch 1, and both selected epoch 2:

   ```
   rung 02   acc_OOD:  0.5833   0.5917 <- selected   0.5833
             number:  +0.0317 <- best  +0.0226       +0.0038

   rung 06   acc_OOD:  0.5785   0.6078 <- selected   0.6045
             number:  +0.0151 <- best  +0.0128       +0.0000
   ```

   `acc_OOD` rises while `number` falls, because acc_OOD is dominated by the buckets that
   already work. The data card already warned that `acc_OOD` flatters (the OOD floor is
   ~12 pts higher); this is the first measurement of what that costs **at the selection
   step**, which is where it actually bites — we have now shipped two rungs selected
   against our own target bucket.

## Next move (cheapest first)
1. ✅ **DONE (T7, 2026-07-19)** — evaluated epoch 1 of BOTH arms on the full 6252
   (50m51s on a 5090, `06b_epoch1_eval.ipynb`). **Answer: no, epoch 2 is better.** See the
   retraction at the bottom.
2. Only then revisit epochs/selection. Any change to the selection criterion is a second
   variable against rung 02 and must be run as its own single-variable comparison.
3. `vit_lr` drops down the queue. It was never supported by literature (VP-LoRA reports
   +2 on the visual path); it was supported by a decay that turns out not to be the ViT's.

## Sources
- `experiments/06-vit-lora/_tools/epoch_trajectory.py` — the diagnostic, zero GPU.
- `runs/{02_lora_sft_v1,06_vit_lora_v1}/sel/ood_checkpoint-*/results.csv` — per-epoch evals.
- Floor + margin via `frame.metrics.template_floor` (RULES §1); gold coverage asserted.
- Related: [[vit-lora-partial]] (the rung this came out of), [[eval-canonical]].
- ~~Qualifies~~ **CONFIRMS** `context/RULES.md` §6 — see the retraction.
- T7 (the run that retracted it): `experiments/06-vit-lora/06b_epoch1_eval.ipynb`, `RESULTS_epoch1.csv`.


---

## ⚠️ Retraction (2026-07-19) — the selection claim was read off an OOD-only slice

T7 (`06b_epoch1_eval.ipynb`) evaluated epoch 1 of BOTH arms on the **full 6252**. The
result contradicts this file's second finding:

| | `bucket_mean` | `margin_ID` | `margin_OOD` | `number` ID | `number` OOD |
|---|---|---|---|---|---|
| rung 02 ep1 | 0.5282 | 0.1505 | 0.1235 | +0.0495 | **+0.0317** |
| rung 02 **ep2** ✅ | **0.5486** | **0.1838** | **0.1320** | **+0.0911** | +0.0226 |
| rung 06 ep1 | 0.5345 | 0.1692 | 0.1188 | +0.0781 | **+0.0151** |
| rung 06 **ep2** ✅ | **0.5667** | **0.2074** | **0.1480** | **+0.0859** | +0.0128 |

**What survives.** `number`'s margin decays across epochs **on OOD**, in both arms —
including the arm whose ViT was frozen. The OOD numbers reproduced exactly (+0.0317 /
+0.0226 / +0.0151 / +0.0128). The `vit_lr` hypothesis stays disconfirmed and the 7.5 h
re-run stays unspent. That was this file's job and it did it.

**What does NOT survive.** "The best checkpoint for `number` is epoch 1 in BOTH arms" and
"acc_OOD selection picks against `number`". On the full set epoch 2 wins `number` on **ID**
by more than epoch 1 wins on **OOD** (rung 02: +0.042 vs +0.009; rung 06: +0.008 vs
+0.002), and epoch 2 also wins `bucket_mean`, `margin_ID` and `margin_OOD` in both arms.
**We did not own a better checkpoint.**

**The error, kept because it is the instructive part.** The per-epoch evidence came from
`runs/*/sel/`, which exists **only for OOD** — those evals were generated to select by
acc_OOD and were never run on ID. Reading them as if they described checkpoint quality is
the exact trap `RULES` §11 and the data card warn about (`acc_OOD` is not the model's
quality), committed inside a file written to prevent misreadings. **An OOD-only slice
cannot support a claim about a checkpoint; say which slice a number came from.**
