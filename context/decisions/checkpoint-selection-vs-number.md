# Finding: training ERASES `number`, and acc_OOD selection picks the checkpoint that erased more

- **Status:** MEASURED (zero GPU, from artifacts we already had) · 2026-07-18
- **Applies when:** choosing a checkpoint, choosing how many epochs to train, or reading
  `acc_OOD` as if it tracked the thing we are trying to fix.
- ⚠️ **This is a measurement that QUALIFIES `context/RULES.md` §6** ("checkpoint selection
  by acc_OOD, per-epoch"). It does not overturn it — §6 exists for a real reason (OOD is
  ~50% of the score). It shows §6 has a cost nobody had priced. **Rule change, if any, is
  Rodrigo's call; this file is the measurement his process asks for first.**

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

2. **`acc_OOD` selection systematically picks against `number`.** In BOTH arms the best
   checkpoint for `number` is epoch 1, and both selected epoch 2:

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
1. **Evaluate rung 06's `checkpoint-860` (epoch 1) on the full 6252** — the merged weights
   are already on volume `gf78k60nlt`, so this is **~24 min of GPU** (measured: full 6252
   eval = 21m34s inference + 2m54s judge on a 5090), not a training run. It answers
   whether epoch 1 is the better *model* overall, in which case we already own a better
   checkpoint and never trained for it.
2. Only then revisit epochs/selection. Any change to the selection criterion is a second
   variable against rung 02 and must be run as its own single-variable comparison.
3. `vit_lr` drops down the queue. It was never supported by literature (VP-LoRA reports
   +2 on the visual path); it was supported by a decay that turns out not to be the ViT's.

## Sources
- `experiments/06-vit-lora/_tools/epoch_trajectory.py` — the diagnostic, zero GPU.
- `runs/{02_lora_sft_v1,06_vit_lora_v1}/sel/ood_checkpoint-*/results.csv` — per-epoch evals.
- Floor + margin via `frame.metrics.template_floor` (RULES §1); gold coverage asserted.
- Related: [[vit-lora-partial]] (the rung this came out of), [[eval-canonical]].
- Qualifies: `context/RULES.md` §6.
