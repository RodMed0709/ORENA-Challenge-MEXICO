# Quick task 260727-jo9 — SUMMARY

**Status:** COMPLETE · **Date:** 2026-07-27 · **Branch:** `task/r3-rung16`
**Commits:** `e8a8219` (notebook) · `7d98489` (plan + blocker) · `d51e688` (results + verdict)

## What was asked

Evaluate rung 06's `checkpoint-2580` (epoch 3) on the full 6252-question FRAME eval set to
supply the epoch-matched control rungs 14 and 15 never had, then record the verdict.

## What was delivered

| Task | Outcome |
|---|---|
| 1 — build `06c_epoch3_eval.ipynb` | ✅ `e8a8219`, 11 cells, `parameters`-tagged, single arm |
| 2 — run it on an RTX 5090 | ✅ smoke then full, 31 min, all gates green |
| 3 — record the verdict | ✅ `RESULTS_epoch3.csv`, `RESULTS_epoch3_paired_ci.csv`, README, decision note, NOW.md, ledger |

**Result: rung 06 ep3 `bucket_mean` = 0.5724** (`acc_ID` 0.5595, `acc_OOD` 0.6045,
`margin_ID` +0.2225, `margin_OOD` +0.1448). Gold coverage 6252/6252; hybrid cross-check
against the vendor scorer agrees (0.5724 vs 0.5724).

**The pre-registered branch that fired:** ≥ 0.5699 → **rung 15 dies outright**. Against the
epoch-matched control, rung 15 is +0.0017 ID / −0.0078 OOD, **0 of 6** paired cells excluding
zero. Rung 14 is now the only rung with a significant cell — `ID ALL −0.0239 [−0.0433, −0.0035]`,
significantly **worse** than the control.

**The second clause did NOT fire cleanly.** "We may gain a free checkpoint" is true on the
headline (+0.0057 over ep2, ladder's best) and false on the statistics (0/6 cells exclude zero).
The shipped checkpoint does not move.

**The finding that outlives the adjudication:** at ep3, `number` on OOD scores exactly the
trivial floor to 16 digits (0.46907993966817496) — rung 05's black-image signature. The OOD
counting margin decays monotonically across epochs (+0.0151 → +0.0128 → 0.0000) while ID
counting climbs (+0.0781 → +0.0859 → +0.1068).

## Deviations from the plan, and why

1. **Executed inline, not via a worktree-isolated `gsd-executor`.** Task 2 creates a billing GPU
   pod and writes to a shared network volume — global side effects that must not run unattended
   in an isolated worktree.
2. **GPU stock blocked the run mid-task.** EU-RO-1 had no RTX 5090 and the network volume pins
   the DC. Another card was refused on method: the measured cross-GPU drift (±0.003 on
   `bucket_mean`) is the size of the difference being adjudicated. The user chose to wait, then
   provisioned a 5090 themselves.
3. **The pod's `sshd` never came up** (port refused) while Jupyter answered. Shell access went
   through the Jupyter terminal websocket instead of reconfiguring the pod.
4. **Optional step 5 (re-evaluating ep2 on the same card) was skipped.** The primary result did
   not depend on it and the pod was billing.

## Known issues surfaced, NOT fixed here

- `frame.ledger._discover_stratified` globs `runs/**/stratified.json` recursively and tier 1 does
  not dedup → two `stratified.json` under one run dir produce two identical rows. Triggered by the
  untracked stray `experiments/02-lora-sft/runs/02_lora_sft_v1/eval_best/`, which was **parked and
  restored, never deleted**.
- The committed ledger still carries 8 phantom rows (rung 10 ×3, rung 12 ×4). Root fix is on
  `task/audit-rung12`, unmerged. The rebuild here changed neither count.
- `merged/checkpoint-2580` (17 GB) was left on the volume: the notebook only reclaims a merge it
  created itself, and the smoke created this one. It is the ep3 weights, kept deliberately.

## Cost

One pod (`wmlmxst3jcf68l`, RTX 5090, $0.99/hr), ~1 h wall clock, **~$1**. Pod stopped; all pods
confirmed EXITED.

## Standing caveat

No run has ever been repeated with a different seed. Every delta above sits inside a variance band
that has never been measured. The user declined seed repeats this session — recorded, not overturned.
