# Rung 47 — `C_epochs`: was rung 42's +0.0402 the corpus or the epochs?

**The ladder.** Rung 21 designed arm `C_epochs` (single variable `--num_train_epochs`, off
A2 `21_lr_2e4_v1`), VRAM-probed it in `RESULTS_vram_C_epochs.json`, and never trained it.
Rung 42 then changed the corpus **and** ran two epochs the control never ran, won
**+0.0402**, and shipped as submission 03. Rung 47 trains A2's own corpus (14,415 rows) to
5 epochs on UNAM so that `ep4 − ep4` has the corpus as its only difference.

The bundle rung 42 never unpacked:

| comparison | Δ `bucket_mean` |
|---|---|
| rung 42 ep3 − A2 ep3 (matched epoch) | **−0.0079** — its corpus *loses* |
| rung 42 ep4 − A2 ep3 (what shipped) | **+0.0402** |
| rung 42 ep4 − ep3 (its own epoch effect) | **+0.0482** |

⇒ the whole gain sits at epoch 4, an epoch the control never ran.

## The arm

`47_a2_ep5_v1` — UNAM, GPU 0, `tmux leo-rung47`, launched 2026-08-18 00:33:56, ~20 h.
`~/storage/rung47/OWNER.md` is the do-not-kill note. Its `diff_vs_A2.json` is
`{"--num_train_epochs": ["3", "5"]}` and nothing else — written by the launcher, so the
single-variable claim is evidence, not assertion.

`--save_strategy epoch` over 4,505 steps ⇒ 901 per epoch.

| epoch | step | role |
|---|---|---|
| 1, 2 | 901, 1802 | trajectory — scored only when free |
| **3** | **2703** | 🔑 **the control.** Must reproduce A2's **0.6342** within ±0.01 |
| **4** | **3604** | 🎯 **the answer.** Against rung 42's ep4 **0.6744** |
| 5 | 4505 | in rung 42 this epoch already fell (0.6592) |

## Scoring

`01_eval_epochs.ipynb` → `_tools/eval_arm47.py`, which imports rung 45's `eval_arm45.py`
rather than copying it. Eval set is rung 42's **8 held-out videos / 1,283 questions**
(`RESULTS_split_42.json`) — the only set on which rung 42 is legal, and A2's corpus
promoted no test video, so it is clean for this arm too.

Runs in `~/storage/envs/orena-train` on **GPU 1** while the arm trains on GPU 0. Read-only
gates verified 2026-08-18 before any GPU spend: judge resolves offline from
`~/storage/hf_cache`, 6,252 test items load, split gate 8 videos / 1,283 q, frames-cache
gate 1,283/1,283.

The sweep is **incremental**: `score_epoch` reuses an existing `results.csv`, so scoring
ep3 now and re-running for ep4 later costs zero GPU on ep3. `RESULTS.csv` is written only
when the sweep contains **both** decisive epochs — a partial write would record a
`selected` column that means "best of what had trained by then", and the next run would
append duplicates on top of it.

## 🔴 ep3 controls three things at once

A2's 0.6342 was produced on a RunPod pod. Between it and this rung: `transformers`
**4.57 → 5.12.1**, frames **decord-off-`.mp4` → JPEG q95 cache** (UNAM holds every test
frame and zero `.mp4`, so the cache is forced — rung 45's declared confound), and
**A100/L40S → RTX 6000 Ada**. A green ep3 bounds all three jointly; a red ep3 does *not*
say which one moved.

## 🟢 The read that survives a red control

`47_ep4 − 42_ep4` crosses the two stacks. The epoch effect **inside** each arm does not:

    rung 42, merged corpus:  ep4 − ep3 = +0.0482
    rung 47, A2's corpus:    ep4 − ep3 = (this rung)

Both are internal to one arm on one stack, so stack, JPEG round-trip and GPU cancel inside
each. A rung-47 rise of ≈ +0.048 ⇒ the epoch effect is corpus-independent ⇒ rung 42's
advantage was **epochs**. This is the primary read.

## 🔴 A2's per-question archive no longer exists

It lived on the pod at `repo_rodri/.../21_lr_2e4_v1/ep3_full/`. Settled 2026-08-18 by a
**full-bucket S3 scan** plus UNAM, which has no `results.csv` outside rung 45's.
**The ep3 control is a scalar check with no paired CI.**

⚠️ The scan turned up a convincing decoy — record it before re-chasing it:
`tmp/leo_backup_20260806/step5_seed42/full/greedy/results.csv` is A2 greedy, but it is
rung 10's stratified subsample: **600 rows over 91 videos, zero of the 8 held-out ones**,
~0.83 accuracy. Overlap with the 1,283 is **0**. Also worth knowing for other rungs:
`tmp/leo_chain/critico/` and `tmp/leo_backup_*/` hold `runs/` trees the `repo_*/` prefixes
cannot, but mostly `summary.csv` / `stratified.json`, rarely per-question `results.csv`.
Rung 42's archive *is* in S3 at `evidence_42/ep{1..5}_full/results.csv` (155 KB each,
verified present) and the notebook fetches it before the GPU — rung 45's scar.

## Files

| file | what |
|---|---|
| `01_eval_epochs.ipynb` | the sweep — gates, control fetch, merge → eval → reclaim, verdicts |
| `_tools/eval_arm47.py` | importable library; arm/control tables, gates, DiD, paired CI |
| `RESULTS.csv` | per-epoch cells (written only on a full sweep with ep3 **and** ep4) |
| `RESULTS_class_f1.csv` | class-balanced F1 on `fo_class` (RULES §9b) |
| `RESULTS_paired_ci.csv` | video-clustered paired CI vs rung 42's same epoch |
| `RESULTS_epochs_vs_corpus.json` | the ledger: control verdict, DiD, declared confounds |
