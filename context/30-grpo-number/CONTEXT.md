# context/30-grpo-number — GRPO on `number`

> Curated context for rung 30. Artifacts live in `experiments/30-grpo-number/`; the
> pre-registration and the smoke findings are in that folder's `README.md`.
> Last updated: **2026-08-08**.

## RESUME HERE — the state a fresh session must load first

## 🔴 THE RUNG IS CLOSED — NO-GO. Read `experiments/30-grpo-number/README.md` §RESULT.

`aggregation_ID` **0.4932 (GRPO) vs 0.5026 (SFT control) = −0.0094**, full 6,252-question eval,
both arms `rc=0`. S8(1) requires the primary CI to exclude zero *in the arm's favour*; the point
estimate is negative, so it fails without a CI, and the gap is an order below the MDE of 0.036.
No further GPU is owed: the pre-registration fixed the verdict at `checkpoint-600` with no
fallback, so the collapse sweep cannot rescue a failed primary.

**Pod `vugto0zhhtm97z` is EXITED** — the unattended chain stopped it on completion, at the
operator's explicit instruction, 22:01 UTC 2026-08-08. ⚠️ A stopped pod holds no GPU reservation
and a restart may fail with *"not enough free GPUs on the host machine"*; that already happened
once. Everything survives on network volume `gf78k60nlt`, reachable **without any pod** over the
S3 API (`boto3`, endpoint `RUNPOD_S3_ENDPOINT`, bucket = volume id) — note `HeadObject` returns
403 there, so use `get_object`, not `download_file`.

**Env:** `source /workspace/envs/infer/bin/activate`. `ms_swift==4.4.1`, `transformers==4.57.6`,
`torch==2.8.0+cu128`. The only package added for this rung is `msgspec==0.21.1`.
**Repo checkout:** `/workspace/repo_rodri` (sync with `git fetch origin && git reset --hard origin/main`).

### What is done

| | |
|---|---|
| GRPO API mapped, source vendored | ✅ `experiments/30-grpo-number/RESULTS_msswift_source.txt` |
| `number` subset built | ✅ 4,436 train / 493 holdout, from A2's exact corpus |
| Reward plugin | ✅ `_models/grpo_reward.py`, 9/9 unit cases, both guards raise |
| Pre-registration under S8/R9 | ✅ committed **before** the run (`d770ccd`) |
| Smoke, 10 steps | ✅ rc=0, and it caught the reference-policy trap |
| **GRPO full, 600 steps** | ✅ **rc=0**, 2h40 at 15.9 s/it, **6 checkpoints written** |
| First SFT control attempt | 🔴 **VOID — trained on nothing.** See the trap below. Kept as evidence at `runs/30_grpo_v1_control_full_VOID_no_gradient` |
| Control fix + smoke | ✅ `f5aec95`; smoke 10/10 steps with gradient, `loss_mean` 0.276 |
| **Step-matched SFT control, relaunched** | ✅ **rc=0** 20:06 UTC 2026-08-08, 1h45. `training_check`: **600/600 steps with gradient**, `loss_mean` 0.240, **6 checkpoints** — matching GRPO's six |
| **Independent review gate** | 🟡 **GO-WITH-FIXES** — `_docs/REVIEW_2026-08-08.md`. One blocking fix left (`_assert_learned` reports success when it cannot find its log); two earlier blockers were closed in flight by `fa811ab` and this file's own §2. The §IV.2 tension `fa811ab` referred to the review is ruled on in the writeup |

### 🔴 The second trap — a control that completes `rc=0` having moved no weight

Both arms read `runs/30_grpo_v1/grpo_train.jsonl`, and that file is GRPO-format by construction:
`build_number_subset.py` `to_grpo` **strips the assistant turn** and moves the gold to a `solution`
column, because in GRPO the model generates the answer and the reward reads `solution`.
`swift sft` masks every token that is not assistant content ⇒ no labels ⇒ no gradient.

The first control ran **1h20 on the 5090 with `'loss': 0.0` and `'grad_norm': 0.0` on 492 of 492
logged steps**, 18 minutes from finishing `rc=0` with an adapter that had learned nothing. Nothing
looked wrong from outside — checkpoints on disk, LR annealing on cue, plausible elapsed time.

⚠️ **A checkpoint diff does not detect it — measured, not assumed.** AdamW's decoupled
`weight_decay` (0.1) moves weights at zero gradient, so the dead `checkpoint-100` differs from A2
on **720/720 tensors**; a "did the checkpoint change?" guard would have passed it. Only the
magnitude separates them — `sum|Δ|` **1.34** dead vs **252.2** real (189×), `max|Δ|` 6.6e-07 vs
5.6e-05. The real run's 5.6e-05 is also the arithmetic sanity check: Adam steps ≈ `lr` 1e-6 over
100 steps ≈ 1e-4. **The instrument is the `grad_norm` log, not the weights.**

⚠️ **This is the same OUTCOME the engine already warned about for a resumed cosine** (lr 0.0 at
step 2703 ⇒ control learns nothing ⇒ GRPO wins for free), reached by a completely different route.
That is the argument for making it a property of the run instead of a note in prose:

* `_materialize_control_data` re-attaches the assistant turn, derived **from `grpo_train.jsonl`**
  so the seeded train/holdout split is preserved byte-for-byte.
* `_assert_supervised` — pre-flight, raises before the GPU is touched.
* `_assert_learned` — reads the run's own `grad_norm` log, raises below 50% nonzero steps.
  **Generalise this**: `rc=0` is not evidence that anything was learned, in any rung.

### What is NOT done — pick up here

0. ✅ **DONE — the verdict pair is scored and the rung is NO-GO** (see the top of this file).
   The three items below are what the pre-registration *would* have owed had the primary
   survived; they are recorded as not-owed rather than deleted, so the closure is auditable.

1. ⛔️ **Superseded — score the six checkpoints.** `runs/30_grpo_v1_grpo_full/ckpt/*/checkpoint-{100..600}`.
   This is not optional bookkeeping — with `beta=0` there is **no KL anchor**, so per-checkpoint
   evaluation *is* the collapse guard. Score `object_recognition_{ID,OOD}` at every one; it is
   the declared veto cell. Needs the GPU the control now holds.
2. ✅ **The control is trained and verified** (`runs/30_grpo_v1_control_full/`) — same rows, same
   600 optimizer steps, **same lr 1e-6**, fresh cosine. Matching the LR is deliberate: it makes
   the OBJECTIVE the single variable. Giving the control A2's 2e-4 would change two things.
   `training_check` 600/600 nonzero-gradient steps, `loss_mean` 0.240.
   🔴 Never `--resume_from_checkpoint`, and never read `rc=0` alone as evidence.
   **Both arms now hold six checkpoints; the comparison is unblocked and the GPU is idle.**
3. **The holdout is now the ONLY live question here.** `grpo_holdout.jsonl` (493 rows, seed 42)
   never entered training. The arm is dead, but the *premise* is not: `pass@8` 0.945 vs greedy
   0.705 was measured on TRAIN data the model saw for three epochs. The holdout separates
   reachable capability from memorisation slack, and that answer outlives GRPO — it decides
   whether any future arm should chase this headroom at all. Cheap: `number` is scored by
   `Number.verify`, so it needs no judge.
4. **Then** adjudicate against the pre-registration: primary cell `aggregation_ID`, MDE ≈0.036,
   S8's three clauses, veto on `object_recognition`.

### Final numbers from the GRPO run (600/600, rc=0)

`reward` 0.6875 at step 588 (early steps sat at 0.31–0.44) · `frac_reward_zero_std` around 0.3 ·
`completions/mean_length` 2.0, `clipped_ratio` 0.0 · memory 29.1 GiB stable, no leak ·
15.9 s/it (faster than the smoke's 19.5 — `beta=0` skips the reference forward pass).
⚠️ The reward rise is **across different batches** and is NOT evidence of a gain. Only the
scored eval against the control can say that.

## Why this rung exists

The August thread closed GRPO as *"diluted — it only touches `number`"*. The arithmetic says
otherwise: `number` is **80.4%** of `aggregation`, `aggregation` is **2 of the 4** populated
buckets, so `number` carries **~40% of the headline**. +4.5pp on `number` clears S8 on
`aggregation_ID`; +7.5pp clears S1's ~0.03; 50% capture of the measured 24pt headroom
(pass@8 0.945 vs greedy 0.705) lands **0.5770** against a rank-1 of 0.5653.

## Findings this rung produced, independent of its own result

1. 🔴 **The reference-policy trap.** With a PEFT model and no explicit `ref_model`, ms-swift's
   KL reference is `null_ref_context` → `disable_adapter()` (`rlhf_mixin.py:186-194`) — **the raw
   base model, not A2**. A KL penalty against that pulls the policy back toward the un-fine-tuned
   checkpoint, against the +0.317 the campaign rests on. Caught because `kl` read 4.06 **at step
   1, before any update**, and pinned at exactly 5.0 (one of two tokens saturating the ±10
   per-token clamp, `grpo_trainer.py:964`). `ref_adapter_name` is not exposed in 4.4.1 and
   `--ref_model <merged A2>` needs 16 GB we do not have. ⇒ `beta = 0`.
2. 🟢 **The entropy gate replicates on a third instrument.** `frac_reward_zero_std` averaged
   **0.30** over the smoke against step 5's `zero_advantage` of **0.280/0.270**. Different
   instrument, different slice, same number.
3. 🔻 **`--train_type` does not exist on this build** — renamed `tuner_type` (`base_args.py:93`).
   `CLAUDE.md` and the official Qwen3-VL recipe both still document the dead flag.
4. 🔻 **`model_type` must be pinned** — the volume weights match three registered types.
5. 📌 **32 counting questions carry golds malformed for `Number`** (`'2.'`, `'Two.'`,
   `'Intestine: 1.'`), *all* under the phrasing *"Please provide a single integer"*, which
   appears nowhere else in the corpus. A candidate source of the trailing-period habit probe 16a
   measured at 86.7% ID. **Data defect, own rung, not fixed here.**
6. ⚠️ **A first reading of the smoke was wrong and is corrected in the README**: one step showed
   `frac_reward_zero_std` 0.0 and it was reported as "zero groups without gradient". Across ten
   steps three produce no gradient at all. 0.30 is the honest figure.

## Ties

[[entropy-gate-scopes-phase-c-to-number]] scoped phase C here · [[vcd-has-nothing-to-subtract]]
established the model looks and fails after looking · [[local-eval-vs-judge-calibration]] is why
the primary cell may not be local `bucket_mean` · [[significance-rule]] S3/S8 govern the verdict ·
[[multimodal-optimizer-is-an-identity]] is why no `--optimizer` flag is passed.
