# CONTEXT — 06 LoRA reaches the ViT

## Objective
Answer one question and only one: **is the vision tower the ceiling?** Rung 02 lifted
`fo_class` and `number` by +0.42/+0.29 with the ViT frozen, yet the model still reports
**1.70 objects when there are 2** — and it does so almost identically on ID (1.704) and
OOD (1.698). If that is a perception limit, adapting the ViT must move it. If it does
not, the defect is in the data/labels or in how the LLM reads the visual tokens, and the
capacity branch (bigger model, higher resolution) is the wrong place to spend.

## Setup-config
- The ONE variable: `--freeze_vit false`. Verified against the **installed** ms-swift
  (**4.4.1**), not the docs:
  - `swift/pipelines/train/tuner.py:91` `get_target_modules` → resolves `all-linear` and
    passes `freeze_vit=args.freeze_vit` (:101) into
  - `swift/utils/transformers_utils.py:221` → `if not freeze_vit: modules +=
    model_arch.vision_tower`.

  So the flag **ADDS `vision_tower` to the LoRA target modules — it does not unfreeze
  the tower.** With `freeze_aligner=True` the aligner is never added, and :233 strips
  aligner submodules out of the vision regex — which is exactly the negative lookahead
  observed in the run's `adapter_config.json` (`(?!(model.visual.merger|
  model.visual.deepstack_merger_list))`).

  ⚠️ **Landmine at `tuner.py:93`:** `if isinstance(args.target_modules, str): return
  args.target_modules` — a *string* `target_modules` returns early and `freeze_vit` is
  silently ignored. It does not fire here (the CLI parses to a list), but a future rung
  passing a string would get a frozen ViT while believing otherwise.
- Baseline arm = rung 02's saved predictions on the same 6252. **No GPU re-run** — the
  A/B is exact because both arms answer the same qIDs under greedy decoding.
- Everything else byte-identical to rung 02: r=8, α=32, dropout 0.1, lr 2e-5, 3 epochs,
  bf16, `max_pixels`, seed 42, same frozen split, same judge, same parser.
- Engine: `_models/vit_lora_train.py` (rung-02 recipe + the freeze_vit toggle + the
  broken-run guard). Metrics: `_models/multiplicity.py` (dice@k / paired_delta / power).
- Ran on an RTX 5090 (32 GB), 7 h 30 m, 21.5 GiB peak, 2580/2580 steps.

## Decisions
- **The rule was pre-registered, then REWRITTEN (v2, 2026-07-16) because v1 could not be
  falsified.** v1 demanded *"`number` acc@2 rises ≥ +0.05"* while the measured per-cell
  noise was ±0.06–0.13 — the threshold sat inside the noise, so any outcome would have
  "passed" or "failed" by chance. v1's target was also the pooled 8-template ID+OOD
  number that `08-data-card` §3/§4b invalidated. See `local/specs/vit-lora/spec.md`.
- **`dice@k`, not accuracy.** `acc_number` is not interpretable (8 templates, floors
  0.24–1.00, four degenerate). Only the 3 non-degenerate templates count, selected by
  criterion (`answer.nunique() >= 4`), never by a hand-written list.
- **Paired delta over videos, not two independent CIs.** The real n is 38 videos — 8–9
  in OOD. Pairing removes between-video variance; comparing independent CIs would have
  thrown away the only power available. This is what made all four cells clear the
  power rule.
- **A power rule with a reachable positive.** A cell speaks only if its delta CI
  half-width < 0.10. Requiring all four cells while two are OOD with 8–9 videos would
  have made "the ViT was the ceiling" unreachable *by construction* — the verdict would
  always have fallen into the branch that kills the capacity front, for lack of videos
  rather than lack of effect.
- **`vit_lr` left at default** → the ViT trained at the LLM's 2e-5. Pre-registered as
  making a weak result ambiguous (ceiling vs recipe). This is the single biggest caveat
  on the verdict and the reason the follow-up below is cheap and obvious.
- **Checkpoint selection stays `acc_OOD` per-epoch**, as in rung 02, even knowing
  `acc_OOD` flatters (the OOD floor is 12 pts higher). Changing the criterion would have
  been a second variable. Both arms selected epoch 2 (`checkpoint-1720`).
- **Broken-run guard, not an early-stop.** A pure stdout observer that aborts only on
  NaN/inf or a first `eval_loss` ≥ first `train_loss`. When it does not fire, the run is
  byte-identical to an unguarded one, so the A/B is untouched. It guards against a
  *wasted* run, never against a faithful negative. It did not fire.

## Results
🟡 **PARTIAL.** `number` moved (OOD Δ +0.091, CI [+0.009, +0.160], excludes zero);
`fo_class` moved in neither cell. **No cell reached the +0.10 relevance threshold.** All
four cells had power, so `SIN POTENCIA` is ruled out. Full table + the closed branches:
`experiments/06-vit-lora/README.md`.

`bucket_mean` 0.5486 → **0.5667**; `margin_ID` +0.0235, `margin_OOD` +0.0160. Real,
small, and **not the verdict**.

Two findings that argue against reading this as a clean win: `number` ID raises `dice@2`
while *lowering* `acc@2` (−0.079, i.e. overshooting 2→3), and `aggregation` OOD flipped
224 verdicts to net **exactly zero** while sitting only +0.022 above its trivial floor.

**Provenance.** The committed numbers come from the second of two eval passes. The first
compared rung 06 against the zero-shot baseline rather than against rung 02, and did not
run the pre-registered `dice@2` bootstrap; it was discarded. The pass that produced these
artifacts reproduces the rung-02 baseline on 7/7 pre-registered targets (G3) before any
comparison is made — which is why a stale `paired_delta.csv` referencing zero-shot deltas
is not part of this run.

## Preconditions for reading these numbers
- **Margins are only valid because gold coverage was asserted first** (6252/6252).
  `frame.metrics.template_floor` keeps rows without gold in the denominator while
  dropping them from the numerator, so partial gold understates the floor and overstates
  the margin. The gate runs ahead of every margin read here.
- **The ledger rows for this run are reproducible from a clean clone** — `stratified.json`
  is versioned (a `.gitignore` exception added with this rung) rather than living only in
  a gitignored `runs/`. 00-baseline and 02-lora-sft were backfilled the same way, each
  verified to reproduce its already-committed row to 1e-9 before being registered.
- **`RESULTS.csv` is ledger-shaped (one row per run); the A/B contract lives in
  `RESULTS_arms.csv`.** They are split because `frame.ledger` treats an `arm` column as a
  run name, which would file this experiment's baseline arm as a run of its own.

## 🔻 Update 2026-07-30 — the `vit_lr` follow-up ran, and it did not settle the question

Rung 21's arm A3 ran the lever this section asks for: `vit_lr 2e-5` under an LLM lr of 2e-4
(`experiments/21-recipe-sweep/`, run `21_vitlr_v1`). **It lost** — proxy −0.0278, four of
thirty paired cells excluding zero, all four favouring the control.

⚠️ **But it does not close this rung's question, for two reasons.**

1. **A3 moved TWO flags.** `--optimizer multimodal` is emitted iff `vit_lr` is set, and the
   control it was compared against (arm A2) **never passed it**. So "lower `vit_lr`" and "a
   different optimiser" changed together. The defensible reading is *lowering `vit_lr` under
   the multimodal optimiser costs 0.028*, not *the tower wants the high rate*.
   ⇒ **[[vit-lora-partial]] stays OPEN.** The fix is a 20-step optimiser probe (~$0.30).
2. **Two points, 10× apart.** Nothing between 2e-5 and 2e-4 ran.

🔴 **The operational trap this uncovered is worth more than the arm.** `--vit_lr` is a
**silent no-op unless `--optimizer multimodal` is also passed** (ms-swift 4.4.1). A run that
sets `--vit_lr` alone trains the tower at the LLM's rate and reports nothing — so **this rung's
"`vit_lr` left at default" caveat was accidentally correct**, and any earlier plan to "just set
`--vit_lr`" would have produced a fake null.

🟢 **What DID change here:** rung 21 showed the whole recipe was under-trained
([[undertrained-was-real]]), which re-frames this rung's weak-positive. The tower was being
driven at 2e-5 — one fifth of what the campaign now runs — so "weak positive" was partly a
statement about the learning rate, not about the tower's capacity.

## Next
1. 🔴 **The optimiser probe** — 20 steps of arm A2's config with and without
   `--optimizer multimodal` at `vit_lr == learning_rate`, comparing loss traces. Until it
   runs, the A3 result cannot be attributed and this rung's question stays open.
2. Until (1) is done, **do not read this run as "the ViT was not the ceiling"**, and do not
   read A3 as "the ViT wants the high LR" either — the pre-registered rule supports neither
   branch, and neither does the evidence.
3. The capacity branch (`context/decisions/qwen-size-ladder.md`) stays gated: it requires
   the 8B to plateau AND the cheap levers to be spent. Neither holds today.
4. `05-bottleneck-audit` and `03-prompt-variants` remain `needs_backfill` in the ledger —
   their predictions are not on the volume, so they cannot be made rich.
