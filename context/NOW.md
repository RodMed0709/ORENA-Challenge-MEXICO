# context/NOW.md — what is happening RIGHT NOW

> The living current-state of the project. Updated as things change. Read this + `context/INDEX.md`
> to get oriented fast. (Supersedes the older `HANDOFF.md` baseline-run handoff, kept as history.)
> Last updated: **2026-07-19**.

## 🔴 2026-07-19 — the strategy moved. Read these three before anything else.

**A zero-GPU session relocated the target and killed a lever.** Nothing was trained; every number
below came from artifacts already committed.

1. **The gap is the `number` FORMAT, not the classes** ([[the-gap-is-the-number-format]]).
   `aggregation × ID` is **80.4 % `number`**; `fo_class` does not appear in the bucket at all.
   Answering 100 % of `binary` only reaches 0.4492 — **no path to the target avoids lifting
   `number` 0.327 → ~0.482.** ⚠️ This **re-scopes [[class-imbalance-not-counting]]**: the
   `sponge`/`gallstone`/`clip` work measures `object_recognition`, the bucket we **lead by
   +14.9**. It defends the advantage; it does not close the gap.
2. **Post-hoc count calibration is DEAD** ([[count-calibration-dead]], probe 05c). Three
   pre-registered rules fail. Mechanism: true values **2, 3 and 4 share the same modal prediction
   (1)**, so a LUT trades one error for another. Not fixable with more data.
3. **The 5 s cap is POOLED, not per-question** ([[latency-budget-is-pooled]], read from the
   official submission template): `120 s setup + B × 5 s`, and the `latency` we emit is not
   scored. Measured p99 is **0.352 s**. **Self-consistency and higher `max_pixels` are
   affordable** — both had been closed against a ceiling that does not exist as modelled.
   🔴 **The risk inverts to COLD START:** imports + weight load + CUDA-graph capture all eat the
   120 s setup allowance.

Also: **the +77 % secondary-label pool is 89.6 % `fo_class` and 0 % `number`**
([[secondary-labels-are-fo-class]]) — the top-ranked data lever adds nothing to the format that
owns the gap, and survives only as a *multiplicity-transfer* hypothesis judged on `number`.

**New:** 🤖 **`context/MEASURED.md`** — generated (`python -m frame.measured`), answers *"has this
already been measured?"* over four sources. **Read it before proposing an experiment.** It exists
because this session re-derived **four** pieces of already-committed work.

## Live fronts
- **Leo (legokna)** → **rung 06 ViT-LoRA**: **TRAINED + EVALUATED + SCORED. Verdict 🟡 PARTIAL.** Selected `checkpoint-1720` (epoch 2) by acc_OOD — same index rung 02 selected. Canonical **`bucket_mean` 0.5667** (vs 02's 0.5486), `margin_ID` +0.0235 / `margin_OOD` +0.0160, registered in the ledger. The pre-registered target was `dice@2`, NOT bucket_mean: `number` moved (OOD Δ +0.091, CI [+0.009,+0.160]) but `fo_class` did not, and **no cell reached the +0.10 relevance threshold** → PARTIAL, one format only. All four cells had power. **Do NOT read this as "the ViT was not the ceiling"** — `vit_lr` ran at the LLM's 2e-5, so the run cannot separate ceiling from recipe. ⚠️ **But `vit_lr` is no longer the next move:** [[checkpoint-selection-vs-number]] disconfirmed its rationale. **T7 then closed the epoch question:** epoch 1 of both arms measured on the full 6252 — **epoch 2 wins**, so acc_OOD selection was right and we did not own a better checkpoint. Full account: `experiments/06-vit-lora/README.md` + `context/06-vit-lora/CONTEXT.md`. **On `main` @ `9d2f1c7`. Rung 06 is CLOSED.**
- **Rodrigo** → the MLOps/consistency system (below) + planned **R1 CoA-format SFT** ([[next-move-rodrigo-coa-format]]).

## Done this session (all on `main`, pushed)
- **The brain** — `context/INDEX.md` (map), `context/RULES.md` (DO/DON'T incl. reading rules 10-13), `context/decisions/` (ViT-swap NO-GO, Qwen ladder, Rodrigo CoA front, eval-canonical). `CLAUDE.md` points here every session.
- **Canonical eval** — `src/frame/metrics.py` (`stratified_report` + 5 RAISING gates), wired into `run.py` with a HYBRID SDK cross-check (`ref.ood` stamped from qID → SDK pre_eval becomes correct → asserts our `bucket_mean` ≈ SDK's). `delta.py` de-duplicated. Killed the leaf→group drop-bug + the all-False-`ood` mislabel + the temporal-orphan inflation.
- **Results ledger** — `results/` tiers (summary/detailed/by_run) + root `RESULTS.md`, auto-built by `frame.ledger`, never hand-edited.
- **Data card hooked into the brain** — INDEX + reading rules.
- **Margin/floor enrichment of the ledger (DONE, `task/results-margin`)** — the template-aware floor + normalisation are now ONE implementation in `frame.metrics` (`template_of` / `template_floor`); `build_card.py` (rung 08) imports them, so the card's §4b and `results/` cannot drift. `stratified_report(gold=…)` populates `floor`+`margin` on `by_format`/`by_bucket`/`by_bucket_format` and `floor_ID/OOD`, `margin_ID/OOD` at the top level; `assert_floors_vs_eval_set` now verifies floor∈[0,1] and `margin==acc−floor` (it no longer raises on a below-floor run — that is a finding, not malformed input). `results/summary.csv` gains `margin_ID`/`margin_OOD`, `results/detailed.csv` gains `floor`/`margin`, `RESULTS.md` leads with "read margin, not raw accuracy". Rescored 00-baseline + 02-lora offline (predictions local, gold = `ledger.gold_from_frame_parquets("external_data/orena-data")` from the S3 val parquets). This makes reading-rules 10–13 something `results/` SHOWS, not just says.

## Real numbers (canonical, recomputed offline via S3 — no pod)
- **Epoch 1 of both arms measured on the full 6252 (T7, 2026-07-19): epoch 2 wins everywhere but `number`-OOD.** rung 02 ep1 0.5282 vs ep2 0.5486 · rung 06 ep1 0.5345 vs ep2 0.5667. `acc_OOD` selection was right; `RULES` §6 CONFIRMED, not qualified.
- **rung-06 ViT-LoRA: `bucket_mean` 0.5667** (acc_ID 0.5444, acc_OOD 0.6078) — the ladder's best. **Real skill: margin_ID +0.207, margin_OOD +0.148** (+0.024 / +0.016 over rung 02).
  - ⚠️ **The headline is not the verdict.** The pre-registered target was `dice@2` per cell, and it says **PARTIAL** (one format only, nothing at the +0.10 relevance threshold). Reading 0.5486 → 0.5667 as "the ViT was the ceiling" is the misreading this rung exists to prevent.
  - The single variable, measured from the adapters: **+3,849,984 visual params** (21,823,488 → 25,673,472). Language side byte-identical between arms.
- **rung-02 LoRA: `bucket_mean` 0.5486** (acc_ID 0.5209, acc_OOD 0.5918). The old `pre_eval 0.708` was inflated by one temporal_grounding n=1 question — corrected in RESULTS.csv/README.
  - **Real skill (MARGIN over the template-aware floor): margin_ID +0.184, margin_OOD +0.132** (floors 0.337 ID / 0.460 OOD). By margin the model adds LESS on OOD even though acc_OOD > acc_ID — matches data card §4b.
- **00-baseline zero-shot: `bucket_mean` 0.2557** — **below floor everywhere** (margin_ID −0.088, margin_OOD −0.191): a weak zero-shot model legitimately under the trivial constant.
- rung-05 arms: a0_real 0.550 / a2_shuffled 0.334 / a1_black 0.275 (from committed CSVs).

## Key findings baked in (from the data card, rung 08)
- 🔴 **We have never read `secondary_capabilities`** ([[unused-metadata]], 2026-07-19). 89.8% of train questions carry them, and **aggregation appears as a SECONDARY label on 4,238 more train questions (+77%)** — the supervision pool for the bucket we need to lift is **9,762, not 5,524**, at zero annotation cost. Ranking stays primary-only, so this changes what we can TRAIN on, not what we are SCORED on. Also unused: `generation` (automatic 78% / anchor 15% / manual 6.4%, same proportions in train and val). ⚠️ **`clinical_relevance` is all-False in BOTH splits** — a second landmine beside `ood`; never filter on either from public data.
- 🔴 **`aggregation` IS the gap, and it is not a ceiling** ([[aggregation-is-the-gap]], 2026-07-19). First external reference (public leaderboard, a **participant** not a baseline): a **Qwen3.5-4B scores 0.5438 on `aggregation×ID` vs our 0.4188** — 12.5 pts ahead on the bucket worth 50% of the exam — while we lead `object_recognition` by +14.9. The split direction argues against a dataset artifact. **Matching their aggregation alone puts us at 59.1%.** Also: the leaderboard's `pre_evaluation_score` is the mean of **populated** buckets and **every OOD bucket is `null`** — the vara is currently ID-only.
- 🔴 **The class set is OPEN and bigger than our data** ([[open-class-vocabulary]], 2026-07-19). `overview.md:17` says "such as … **and similar objects**"; the organizers' predefined list is **10 classes**, of which **`mesh` and `foreign object` have ZERO examples in train AND val**, and `silicone loop` exists only in train. **Our 8 classes are an artefact of our batch, never a definition of the task — do not optimise the training class mix against val frequencies.** Separately: `overview.md:125` says questions carry a class list, but **70% of ours carry none** → possible train/test regime mismatch, and an argument for open-vocabulary output (FICHAS lever #2, HIGH, untried).
- 🔴 **The FO failure is per-CLASS, not per-count** ([[class-imbalance-not-counting]], 2026-07-19). `gallstone` recall **0.000** in both arms (14 training examples); `needle` 0.511; `sponge` 0.633 **with 558 examples**. Training carries a **phantom class** — `silicone loop`, 435 train examples, **zero in val** — emitted ~27 times as guaranteed false positives. `clip` precision 0.619 (212 of 327 FPs). **68% of omissions are on classes with >400 examples**, so data rebalancing has a low ceiling; the prize is `sponge` perception. Also: **the ViT LoRA raised `needle` recall +17.8 pts** — `bucket_mean` averaged that away.
- **`acc_OOD > acc_ID` is an ARTIFACT** — the OOD floor is ~12 pts higher; read MARGIN over the template-aware floor. By margin the model adds *less* on OOD.
- **`acc_number` is not interpretable** (8 templates, 4 degenerate) — use the hierarchical estimate.
- **Effective n ≈ 38 videos**, not 6252. `procedure_type`/`generation` reach the model but `procedure_type` as a model lever risks OOD (unseen procedures break it) → analysis-only stratifier.

## In progress
- **Rung 10 — self-consistency (k-voting on `number`)** · branch **`task/self-consistency`**,
  pushed, **NOT merged**. Local half done and unit-checked; **nothing has run, there is no
  `RESULTS.csv`.** `experiments/10-self-consistency/README.md` + `context/10-self-consistency/`.
  - ⚠️ **The branch changes SHARED code** (`src/frame/engine.py`, `config.py`): sampling flags
    whose defaults are meant to reproduce greedy byte-identically. **That guarantee could not be
    verified — there is no local GPU.** It is deliberately held off `main` until the bit-identity
    gate passes on a pod. **If you pull the branch, know the guarantee is by design, not by test.**
  - Order on the pod: **(1)** bit-identity gate with the flag OFF · **(2)** T0 diversity probe
    (~20 min) · **(3)** STOP for go/no-go before spending T1.
  - 🔴 **Why T0 first:** voting moves toward the mode, and ours is *measured biased* (05b: 81.5 %
    of `number` errors are under-counts; 05c: true 2/3/4 share modal prediction 1). Aggregating a
    biased distribution reinforces it — how calibration died. The decisive T0 column is
    `mode_closer_than_greedy`, **not** entropy.

## Pending / blocked
- **Rung 06's successor: DECIDED — it is rung 10** (above). The two candidates cleared on 07-18
  (`vit_lr`, epoch-1 checkpoint) stay dead.
- **Superseded note — the old text of this bullet said:** "Next experiment for rung 06: UNDECIDED. Two candidates were cleared out of the way today, both cheaply: the 7.5 h `vit_lr` re-run (rationale disconfirmed — `number` decays with the ViT frozen too) and the epoch-1 checkpoint switch (**T7 measured it: epoch 2 is better in both arms, we did not own a better checkpoint**). **The two roadmaps are now reconciled in THE_MAP §"What comes next"** — they disagreed for three days and nobody could see it. Its read: **measuring p99 on a real L40S is the only step BOTH documents demand** (Bloque-A makes it a hard gate on the whole capacity branch; no question has ever run on the target hardware). The rank probe is single-sourced. 🔴 **Constrained decoding is measured dead** — `number` is 100% bare integers in all three rungs including zero-shot. See [[checkpoint-selection-vs-number]] **including its retraction**."
  - ✅ **What still holds:** both dead candidates stay dead; constrained decoding stays measured dead.
  - 🔴 **What changed 07-19:** *"measuring p99 on a real L40S is the only step BOTH documents
    demand"* was answered from the **official template instead** — the budget is POOLED, and our
    measured p99 is 0.352 s ([[latency-budget-is-pooled]]). L40S confirmation is now a
    verification, **not a gate**. THE_MAP's capacity branch and its resolution branch were both
    costed against a per-question ceiling that does not exist as modelled; **both need re-costing.**
- **05-bottleneck-audit + 03-prompt-variants rescore** — their predictions are NOT on the volume (only notebook/logs) → stay `needs_backfill`.
- **Phase 0** (offline Docker + first leaderboard submission) — still open. ⚠️ **Jul 15 was the pre-eval OPENING, not a deadline** — the real dates are **Sep 1** (pre-eval closes) and **Sep 8** (final submission). This line used to read "was due Jul 15", which made an open task look overdue.

## Infra / workflow
- **main = shared truth; one branch per task; merge to main when done.** Never work on Leo's `task/vit-lora`.
- **Offline rescoring via RunPod S3** (`get_object`, region eu-ro-1, creds in `.secrets.env`) — no pod needed to recompute metrics from saved predictions.
- **Local-first → push to GitHub.** Never `pull` on a pod while it trains; the shared volume repo is a single checkout (coordinate its branch).
- 🔴 **`stratified.json` is now VERSIONED (`.gitignore` exception, on `main` @ `da56eba`).** Before this, `results/` was committed but rebuilt from files living in gitignored `runs/`, so **`build_results_ledger` on a clone missing another run's artifacts silently downgraded that run's committed row to `needs_backfill=True` with every floor/margin → NaN.** 00-baseline and 02-lora-sft were backfilled and each reproduces its committed row to 1e-9. **Their `source_commit` moved 708a4cb → `da56eba`** (the JSON is newly tracked — the numbers are unchanged).
- ⚠️ **`frame.ledger` treats an `arm` column as a run name** (`ledger.py:67`) — an experiment CSV shaped per-arm injects phantom rows into the shared ledger. Rung 06 works around it by splitting `RESULTS.csv` (ledger-shaped) from `RESULTS_arms.csv`; the edge is still there for the next one.
- ⚠️ **`frame.metrics.template_floor` overstates margin when `gold` is incomplete** — rows without gold leave the numerator but stay in the denominator, warned only via `logger.warning`. Assert gold coverage before reading any margin.
- Pods: all OFF except a read-pod (`eu5j5t7qobk1k2`). Volume `gf78k60nlt` (EU-RO-1) holds data + all run artifacts.
