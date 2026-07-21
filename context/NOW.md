# context/NOW.md — what is happening RIGHT NOW

> The living current-state of the project. Updated as things change. Read this + `context/INDEX.md`
> to get oriented fast. (Supersedes the older `HANDOFF.md` baseline-run handoff, kept as history.)
> Last updated: **2026-07-20**.

## 🔴 2026-07-20 — rung 10 is CLOSED, and it kills a whole FAMILY of levers.

**Self-consistency is measured dead** ([[self-consistency-dead]]). Three pre-registered arms on the
full 2094 `number`: every one negative, and **k=16 significantly HARMS OOD** (−0.0430, CI excludes
0). Doubling k doubled the harm — the signature of a mode sitting on the wrong value. On OOD the
voted answer falls **below the trivial floor**. It died on **quality, not latency** (k=8 ≈ 1.15 s/q
against a pooled budget that affords it).

🔴 **The generalisation, and it is the important part.** This is the **third independent
measurement** of one fact, after [[count-calibration-dead]] and [[naming-equals-counting]]:
**the deficit is UPSTREAM of the output.** Anything that aggregates, re-encodes, re-ranks, votes on
or remaps what the model has **already emitted** is closed by measurement. Of the six ideas in the
group that owns the gap, **three are now dead** (voting, calibration, enumerate-then-count).
**The next lever must attack perception or supervision — not the output.**

⚠️ **Arm C was a trap the ID-AND-OOD conjunction caught**: +0.0284 in ID, within reach of the bar,
while significantly damaging OOD. **Never relax that conjunction.**

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
- **Leo (legokna)** → **rung 12 image processing: OPEN ON PURPOSE.** Branch A (unsharp ×3 at inference, on the fine-tuned model) is a **faithful NEGATIVE and monotonic in dose** — −0.056 ID / −0.058 OOD, both significant; the identity gate passed 50/50 byte-identical. 🔴 **The method correction it produced is the important part: any inference-only test of an INPUT intervention is biased toward the negative** on a model fine-tuned without it. That re-scopes branch A itself and, retroactively, rung 11 — it does **not** touch the output family (voting/calibration/enumeration), which died with no mismatch at all. **The rung stays open because branch B decides it** — the same two arms on the **ZERO-SHOT** model (~52 min of 5090, all code exists, only `model_path` changes), far less locked to our frames' appearance. ⚠️ Poor instrument (`bucket_mean` 0.2557, **below floor everywhere**): read it for DIRECTION, never magnitude. 🔴 **But branch B cannot be the fair test either — it is still inference.** The honest test of the whole input-side family is to **TRAIN with the transform**, which is why **idea 2 (proportional subsampled-training harness, `local/hallazgos/ideas-mejora.md` §2) is escalated from convenience to ENABLER**: it takes a run from **7.5 h to ~1.5 h**, costs only CPU hours, and unblocks the input family and the re-scoped rung 11 alike. Subsample **questions within ALL videos** — dropping videos would destroy the effective n of 38. Its known limit: the LR optimum shifts with size, so it serves **relative** comparisons, not absolute values. **On `task/image-processing` @ `ed7a955`, pushed, deliberately NOT merged.**
- **Closed by Leo, on `main`:** rung 06 ViT-LoRA (🟡 PARTIAL, `bucket_mean` 0.5667, `@ 9d2f1c7` — full account in `experiments/06-vit-lora/README.md` + `context/06-vit-lora/CONTEXT.md`; ⚠️ do NOT read it as "the ViT was not the ceiling", `vit_lr` ran at the LLM's 2e-5 so it cannot separate ceiling from recipe, and [[checkpoint-selection-vs-number]] disconfirmed `vit_lr` as a next move) · rung 10 self-consistency (dead, see above) · rung 11 resolution (**dead at the gate, zero GPU** — the "100%/0%" partition was an n=50 artefact, and the axis is irresolvable anyway: 130 videos, 0 with more than one resolution, so resolution is perfectly confounded with video).
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
- 🔴 **`number` per TEMPLATE (new read, 2026-07-20, from rung 10's `RESULTS_templates.csv`).** The
  2094 are not one block: **five templates are already maxed and 1947 questions carry the whole
  fight.** Margin over the template-aware floor:

  | template | n | distinct true | floor | acc | **margin** |
  |---|---|---|---|---|---|
  | *How many **Clips**…* | **681** | 12 | 0.239 | 0.266 | **+0.026** |
  | *…foreign object **instances**…* | **830** | 11 | 0.286 | 0.336 | **+0.051** |
  | *…foreign object **classes**…* | 436 | 4 | 0.608 | 0.665 | +0.057 |
  | *How many Sponges…* | 83 | 2 | 0.904 | 0.928 | +0.024 |
  | Drains / Needles / Bags / Specimens | 64 | 1 | 1.000 | ~0.99 | 0 (degenerate) |

  **`Clips` is the single largest hole in the exam**: 681 questions, 12 distinct true values, and
  the model beats "always answer the mode" by **+2.6 pts**. Read with rung 05 (black image returns
  the `number` floor to 16 digits) the diagnosis is: **we are a good object RECOGNISER that does not
  INDIVIDUATE instances** — and the exam weights individuation at 50 %.

## Key findings baked in (from the data card, rung 08)
- 🔴 **We have never read `secondary_capabilities`** ([[unused-metadata]], 2026-07-19). 89.8% of train questions carry them, and **aggregation appears as a SECONDARY label on 4,238 more train questions (+77%)** — the supervision pool for the bucket we need to lift is **9,762, not 5,524**, at zero annotation cost. Ranking stays primary-only, so this changes what we can TRAIN on, not what we are SCORED on. Also unused: `generation` (automatic 78% / anchor 15% / manual 6.4%, same proportions in train and val). ⚠️ **`clinical_relevance` is all-False in BOTH splits** — a second landmine beside `ood`; never filter on either from public data.
- 🔴 **`aggregation` IS the gap, and it is not a ceiling** ([[aggregation-is-the-gap]], 2026-07-19). First external reference (public leaderboard, a **participant** not a baseline): a **Qwen3.5-4B scores 0.5438 on `aggregation×ID` vs our 0.4188** — 12.5 pts ahead on the bucket worth 50% of the exam — while we lead `object_recognition` by +14.9. The split direction argues against a dataset artifact. **Matching their aggregation alone puts us at 59.1%.** Also: the leaderboard's `pre_evaluation_score` is the mean of **populated** buckets and **every OOD bucket is `null`** — the vara is currently ID-only.
- 🔴 **The class set is OPEN and bigger than our data** ([[open-class-vocabulary]], 2026-07-19). `overview.md:17` says "such as … **and similar objects**"; the organizers' predefined list is **10 classes**, of which **`mesh` and `foreign object` have ZERO examples in train AND val**, and `silicone loop` exists only in train. **Our 8 classes are an artefact of our batch, never a definition of the task — do not optimise the training class mix against val frequencies.** Separately: `overview.md:125` says questions carry a class list, but **70% of ours carry none** → possible train/test regime mismatch, and an argument for open-vocabulary output (FICHAS lever #2, HIGH, untried).
- 🔴 **The FO failure is per-CLASS, not per-count** ([[class-imbalance-not-counting]], 2026-07-19). `gallstone` recall **0.000** in both arms (14 training examples); `needle` 0.511; `sponge` 0.633 **with 558 examples**. Training carries a **phantom class** — `silicone loop`, 435 train examples, **zero in val** — emitted ~27 times as guaranteed false positives. `clip` precision 0.619 (212 of 327 FPs). **68% of omissions are on classes with >400 examples**, so data rebalancing has a low ceiling; the prize is `sponge` perception. Also: **the ViT LoRA raised `needle` recall +17.8 pts** — `bucket_mean` averaged that away.
- **`acc_OOD > acc_ID` is an ARTIFACT** — the OOD floor is ~12 pts higher; read MARGIN over the template-aware floor. By margin the model adds *less* on OOD.
- **`acc_number` is not interpretable** (8 templates, 4 degenerate) — use the hierarchical estimate.
- **Effective n ≈ 38 videos**, not 6252. `procedure_type`/`generation` reach the model but `procedure_type` as a model lever risks OOD (unseen procedures break it) → analysis-only stratifier.

## In progress
- 🔶 **Rung 12 — image processing. OPEN. Branch A closed NEGATIVE; branch B is what decides it.**
  Branch `task/image-processing`, **not merged**. `experiments/12-image-processing/README.md`.
  - **A (fine-tuned base, rung 06 ckpt-1720):** unsharp at ×1 and ×3, inference only, measured on
    `fo_class`. **No arm rises; ×3 harms significantly in ID AND OOD** (−0.0558 [−0.0949,−0.0176] ·
    −0.0582 [−0.0887,−0.0277]) and the damage is **monotonic in dose**. Identity gate 50/50 byte for
    byte; control reused from rung 06 and gated to its canonical `fo_class`.
  - 🔴 **The generalisable part — appearance rarity is real and now quantified.** Rung 05 warned a
    black frame degrades *"by rarity, not only by absence of information"*; this is the first clean
    dose-response of it. ⚠️ **Therefore branch A does NOT show enhancement fails to help
    perception** — it cannot separate that from the fine-tune penalising an unfamiliar appearance.
  - 🔴 **Method consequence that reaches back:** **any inference-only test of an INPUT-side
    intervention is biased toward negative** on a model fine-tuned without it. That applies to
    **rung 11** too. It does **NOT** apply to the output-side family (voting, calibration,
    enumerate-then-count), which died with no train/test mismatch. ⇒ **The honest test of the
    input-side family is to TRAIN with the transform**, which raises the value of a cheap
    subsampled-training harness from convenience to enabler.
  - **B (next): the same arms on the ZERO-SHOT model**, far less locked to our frames' appearance.
    ⚠️ Poor instrument (`bucket_mean` 0.2557, **below floor everywhere**) — read it for direction,
    never magnitude.
  - **Tooling built and reusable:** `_models/build_frame_index.py` — one entry per cached frame with
    its questions, their results in three runs, and photometric statistics. It killed two candidates
    (global white-boost, CLAHE) for **zero GPU** before any arm ran.
- **Rung 10 — self-consistency: CLOSED, FAITHFUL NEGATIVE. Merged to `main` @ `e520dcf`.**
  `experiments/10-self-consistency/` + `context/10-self-consistency/CONTEXT.md`.
  - ✅ **The bit-identity gate PASSED** — `n_samples = 1` reproduces rung 06's `predictions.json`
    **50/50 byte-for-byte across four independent model loads** on the GPU class that produced the
    reference. The shared-code change in `src/frame/{engine,config,parsing}.py` is verified
    flag-off-identical, so the A/B was single-variable and the branch was safe to merge.
  - `src/frame/metrics.py` gained **`paired_delta_ci`** — an A/B on the same questions needs the
    bootstrap of the paired DIFFERENCE; two independent CIs discard the pairing and read far too wide.

## Pending / blocked
- **Rung 06's successor: rung 10 ran and returned a faithful negative.** The two candidates cleared
  on 07-18 (`vit_lr`, epoch-1 checkpoint) stay dead.
- 🔴 **Rung 11 — the resolution axis (6b) is CLOSED too, at the gate, for zero GPU**
  ([[resolution-is-not-the-gap]]). The "100 %/0 % partition" was an artefact of a non-random n=50:
  over the full 15,213-frame cache `lapchole` (ID) has **six** resolutions and its minimum (230k px)
  is **below** `heico`'s uniform 518k. **"OOD gets 56 % of ID's visual tokens" is wrong** — ~66 % by
  mean, inverted in the tails. And **130/130 videos have exactly one resolution**, so resolution is
  **perfectly confounded with video identity** and this dataset cannot answer the question at all.
  Accuracy across resolution cells is non-monotonic. **6b, 11b and 11c die unrun.**
- **Next lever: UNDECIDED, and the constraint has tightened.** It must act upstream of the output
  (rung 10) — and rung 11 plus rung 05 (`number` returns its floor **to 16 digits** on a black
  image) together argue that levers acting on the *image itself* have little to act on for the
  format that owns the gap. ⚠️ **Phase 0 (offline Docker + first leaderboard submission) is still
  open, and every lever is being judged against a score we have never confirmed transfers to the
  organizers' hardware, engine and batching.**
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
