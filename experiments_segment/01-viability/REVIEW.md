# Rung 01 — independent review of `PLAN.md` v1

> Three independent read-only reviewers, 2026-08-14. **Verdict: NO-GO** (2 NO-GO, 1 GO-WITH-FIXES).
> `PLAN.md` v1 is superseded. Nothing was launched. No GPU was booked.
> This file adjudicates the findings, including where two reviewers disagreed.

## Why NO-GO and not GO-WITH-FIXES

Four defects each **individually** void the run, and three of them are **silent** — the run
would exit `rc=0` with a healthy `grad_norm`, a falling loss, and every one of V1–V8 passing:

1. the temporal stamp is discarded downstream of the field the plan sets,
2. the run trains at 1/8.8 the visual resolution of the checkpoint it continues from,
3. `Request.start_time` may be `0.0` and nothing detects it,
4. the pre-registered headroom gate used the floor `RULES.md:81` explicitly forbids.

The plan's structure survives — from-base control, pre-registered gate, epoch matching, the
`G-SEG-01` design. Its mechanics do not.

---

## A. Blocking, mechanical (the chain does not run, or runs wrong)

| # | Finding | Evidence | Fix |
|---|---|---|---|
| **A1** | **`--attn_implementation` is not an ms-swift flag.** It is `--attn_impl`. `HfArgumentParser` rejects unknown args ⇒ `swift sft` exits in seconds, both arms. | `swift/arguments/base_args/model_args.py:79`; our own working engine already uses the right name at `experiments/06-vit-lora/_models/vit_lora_train.py:112` | rename |
| **A2** | **The per-row `fps` mechanism does not work.** `chat_template_kwargs` *is* a real per-row field and *is* spread into the payload — then `sample_fps` is **unconditionally overwritten** with the module global (`FPS = 2.0`) on exactly the frame-list branch. Second, independent kill: `fps` is only read inside `smart_nframes`, which the frame-list branch never calls. ⇒ a 299 s clip is presented as frames 0.5 s apart when they are 8.6 s apart — a **17× wrong clock** on 38.2% of the corpus. **V4 as written cannot catch it**: it asserts the JSONL field, and the JSONL field is correct. | `swift/template/templates/qwen.py:348-351`; `qwen_vl_utils/vision_process.py:31, :171, :415-446` | use `chat_template_kwargs: {"raw_fps": X}` — it survives (`vision_process.py:441` → `video_metadata['fps']` → `mm_processor_kwargs`, `qwen.py:359`). **V4 must assert on the effective `video_metadata`, not on the JSONL.** |
| **A3** | **D3 and D4 disagree about resolution by 3.9×, and D4 wins.** `--model_kwargs` is an env-var setter and `video_max_token_num` is effective **per frame** ⇒ the run trains at **128 tok/frame ≈ 362×362**, not D3's 510. Rung 21's `checkpoint-2703` was trained at up to **1,125 tok/frame (1280×704)**. Arm (a) would continue it at **1/8.8** the visual resolution — a confound on the one variable the rung exists to measure. Sub-gate A scored a configuration the launch does not create. Also `"fps_max_frames":36` is a **total no-op** on frame lists. | `swift/arguments/base_args/base_args.py:205-210`; `swift/model/models/qwen.py:679-713`; `vision_process.py:407, :451, :173` | decide the resolution explicitly and make D3 and D4 emit the same number; drop the dead key |
| **A4** | **L5's 16 h watchdog kills the paid run.** §3 budgets the 3-epoch winner at **32.3 h** at its own centre estimate; L5 stops unconditionally at 16 h. As written it SIGTERMs at ~1.5–1.9 epochs, **before eval and before commit** — ~$22 spent, nothing scored. Two arms in one chain (18.5–21.6 h) also die. | `PLAN.md:201` vs `PLAN.md:300` | wall must be derived from the measured s/it, with margin, and asserted against the budget at render time |
| **A5** | **Gate 13 raises by construction.** It pre-registers the K histogram as *exactly* `{10:6, 29:12, 119:27, 299:36}`. There are **149 distinct durations** (min 1 s, max 300 s); those four cover 97.08%. And the adaptive branch applies only to the 38.55% of rows carrying the literal token — a population in which **no `dur=10` row exists at all**. | measured over all 8 parquets | histogram keyed on the formula, not on four literals |

## B. Blocking, methodological (the number would not mean what we say)

| # | Finding | Adjudication |
|---|---|---|
| **B1** | **Sub-gate B used the train prior. `RULES.md:81-82` forbids it verbatim** (*"vs the EVAL set … Never against the train prior"*), and the compliant floor is already implemented (`metrics.py:353-357` `_slice_floor` → `template_floor`). Recomputed the canonical way, the SEGMENT trivial `bucket_mean` is **0.5803**, and **6 of 10 buckets exceed the gate's own 0.40 threshold** — `aggregation_ID` **0.9130**, `complex_reasoning_ID` 0.9074, `event_understanding_ID` 0.8191. | **The gate FAILS, and the deeper finding is worse than the gate:** with 21 distinct templates over 23 questions, the per-template floor is nearly the identity, so **margin is uninterpretable on the small ID buckets**. RULES §10 says judge by margin. A new instrument is owed before those buckets can be read at all. |
| **B2** | **`Request.start_time` is the load-bearing unverified claim.** D2 (the whole time parameterisation) and D3 (the whole frame policy) both hang off it. No real SEGMENT payload has ever been seen. If the container ships a pre-cut clip and sets `start_time = 0.0`, the post-processor adds zero ⇒ every `2a` stays relative ⇒ `temporal_grounding` = 0.000 on **both** halves = 20% of the headline, **and** `dur = 0` ⇒ K=1 frame. All of V1–V8 pass. | **Make it self-defending**: take `dur` from the clip container's own duration, treat `start_time == 0.0` as a sentinel that falls back to absolute targets, and assert the distribution of `start_time` at load. Also: the *"observed SEGMENT layout"* (`PLAN.md:45, :90`) came from a mis-globbed fixture whose source **does not name the track** — it may be PROCEDURE's. Downgrade it from observation to conjecture. |
| **B3** | **D3's K is a factor of 2 short for *localisation*.** The formula is exact as a **coverage** floor (nearest-frame reach measured 1.0000 at D3's K), but a **perfect** detector that names the frame it saw the event in scores **0.5134**, not 1.0 — the event is bracketed in an interval of width `2·thr` and the model must name its midpoint with zero error budget. `spacing ≤ thr` gives 1.0000 at 2× the token bill. | The plan's *"buys the full `time` ceiling (0.963)"* rests on an **unstated interpolation assumption**. Either double K and pay, or state the assumption and pre-register it as a risk. |
| **B4** | **D4 is not a single variable.** Arm (a) changes initialisation **⊕ 2,703 prior gradient steps on the same 130 videos**; arm (c) controls the weights, not the optimisation distance. And `[[gen36-fails-the-8b-recipe-not-the-backbone-test]]` already **measured** A2-verbatim onto a changed initialisation at **−0.0334 [−0.0642, −0.0027]**. | A sufficient control needs a third arm (from-base at ≈4.1 epochs) and it is not budgeted. Either budget it, or **drop "static→dynamic continuation" to a leaderboard claim and stop calling it a method claim.** |
| **B5** | **The 1-epoch screen cannot select the 3-epoch winner.** Cosine anneals over the *planned* steps, so the two runs differ from step 1 and neither contains the other (`undertrained-on-both-axes.md:105-112`). Rung 38's own control: `bucket_mean` **0.5592 at ep1 vs 0.6496 at ep3**. | The $88 fork sits on an instrument the repo has already measured as unable to resolve it. Either read the verdict at 1 epoch (epoch-matched, legitimate) and call 3 epochs a deployment run, or budget both arms to 3. This also repairs **V7**, which currently asserts epoch-matching on a design that guarantees its absence. |
| **B6** | **D7 drops half of S8's win condition.** `RULES.md:208-212` requires the result to hold **on ID and OOD jointly**; D7 declares ID-only cells and states no OOD no-fall condition. `RULES.md:85-87` (§6) killed rungs 14 and 15 on exactly this. Also undisclosed: `temporal_grounding × ID` has **no calibration point in existence** — no FRAME bucket, no submission, no deflation factor. | add the conjunction; disclose the missing calibration |
| **B7** | **Judge exposure is 44.9%, not 30%.** Averaged over the ten buckets that constitute `bucket_mean`. The plan counted the three ID buckets it noticed and dropped `cr_OOD` 0.6554, `obj_ID` **0.3818**, `obj_OOD` 0.3615, `eu_OOD` 0.2690. D7's own fallback primary cell, `object_recognition × ID`, is **38.2% judge-decided** against a substitute judge, undisclosed. | correct sub-gate E; disclose in D7 |
| **B8** | **S8's reachability curve blends two leaves.** The *"3.69% unreachable at any K"* is **not scatter**: `2a` contributes **0/7,384** and `2b` contributes **257/261 (98.5%)**. Asking whether a sampled frame sits within `thr` of an *elapsed span* is a category error. | Honest read: **`2a` ceiling = 1.0000; `2b` ceiling = unmeasured.** S8 contains **zero** information about the leaf it appears to be about. |
| **B9** | **D2 creates three clocks in one prompt on 13.93% of `time` rows.** 1,065 questions embed their own `hh:mm:ss` anchor (751 train / 314 test, **all `2a`**, **100%** inside `[start,end]` ⇒ absolute). Under D2 that prompt carries an absolute window, an absolute in-text anchor, and a relative target. Second-order and worse: D2 collapses `2a`-relative (≤299 s) and `2b` (≤283 s) into the **same output range**, destroying the magnitude signal that separates them today and making a router error unrecoverable. | handle the anchor rows explicitly; price the lost separability |

## C. Structural (the result cannot be recorded)

| # | Finding | Fix |
|---|---|---|
| **C1** ✅ | **`.gitignore` excluded the run *directory*, and git cannot re-include a file whose parent is excluded** ⇒ `stratified.json`, the artifact the whole ledger is built from, could **never** be committed for SEGMENT. The exact bug the root `.gitignore` comment records having already suffered on 2026-07-18. Also measured: the root pattern `runs/**` is anchored to the repo root and **does not reach** the new tree. | **FIXED** in the root `.gitignore`, mirroring the existing exclude-contents / re-include-dirs pair. Verified both ways: `log.txt` ignored, `stratified.json` committable. |
| **C2** | **`assert_bucket_set` is decorative.** `stratified_report` applies `min_bucket_n` to a **local** frame used only to compute `bucket_mean` (`metrics.py:367-376`) and returns the **unfiltered** `by_bucket` (`:449`) ⇒ the gate sees 10 and passes on a run that averaged 7. It is structurally blind to the defect it was written for. | `stratified_report` must also return the kept set; assert on **that**. And `assert_bucket_counts` **already exists** (`metrics.py:549-573`) — RULES §1 says extend it, not add a weaker sibling. |
| **C3** | **`P6` is neutralised by `L7`.** With `set +e` (correct — a failing arm must reach its trap), papermill's non-zero exit does **not** stop the chain. A raising gate becomes a log line. | explicit exit-code test per gate |
| **C4** | **`frame.ledger` and `frame.measured` cannot see the new tree.** `ledger.py:250, :469, :609` glob `experiments/*`; `measured.py:248, :283, :289, :315` likewise. Worse, the naive fix is **harmful**: `_rung_key` (`measured.py:182-190`) collapses SEGMENT `01` and FRAME `01-ood-split` to the same key, and `_dedupe_ladder` keeps the longest ⇒ **silently drops the SEGMENT rung**. And `ledger.py:488-490` sorts a 10-bucket `bucket_mean` into the same table as 4-bucket ones with no `track` column. | teach both about the tree, add a `track` prefix to `_rung_key` and a `track` column to `_TIER1_COLS` |
| **C5** | **`_tools/` is folder-private** (`EXPERIMENT_REPO_STRUCTURE_SPEC.md:421-423`); `PLAN.md:288` imports `experiments/40-.../\_tools/chain40.py` from outside. Compounded: that experiment **does not exist on `main`**. | promote the renderer to `src/frame/` in Stage 1 |
| **C6** | **Sub-gate C omits `fo_class`** — 25.1% of SEGMENT, spanning 4 of 5 groups, and the format RULES §8b was written for. Stage 2 mints **training targets** with no accepted-set assertion anywhere. | add the `FOType.names()` clause |
| **C7** | **Frame budget omits the test half.** 18,478 distinct clips (train 12,817 + test **5,661**); router-aware total **226,007 frames**, worst case 665,208 — a **14.9×** growth of `frames_cache`, with ~zero overlap with FRAME's 15,213 point-timestamp frames. | resize; and note 226k small files on MooseFS is a **time** cost, so Stage 2 is zero-GPU but not zero-pod-hours |

## D. Hardware — the item that may block regardless of code

🔴 **A reviewer reports that no A100 (and no L40S / H100 / RTX PRO 6000) is bookable in
`EU-RO-1`**, the datacenter that holds the volume — only the RTX 5090 32 GB, the card D8
explicitly rejects. Network volumes are datacenter-locked.

⚠️ **I could not independently confirm this** — the GraphQL endpoint returned `403` on every
query shape I tried. **Confirm before booking.** What I did confirm directly:

- volume `gf78k60nlt` = `ORENA-CHALLENGE`, **`EU-RO-1`**, **670 GB** — the `.secrets.env` comment
  saying 300 GB is stale by 370 GB.

If it holds, it forces a fork the plan never considered:

| option | consequence |
|---|---|
| train on the 5090 32 GB | fits **only** at 128 tok/frame (peak ~29.6 GB) — i.e. **A3's gutted resolution becomes mandatory**, and arm (a) continues a 1280×704 checkpoint at 362×362 |
| wait for A100 stock in EU-RO-1 | unknown delay against a 2026-09-08 deadline |
| move the data to another datacenter | a second 670 GB volume + a 270 GB dataset transfer |

## E. Disk — GO, and two free wins

Measured over the S3 gateway (155,168 objects, full recursive listing — not `df`, which lies on
this volume): **540.63 GB used of a ~640 GB working wall.** `merged/checkpoint-2703` **verified
complete** (4 shards, 17.5503 GB) ⇒ S4 holds and the continuation still costs zero merge time.
Expected SEGMENT footprint **47.9 GB** (worst case 98.9). After deleting one of the **five**
`checkpoint-901` merges: **+69 GB headroom expected, +18 GB worst case. GO.**

Unlisted free wins: `orena-data/heico/.cache/.../*.incomplete` = **12.96 GB** of aborted
downloads; `hf_cache/models--Qwen--Qwen3.6-27B` = **55.59 GB** for a backbone already ruled
NO-GO; **16.8 GB** of stray `optimizer.pt`.

## F. Alarms that were wrong, and claims that survived

**Two of the plan's own 🔴 were non-threats** — worth recording so they are not re-raised:
- `--load_args false` is **already false for training** (`swift/arguments/sft_args.py:174`), and
  `get_ckpt_dir` gates on `args.json` existing, which a **merged** dir has none of. Harmless no-op.
- `--vit_gradient_checkpointing true` is already `not freeze_vit` (`sft_args.py:211-212`).
- **V2's rationale is misattributed**: `--target_modules` is typed `List[str]`, so the CLI cannot
  produce the string case that triggers the early return at `tuner.py:91`. The real trigger is
  `--target_regex`. Keep the gate; fix its stated cause.

**A reviewer error, adjudicated:** the adversarial review argues the factor-28→32 correction
*raises* tokens by (32/28)². It is the other way — 960×540 gives **646** tokens at factor 28 and
**510** at 32. The *conclusion* survives (sub-gate A must be OPEN until the factor is confirmed),
but the risk is the factor being **28**, not 32.

**Survived a genuine attack** (independently re-derived, recorded as survived):
- **S1** — FRAME-train ∩ SEG-train = 92/92; FRAME-train ∩ SEG-test = **0**; SEG-test == FRAME-test.
  And the qID prefix at `data.py:96` is **load-bearing, not cosmetic**: the raw `id` column is a
  bare int and seg/frame ids **collide on 33 values**.
- **S6** — stronger than stated: `2a` in-window **7384/7384 = 1.000000**; `2b` **4/261 = 0.0153**.
- **S9** — survives **under the literal-token reading only**. ⚠️ The plan's phrasing reads naturally
  as a timestamp *regex*, and that implementation gives recall **0.1393** with **2,346 false
  positives** (e.g. *"What types of foreign objects are seen between 00:45:40 and 00:46:41?"* is
  `fo_class`). Sub-gate C must pin the literal-string form or it certifies a different function
  than the one that ships.
- **S11**, the `Time` threshold formula, `TRACK_MAX_LATENCY[SEGMENT] = 15.0` (so **B1 is real**),
  template overlap 0.72%, and D8's arithmetic (859 steps, 32.3 h, $88) — all confirmed.
- **Multi-timestamp cardinality** — attacked and largely held: only **44/2,360** test `time` rows
  (1.86%) carry >1 timestamp, all one template, where always-emitting-one is right 86.8% of the time.
- **The open item the plan could not close, closed by a reviewer:** the `2a`/`2b` router
  `"for how long" | "How much time passes"` (case-insensitive) separates **261/261 `2b` from
  0/7,384 `2a`** — exact on all 7,645 rows, zero template overlap between the leaves.

## G. Silent-failure modes none of V1–V8 would catch

1. `args.json` is never read back ⇒ a leaked `learning_rate`/`num_train_epochs` is invisible.
   **Diff both arms' emitted `args.json`; assert the only difference is `--model`.**
2. The two arms can load **different processors** (`merged/checkpoint-2703` carries
   `longest_edge: 16777216`; stock may not) ⇒ different visual resolution, single variable
   destroyed. **Assert identical `video_grid_thw` distributions over the same N rows.**
3. `frames_cache` is keyed on FRAME's **point** identity (`data.py:117`); nothing verifies a
   cached file's content against the requested timestamp, nor cross-row collisions between
   K-frame windows. Trains on the wrong pixels. Precedent: the stale-image trap.
4. **No post-run `time`-legality gate.** `evaluator.py:338-344` returns `False` behind a
   `logger.debug` on a parse failure. A post-processor emitting well-formed-but-illegal strings on
   100% of `time` rows yields `tg = 0.000`, indistinguishable from the floor, and **V5 will not see
   it** — it counts `"Inference Error:"` sentinels and this is not one. **Add V9: `time` legality
   ≥ 99%, RAISE.**
5. **No per-leaf read.** A total `2b` collapse (87 test rows) moves `tg` by ~3.6% and is invisible
   in every cell D7 declares.
6. **No judge-identity assertion across arms.** 44.9% of the headline is judge-decided; a silent
   judge fallback in one arm only shifts the comparison and all V pass.
7. **`V1` is a liveness gate, not a correctness gate.** LoRA `B` is zero-init, so `grad_norm` is
   non-zero from step 1 regardless of whether the adapter reached anything that matters — which is
   why V2 exists. A run on a mis-routed target or a wrong clock has non-zero grads and a smoothly
   falling loss. And *"a falling loss"* has **no declared threshold**.
8. **`V3` is decorative** — *"dump the effective `video_grid_thw`"* is an observation with no
   expected value. Stage 2 already declares the histogram; **V3 should assert against it.**
9. **`P1` cannot fail** — `du -sx` instead of `df` is right, but **no threshold is declared**.
10. **`P4`'s shrink clause is the wrong assertion.** Rung 15's smoke shrank perfectly and covered
    nothing (16 rows, zero `number`). **The coverage clause is the gate**; the argv check is lesser.

## H. Traps in the renderer the contract does not cover

- 🔴 **The key is read before the trap is registered** (`chain40.py:99` vs `:119`). With `set +e` a
  missing key file leaves `K=""` silently, and then **neither the trap nor the watchdog can stop the
  pod** — it bills until a human notices. This is the exact failure the whole design exists to
  prevent, and L1/L2/L9 do not cover *"the key was empty"*. **Read, assert non-empty, then trap.**
- 🔴 **`chain40` never sets `git config user.name/email`** (`chain.py:164-165` does). On a fresh pod
  `git commit` fails and `:172` swallows it with `|| echo "nothing to commit"` ⇒ combined with
  commit-only-no-push, **the results exist nowhere but as untracked files.** Silent.
- **The eval's return code is never checked** — `:197` commits a message asserting a score that may
  not exist.
- **`git add -A {exp_dir}`** (`:171`) is a regression from `chain.py:166-168`'s allow-list.
- **`chain40` has no smoke stage at all** (`-p SMOKE False` hardcoded at `:181`) and **no GPU
  exclusivity check** (that lives only at `chain.py:107-131`). Both must be ported.
- The watchdog watches a log **the script never writes** (`:287` vs the human-typed redirect at
  `:331`), and there is a **race** where a fast exit deletes the key before the watchdog reads it.
- After 5 failed stop attempts `stop_pod` deletes the credential a manual retry would need (`:117`).

## I. What the plan owes the brain, beyond its own §8

- **RULES §4 itself is FRAME-only** (*"mean over the 4 real buckets; drop `temporal_grounding` n=1"*)
  and would mis-instruct a SEGMENT session. Same for the generated prose at `ledger.py:535, :627`.
- **RULES §4b-i (pooled latency) needs its decision note NOW**, not *"when the template is
  obtainable"* — D3/D8 are already designed against the contradiction, and `RULES.md:280-283` says a
  rule changes only by a note plus the same-commit edit.
- **Three notes, not one.** `PLAN.md:368` bundles three unrelated verdicts; one note cannot carry
  three `verdict:` fields, and `assert_decisions_indexed` RAISES on missing frontmatter — the one
  gate the new tree does **not** escape (`measured.py:48` is root-anchored).
- **`context/INDEX.md` links** for every note (`INDEX.md:125`), **`MEASURED.md` regeneration**,
  **`NOW.md`/`HANDOFF.md`** (opening a second track is the largest state change of the campaign),
  and **`INDEX.md:112`**, which the tree move falsifies.
- **RULES §9 is unsatisfiable for SEGMENT until C4 is fixed** — no SEGMENT row can reach the ledger
  at all. Either make the ledger fix blocking in Stage 1, or state plainly that SEGMENT results live
  outside the ledger and for how long.
- **`AGENTS.md` ≡ `CLAUDE.md` is already broken before any edit**: identical after newline
  normalisation, but `CLAUDE.md` has 115 CRLF and `AGENTS.md` has 0. Needs a `.gitattributes`.

## J. Citation errors in v1 (fix so an implementer is not misled)

- **B2's mechanism is wrong.** The slice is at `run.py:204` and the sort at `run.py:207` — **the
  slice precedes the sort**. The 100%-OOD purity comes from the dataset loop order at `data.py:93`,
  not the sort key. The conclusion holds; someone fixing the comparator changes nothing.
- **L1** cites `chain.py:66`; it is **`:62`**. **P4** cites `recipe_sweep_train.py:275`; the
  statement is the comment at **:272-274**. Substance confirmed in both.
- `PLAN.md:84`'s *"max gold `04:55:57`"* is the max `timestamp_end`; the max `time` **gold** is
  `04:47:37`.
- `experiments/40-gen36-recipe-connector/` is not on `main`.
- **ms-swift 4.x deleted `swift/llm/` entirely** — every `swift/llm/...` path in this lineage 404s.

---

## The one check that retires the most risk, for zero GPU

**Encode ~20 real rows through ms-swift's own `qwen3_vl` template on CPU, against the
`merged/checkpoint-2703` processor** (not the base — that is what arm (a) loads). Print per row:
total tokens, `video_grid_thw`, frames actually consumed, and the resulting
`video_metadata['fps']`. It settles **A2, A3, the `<video>` tag question, and the real
sequence-length distribution** — which in turn fixes s/it, VRAM, the GPU choice and the watchdog
wall — in minutes, before anything is booked.

⚠️ ms-swift is **not installed locally**, so this needs an environment that has it. It does not
need a GPU.
