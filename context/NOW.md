# context/NOW.md — what is happening RIGHT NOW

> The living current-state of the project. Updated as things change. Read this + `context/INDEX.md`
> to get oriented fast. (Supersedes the older `HANDOFF.md` baseline-run handoff, kept as history.)
> Last updated: **2026-07-30**.

## 🟢 2026-07-30 — the recipe sweep is CLOSED, and we have a new best checkpoint

⚠️ **Adversarially audited the same day; three headlines were corrected.** The corrections are
in place below and collected in the decision note's "What the audit changed" table.

**Read [[recipe-axis-is-the-learning-rate]].** Five arms, one flag each, zero data change in
any of them. **Only the learning rate is real.**

| arm | flag | baseline | Δ proxy @ep3 | paired cells (of 30) |
|---|---|---|---|---|
| `A_lr` | lr 2e-5 → **1e-4** | rung 18 | **+0.0480** | **21 sig, all pro-arm** |
| `A2_lr` | lr 1e-4 → **2e-4** | `A_lr` | **+0.0203** | 3 sig, all pro-arm |
| `B_rank` | r 8→32, α 32→128 | `A_lr` | +0.0193 | **0 sig** |
| `D_clip` | `max_grad_norm` 1.0 → 10.0 | `A2_lr` | −0.0051 | 3 sig, **all at ep1/ep2** |
| `A3_vitlr` | `vit_lr` 2e-4 → 2e-5 | `A2_lr` | **−0.0278** | **4 sig, all pro-CONTROL** |

🟢 **BEST CHECKPOINT OF THE CAMPAIGN — `21_lr_2e4_v1/checkpoint-2703`** (arm A2, epoch 3):
proxy **0.6104**, `bucket_mean` **0.6496**, `margin_OOD` **0.2343**. Rung 06 ep3's 0.5724 had
stood since 13 July; nineteen rungs of data work did not move it and one flag did.

🔻 **Lowering the ViT learning rate costs 0.028 — but the ViT reading is DOWNGRADED.** A3 loses
significantly (4 of 30 cells, all pro-control). ⚠️ **It is a TWO-flag arm**: `--optimizer
multimodal` is emitted if and only if `vit_lr` is set, and **A2 never passed it**, so the two
checkpoints differ in two things. The single-variable gate diffs against a *synthetic* baseline
config and omits `optimizer` from its artifact check. Defensible claim: *lowering `vit_lr`
under the multimodal optimizer costs 0.028.* **NOT** "the tower wants the high LR", and
[[vit-lora-partial]] stays **OPEN**. 🔴 **Open action: a 20-step probe of A2's config with and
without `--optimizer multimodal` at `vit_lr == learning_rate`.**

⚠️ `--vit_lr` is a **silent no-op unless `--optimizer multimodal` is passed** — the trap is real
and still worth more than the arm it guarded.

🔻 **Rank is OPEN — and by the pre-registration it is a WIN.** `PLAN.md:114-116` says a win =
proxy rises AND `margin_OOD` does not fall. B at ep3: **+0.0193 / +0.0155** — it passes, at ep2
and ep3. The CI gate that demoted it was the *noise instrument*, never a decision rule. B and A2
were **never compared to each other**.

🔻 **And A2's edge is not where the leaderboard looks.** The proxy is ID-only; on the ID cell
A2 is +0.0221 **[−0.0017, 0.0449]** and B is +0.0212 **[−0.0039, 0.0468]** — *neither excludes
zero*. A2 ships for being top-scoring and 4× cheaper, not for a significant edge over B.

⚠️ **The cost the headline hides:** A2 appears to give back most of arm A's class-balanced F1 on
ID (0.6906 → **0.5474**) while exact-match rises — the `Clip` attractor. 🔻 **No paired CI on
macro-F1 exists**, so this is a point estimate quoted against the rung's own rule.

⚠️ **No multiplicity correction** anywhere: 150 paired cells at 95% ⇒ ~7.5 false positives
expected. Bears on A2's 3 cells and D_clip's 3, not on arm A's 21 of 30.

**Next:** submit A2 ep3 (9 of 10 slots left), and rebase rung 22 (loss-mass) onto A2 — it was
designed against a 2e-5 recipe whose gradient behaviour it no longer describes.

## 🔴 2026-07-29 — rung 21 is the RECIPE now, and it is training

**The two optimiser rungs swapped places.** `21-loss-mass` became `22-loss-mass`; rung 21 is
`21-recipe-sweep`. The argument is the leaderboard proxy, `mean(aggregation_ID,
object_recognition_ID)`: loss-mass moves gradient from `fo_class` (71% of `object_recognition`)
to `number` (80.4% of `aggregation`), so it is **structurally near-zero-sum on exactly the
number that gates co-authorship** — its own pre-registration says the modal outcome is a wash.
The recipe is not a trade: it adds optimisation distance to both buckets. And the ordering
matters, because whether taking gradient from `fo_class` costs anything depends on whether
`fo_class` has saturated, which is what lr and epochs move.

**IN FLIGHT — arm A: `--learning_rate` 2e-5 → 1e-4.** One flag. 3 epochs, 2,703 steps, ~7.6 h
at the measured 10.1 s/it, peak 22,210 MiB of 32,607. Run `21_lr_1e4_v1`, log
`/workspace/tmp/21_full_A.log`. **Control = rung 18's already-scored per-epoch series**
(0.5255 / 0.5488 / **0.5721**) on the **same `train.jsonl`, used in place**, sha256
`180e28f0…8e8b` asserted. Zero data change. Score with `21b_epoch_eval.ipynb -p EPOCH n`.

⚠️ **Named before it ran:** our LoRA reaches the ViT at the LLM's own LR while the Qwen3-VL
default puts the tower 5–10× lower. A collapse in arm A may be the **tower**, not the recipe —
the pre-registered diagnostic is **A2 = lr 1e-4 with `vit_lr` held at 2e-5**. Arm B (rank 8→32,
α 32→128 so α/r stays 4) is **not launched until A is read**.

🟢 **The loss-mass mechanism is now MEASURED, not inferred** (`22-loss-mass/RESULTS_preflight.json`):
the trainer's `num_items_in_batch` equals the token sum of all 16 micro-batches, to the token,
every step. ⚠️ The decision note's original "confirmed at the source" quotation was incomplete —
`seq2seq_trainer.py:196` sits under `if num_items_in_batch is None:` and, had the guard been
open, would have meant training was ALREADY per-sample and the whole 62.4/20.8 table an
artefact. Free findings from the same probe: `max_grad_norm` is **1.0** (never set by us), and
`swift/plugin/loss_scale/` **does not exist** in ms-swift 4.4.1 — the hook is `compute_loss_func`.

🟢 **`frame.metrics.class_f1_report`** landed (`0674f02`) — class-balanced F1 on `fo_class`,
validated against probe0 (rung 06 ep3 ID: n=920, exact 0.6391, macro 0.5116). It is what can
see the tail that a 0.6478 headline hides (`Gallstone` recall 0.036).

## 🔴 2026-07-27 (pm) — submission 01's metrics decoded: a 4B beats us, and we were reading the wrong number

**Read [[leaderboard-metric-vs-our-headline]] before quoting any number as "our score".**

🔴 **We were comparing incomparable quantities.** `pre_evaluation_score` is an unweighted mean over
**populated** buckets and only **two** populate on the pre-eval set, both ID — while our
`bucket_mean` averages **four** (ID+OOD). The right local comparator is **mean-ID = 0.5281**, not
0.5667, so the real local↔leaderboard gap is **−0.057**, not −0.096. Now RULES §4b.

🔴 **On identical questions a 4B beats us 0.5163 vs 0.4710.** The exact rationals (`343/754`,
`607/1246` ours; `410/754`, `609/1246` theirs) recover **B = 2000 over 20 videos** and prove the
same set. The whole margin is `aggregation` — **67 questions** — and `object_recognition` is a
**two-question tie**. ⇒ **[[aggregation-is-the-gap]]'s "+14.9 lead on `object_recognition`" is
RETRACTED**; it had compared our local val against their platform score. The `aggregation`
prescription survives and is better supported.

🟢 **Latency is a non-issue and the alarm was arithmetic.** `mean_latency_s × throughput = 20.0`
**exactly** (both teams) — one number, not two. Real cost **0.79 s/question** against a **5.06**
ceiling: **6.4× headroom**. Corroborated by running the container on the organizers' fixture
(0.78 s/q warm, 31.9 s setup of a 120 s allowance).

🟢 **`heico`=OOD is the organizers' own design, not our invention** (RULES §3). Their public
partition puts **Sigmoid Resection — a procedure absent from ALL training** — in `heico` test,
which is exactly the official *"OOD tag with respect to procedure type"*. The empty `ood` column
is a publication choice. ⇒ OOD work is **not** wasted: the final ranking weights ID and OOD
**equally** (`challenge_design.txt:2001`), and it is **Copeland with significance tests**, so a
+0.003 lever buys nothing (RULES §4c).

🔴 **Zero training examples for `event_understanding` and `complex_reasoning`** — two of five
groups, `primary_capability` is only `1a,1c,1d,1e,2a,3a` across all 20,000 public questions
(RULES §4d).

⚠️ **Selection bias, unmeasured.** `val_id ∪ val_ood` IS the whole 6252 set; every rung selects by
`idxmax(acc_ood)` over **10 videos** and then reports on a set containing those same questions.
Biases absolute numbers upward; rung-vs-rung deltas largely cancel. `kfold_lopo` (`split.py:342`)
exists and is unused.

**The submission itself was NOT broken** — an adversarial audit cleared frames, prompt, generation,
offline and Dockerfile. It has since been hardened anyway (`batch.json` layout now read,
case-insensitive frame matching, loud failure, CUDA hard-fail) and has produced a real
`answer.json` against the organizers' fixture for the first time.

🔒 **Open:** we do **not** know where the baselines sit. `challenge_design.txt:453` gates the final
stage on beating **both**, `:375` says they would be "clearly identified" on the leaderboard, and
the visible leaderboard shows 13 rows, all participant teams. Worth asking the organizers.

## 🟢 2026-07-27 — the missing control was run: rungs 14 and 15 are CLOSED, and epoch 3 erases OOD counting

**Rung 06 epoch 3 = `bucket_mean` 0.5724** (`experiments/06-vit-lora/06c_epoch3_eval.ipynb`, RTX
5090, full 6252, protocol identical to `eval_best`, 31 min, **zero training**, ~$1 of pod). All
gates green, gold 6252/6252, hybrid cross-check agrees with the vendor scorer. This is the number
[[epoch-matched-control]] said had to exist before anything else could be read.

🔴 **Both team rungs are closed.** Rung 15's 0.5699 was a win over rung 06's *wrong epoch*, not
over rung 06: against the epoch-matched control it is +0.0017 ID / **−0.0078 OOD** with **0 of 6**
paired video-clustered cells excluding zero. Rung 14 is now the only rung with a significant
cell — `ID ALL −0.0239 [−0.0433, −0.0035]`, i.e. **significantly worse** than the control. The
14+15 fusion is worse-motivated than when proposed: a measured-negative plus a measured-null.

⚠️ **Rung 06 ep3 does NOT become the shipped checkpoint.** It is the ladder's best headline
(+0.0057 over ep2) and a **statistical null** — 0 of 6 paired cells exclude zero. The standard
that closes 14 and 15 closes this too. Submission 01's ep2 checkpoint stands.

🔴 **The finding that outlives the adjudication:** at ep3 `number` on OOD scores
**exactly the trivial floor to 16 digits** (0.46907993966817496 vs floor 0.46907993966817496) —
the signature rung 05 recorded for a **black image**. Across epochs the OOD counting margin decays
monotonically (+0.0151 → +0.0128 → **0.0000**) while ID counting climbs (+0.0781 → +0.0859 →
+0.1068). Training **erases** OOD counting, and epoch 3 is where the erasure completes.

**Two repo defects surfaced, neither fixed here.** (1) `frame.ledger._discover_stratified` globs
`runs/**/stratified.json` recursively and tier 1 does **not** dedup, so two `stratified.json` under
one run dir produce two identical rows — triggered by an untracked stray
`experiments/02-lora-sft/runs/02_lora_sft_v1/eval_best/`, which was parked (not deleted) for the
ledger rebuild. (2) The rung-10/rung-12 phantom rows are **still in the committed ledger** on this
branch (8 extra rows); the root fix lives on `task/audit-rung12`, **unmerged**.

⚠️ Still no seed repeat, ever. Every delta above sits inside a variance band we have never
measured. The user declined seed repeats this session.

## 🟢 2026-07-25 (pm) — the whole repo is consolidated onto `main`, and rung 16 → 17

**`main` is now the single source of truth — it carries EVERYTHING** (merge commit `abdcbcf`,
then `9d663e4`): submission 01 (rung 06), the **CoA line** (rung 09 — `gen_onpod.py`,
`coa_scaffold_gen.py`), and the full **R2 wave** (experiments 13/14/15/16→17 + result CSVs +
robustness analysis + the two literature corpora). No more sibling divergence — **everyone
branches off `main` going forward** (Leo took over the CoA line; Rodrigo owns the rest).

**How it happened.** Two entangled sibling branches existed — `task/r1-coa-sft` (Rodrigo, +12,
held rung 09 alone) and `task/r2-lit-levers` (mostly Leo, +38, held the wave). Neither contained
the other, and Leo had meanwhile moved to committing **directly on `main`** (31 commits incl.
submission 01). Consolidation: FF `main → r1` (brought CoA + all of main), then merged `r2 → main`
(brought the wave). Three brain conflicts were **union-resolved** (INDEX.md, RULES.md kept both
sides; NOW.md kept both dated sections newest-first) — **brain polish is still deferred**, this is
functional-not-pretty. PR #1 auto-marked MERGED.

**Branches after the sweep.** Deleted (local + GitHub): `task/r1-coa-sft`, `task/r2-lit-levers`
(their work is all in `main`), plus 12 fully-merged stale branches earlier the same day. **Kept,
untouched:** `task/audit-rung12` (+32, decision pending), `origin/task/image-processing` (+28,
🔒 do-not-touch per user), `origin/task/enumeration-probe` (+9). Local safety refs `backup/*`
remain, deletable anytime.

**Rung 16 → 17 renumber.** The CoA generator perception probe (`generator-probe`) moved from rung
16 to **rung 17** (`experiments/17-generator-probe/`, notebook `17_generator_probe.ipynb`,
`context/17-generator-probe/`) to **free the 16 slot for the 14/15 continuation work**. Pure
renumber — folder + context + notebook + every internal label/path. **Agent-audited clean:** no
dangling refs, NOW.md inbound links repointed, outbound dep `../09-coa-sft/_tools/gen_onpod.py`
resolves, notebook JSON valid. The ladder now skips 16 (unused), normal for these skip-numbered
ladders. Rung 17 is still **PRE-REGISTERED, UNRUN** (needs ≥80 GB).

**Correction booked this session:** rung 14 (appearance-aug) is a **statistical** null, NOT
"flat/did nothing" — its best checkpoint lifts `margin_OOD` **+0.0078** (OOD `fo_class` **+0.017**)
at an ID cost of −0.0116, a real ID↔OOD robustness trade in the designed direction, just inside
the noise band (CI [−0.0038, +0.0195] crosses 0). rung 15 lifts `number` **on ID only**
(`number_margin_ID` +0.117 vs rung 06's ~0), evaporates OOD. Both are directional signals worth
stacking/dosing, not dead ends — see [[epoch-matched-control]].

## 🟢 2026-07-25 — the first submission is UPLOADING, and the team's two rungs are read.

**Submission 01 is going up** ([[submission-01-rung06]]): algorithm `Qwen3VL-8B-FT-ViT-LLM-v1`,
the merged rung-06 checkpoint (ep2 / step 1720), 17 GB tarball, offline, built from the official
template. **The leaderboard score is the distance-to-baselines number the whole campaign has
lacked** — it decides whether levers of the +0.003 size are worth a 7.5 h run at all.

🔴 **It nearly shipped broken.** The platform's algorithm interface declares `batch-frames` as a
**ZIP at `/input/batch-frames.zip`**, while the organizers' own template documents
**`/input/frames/<qID>.png`** and ships a plain-directory fixture. Our `inference.py` followed the
template; had the ZIP arrived, the per-question `except` would have written a complete
`answer.json` of **empty answers** — a silent zero costing 1 of 10 submissions. The container now
**accepts both layouts** and **logs the `/input` inventory before loading weights**, so a failure
names its own cause. Prompt verified **byte-identical** to the scored engine across all paths.

🔴 **The team's rungs 14 and 15 do NOT beat rung 06 — and the ep3 comparison has no control**
([[epoch-matched-control]]). Rung 14 (appearance-aug) is a faithful **NULL**: every epoch below
rung 06, pre-registered `margin_OOD` +0.0078 with a video-clustered CI **[−0.0038, +0.0195]**.
Rung 15 (count-target) ep3 reads **0.5699 vs 0.5667**, but it is **ID-driven** — `margin_OOD`
**−0.0110**, `number` margin OOD **−0.0053 (below floor)**, its own pre-registered target — and
**0 of 10** paired cells exclude zero. The structured target parsed perfectly (0.0 % malformed),
so the **format worked and the counting did not**. ⚠️ **Correction to the 07-24 note:** the
"rung 15 = 0.5612" reading was **epoch 2 of a still-running eval**, not the rung.

🔴 **The load-bearing gap: rung 06 never evaluated its own ep3** while `eval_loss` was still
falling (0.3204 → 0.2925 → 0.2782). Its `checkpoint-2580` adapter is on the volume, unmerged and
unscored — so **both team rungs compare their ep3 against rung 06's ep2**, which is exactly where
rung 15's only win lives. Closing it is **merge + eval, ~1 h, zero training**, and it settles both
rungs at once. **Do it before any new training run**, especially before the proposed **14+15
fusion** — which would combine two nulls, break single-variable attribution, and inherit the same
missing control. New rule: **RULES §6b — every epoch evaluated, comparisons epoch-matched.**

⚠️ **And the thing none of our tooling can currently answer: no run has EVER been repeated with a
different seed.** We hold **no variance estimate**, so a +0.003 "effect" is formally
indistinguishable from noise. This is the open question legokna raised for the next phase — the
levers may be mis-aimed rather than the training wrong.

## 🔴 2026-07-24 — Wave R2: literature-grounded levers, and two project premises falsified

**The whole session's thesis.** Two paths were on the table — preprocess the image (Leo's rung 12
line) or CoA (change the training target). A literature sweep (87 byte-verified PDFs, new corpora
`literature/preprocessing/` + `literature/vlm-techniques/`) found that **both configurations we were
about to run are published as wrong**, and re-scoped the plan onto the axes the literature says DO
win. See [[coa-sft-published-null]] and [[inference-only-input-tests-biased]].

### Running RIGHT NOW — pod `m7s3xq835y9hd4` (RTX 5090, EU-RO-1), unattended, self-stopping
Driver → `r15` → `r14`; finisher then runs `r16`, rebuilds the ledger, commits+pushes `results/`,
and stops the pod. ~10 h left. Details + the finisher's traps in [[lora-training-in-progress]].
⚠️ GPU note: two prior pods hung in provisioning; the wave first ran on a PRO 6000 Workstation
(27.5 s/step, the slow card — my error, [[train-on-powerful-gpu]] had the measurement) and was cut
over to the 5090 (~12 s/step for ViT+LLM training).

### The four R2 rungs — single-variable vs rung 06 (0.5667), pre-registered before any number

| rung | the one variable | what it attacks | status |
|---|---|---|---|
| **13** `13-wise-ft` | interpolate weights base↔rung06 (no training) | recover `number` erased by training | 🔴 **DONE — NEGATIVE** |
| **15** `15-count-target` | `number` target `"3"` → `{"label":…,"counts":3}` | counting collapse | 🔄 trained (100%), evaluating |
| **14** `14-appearance-aug` | colour/WB augmentation DURING LoRA | the OOD half | ⏳ next |
| **17** `17-generator-probe` | 32B answers with NO gold, scored canonically | is the CoA teacher a real perceiver? | ⏳ needs ≥80 GB — deferred |

**rung 13 result (in the ledger):** all three α NO-WIN — α=0.50→0.5016, 0.70→0.5530, 0.85→0.5645
(vs 0.5667). `number` margin never rises strictly in BOTH distributions and no paired video-clustered
CI excludes 0. Interpolation does not buy back counting for free. Faithful negative, zero training GPU.

### 🔴 The two falsified premises (the session's real product)

1. **CoA-format SFT is a published wash — [[coa-sft-published-null]].** exp 09 asserted "there is NO
   `SFT+CoA, no-RL` row in the source." **It exists**, in Chain-of-Adaptation (arXiv:2603.20116,
   Qwen3-VL-8B, our exact scaffold, verified in the PDF): scaffold-SFT-without-RL scores **62.0 vs
   bare-gold SFT 65.7** on EndoVis2018 — a wash that *loses*; **RLVR-with-no-tags already beats SFT
   (67.4)**; the +18 is RLVR's. The false premise came from `Bloque-A-Modelo.md`, an untracked local
   summary nobody could audit. ⇒ CoA is **re-scoped, not cancelled**: it is a cold start for a later
   **RLVR** rung (the only thing with +18 on our backbone). Rodrigo reserved that decision.
2. **rung 12's image-processing screen measured the wrong thing — [[inference-only-input-tests-biased]]**
   + [[pooled-screening-manufactures-winners]]. It ranked descriptor separability, which Awad 2025
   states does NOT predict downstream gain; and its inference-only test was biased-to-negative by a
   design Jong 2025 / Medeiros 2026 already characterised. The three axes that DO win
   (train-with-transform, geometry, frame-selection) — we tried one, badly, and two never. rung 14
   is the honest version of the first.

### Repo integrity fixed this session (numbers that existed only as prose)
- **rung 12c recovered from the pod volume** — its trained A/B (`+0.021`) was never committed; now
  reproducible. Corrected a mis-reported delta (OOD +0.0005 → **+0.0040**), verdict unchanged (null).
- **Ledger de-poisoned** — rung 12's per-arm CSV injected 8 phantom rows, rung 10 appeared 3×; root
  fix in `frame.ledger._tier1_rows_from_csv` (21→19 rows, dedup within file). Five decision notes +
  six document corrections for Leo's campaign, on branch `task/audit-rung12`.
- 🔴 **No archived result is bit-reproducible — [[archived-results-not-bit-reproducible]].** ~0.5 %
  of stored answers change on a GPU swap (measured: rung 13's α=1.0 gate, 199/200 vs the 5090 while
  the weights were bit-identical). Identity gates must build their control on the same machine.

### Open / next (Rodrigo)
- **rung 17 (32B blind perception probe)** — the gate for the whole CoA/RLVR line. Needs a ≥80 GB pod.
  Everything built + pushed; on a big pod: `papermill 17_generator_probe.ipynb -p SMOKE False`.
- **rung 15 caveat to read with its result:** it replicates Gautam 2025's structured count *format*
  but NOT the *pointing* (coordinates) — our dataset has no boxes. ⚠️ **Corrected 2026-07-27:** the
  pointing did **not** drive the 9.86→0.26; that is v05's **counting-ONLY** arm. Per its Table I the
  joint count+point arm reaches only **1.52**, so dropping pointing is if anything **favourable** for
  counting (different evaluation subsets, so not a strictly paired ablation). Partial replication of
  the *format*, not a weakened copy of the better arm.
- 🔒 **Rotate the GitHub PAT** — it is in plaintext in the pod's git remote URL.

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
- **Leo (legokna)** → **submission 01 uploading** ([[submission-01-rung06]]); next, the **CoA rungs
  delegated to us** (`experiments/09-coa-sft`, on the pod volume, not yet in this tree). Then the
  question raised 07-25: **the levers may be mis-aimed rather than the training wrong** — every
  model-side result lands inside a noise band we have never measured (no seed repeat, ever), while
  the ceiling keeps pointing at the DATA (see [[rung12-dominated]] below and
  [[epoch-matched-control]]).
- **Team (RodMed)** → rungs 13–16 on branch `task/r2-lit-levers`. **13 WiSE-FT = NO-WIN** (3 α),
  **14 appearance-aug = NULL**, **15 count-target = no win under the ID-AND-OOD conjunction**,
  17 generator-probe open. A **14+15 fusion** is under discussion **and is theirs to decide** — our
  input is [[epoch-matched-control]]: run the missing ep3 control first, and note that combining
  two nulls breaks single-variable attribution.
- 📕 **CLOSED — rung 12 image processing.** Kept below as the reasoning record; the input-side
  family is **dominated by the ViT-unfreeze**, and its living heir (train-time augmentation) was
  the team's rung 14, now measured NULL. Branch A (unsharp ×3 at inference, on the fine-tuned model) is a **faithful NEGATIVE and monotonic in dose** — −0.056 ID / −0.058 OOD, both significant; the identity gate passed 50/50 byte-identical. 🔴 **The method correction it produced is the important part: any inference-only test of an INPUT intervention is biased toward the negative** on a model fine-tuned without it. That re-scopes branch A itself and, retroactively, rung 11 — it does **not** touch the output family (voting/calibration/enumeration), which died with no mismatch at all. **The rung stays open because branch B decides it** — the same two arms on the **ZERO-SHOT** model (~52 min of 5090, all code exists, only `model_path` changes), far less locked to our frames' appearance. ⚠️ Poor instrument (`bucket_mean` 0.2557, **below floor everywhere**): read it for DIRECTION, never magnitude. 🔴 **But branch B cannot be the fair test either — it is still inference.** The honest test of the whole input-side family is to **TRAIN with the transform**, which is why **idea 2 (proportional subsampled-training harness, `local/hallazgos/ideas-mejora.md` §2) is escalated from convenience to ENABLER**: it takes a run from **7.5 h to ~1.5 h**, costs only CPU hours, and unblocks the input family and the re-scoped rung 11 alike. Subsample **questions within ALL videos** — dropping videos would destroy the effective n of 38. Its known limit: the LR optimum shifts with size, so it serves **relative** comparisons, not absolute values.
  - 🆕 **12d (2026-07-21) — 32 transforms screened for ZERO GPU, and the load-bearing result is methodological** ([[context/12-image-processing/CONTEXT.md]]). `tophat` came **first** pooled (+0.0043) and collapses to **−0.0172 measured inside each video**: its advantage was between videos, because videos containing an object are videos that *look different*. 🔴 **Pooled screening of image transforms manufactures winners** — the same confound as rung 11, and the ≥3-videos gate does not prevent it. `within_video()` is now the standard gate and the primary metric. Against a raw-image separation of 0.2181: **every pipeline ending in an edge operator is strongly negative (−0.018 to −0.054)**, the only two survivors both **preserve** the image (`despec+clahe` +0.0052, `despec+bilateral+unsharp` +0.0031), and `homomorphic` was **mis-calibrated, not dead** (−0.032 → +0.0012 as it softens). **Nothing is validated** — `despec+clahe` is p=0.045 on one comparison out of 13 and fails Bonferroni. Two defects are recorded, not hidden: the negative class is **contaminated** (`scene_inventory` is `partial: true` on **100 %** of frames) and **conditional effects are averaged away**. The screen's record is **four candidates killed, none validated** → it is an instrument of **exclusion, not selection**.
  - ⚠️ **Also measured 12d:** of **8,969 `fo_class` questions, ZERO have gold `none`** although the prompt offers it. The dataset contains **no negative case** — an irreducible ceiling for any perception-side lever, and the reason a human reviewer finds frames with no visible object that the label still asserts.
  - 🆕 **12d bis (2026-07-21) — the "edge family is strongly negative" headline is RETRACTED, and it is a second instrument defect.** `wv_delta` ranks the **best of 18 descriptors** per cell (`idxmax`, `transform_bank.py:483`). Split max from mean: the edge family raises **every** descriptor (`identity` 0.1069 → 0.1245 `bilateral+morphgrad`, 0.1311 `despec+tophat`, 0.1380 `clahe+despec+morphgrad`) while lowering the best one. It does not destroy information — it **redistributes** it onto one axis, and a *maximum* reads compression as loss. Symmetrically the two "winners" barely move the mean (`despec+clahe` 0.1092): they rank first by **preserving** the standout descriptor, not by adding signal. ⇒ the negative stands as a claim about this statistic, **not** about information, and a VLM consumes pixels rather than one descriptor. **Only `homo+sobel` dies on both readings** (mean 0.088 **and** max 0.164, both below identity).
  - **Five transforms selected for the next stage (legokna):** `despec+clahe`, `despec+bilateral+unsharp`, `homo_soft`, `bilateral+morphgrad`, `homo_soft+morphgrad` — the last two are the edge-family rivals the two metrics disagree on, and the pair with `homo_soft` **isolates `morphgrad` as a single variable**. ⚠️ Ranks 4–5 of the screen are the **null anchors**: only 3 of 14 pipelines beat doing nothing. ⚠️ `clahe+despec+morphgrad` leads the *mean* metric (0.1380) and was passed over on the CLAHE vein-noise objection — it is the candidate that reading would pick. ⚠️ `despec+bilateral+unsharp` contains `unsharp`, the only bank member with a real model measurement (branch A **−0.056, monotone**): screen and model **disagree in sign**.
  - 🔴 **12e (2026-07-21) — the conditional hypothesis is a faithful NEGATIVE, and open point ② is closed.** It was the last cheap explanation for the rung: that `wv_delta` averages away a sign-flipping effect (helps on conspicuous objects, hurts on camouflaged ones), which would have explained branch A's −0.056 too. **First, the literal test is NOT MEASURABLE** — splitting 235 cells by the model's right/wrong verdict leaves **14** clearing the gate (19 using every format), because the model was asked about only **4,486 of the 15,213** indexed frames. That negative is recorded, not worked around. **Measured instead:** inside each video, does a transform's descriptor separate frames the model gets right from those it fails (37/38 videos, 3,977 frames, null band permuting the verdict within video, Bonferroni |z|>3.1)? **On the primary max statistic nothing clears the band and `null_jpeg` ranks FIRST** — the textbook signature of no effect. Two transforms clear on the *mean* statistic (`bilateral+morphgrad` +3.91σ, `homo_soft+morphgrad` +3.28σ) **but collapse when restricted to frames containing the class**, so the parsimonious reading is **class/scene composition, not conspicuity**. ⇒ **None of the five selected transforms has evidence of touching what the model actually gets wrong**, which *lowers* the case for spending pod on them as they stand. Does **not** close 12c, the train-with-transform test, or the contamination defect.
  - 🟢 **12c-res (2026-07-21) — the aux view can be sent at HALF resolution for free, zero GPU.** Settling this inside the training A/B would have been fatal (a negative composite arm could not be told from a map crippled by downscaling). `bilateral+morphgrad` Δ mean vs identity: full +0.0176, **half +0.0199**, quarter +0.0187, **1/16 −0.0113**. 🔴 **The control is what makes it readable:** the first run also said halving the *raw photograph* costs nothing (+0.0009), and the descriptors are tile statistics ≈ scale-invariant by construction — so the instrument was suspected blind before it was believed, and retested at an extreme dose. It is **not** blind (monotone to 1/16, where the map collapses): **plateau to 1/4, then a cliff**. ⇒ composite token penalty falls ~2× → **~1.25×**; compute the map at native resolution and **then** shrink (`half_post` +0.0199 vs `half_pre` +0.0165); and the map choice is scale-invariant, supporting `bilateral+morphgrad` alone. Also: the `aux_view` flag is in (`engine.py`/`config.py`, default OFF byte-identical, `gate.aux_view_payload_gate` green without a GPU).
  - **Open, in priority order (2026-07-22):** ① **12c is untested** — the screen *replaces* the image with the edge map; the proposal is image **plus** map, the only configuration in which the edge family can still work. ② **Split by model right/wrong** — does the sign of a transform invert between frames the model already gets right and the ones it fails? Zero GPU, and it would explain the whole rung if appearance help lands where the model already succeeds. ③ **Negative-class contamination** — the most serious defect of the screen, with no known fix short of annotation.
  - **On `task/image-processing`, pushed, deliberately NOT merged.**
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
