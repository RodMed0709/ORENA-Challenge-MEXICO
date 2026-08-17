# context/RULES.md — operating rules the brain enforces

> The DO / DON'T rules that keep results **consistent** and stop recurring mistakes —
> human OR Claude. Read with `context/INDEX.md` before touching evaluation, training,
> or results. A rule here is not advice; it is how we avoid re-breaking what we already
> fixed. **When a rule and a number disagree, the number is the suspect — not the rule.**
>
> These complement `CONSTITUTION.md` (broad binding law): CONSTITUTION is the project's
> constitution; this file is the operational checklist for the traps we have actually hit.

## EVAL — how metrics are computed (canonical, non-negotiable)

1. **One source of truth.** Score ONLY via `frame.metrics.stratified_report`. NEVER
   re-derive buckets / floors / accuracy inline in a notebook. Ad-hoc re-derivation is
   exactly how the same bug came back in rung 07. If the module lacks something, EXTEND
   the module — do not reimplement beside it.

2. **Leaf→group ALWAYS via `Capability.group`** (`split._group`). `results_df["primary"]`
   is a taxonomy LEAF (e.g. `object_identification`); the scored bucket is a GROUP
   (`object_recognition`). NEVER filter group names against the leaf column — it silently
   drops rows (964 lost in rung 07).

3. **ID/OOD ALWAYS from the qID prefix** (`heico`=OOD, `lapchole`=ID). NEVER trust
   `results_df["ood"]` — it is all-False on the public data (organizers populate it only
   in their private test split).

   🟢 **This is a faithful reconstruction of the organizers' design, NOT a convention we
   invented** — measured 2026-07-27, and worth knowing because it is the difference between
   an arbitrary proxy and the official axis. `challenge_design.txt:1018` defines the tag as
   *"In-distribution (ID) vs Out-of-distribution (OOD) **with respect to procedure type** and
   question"*, and they encoded exactly that in how they partitioned the public data:

   | dataset | train procedures | test procedure |
   |---|---|---|
   | `heico` | Proctocolectomy, Rectal Resection | **Sigmoid Resection — absent from ALL training** |
   | `lapchole` | Lap. Cholecystectomy | Lap. Cholecystectomy (same) |

   So `heico` test is genuinely unseen-procedure OOD and `lapchole` test is ID, by their own
   definition. The empty `ood` column is a publication choice (they do not hand you the tag),
   not an absence of design. `split.py` records it as `organizer:heico-test(Sigmoid Resection)`.

4. **Headline = `bucket_mean`** (mean over the 4 real buckets; drop `temporal_grounding`
   n=1). `pre_evaluation_score` is REFERENCE ONLY — it is broken on our split: an
   unweighted bucket mean that a single n=1 question inflates (it lifted rung-02 from an
   honest **0.550** to a reported **0.708**).

4b. 🟢 **RETRACTED AND REPLACED 2026-07-30 — the headline IS `bucket_mean`.** The platform's
   own `_docs` defines `pre_evaluation_score` as the *"unweighted mean accuracy over the 10
   buckets (5 capability groups **× in-/out-of-distribution**)"*, and its four populated
   buckets reproduce the reported score **exactly to 1e-15**. **Score `bucket_mean`. It was
   always the right comparator.**
   ⚠️ The retracted version of this rule said the headline was an ID-only mean over 2
   populated buckets, and prescribed `mean(aggregation_ID, object_recognition_ID)` as the
   comparator. That came from a **partial** payload that reported everything under two ID keys.
   Rung 21 was designed and reported against that proxy; **no verdict flipped** (the arms rank
   identically under both), but the offset, the `object_recognition` collapse size and every
   "mean-ID" figure are withdrawn. See [[leaderboard-metric-is-bucket-mean]].
   ⚠️ Still true and still load-bearing: a `null` bucket means **EMPTY, not broken**, and empty
   buckets are **excluded from the mean**, not counted as zero. FRAME populates only
   `aggregation` and `object_recognition`; the other three groups belong to the video tracks.

4b-i. **Latency is pooled PER BATCH and we are 12.5× under it.** `120 s setup + B × 5 s` per
   batch of B=20; submission 01 used **17.56 s** of 220 and reported
   `mean_latency_per_question_s: 0.0` (the whole job fit inside the setup allowance).
   🔴 **Overrunning by 20% forfeits the ENTIRE batch**; smaller overruns forfeit questions
   *proportionally*, chosen deterministically and stratified across buckets — **not** the
   individually slow ones. One slow question does not cost itself, it costs a share of the
   batch.

4c. **The FINAL ranking is not a mean at all.** It is **Copeland over buckets with pairwise
   significance tests** — non-significant deltas collapse to the SAME rank
   (`challenge_design.txt:1010-1040`) — weighting ID and OOD **equally** (`:2001`). Mean
   accuracy applies only to clearing a baseline in pre-eval (`:1039`). ⇒ A lever worth +0.003
   buys nothing under the final ranking, and OOD work is not wasted merely because the
   pre-eval cannot see it.

4d. ⚠️ **We hold ZERO training examples for `event_understanding` and `complex_reasoning`.**
   Across all 20,000 public questions `primary_capability` is only `1a, 1c, 1d, 1e, 2a, 3a`.
   The SDK defines five groups and the platform reports ten buckets. Any claim about those two
   groups is **unmeasurable with the data we have** — do not assert one.

5. **Trivial floors: vs the EVAL set, split ID/OOD.** Never against the train prior, never
   read off a suffix-collided merged dataframe.

6. **Checkpoint selection by acc_OOD, per-epoch.** Never last-epoch-by-default. Discard any
   checkpoint that wins ID but drops OOD (chole-overfit / OOD collapse). OOD = 50% of score.

6b. **EVERY epoch must be evaluated, and cross-rung comparisons must be EPOCH-MATCHED.**
   "Selected per-epoch" is not satisfied by scoring a subset: rung 06 never benchmarked its
   own ep3 while `eval_loss` was still falling, so rungs 14 and 15 both compared *their* ep3
   against rung 06's **ep2** — and rung 15's only apparent win lives exactly there. An
   unevaluated epoch is a **missing control**, not a discarded one. Before quoting a delta,
   check that the control was scored at the same epoch. See [[epoch-matched-control]].

7. **Gates RAISE — never disable one.** A gate that fires is a FINDING, not an obstacle.
   NEVER change an expected value to make a broken number pass; fix the code. (Lesson from
   commit `881d057` — the reverted "fix" that papered over a real 964-row drop.)

8. **SMOKE must be stratified** across `heico`+`lapchole`, never a single-dataset prefix.

8b. **NEVER emit a class token outside `FOType.names()`.** The predefined list the organizers
   put INSIDE the prompt and the scoring registry disagree on their 10th element
   (`foreign object` vs `Absorbable Hemostatic Agent`). An unrecognised token does not merely
   score 0 — `FOType.from_name()` **RAISES** (`vendor/orena-focus/src/focus/foreign_objects.py:154-161`)
   and `verify()` marks the answer wrong. Read the accepted set from `FOType.names()` at
   runtime — **never hard-code it** — and suppress or map anything else at the answer
   boundary. Applies to prompts, training targets and post-processors alike.
   See [[open-class-vocabulary]].

8c. **A failed generation is NOT a wrong answer — G-INFER raises.** `engine.predict` swallows
   every exception and returns `"Inference Error: …"` so one bad frame cannot kill a long run.
   The cost is that a *total* engine failure also returns cleanly and scores as incapacity:
   rung 23a's first smoke "completed" at `bucket_mean` 0.0000 because all 24 calls hit a
   missing FP8 kernel, and the only tell was an impossible 19 q/s. `run_baseline` now counts
   those sentinels before evaluation and raises above 1 % (`run.py`, G-INFER). **Never read a
   score from a run whose error rate was not logged**, and never "fix" this gate by relaxing
   the threshold — a run that trips it has no result to report.

9. **Every result → the ledger.** Regenerate root `RESULTS.md` via `frame.ledger`; every
   number must be reproducible from a commit.

9b. 🔴 **A run that scored `fo_class` may NOT publish without a class-balanced macro-F1.**
   `fo_class` accuracy is exact SET equality, so it is dominated by the head of a long-tailed
   class distribution and **cannot see a tail collapse**. Measured three ways: on our exact
   backbone SFT lifts F1 58.7 → 62.4 while crushing **F1cls 20.7 → 15.3**
   (`literature/vlm-techniques/FICHAS.md` §v01); our own rung 18 ep3 scores **0.6478** while
   `Gallstone` recalls **0.036**; and the failure inside `object_recognition` is per-CLASS, not
   per-count ([[class-imbalance-not-counting]]). Compute it with `frame.metrics.class_f1_report`
   — never by hand (§1) — and gate the publication with
   `frame.metrics.assert_class_f1_reported`, which RAISES on a missing or NaN value. Binding for
   every rung, and *especially* for anything that changes the training target, which is the
   intervention that produces the collapse. ⚠️ **Read the `per_class` table beside the scalar:**
   rung 21's `+0.174` macro-F1 was **82% one `Needle` question** flipping.
   See [[class-balanced-f1-is-mandatory]].

## READING results (from the data card, rung 08 — read `experiments/08-data-card/`)
A raw accuracy is meaningless without its trivial floor. Read numbers this way:

10. **Judge by MARGIN over the template-aware floor, not raw accuracy.** The floor = what a dumb
    constant (the per-template modal answer) scores by exploiting the answer distribution alone.
    Margin = accuracy − floor = the real skill added. A high floor makes a raw number look good.
11. **`acc_OOD > acc_ID` does NOT mean better generalization.** The OOD floor is ~12 pts higher
    (answers are easier to guess), so by margin the model adds *less* on OOD. Never read the raw
    OOD>ID gap as "generalises well".
12. **`acc_number` is NOT interpretable — do not quote it.** It averages 8 templates with floors
    from 0.24 to 1.00 (4 degenerate). Use the SDK hierarchical estimate + per-template margins.
13. **Effective n ≈ 38 videos, not 6252.** Questions are not independent (they cluster on 38
    videos); trust the video-level hierarchical CI for "will this hold on a new video?".

13b. 🔴 **A rank correlation is a property of the SLICE, not of the model. Never compare two
    `r` values computed on different gold ranges.** Range restriction attenuates Spearman
    toward zero by construction, so the slice must be named every single time. Measured:
    rung 06 ep3 on the `Clips` template scores **0.5866** over the full gold 1–12,
    **0.4103** restricted to gold 3–6, and **0.8303** on the gold-stratified 16b sample —
    one model, one gold, three numbers. This is how "a human orders better (0.72) than our
    model (0.43)" got written down: a full-range human number against a range-restricted
    model number. On the same frames the model scores **0.8303** and out-ranks the human.
    Same discipline for bias, which moves −0.646 → −1.083 → −1.606 across those same slices.
    Score rank ONLY via `frame.metrics.count_rank_report`, quoting `template_pattern` and n.
    See [[model-out-ranks-the-blind-human]].

13c. **Rank and score are different questions and must not be cashed into one another.** Rung
    06 ep3 orders OOD counts at **r = 0.4997** while its `number` OOD margin is **exactly
    0.000000** (accuracy 0.469080 == floor 0.469080). Ordering skill is real and does not
    convert into exact-match points. ⇒ a lever that raises `r` has NOT thereby raised
    `bucket_mean`; report the rise as a rise in `r`.

## SIGNIFICANCE — when a delta is acted on (BINDING)

Full rule and its price in [[significance-rule]]; the calibration it rests on is
[[local-eval-vs-judge-calibration]]. Adopted 2026-08-05, replacing the unsigned 2026-08-01
draft whose ε = 0.05 floor was **larger than the whole competitive field** (rank 1 → rank 11
spans 0.0365; adjacent ranks near the top differ by 0.002–0.009).

S1. **Large ships, small gets discussed.** A local delta **≳ 0.03** on the declared primary
    cell is acted on: direction transferred in **8 of 8** bucket comparisons over both scored
    submissions and never reversed. Below that there is **no evidence of transfer** — we have
    never shipped a small move — so it does not run automatically; it becomes *"is this worth
    1 of 8 slots?"*, a team call with the cost stated. ⚠️ Evidence, not a guarantee: both
    calibration points are large moves. Read the result on return.

S2. **The threshold is derived and it moves.** ~0.03 is what survives the measured deflation
    (÷1 `agg_ID`, ÷1.5 `obj_ID`, ÷2 `obj_OOD`, ÷3.8 `agg_OOD`) and still moves rank. Each new
    submission adds a calibration point and may move it.

S3. **One primary cell, declared before the run.** 🔴 It may NOT be local `bucket_mean` —
    that number overstates the judge by **+0.12** and **inverts the bucket ordering**
    (`obj_OOD` is our best bucket locally, our worst on the judge). Use the ID cells or an
    explicitly deflated OOD cell.

S4. **Below |Δ| = 0.01 nothing is readable.** The seed band is borrowed literature
    ([[seed-variance-is-small-when-clean]]) and **no run has ever been repeated under a
    different seed**. A delta that is noise has no sign to transfer. One seed repeat on the
    best arm retires this; nothing else does.

S5. **Reserved vocabulary.** `WIN`/`NULL`/`LOSS` apply to the primary cell only; every other
    cell is exploratory (*"consistent with"*). ⚠️ The 10 cells per epoch are **not
    independent** — 3 are aggregates of the other 7, and `ALL` is **64% OOD by question
    count** while the challenge weights ID and OOD equally. Significance on `ALL` is not
    significance on the metric that pays.

S6. **Retroactive one way.** Cells already read are not relabelled; but no NEW argument leans
    on a secondary cell as a win. A load-bearing old cell is re-read under S3–S5 first.

S7. **A faithful NULL is published.** No re-cutting a rung after the fact to find a cell that
    clears the bar. Post-hoc slicing IS the multiplicity problem.

S8. **Multiplicity is asymmetric — one cell may WIN, every cell may VETO.** Only the cell
    declared under S3 can grant a win. **Any** cell whose CI excludes zero *in the control's
    favour* takes one away. A rung wins only if (a) the pre-registered cell's paired CI
    excludes zero in the arm's favour, (b) that holds on **ID and OOD jointly** — never the
    aggregate alone, never one side — and (c) no cell anywhere shows significant harm.
    🔑 A positive point estimate whose CI includes zero is a **NULL**, not weak evidence.
    ⚠️ Significance ≠ worth: clearing S8 says the effect is real, S1 says whether to spend GPU
    on it. Rung 24 cleared S8 (CI [+0.0023, +0.1225]) and scaling it was still correctly
    declined. 🔴 S8 gates **fishing, not resolution** — it does not rescue a cell whose
    draw-to-draw noise exceeds the effect (`number` greedy moved 10.5 pts between two n=200
    draws; one dropped video moves `acc_OOD` by a median 0.024). S4 and the jackknife still
    bind. Proposed by **Yingyu**; full statement and the grid split in [[significance-rule]].

## COMPLIANCE — challenge data never leaves the secure environment (BINDING, DUA)

14. **No challenge data — frames OR annotations (questions + gold) — to any external API.**
    Gemini/GPT/Claude/DeepSeek and any MCP-backed API are third parties; sending data to them
    violates DUA (2)/(5). The PURPOSE (data-gen vs training) is irrelevant — transmission is
    the violation. Any model that touches the data must have **downloadable weights** and run
    **on-pod** (secure env). The generator's license need not be releasable (we ship only
    Qwen, documented use); it must only permit **using its outputs to train** our model. Full
    reasoning + retroactive flag (this session leaked annotations to DeepSeek via MCP):
    [[no-external-api-for-challenge-data]].

[redacted]
    Staged there 2026-08-17 by the lead's explicit decision, with `drwx------` and a team-only
    account as the mitigations; the agreement requested 2026-08-12 has not arrived. This changes
    **nothing** about §14 — UNAM is our own secure environment, not a third party, and no frame or
    annotation may leave either machine for an external API. It is recorded here so the next
    person does not read `UNAM_SERVER.md`'s prohibition as the current state, and so the open
[redacted]

## EXTERNAL DATA — permitted, with three obligations (BINDING, challenge rules)

Source of truth is the tracked PDF at the repo root,
**`ORena-FOCUS-challenge-design-FRAME-track.pdf`** — cite it and its page, never a paraphrase.
Rationale + the organizers' written answer (2026-08-03): [[external-data-policy]].

15. **A licence is NOT the test — `CC BY-NC-SA` and `CC BY-SA` are permitted.** Confirmed in
    writing by the organizers. Public datasets and publicly released pre-trained models may be
    used (§Training data policy, p.7). **Do not re-litigate this**; the 2026-07-31 sweep's
    DUA-based argument is superseded by the organizers' own answer.

16. **🔴 The date gate: any external dataset or pre-trained model must have been publicly
    accessible by 2026-07-15**, the pre-evaluation launch (§Training data policy p.7 +
    §Schedule p.9). Released later = ineligible, however good or however permissive. **Check the
    date BEFORE the licence** — it is the cheaper kill.

17. **All training/fine-tuning data must be specified** in the method description. The organizers
    call this *"both necessary and sufficient"*. Every rung that touches external data owes its
    row in that list.

18. **Annotation publication splits by scope, and the two halves are opposites.** Annotations we
    create on **third-party public** data MUST be published with the submission (§Training data
    policy, p.7). Annotations on **challenge** data (`heico`/`lapchole`) MUST NOT be published
    (DUA clause 3, §Data usage agreement p.9). SAM 2 masks over our own videos are probe output
    and stay in the secure environment; masks over a public corpus that train the model ship.

19. **Weights-as-derivative-work is unsettled, and it does not gate us.** The organizers decline
    to rule and note the conservative view is that weights ARE derivative. Irrelevant here: the
    award criterion is *"make their model … public"* and **names no licence** (§Award policy,
    p.7), so ShareAlike inheritance would constrain WHICH licence we release under, never
    whether we may release. Do not spend another session on this question inside this challenge.

## BEFORE proposing an experiment — check `context/MEASURED.md` (BINDING)
Read **`context/MEASURED.md`** before proposing any measurement, probe or rung. It is
**generated** (`python -m frame.measured`) from four sources — decision notes, the ladders,
`RESULTS*.csv`, and the cuts already present in `stratified.json` — so it cannot drift.
Rationale in [[measured-index]]: four sessions in a row re-derived work that was already
committed and well written, including a complete probe spec whose run had closed.
**A row marked ⚠️ has been narrowed or partly withdrawn — read its note, never the row alone.**
New decision note → it MUST carry frontmatter (`question`/`verdict`/`status`); the gate
`frame.measured.assert_decisions_indexed` RAISES otherwise, and is never disabled.

## Cross-tool instruction files — `AGENTS.md` ≡ `CLAUDE.md` (BINDING)
The repo ships two agent-instruction files with **identical content**: `CLAUDE.md` (auto-loaded by
Claude Code) and `AGENTS.md` (auto-loaded by Codex and other agents). They are two names for the
SAME project instructions so every teammate's tool lands equally oriented. **Keep them byte-identical:
whenever you edit one, copy it to the other in the SAME commit. Never let them drift.**

## How a rule changes
A rule changes ONLY by: (a) a new measurement that contradicts it, recorded as a
`context/decisions/*.md` note, then (b) editing this file in the SAME commit, linking that
decision. Never silently, never mid-experiment to make a number look better.
