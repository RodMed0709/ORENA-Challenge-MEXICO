# context/INDEX.md — the brain (start here)

> Root map of what we know and where it lives. Obsidian-style: this file is **short**;
> it POINTS, it does not CONTAIN. Read it BEFORE proposing an architecture change, a
> model swap, or a new experiment — so we don't re-litigate settled calls.
>
> **Workflow:** local-first → push to GitHub. Never `pull` on the pod while it trains
> (avoid stepping on a teammate's run). GSD = how we *do* work; this brain = what we *know*.

## HAS THIS BEEN MEASURED? — ask here FIRST
- **`context/MEASURED.md`** — 🤖 **GENERATED** (`python -m frame.measured`, never hand-edited). One table of every question this project has already answered: settled verdicts, what each rung changed, which runs produced a number, and **which stratified cuts are already computed**. **Read it before proposing an experiment.** Four sessions in a row re-derived work that was already committed — not for lack of documentation, but for lack of a way to ask. Rows marked ⚠️ have been narrowed or partly withdrawn: **read their note, not the row.**

## NOW — what's happening
- **`context/NOW.md`** — the living current-state: fronts, real numbers, findings, what's in progress. **Read it first.** (`HANDOFF.md` is the older baseline-run handoff, kept as history.)
- Live fronts:
  - **Leo (legokna)** → rung 06 **ViT-LoRA** — scored, verdict **PARTIAL** ([[vit-lora-partial]]); next is a lower `vit_lr`. See [[NOW]].
  - **Rodrigo** → MLOps consistency system + **R1 CoA-format SFT** — see [[next-move-rodrigo-coa-format]].

## STRATEGY — the north star
- **`THE_MAP.md`** — unified FRAME strategy: the 4 buckets, Block A/B, §0 roadmap + Phase-3 status, Block-A diagnosis (folded from the retired `BLOCK_A_MODEL.md`).
- **`ATTACK_LADDER.md`** — original master ladder (recipes valid; THE_MAP supersedes the sequencing).
- **`CONSTITUTION.md`** — hard rules, version pins, structure §VIII–IX.

## RULES — how we stay consistent (read before eval/training/results)
- **`context/RULES.md`** — the DO/DON'T checklist that stops recurring mistakes (human or Claude). EVAL rules: score only via `frame.metrics`, leaf→group via `Capability.group`, ID/OOD from qID, headline `bucket_mean`, gates RAISE (never disable). Complements `CONSTITUTION.md`.

## DECISIONS — settled verdicts (do NOT re-litigate)
- [[viT-swap-nogo]] — swapping Qwen3-VL's ViT = **NO-GO** (license + full realignment).
- [[qwen-size-ladder]] — next size up = **30B-A3B MoE FP8** (measured wildcard, not the cheap step). 🔒 Gated by [[vit-lora-partial]].
- [[unused-metadata]] — 🔴 **90% of questions carry `secondary_capabilities` we never read**: aggregation supervision is **+77% larger** than we thought (9,762 train questions, not 5,524) at zero annotation cost. Also `generation` (automatic/anchor/manual) unused, and **`clinical_relevance` is all-False** — a second landmine beside `ood`.
- [[aggregation-is-the-gap]] — 🔴 **THE target.** First external reference: a **4B model beats us by 12.5 pts on `aggregation`** (50% of the exam) while we lead `object_recognition` by +14.9. Counting is **not a ceiling** — a smaller model already does it better. **Closing that one gap alone ≈ 59%.**
- [[the-gap-is-the-number-format]] — 🔴 **WHERE the gap is.** `aggregation × ID` is **80.4% `number` format**, and `fo_class` is **not in `aggregation` at all** (it sits entirely in `object_recognition`). Arithmetic: answering 100% of `binary` only reaches 0.4492 — **no path to the target avoids lifting `number` 0.327 → ~0.482**. And `number` is the format that clears its floor by only +8.0 pts (+0.013 on OOD) and returns the mode floor on a black image. **Re-scopes [[class-imbalance-not-counting]].** Zero GPU — the number had been committed for days.
- [[count-calibration-dead]] — 🔴 **post-hoc count calibration is DEAD**, measured three ways (05c): oracle LUT gain **+0.026** vs the **+0.155** needed, ID↔OOD transfer **−0.016** (negative), and `argmax_injective` **False**. Mechanism: true values **2, 3 and 4 all share the same modal prediction (1)** — a LUT can only send `pred=1` one place, so it trades one error for another. **Not fixable with more data.** Leaves synthetic-counting SFT as the only lever left in the group that owns the gap.
- [[open-class-vocabulary]] — 🔴 **the class set is OPEN** (`overview.md:17`: "such as … and similar objects"); the organizers' predefined list is **10 classes, not 8**, and **`mesh` + `foreign object` have ZERO examples anywhere**. `silicone loop` is train-only. **Never reshape training classes against val frequencies.** Also: 70% of questions carry no class list although the spec says they do — a possible train/test regime mismatch.
- [[class-imbalance-not-counting]] — ⚠️ **RE-SCOPED: measures `object_recognition` (where we LEAD), not the gap** — see [[the-gap-is-the-number-format]]. Within that bucket: the failure is **per-CLASS, not per-count**: `gallstone` recall **0.000** (14 train ex.), `needle` 0.511, and a **phantom class** (`silicone loop`, 435 train ex., 0 in val) that is a guaranteed FP. **The ViT LoRA DID work** — `needle` +17.8 pts — `bucket_mean` hid it. But **68% of misses are on classes with >400 examples**, so rebalancing has a low ceiling.
- [[checkpoint-selection-vs-number]] — training erodes `number` on OOD **in both arms, so it is not the ViT** — killed a 7.5 h `vit_lr` run before it was spent. ⚠️ **Its selection claim is RETRACTED**: measured on OOD only; on the full 6252 epoch 2 wins and **RULES §6 is CONFIRMED**.
- [[vit-lora-partial]] — LoRA on the ViT = **PARTIAL**. Settles the *reading*: it neither licenses nor kills the capacity branch. **The ceiling question stays OPEN** until a lower `vit_lr` runs.
- [[next-move-rodrigo-coa-format]] — Rodrigo owns **CoA-format SFT (R1)**; RL is **deferred**.
- [[eval-canonical]] — ONE scoring module (`frame.metrics`): leaf→group via `Capability.group`, ID/OOD from qID; headline = `bucket_mean`, pre_eval reference-only.

## KNOWLEDGE — dense notes (open only when needed)
- **Data card — know your data (READ before quoting any number)** → `experiments/08-data-card/` (README + `tables/*.csv`). The measurement-trap map. Sharpest traps: effective n ≈ **38 videos** (not 6252); the unit is the **TEMPLATE** (188), not `answer_format`; **`acc_number` is NOT interpretable** (8 templates, 4 degenerate — use the hierarchical estimate); **`acc_OOD > acc_ID` is an ARTIFACT** (OOD floor is 12 pts higher → read **MARGIN over the template-aware floor**, not raw acc); `procedure_type`/`generation` reach the model (free stratifiers — but `procedure_type` as a model lever risks OOD, unseen procedures break it). The canonical eval (`src/frame/metrics.py`) already closed its capability-mapping + orphan + two-estimator open items (§12).
- **Evaluations** → `src/frame/metrics.py` + gates (canonical scoring: leaf→group + qID ID/OOD).
- **Results ledger** → `RESULTS.md` + committed `results/` tiers, auto-built by `frame.ledger.build_results_ledger` (never hand-edited, cannot drift). **Tier 1** `results/summary.csv` (one row per experiment/run — headline compare), **Tier 2** `results/detailed.csv` (run × capability_group × {ID,OOD} × answer_format — accuracy/n/floor/CI), **Tier 3** `results/by_run/<experiment>__<run>.csv` (full per-run canonical breakdown). Source = each run's canonical `stratified.json` (dict from `frame.metrics.stratified_report`, dropped via `frame.ledger.register_run`); runs lacking one are Tier-1-only from their `RESULTS.csv` and flagged `needs_backfill` (need a pod pass of `stratified_report` to go rich).
- Per-experiment context: `context/<id>/CONTEXT.md` (`00-baseline` … `08-data-card`).
- Experiment designs: `context/EXPERIMENT_DESIGNS.md`.
- Papers: `literature/INDEX.md` + `literature/FICHAS.md` (the technique fichas).
- Challenge spec: `context/challenge/`.

## PROCESS — how we work (GSD / spec-driven)
- `.planning/` — PROJECT / ROADMAP / REQUIREMENTS / STATE + per-phase plans.
- GSD drives the *doing* (phase → plan → execute → verify). This brain holds the *knowing*
  (what we decided and why). The bridge is this INDEX.

## Keeping this alive (rules)
- **New settled verdict** → add one short file in `context/decisions/`, link it here.
- **State changed** → update `HANDOFF.md` (the NOW), NOT this index.
- **Decision file format:** `Question / What we sought / What it gave us / Verdict / Sources`.
- Keep this INDEX to one screen. If it grows, it's carrying content it should only point to.
