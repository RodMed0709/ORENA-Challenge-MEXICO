# context/INDEX.md — the brain (start here)

> Root map of what we know and where it lives. Obsidian-style: this file is **short**;
> it POINTS, it does not CONTAIN. Read it BEFORE proposing an architecture change, a
> model swap, or a new experiment — so we don't re-litigate settled calls.
>
> **Workflow:** local-first → push to GitHub. Never `pull` on the pod while it trains
> (avoid stepping on a teammate's run). GSD = how we *do* work; this brain = what we *know*.

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
- [[checkpoint-selection-vs-number]] — 🔴 training ERASES `number` (to +0.000 by epoch 3, **with the ViT frozen too**) and **acc_OOD selection picks the checkpoint that erased more**. Qualifies RULES §6. Killed a 7.5 h `vit_lr` run before it was spent.
- [[vit-lora-partial]] — LoRA on the ViT = **PARTIAL**. Settles the *reading*: it neither licenses nor kills the capacity branch. **The ceiling question stays OPEN** until a lower `vit_lr` runs.
- [[next-move-rodrigo-coa-format]] — Rodrigo owns **CoA-format SFT (R1)**; RL is **deferred**.
- [[eval-canonical]] — ONE scoring module (`frame.metrics`): leaf→group via `Capability.group`, ID/OOD from qID; headline = `bucket_mean`, pre_eval reference-only.
- [[no-external-api-for-challenge-data]] — 🔒 **BINDING (DUA):** NO challenge frames/annotations to any external API (Gemini/GPT/Claude/DeepSeek, incl. MCP). Data-touching models must have downloadable weights and run **on-pod**; only Qwen is released, the generator is documented not shipped. Qualifies RULES (new §COMPLIANCE).
- [[coa-generator-qwen32b-onpod]] — R1's scaffold generator = **Qwen3-VL-32B, vision, zero-shot, on-pod** (Apache-2.0, same family as the 8B student). Supersedes the text-only "no pixels" line. Carries the adversarial GO-WITH-CHANGES gate: **retire the +16.3 prior** (it's RL+format; R1 tests the untested SFT-only + emit-only-answer cell), eyeball-50-vision first, **matched 2k-bare control**, emit-only-`<answer>` eval + p99, number-aware selection.

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
