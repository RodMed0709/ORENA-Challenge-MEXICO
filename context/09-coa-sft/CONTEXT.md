# CONTEXT — 09 CoA-format SFT (R1)

## Objective
Answer one question: does training Qwen3-VL-8B to emit a **structured reasoning scaffold
that ends in the gold** (instead of the bare gold, as rung 02 did) improve OOD
generalization? Single variable vs rung 02 (`bucket_mean 0.549`, margin_OOD +0.132). The
model reasons internally at inference and emits **only `<answer>`**. Owner: **Rodrigo**;
orthogonal to Leo's rung 06 (perception/ViT). Decision:
`context/decisions/next-move-rodrigo-coa-format.md`.

## Why this format, the RL caveat, and the generator (updated 2026-07-18)
- ⚠️ **The +16.3 is RL+format, not SFT+format** — there is no SFT-only-CoA row in the source
  (`Bloque-A-Modelo.md:301-306`); R1 is that untested cell, and we emit only `<answer>` (the
  paper emitted the full CoA) → the whole thesis is a **weights-level-regularizer bet** the pilot
  exists to test. Retire the +16.3 prior. Full reasoning: [[coa-generator-qwen32b-onpod]].
- **Generator = Qwen3-VL-32B, vision, zero-shot, ON-POD** ([[coa-generator-qwen32b-onpod]]).
  The path here moved through three positions: (1) our own 8B sees the frame — REJECTED (below
  floor, self-distills hallucinations); (2) text-only reverse-gen, no pixels — REJECTED (DUA:
  the deepseek/Claude route sent annotations to external APIs; and it hallucinates the scene,
  fatal for positional questions); (3) **a strong general VLM that sees the frame, run on-pod** —
  a genuine perceiver, Apache-2.0, same family as the 8B student, DUA-compliant. No surgical
  generative VLM has downloadable weights, so a general Qwen is the correct on-pod generator.

## The reverse-generation recipe
Per training row we hand the generator: `question`, **`gold`**, `procedure_type`,
`primary_capability`, `answer_format`, and the **sibling fact-sheet** (every other
(question, gold) on the same frame identity `<ds>__<video>__<frame_index>`). It writes
`<description>/<evidence>/<thought>` that *derives* the gold, procedure-generic, ending in
`<answer>gold</answer>`. The gold is never guessed — it is given.

## Setup-config (parity with rung 02)
- Source: `external_data/orena-data/{heico,lapchole}/data/frame/train.parquet` (LOCAL — no
  S3 needed for Stage 1). 13,748 rows. Columns: `id, video, procedure_type, question,
  answer, answer_format, track, generation, clinical_relevance, ood, timestamp_start,
  timestamp_end, primary_capability, secondary_capabilities`.
- `SYSTEM_PROMPT` = `frame.engine.SYSTEM_PROMPT` **verbatim** (reused, never redefined).
- Frame identity / cache key = `frame.data.frame_cache_name` (`<ds>__<video>__<idx>.jpg`),
  `frame_index = round(ts_to_seconds(timestamp_start) * base_fps)`, base_fps
  {heico:25, lapchole:30}. Sibling grouping uses this identity.
- `ood` column is **all-False** on public data (RULES §3) → OOD-proxy = dataset prefix
  (heico=OOD, lapchole=ID), matching the qID convention.

## Measured facts that shaped the design
- **Sibling coverage:** 10,727 unique frames; only **2,210 (38% of rows) have a sibling**;
  **62% of questions are alone on their frame** → no fact-sheet for the majority. The sample
  oversamples single-Q frames to stress the detachment risk. This 62% is the strongest case
  for a VLM-hybrid generator later if text-only detaches.
- **answer_format counts (train):** fo_class 6294, number 4262, open_ended 1322,
  binary 1230, multiple_choice 640.

## Backend history (Stage 0 done → Stage 1 pivot)
- **Stage 0 (done, local):** a text-only 50-sample generated with deepseek-reasoner AND Claude
  side by side. **DUA-invalid for training** (annotations went to external APIs — the exact
  thing [[no-external-api-for-challenge-data]] now forbids); kept ONLY as a format/methodology
  proof. It already showed the payoff of moving to vision: positional questions are un-derivable
  for a blind generator, and text-only makes coherence slips the literal flag misses.
- **Stage 1 (on-pod, next):** regenerate with **Qwen3-VL-32B seeing the frame**, on-pod. The
  engine's backend is pluggable (`vlm-local`), so this is a config flip, not a rewrite. Kill-gate
  is the eyeball-50 (over-sampling multi-object-OOD + single-Q) BEFORE any 2k generation.

## Landmines / open
- **62% single-Q frames** — the anti-detachment anchor is absent for most rows. Watch it.
- **p99 latency with longer reasoning is UNMEASURED** — the hard gate before any full run.
- **`frame.ledger` treats an `arm` column as a run name** (NOW.md) — if RESULTS.csv is
  shaped per-arm, split it like rung 06 did (`RESULTS.csv` ledger-shaped + `RESULTS_arms.csv`).
- **`number` erosion + acc_OOD selection** ([[checkpoint-selection-vs-number]]) — few epochs,
  number-aware selection at the training stage.
