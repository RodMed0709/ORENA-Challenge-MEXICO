# CONTEXT — 09 CoA-format SFT (R1)

## Objective
Answer one question: does training Qwen3-VL-8B to emit a **structured reasoning scaffold
that ends in the gold** (instead of the bare gold, as rung 02 did) improve OOD
generalization? Single variable vs rung 02 (`bucket_mean 0.549`, margin_OOD +0.132). The
model reasons internally at inference and emits **only `<answer>`**. Owner: **Rodrigo**;
orthogonal to Leo's rung 06 (perception/ViT). Decision:
`context/decisions/next-move-rodrigo-coa-format.md`.

## Why the format, not RL, and why no pixels (settled)
- **Format is the lever, not RL.** CoA ablation: SFT 65.7 → +RL 67.4 (+1.7) →
  +CoA-format 83.7 (+16.3). GRPO on 1×L40S is unvalidated; the +1.7 RLVR variant was killed.
- **Generator does NOT see the frame.** Adjudicated by a vlm-strategist adversarial pass
  (2026-07-18). The user's first instinct (our own 8B sees the frame) was **refuted by our
  own numbers**: the 8B is below floor everywhere (`bucket_mean 0.256`) and hallucinates on
  exactly the multi-object OOD frames R1 targets — self-distilling from a below-floor teacher.
  Every surgical-VLM paper (LLaVA-Surg, SSG-VQA, GP-VLS, Surgical-LVLM) grounded structured
  reasoning on **annotations / known answers via a text model**, never on raw-pixel free
  perception. "Faithful" = anchored on the correct label, not on a perceiver.

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

## Stage-1 backend decision
Generate the ~50 sample with **deepseek-reasoner AND Claude, side by side**, so the eyeball
picks the cheapest sufficient generator for the 13.7k. No Gemini/OpenAI key exists in
`.secrets.env` (only RUNPOD); deepseek is available via the `deepseek-worker` MCP, Claude via
the orchestrator. The engine's backend is pluggable → swapping to a VLM-hybrid or an API
model later is a config flip, not a rewrite.

## Landmines / open
- **62% single-Q frames** — the anti-detachment anchor is absent for most rows. Watch it.
- **p99 latency with longer reasoning is UNMEASURED** — the hard gate before any full run.
- **`frame.ledger` treats an `arm` column as a run name** (NOW.md) — if RESULTS.csv is
  shaped per-arm, split it like rung 06 did (`RESULTS.csv` ledger-shaped + `RESULTS_arms.csv`).
- **`number` erosion + acc_OOD selection** ([[checkpoint-selection-vs-number]]) — few epochs,
  number-aware selection at the training stage.
