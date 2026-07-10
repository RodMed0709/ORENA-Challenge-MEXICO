# CONTEXT — 00-baseline

## Objective
Establish the honest zero-shot baseline number for `Qwen3-VL-8B-Instruct` on the FRAME split, scored with the official `focus.Evaluator`. This is the always-valid floor and the reference every later rung is compared against — and our first read on how far we are from the two baselines to beat.

## Setup-config
- Backbone: `Qwen/Qwen3-VL-8B-Instruct`, zero-shot (no fine-tune).
- Data: `heico` + `lapchole` `data/frame/test.parquet`; single frame per question (timestamp_start == timestamp_end).
- Frame sampling: 1–3 frames from the clip as images (shared `SamplingPolicy`, Phase 2). max_pixels capped for the 5 s budget.
- Metric: `pre_evaluation_score` (unweighted mean over capability-group × {ID, OOD} buckets).
- Judge (open_ended/matching/multiple_choice): `Qwen/Qwen3.5-4B` via `TransformersJudge` (GPU).
- Hardware: RunPod GPU pod, network volume `gf78k60nlt` mounted at `/workspace` (data at `/workspace/orena-data`).

## Decisions
- Vendor the SDK (v0.3.4) as read-only reference; use the installed `focus` package for the run.
- Output guard from day one: greedy, `max_new_tokens ≤ 48`, ≤300 chars, AdversarialDetector pre-scan (heavy per-format canonicalization deferred to Phase 5).

## Results
(pending — needs GPU run; fills the ladder in README.)

## Next
Depends on Phase 1 (harness in `src/frame`) + Phase 2 (inference engine). Then: build `00_zeroshot_qwen3vl.ipynb`, run on a GPU pod, record the number before Jul 15.
