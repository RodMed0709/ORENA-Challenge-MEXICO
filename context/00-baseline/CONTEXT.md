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

## Results (2026-07-10, full 6252-Q FRAME test, 1× A100 80GB)
- **pre_evaluation_score = 0.174** (headline). overall 0.254, raw 0.262, 0 timed out.
- Latency p50/p95/p99 = 0.30/0.49/0.59 s vs 5.0 s cap → ~8× headroom.
- answer_format: binary .597, open_ended .590, mc .534, fo_class .182, number .127.
- group×ID buckets: object_recognition .279, aggregation .223, temporal_grounding .0 (n=1).
- Judge: used real `Qwen/Qwen3-4B` (SDK-default `Qwen3.5-4B` does not exist on HF).
- Implementation reality vs plan: FRAME data has **no pre-extracted frames** and
  `FocusDataset` loads via HF Hub — so we read parquet directly and sample the frame from
  the source video with decord at `round(start_time*base_fps)` (heico 25 fps, lapchole 30).
  qID namespaced by dataset (`heico__`, `lapchole__`) — the two id ranges collide on ≥1 row.

## Diagnosis (drives the next rung)
Low fo_class/number is **not** a formatting problem: fo_class parse-fail 1.1 %, number 0.0 %.
The model is format-clean but **genuinely wrong** — confuses instruments (LigaSure, Harmonic,
Ethicon) with foreign objects, and miscounts. The gap is domain perception → **fine-tuning is
the primary lever.** Cheap adjuncts worth probing first (huge latency headroom): prompt with
explicit FO taxonomy + "instrument ≠ foreign object", higher `max_pixels`, and 2–3 frame
windows. The `pre_evaluation_score` is fragile on FRAME (a single n=1 temporal bucket at 0
costs ~⅓ of the headline) — track overall/raw acc alongside it.

## Next
- Rung 01 candidate (cheap, no train): prompt/taxonomy grounding A/B vs 00, same test split.
- Rung 02: frame resolution / multi-frame A/B.
- Rung 05 (primary prize): LoRA fine-tune (ms-swift, freeze ViT+merger, bf16) on the train
  split; our 8B FT vs the org's fine-tuned 4B baseline.
- Pod `rvfi2btzgqb92i` (A100 80GB) kept RUNNING for follow-on experiments (auto-stop watchdog
  disabled per request — remember to stop it manually to end billing).
