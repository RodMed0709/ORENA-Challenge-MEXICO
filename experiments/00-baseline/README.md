# Experiment 00 — Zero-Shot Baseline

> The always-valid floor: an open VLM answering FRAME questions with NO fine-tuning,
> scored by the official `focus.Evaluator`. This is the number every later rung is measured against.

## Ladder

| Notebook | Rung | Metric (pre_evaluation_score) | Verdict |
|---|---|---|---|
| `00_zeroshot_qwen3vl.ipynb` | 00 | **0.174** | baseline ⭐ |

*Ladder fills in once the first run lands. Best-so-far marked ⭐.*

## Rung 00 — result (2026-07-10)

Full FRAME test (6252 Q: 4000 heico + 2252 lapchole), `Qwen/Qwen3-VL-8B-Instruct` zero-shot,
1 frame per question at its timestamp, greedy, judge `Qwen/Qwen3-4B` (the SDK-default
`Qwen3.5-4B` id does not exist on HF). Run on 1× A100 80GB.

- **pre_evaluation_score = 0.174** (headline; unweighted mean over the 3 populated
  capability-group×ID buckets: object_recognition 0.279, aggregation 0.223,
  temporal_grounding **0.0 on n=1** — a single temporal question drags the macro-mean ~⅓).
- overall mean acc 0.254 · raw acc 0.262 · **0 timed out**.
- **Latency p50/p95/p99 = 0.30 / 0.49 / 0.59 s** (cap 5.0 s) → ~8× headroom for bigger
  models / multi-frame / higher resolution.

Per answer_format: binary 0.597 · open_ended 0.590 · multiple_choice 0.534 ·
fo_class 0.182 · number 0.127.

**Key diagnosis (not what we expected):** the low fo_class/number scores are **not** a
format-compliance problem — fo_class parses fail only 1.1 %, number 0.0 %. The model is
format-clean but **genuinely wrong**: it confuses surgical *instruments* (LigaSure,
Harmonic Scalpel, Ethicon Endo-S) with *foreign objects*, and miscounts. → the gap is
domain perception, so **fine-tuning is the primary lever**; prompt/taxonomy grounding and
frame resolution/multi-frame are cheaper adjuncts (enabled by the latency headroom).

Artifacts: `runs/00_zeroshot_qwen3vl/` (gitignored) — `report.json`, `results.csv`,
`summary.csv`, `predictions.json`, and `qualitative/` (40 mixed cases: frame + question +
GT + our answer, browsable `index.html`).

## What this experiment is

- **One variable:** none yet — this establishes the baseline. Backbone: `Qwen/Qwen3-VL-8B-Instruct`, zero-shot (no training).
- **Data:** FRAME split from `heico` + `lapchole` (`data/frame/test.parquet`), single frame per question.
- **Engine:** `_models/` inference engine implementing `predict(sample) -> str` (samples 1–3 frames, calls the VLM), driven through our `src/frame` harness → `focus.Evaluator(track=Track.FRAME)`.
- **Runs on GPU** (RunPod). Reference code: `orena-focus/examples/inference.py` + `evaluation.py` (vendored, see `VENDORED.md`).

## Layout (per EXPERIMENT_REPO_STRUCTURE_SPEC.md)

```
experiments/00-baseline/
├── README.md            # this file (opens with the ladder)
├── VENDORED.md          # note on the vendored SDK
├── orena-focus/         # vendored SDK v0.3.4 (read-only reference)
├── 00_zeroshot_qwen3vl.ipynb   # (to add) the baseline run notebook
├── _models/             # (to add) inference engine, engines-only
├── RESULTS.csv          # (to add) scorecard, one row per rung
└── runs/                # GITIGNORED — predictions + config_snapshot.yaml
```

Context lives in `context/00-baseline/CONTEXT.md`.
