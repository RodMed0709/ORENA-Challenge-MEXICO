# Experiment 00 — Zero-Shot Baseline

> The always-valid floor: an open VLM answering FRAME questions with NO fine-tuning,
> scored by the official `focus.Evaluator`. This is the number every later rung is measured against.

## Ladder

| Notebook | Rung | Metric (pre_evaluation_score) | Verdict |
|---|---|---|---|
| `00_zeroshot_qwen3vl.ipynb` | 00 | — (pending GPU run) | baseline |

*Ladder fills in once the first run lands. Best-so-far marked ⭐.*

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
