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

### ⚠️ Trust caveats (adversarial review, 2026-07-10)

The number is *correctly computed* from its inputs (reproduced exactly from artifacts), but
**`0.174` is a fragile local-slice figure, NOT a leaderboard-comparable score.** Report the
**robust signals instead: raw accuracy 0.262 / overall 0.254 + the per-answer_format table.**

- **The headline swings on ONE question.** Only 3 of 10 group×ood buckets are populated and one
  is **n=1** (`temporal_grounding`, qID `lapchole__5024415` — an "at the end of the procedure"
  question that a single start-frame *cannot* answer). It scores 0.0 and carries ⅓ of the mean;
  drop/populate it and the headline is ~0.262. A single 0↔1 flip moves 0.174→~0.51.
- **Local test is all in-distribution.** Every row has `ood=False` (the parquet column exists and
  is genuinely all-False — not a parsing bug), so our score averages 3 ID-only buckets while the
  organizers' scorer may average up to 10 including OOD → different denominator, non-comparable.
- **Judge substituted.** We used `Qwen/Qwen3-4B` (SDK default `Qwen3.5-4B` doesn't resolve on HF).
  759 judge-routed questions → headline bounded in [0.131, 0.539] under judge disagreement.
  Confirm the organizers' eval judge before quoting open_ended/multiple_choice accuracy.
- **The authoritative number is the public leaderboard (opens Jul 15).** Treat 0.174 as an internal
  sanity floor on our local, OOD-stripped, Qwen3-4B-judged slice — not the FRAME leaderboard score.

Known issues to fix before the next rung (logged in CONTEXT): use `decord.get_avg_fps()` instead of
hard-coded fps (verified matching on sampled videos, but only 2 of 38 checked); add a counter/log for
the silent last-frame clamp (`data.py`); handle the n=1 unanswerable temporal question (needs
multi-frame / end-of-clip sampling).

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
