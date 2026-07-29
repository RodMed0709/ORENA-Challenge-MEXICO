# 20 — judge swap: is our measuring stick the organizers'?

**Ladder position.** Not a rung. No training, no VLM inference, no new data. An
**instrument check** on top of the finished rung-06 ep3 run (`experiments/06-vit-lora/runs/
06_vit_lora_v1/ep3_full`, bucket_mean 0.5724), which stays the control for every later A/B.

## The question

`src/frame/config.py:93` pins the judge to `Qwen/Qwen3-4B` behind the comment *"real HF id
(SDK default 'Qwen3.5-4B' does not exist)"*. It **does** exist — HF, created 2026-02-27,
6.4M downloads (`context/decisions/wrong-judge-model.md`). So every judged number in this
repo came from a substitute.

That matters because the judged formats (`open_ended`, `multiple_choice`, `matching` —
`focus.data.formats.JUDGE_FORMATS`) are **376 of the 1,296 `object_recognition` ID
questions**, and `object_recognition` is the single bucket that collapsed **−0.150** between
our local val and the platform, which is numerically the whole leaderboard gap
(`context/decisions/leaderboard-metric-vs-our-headline.md`).

## Design

`_tools/rejudge.py` reloads the run's saved `requests.json` / `references.json` /
`predictions.json` and runs the vendor `Evaluator` once per judge over the **same triples,
same process, same GPU**. It does not compare against the archived `results.csv`: verdicts
from another machine carry the cross-GPU drift already recorded in the repo, which would be
indistinguishable from a judge disagreement.

Single variable = **the judge model**. Everything else — predictions, references, latency
track, seed, scoring path (`frame.metrics.stratified_report`) — is byte-identical between
the two arms.

## What each outcome buys

| Outcome | Reading | Next move |
|---|---|---|
| Judges disagree materially | The −0.150 is substantially instrument + a terse output policy | Judge-aware output policy + ship `normalize_answer` in the container. **No training.** |
| Judges agree ≥97 % | The collapse is real capability on unseen centres | Zero-shot screen of a newer-generation backbone (~$5); the leaderboard is monotone in generation, not size |

⚠️ **This is not a score.** Adopting a different judge as the project's instrument is its own
decision note, not a silent config edit.

## 🟢 RESULT (2026-07-28) — the instrument was wrong and it is not the gap

`RESULTS.csv`, from `runs/20_judge_swap_v1/summary.json`. ~7 GPU-minutes total, no training.

| | substitute `Qwen3-4B` | official `Qwen3.5-4B` | Δ |
|---|---|---|---|
| `bucket_mean` (headline) | 0.5724 | 0.5710 | **−0.0014** |
| `object_recognition` ID | 0.6520 | **0.6559** | **+0.0039** |
| `object_recognition` OOD | 0.6485 | 0.6391 | −0.0094 |
| `aggregation` ID / OOD | 0.4346 / 0.5547 | identical | **0.0000** |

`aggregation` does not move at all because **none of its questions are judged** — every judged
row lives in `object_recognition`.

Agreement: **99.2 %** over all 6,252 questions, **93.5 %** over the 759 judged ones.
`multiple_choice` agrees **202 of 202**; all 49 disagreements are `open_ended`, 40 of them OOD,
and the substitute was the lenient one in 32.

⇒ **The −0.150 platform collapse survives the correct judge.** It is behaviour on unseen
centres, not our measuring stick. Verdict and the disagreement anatomy:
`context/decisions/judge-swap-is-not-the-gap.md`; the rows themselves in `DISAGREEMENTS.csv`.

## The judge needs its own environment (build note)

`Qwen/Qwen3.5-4B` declares `model_type: qwen3_5`, unknown to `transformers` 4.57 — which is the
hard floor for Qwen3-VL and cannot be raised (ms-swift caps `<5.13`). Arm 2 therefore runs in
`/workspace/envs/judge35` (transformers 5.14.1, torch inherited from the image), and `main()`
skips any arm whose `results.csv` already exists so each env runs only its own.

⚠️ Installing `accelerate` into that venv pulls `torch 2.13.0+cu130`, which shadows the image's
`2.8.0+cu128` and desyncs it from `torchvision 0.23+cu128`. The error blames the model
(`Could not import module 'Qwen3_5ForCausalLM'`); the real cause is
`RuntimeError: operator torchvision::nms does not exist`. Fix: uninstall `torch` and
`torchvision` from the venv so it falls back to the image's matched pair.
