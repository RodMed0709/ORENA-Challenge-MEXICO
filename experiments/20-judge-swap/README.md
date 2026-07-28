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
