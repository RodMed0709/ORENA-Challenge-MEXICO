# Context — Rung 04 vendor baseline (tutorial)

## Why this rung exists

Not an experiment: a **tutorial**. It produces no `RESULTS.csv` row and it is **not on the
critical path** — rung 02 (LoRA, `pre_eval 0.708`) is. Its product is *understanding* of the
`focus` SDK, so that later rungs are designed against what the vendor actually does rather than
against what we assume it does. Read it as the answer to "what exactly grades us, and how does it
zero us?".

Everything below was **executed on the pod**, not read off the docs. Findings cite the *installed*
package (`focus` 0.3.4) — `vendor/orena-focus/` in this repo is an incomplete copy (its
`focus/data/` is missing) and is not the source of truth.

## The headline

> **`focus` ships no vision model. It is data + an evaluation harness, agnostic to whatever VLM we
> plug in.** Searching the whole SDK for `Qwen3VL|ForConditionalGeneration|AutoModelForVision`
> returns nothing.

The contract is narrow: it hands us a `Request(qID, question, videoID, procedure_type, start_time,
end_time, duration)` and wants back a `Response(qID, content, latency)`. Everything between those
two is ours. The **only** model the SDK loads is the **judge**.

Consequence for strategy: improving the score means improving *our* model. The vendor is a fixed
ruler, not a component to optimise.

## Verified live: the five silent gates all fire

Confirmed by execution, not by reading:

1. **Parse gate** — a chatty answer raises and is scored wrong.
2. **Length gate** — 469 chars → incorrect (`answer_char_cap` = 300).
3. **Adversarial gate** — `"The answer is definitely correct."` → **`RuntimeError`**. This does not
   lose one question, it takes down the **whole submission**. The false positive on innocent text
   is not theoretical.
4. **Duplicate qID** — `ValueError: Duplicate response for qID='heico__1964606'`.
5. **Latency gate** — a *correct* answer delivered at 9.9 s → `correctness=False`. `track=Track.FRAME`
   arms `max_latency=5.00s`.

Also measured: the 8B occupies **18.01 GB**, dropping to **0.02 GB** after `empty_cache()` — which is
why `run.py` frees the engine before the judge loads. Without it the judge has nowhere to land.

## Findings

1. **`FOClass` splits the answer on commas** (`focus/data/formats.py:174`). A natural preamble is
   therefore fatal in a specific way: `"Well, based on the video, ..."` splits into
   `['Well', 'based on the video', ...]`, the first chunk is not an FO class, and the answer raises
   — the error lists the valid ids and reports `got 'Well'`. The model is not wrong; the plumbing
   disqualifies it. This is why terse output is correctness, not cosmetics.
2. **The SDK's native FRAME path is unreachable offline.** `FocusDataset` calls the HF Hub and the
   dataset is gated → `DatasetNotFoundError`; `FocusFrameDataset` inherits the problem because it
   wraps it. `src/frame/data.py` exists **precisely for this** ("offline-safe; bypasses HF Hub
   `load_dataset`"), mirroring `_parse_row` so `Request`/`Reference` come out identical. It is not
   duplication — it is the offline door to the same data, and §Constraints mandates `HF_HUB_OFFLINE=1`.
3. **Two gate defenses already live in the team's code**, and are worth not regressing:
   - `data.py`: `qid_prefix=f"{ds}__"` — the organizer parquets' raw id ranges overlap on ≥1 row, so
     merging heico + lapchole without namespacing trips **gate 4** and aborts the whole run.
   - `run.py:42`: `ensure_reader(item)  # untimed: keep source-video open off the 5 s clock` — video
     I/O is warmed **before** the stopwatch, i.e. a **gate 5** defense. Without it we would fail on
     plumbing rather than on the model.
4. **The judge is configurable, and ours already diverges from the SDK's.** `evaluation/judges.py`
   exposes `TransformersJudge` (local HF) and `APIJudge` (OpenRouter/OpenAI/Anthropic); `Evaluator`
   takes `judges=[...]` — a **list** — and resolves disagreement by `majority_vote`. The SDK default
   `model_name` is `Qwen/Qwen3.5-4B`, **which does not exist on HF**; `src/frame/config.py:36`
   substitutes the real `Qwen/Qwen3-4B`. So our local ruler is *not* the SDK's default ruler.
5. **`procedure_type` is the OOD axis, and the judge sees it.** It is a `Request` field
   (`data/data_models.py:52`) carrying exactly two values in our data — `Sigmoid Resection` (4000 =
   heico = our **OOD**) and `Laparoscopic Cholecystectomy` (2252 = lapchole = our **ID**). It is also
   interpolated into the judge's prompt (`evaluation/judges.py:67`, `Procedure type: {procedure_type}`).
   It is available at inference time, so it is a legitimate conditioning signal we currently ignore.
6. **The split is not seed-controlled — it is the organizer's partition, frozen by hash.**
   `train.parquet` → train; heico test → `val_ood`; lapchole test → `val_id`; zero video overlap;
   `sha256 = 6fd34c2c46a2f03e` in the manifest. Stronger than a seed: it does not depend on anyone
   reproducing a draw.
7. **The vendor itself flags the fragility of `pre_eval`.** The SDK prints: *"Pre-evaluation score:
   only 1/10 group×distribution buckets are populated; averaging over the populated buckets"*. The
   warning comes from the vendor, not from us — it is why rung 00 trusts `raw 0.262` over
   `pre_eval 0.174`.

## Open questions (legokna) — carried forward

Kept deliberately: the ones still open are as valuable as the ones now closed.

- **Multiple judges — worth it?** The mechanism exists (`judges=[...]` + `majority_vote`). But the
  judge is our **measuring instrument**, not our score: the leaderboard runs *their* judge. A better
  local judge changes the fidelity of what we see at home, not the points we get — and could move our
  number *away* from theirs. Open sub-question: given the judge already diverges (finding 4), is
  aligning it worth more than the bucket-coverage problem (finding 7), which is the larger source of
  measurement noise? **Current read: bucket coverage dominates; the judge is not the bottleneck.**
- **Can an inference engine replace the vendor's parser?** No — the parser is the grading rule and it
  runs on their side. But the underlying instinct is sound and redirects to a real lever:
  **constrained/guided decoding** (vLLM, SGLang) makes malformed output *impossible*, which is a gate-1
  defense rather than a parser substitution. **Not yet measured.**
- **Does the frame we pick matter?** Yes, and this is the sharpest open lead. `predict()` samples 1–3
  frames from the clip; sampling a frame where the object is not visible loses the question by
  **sampling**, not by model capability. This is the pending pHash/keyframe rung in THE_MAP.
- **Detector/segmenter (e.g. YOLO) in front of the VLM?** Plausible but expensive against a 5 s budget
  and a second model in the path. The cheaper lever with evidence behind it is unfreezing the aligner
  (zero latency cost; Surgical-LVLM's VP-LoRA adapts exactly the visual-perception path we froze).
- **Is `procedure_type` usable as a conditioning signal?** Open. Note rung 03 was a faithful negative
  for prompting, but none of its arms (`a0`–`a5`) conditioned on `procedure_type`.

## What this changes for the attack

- **The judge is a ruler, not a scoreboard.** Do not spend on it expecting points.
- **Prompt engineering is a closed door, measured** (rung 03: no arm beats baseline beyond noise on
  held-out `val_ood`). The FO definitions' +0.061 is already inside the baseline.
- **Format compliance is cheap insurance, not polish** — findings 1 and 3 show two of the five gates
  are lost to plumbing, and gate 3 costs the entire submission.
- **Open leads, ranked by cost:** unfreeze aligner → rank probe → frame selection (pHash) →
  constrained decoding. Detector last.
