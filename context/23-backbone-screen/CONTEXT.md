# Context — 23-backbone-screen

**Status: CLOSED 2026-07-29. Verdict NO-GO, verdict note
`context/decisions/backbone-generation-is-not-the-lever.md`.** ~$4.40 of pod, no training.
🔒 **23b never ran** — its gate was 23a and 23a failed it.

## Why the rung existed

[[judge-swap-is-not-the-gap]] closed the cheap escape: the −0.150 `object_recognition` collapse
survives the organizers' real judge, so it is behaviour on unseen centres. That left the
backbone as the only step-change lever on the board, and [[latency-budget-is-pooled]] had
already unblocked it (0.79 s/q against a 5.06 s ceiling).

The leaderboard's own shape argued for **generation over size**: 8B gen-3 scores 0.4710, a **4B**
scores 0.5163, first place 0.5591 ([[leaderboard-metric-vs-our-headline]]). A model a third our
size beats us, so parameter count is not what separates us from the top.

## What was run

`Qwen/Qwen3.6-27B` (2026-04-21, Apache-2.0, `Qwen3_5ForConditionalGeneration`), **zero-shot**,
on the full 6,252-question eval set. `SYSTEM_PROMPT` imported not copied; message shape, greedy
decode, `max_new_tokens`, `max_pixels`, `answer_char_cap`, judge and eval split all held at rung
06. Single variable: the backbone.

`_tools/screen_engine.py` exists because `frame.engine.QwenFrameEngine` hard-imports
`Qwen3VLForConditionalGeneration` and this arch **cannot load under `transformers` 4.57 at all**
(`AutoConfig` raises `KeyError: 'qwen3_5'`). It is wired in through `cfg.engine_factory`, whose
default `None` builds the original engine unchanged.

## What it gave us

| | rung 00 zs 8B gen-3 | **23a zs 27B gen-3.6** | rung 06 ep3 FT 8B |
|---|---|---|---|
| `bucket_mean` | 0.2557 | **0.2913** (+0.036) | **0.5724** (+0.317) |
| `margin_ID` | — | **−0.0666** | +0.2225 |
| `margin_OOD` | — | **−0.1367** | +0.1448 |

🔴 **Both margins are negative: the zero-shot 27B scores BELOW the template-aware trivial
floor.** So the +0.036 is movement *beneath* the floor, not skill, and the +0.090 it gains on
`object_recognition` OOD is one sub-floor model beating another. Three generations and 3.4× the
parameters buy less than a ninth of what our own fine-tuning buys.

⚠️ **That floor reading did not exist when the verdict note was first written** — this rung
shipped its decision note before its canonical `stratified.json`, and the note carries a dated
amendment saying so. The procedural lesson is the durable part: **a rung that ships raw accuracy
without its floor has not been read yet** (RULES §10).

## Three things the screen paid for by itself

1. **Size is not latency-bound for FRAME.** 0.585 s/q mean, **p99 0.851 s**, max 1.29 s — a 27B
   is *faster* than our fine-tuned 8B against a 5 s ceiling.
2. **The zero-shot 27B emits tokens our supervision cannot teach.** It answers `0` and names
   `Mesh` unprompted; a numeric zero appears nowhere in our numeric gold
   ([[zero-is-format-localized]]) and `Mesh` has zero examples anywhere in our data
   ([[open-class-vocabulary]]). An argument for external data, not for a bigger backbone.
3. **The 2026 Qwen line is a reasoning model.** Left on its default template it emits a chain of
   thought, `max_new_tokens=64` truncates it before the answer, and the run scores **0.0000** —
   a format failure indistinguishable from incapacity. `enable_thinking=False` is mandatory, and
   is also the right default per the CoT-degrades-grounding literature.

## Deployment facts recorded for whoever revisits this

- **FP8 is mandatory to fit the 48 GB eval GPU** (30.9 GB vs 55.6 GB bf16;
  `challenge_design.txt:437`). The screen ran **bf16** because the hub's `finegrained-fp8`
  kernels carry **no sm_120 build**, so FP8 will not run on a Blackwell RTX PRO 6000. bf16 is the
  *upper* bound of the FP8 build's score, so the NO-GO is not an artefact of precision.
- Installing `accelerate` into a fresh venv pulls `torch 2.13.0+cu130` over the image's
  `2.8.0+cu128` and desyncs `torchvision`; the symptom
  (`Could not import module 'Qwen3_5ForCausalLM'`) blames the model. Uninstall torch and
  torchvision from the venv so it falls back to the image's matched pair.
- The 52 GB of weights were **deleted from the volume** after the verdict (re-downloadable in
  minutes). Nothing on disk depends on this rung.

## Gate this rung added to the repo

**G-INFER** (`src/frame/run.py`, RULES §8c). `engine.predict` never raises — it returns
`"Inference Error: …"` so one bad frame cannot kill a long run. This rung's first smoke therefore
"completed" at `bucket_mean` 0.0000 with **all 24** calls hitting a missing FP8 kernel, and the
only tell was an impossible 19 q/s. `run_baseline` now counts those sentinels before evaluation
and raises above 1 %.

## Bookkeeping debt paid 2026-07-29

Originally shipped a transposed `RESULTS.csv` (buckets as rows, runs as columns), no
`stratified.json`, and no `CONTEXT.md` — so `frame.ledger` read it as 14 blank rows and its
0.2913 was not reproducible from a commit (RULES §9). Canonical `RESULTS.csv` and
`stratified.json` now committed, per-bucket generation deltas moved to
`RESULTS_generation_deltas.csv`. Still **no notebook** (spec §1): the run was driven from a
config cell executed on the pod, and `_tools/screen_engine.py` is only the engine.
