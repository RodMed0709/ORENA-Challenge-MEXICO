# 23 — backbone screen: does a newer generation close the gap by itself?

## Ladder

| run | what it is | `bucket_mean` |
|---|---|---|
| `../00-baseline/` rung 00 | Qwen3-VL-8B, **zero-shot** | 0.2557 |
| `../06-vit-lora/` rung 06 ep3 | Qwen3-VL-8B, **fine-tuned** — the control for every A/B | **0.5724** |
| `23a` | `Qwen3.6-27B` (bf16), **zero-shot** | **0.2913** — 🔴 NO-GO |
| `23b` | `Qwen3.6-35B-A3B-FP8`, zero-shot | 🔒 **does not run** — its gate was 23a |

## 🔴 RESULT (2026-07-29): the migration is dead, and it cost $4.40

Three generations and 3.4× the parameters buy **+0.036** zero-shot; our own fine-tuning buys
**+0.317**. Full table in `RESULTS.csv`, verdict in
`context/decisions/backbone-generation-is-not-the-lever.md`.

One result argues the other way and is kept: the lift is **not uniform** — `object_recognition`
**OOD** gains **+0.090**, 2.5× the headline, and that is exactly the cell that collapsed on the
platform. Better at what we fail at, just not enough to skip training.

Free findings: a 27B runs at **p99 0.851 s**, *faster* than our fine-tuned 8B, so model size is
not latency-bound for FRAME; and the zero-shot model emits `0` and names `Mesh` — two tokens our
supervision contains no example of and therefore cannot teach.

## Why this rung exists

[[judge-swap-is-not-the-gap]] closed the cheap escape: the −0.150 `object_recognition`
collapse between our local val and the platform survives the organizers' real judge, so it is
behaviour on unseen centres, not our measuring stick. The remaining step-change lever is the
backbone — and [[latency-budget-is-pooled]] unblocked it (0.79 s/q measured against a 5.06 s
ceiling, 6.4× headroom).

The leaderboard's own shape argues for **generation over size**: 8B gen-3 scores 0.4710, a 4B
scores 0.5163, first place 0.5591 ([[leaderboard-metric-vs-our-headline]]). A model a third
our size beats us, so parameter count is not what separates us from the top.

## The candidates (all Apache-2.0, all public before the 2026-07-15 eligibility cutoff)

| model | released | weights | fits the 48 GB eval GPU? |
|---|---|---|---|
| `Qwen/Qwen3.6-27B-FP8` | 2026-04-21 | **30.9 GB** | ✅ ~17 GB spare |
| `Qwen/Qwen3.6-35B-A3B-FP8` | 2026-04-15 | **37.5 GB** | ✅ ~10 GB spare, MoE with 3B active |
| `Qwen/Qwen3.6-27B` bf16 | — | 55.6 GB | ❌ |
| `Qwen/Qwen3.6-35B-A3B` bf16 | — | 71.9 GB | ❌ |

🔴 **FP8 is not optional, it is the only way in.** `challenge_design.txt:437` fixes the FRAME
track at **5 s on a 48 GB VRAM GPU** (L40S, `CONSTITUTION.md:76`), and both bf16 builds exceed
that outright. The L40S is Ada CC 8.9 with **native FP8 w8a8**, which `CONSTITUTION.md:70`
already names as the serving lever — so this is the planned path, not a workaround.

⚠️ FP8 also needs Ada / Hopper / Blackwell to be *native*. An A100 is Ampere and would emulate
it, which is why the screen pod is a Blackwell RTX PRO 6000, not the cheaper A100.

## 🔴 The screen runs bf16; the DEPLOYMENT question stays FP8 (measured 2026-07-29)

The first two smokes both failed at the FP8 kernel, not at the model:

| attempt | error | cause |
|---|---|---|
| 1 | `finegrained-fp8 kernel requires the 'kernels' package` | missing dependency |
| 2 | `Triton Error [CUDA]: device kernel image is invalid` | the hub's prebuilt FP8 kernels carry no **sm_120** binary |

The screen pod is an RTX PRO 6000 **Blackwell (sm_120)**, triton 3.7.1, torch 2.8.0+cu128. The
`finegrained-fp8` kernels ship builds for Ada/Hopper, not for this card — so **FP8 could not
run here for a hardware reason, not a model one**.

⇒ The screen loads **bf16** instead (55.6 GB on a 96 GB card, no kernel dependency). That is
sound because a screen measures **capability**, and bf16 is the *upper* bound of what the FP8
build can score. It does **not** answer the deployment question: whether FP8 holds that quality
on the L40S. That check is separate, belongs on Ada/Hopper where the kernels exist, and is a
prerequisite before any submission — **not** before deciding whether this backbone is worth
migrating to.

⚠️ So a win here is provisional in one specific way: it would still need an FP8 confirmation
run on the eval-class hardware before it can ship.

## Design

Single variable: **the backbone**. The `SYSTEM_PROMPT` is imported from `frame.engine`, not
copied; the message shape, greedy decoding, `max_new_tokens`, `max_pixels` and
`answer_char_cap` all stay at rung 06's values; the eval set and the organizers' ID/OOD split
are untouched; scoring is `frame.metrics.stratified_report` as always.

`_tools/screen_engine.py` exists because `frame.engine.QwenFrameEngine` hard-imports
`Qwen3VLForConditionalGeneration`, while the 2026 line ships as
`Qwen3_5ForConditionalGeneration`. It is wired in through `cfg.engine_factory`, a hook whose
default (`None`) makes `run_baseline` construct the original engine — **flag off is
byte-identical**.

## How to read the result — pre-registered

The comparison is **deliberately unfair to the challenger**: a zero-shot 27B against our
*fine-tuned* 8B. So:

- **23a lands near or above 0.5724 zero-shot** → the generation gap is real and large enough to
  justify the migration. Fine-tuning it should then clear our ceiling comfortably.
- **23a lands well below but far above rung 00's 0.2557** → normal. The read is then the
  *shape*: does it fix `object_recognition` **OOD**, the cell that collapsed? A backbone that
  is merely better on ID buys us nothing.
- **23a lands near rung 00** → the generation story is wrong for this task and the migration is
  dead for ~$3.

**Primary target:** `object_recognition` × OOD, the cell the platform punished.
**Always reported:** the margin over the template-aware floor in BOTH ID and OOD — the
conjunction that caught rung 10's Arm C and rung 15's false win.

## 🔴 What a win actually costs (read before celebrating one)

The screen is cheap; acting on it is not.

1. **`transformers` 4.57 cannot load these models at all.** `CONSTITUTION.md:70` says in so
   many words *"do NOT jump to 5.x"* — it breaks `qwen-vl-utils` and ms-swift's `<5.13` cap.
   Engine, container and serving would all have to move.
2. **ms-swift support for `Qwen3_5MoeForConditionalGeneration` is UNVERIFIED.** If it is
   missing there is no path to LoRA-tune the winner with our recipe.
3. **Training a 27B/35B** needs an 80 GB pod with QLoRA NF4 — more expensive than everything
   this project has spent to date.

⇒ A win here is a *decision to spend two to three of the remaining six weeks*, not a free lift.
