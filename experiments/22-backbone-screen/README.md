# 22 — backbone screen: does a newer generation close the gap by itself?

## Ladder

| run | what it is | `bucket_mean` |
|---|---|---|
| `../00-baseline/` rung 00 | Qwen3-VL-8B, **zero-shot** | 0.2557 |
| `../06-vit-lora/` rung 06 ep3 | Qwen3-VL-8B, **fine-tuned** — the control for every A/B | **0.5724** |
| `22a` | `Qwen3.6-27B-FP8`, **zero-shot** | *pending* |
| `22b` | `Qwen3.6-35B-A3B-FP8`, zero-shot | 🔒 gated on 22a |

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

- **22a lands near or above 0.5724 zero-shot** → the generation gap is real and large enough to
  justify the migration. Fine-tuning it should then clear our ceiling comfortably.
- **22a lands well below but far above rung 00's 0.2557** → normal. The read is then the
  *shape*: does it fix `object_recognition` **OOD**, the cell that collapsed? A backbone that
  is merely better on ID buys us nothing.
- **22a lands near rung 00** → the generation story is wrong for this task and the migration is
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
