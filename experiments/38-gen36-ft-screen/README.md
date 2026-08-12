# 38 — gen-3.6 fine-tuned screen

**Ladder:** rung 37 (dead, `G-BOUNDARY` failed) → **38** → 39 connector (queued).

**Status 2026-08-12: `G-VIABILITY` PARTIALLY RUN. Stage 1 NOT started.** No training arm exists.
Everything below cost **zero training GPU** — one env build, one probe, one 2-step smoke on a 2B.

## 🔑 The strongest reason, and it was already in the repo

**Rank 1 on the leaderboard is an entry named `Qwen3.6 Finetuned`, scoring 0.5653**
([[local-eval-vs-judge-calibration]] table). We are rank 11 at 0.5288 — **+0.0365 away.**

[[backbone-generation-is-not-the-lever]]'s amendment named exactly two things that would reopen the
question, the first being *"verified confirmation that a leading entry **fine-tunes** a gen-3.6
model and that its score comes from there"*. **A leaderboard row named after the model, sitting at
rank 1, is that confirmation** — so this rung is not a speculative screen; it is following the
leader's demonstrated route. ⚠️ What the row does not tell us is their size, their recipe, or
whether the vision tower was trained.

## Why this rung exists

[[backbone-generation-is-not-the-lever]] was amended 2026-08-09 at Rodrigo's instruction: its NO-GO
is **zero-shot only** and stands **OPEN**. The reopening test it names is *one epoch of the A2
recipe on a gen-3.6 model, read against A2's own epoch-1 checkpoint*. This rung is that test,
gated by a viability stage so the 3 GPU-h and 56 GB download are only spent if the route exists.

## `G-VIABILITY` — five asserts, three already answered off-GPU

| # | question | verdict |
|---|---|---|
| **V1** | which gen-3.6 multimodal models exist; smallest? | ✅ **four** (27B, 35B-A3B, ±FP8). **Smallest is 27B** ⇒ the "generation + size" confound is unavoidable for gen-3.6. **Every** gen-3.5/3.6 config carries `model_type=qwen3_5` |
| **V2** | does ms-swift resolve the architecture? | 🔴 **FAIL** — `RESULTS_viability_v2.json`. See [[ms-swift-cannot-train-gen35]] |
| **V3** | which transformers version? | ✅ 5.5–5.12.x — inside ms-swift's `<5.13.0` cap **and** shipping `qwen3_5` |
| **V4** | does LoRA fit? | ✅ 4B 10 GB · 9B 22 GB · 27B 56 GB · 35B-A3B 74 GB (Unsloth's table). **Serving FP8 27B = 30.9 GB ⇒ fits the 48 GB eval GPU** |
| **V5** | `enable_thinking=False` | ⬜ not yet run |

## What replaced the dead route

V2's failure moved the trainer, not the rung. `RESULTS_smoke_unsloth.json` — six stages on
`Qwen/Qwen3.5-2B` (same `model_type`, same class ⇒ same code path as the 27B), public CholecT50
frames, **zero challenge data**:

| stage | result |
|---|---|
| load | `Qwen3_5ForConditionalGeneration`, 4.13 GiB, banner *"Fast Qwen3_5 patching"* |
| LoRA | 11,555,328 / 2,224,796,992 = **0.519 %** trainable |
| **module audit** | visual **96** · language_model **186** · **merger 0** · **deepstack 0** |
| 2 steps | loss 2.441 → 2.276, peak **4.51 GiB** |
| adapter / merge | 63.2 MB / 4.25 GiB, both OK |

⇒ [[unsloth-is-the-route-to-gen35]] and [[the-merger-is-unreachable-by-default]].

## ✅ The eval path is CLEARED — and it cost two bugs to find out

`RESULTS_eval_path.json`, replicating `screen_engine.py:48-115` line for line against the smoke's
Unsloth merge:

```
AutoProcessor            → Qwen3VLProcessor                        ✅
AutoModelForImageTextToText(dtype="auto")
                         → Qwen3_5ForConditionalGeneration, 2.21B, bf16   ✅
apply_chat_template(enable_thinking=False) + generate
                         → prompt_tokens=443, answer '0', no CoT   ✅
VERDICT: PASS
```

Two defects found, both of which would have surfaced mid-pod-session:

1. 🔴 **`screen_engine.py:69` was broken for `transformers` 5.x** — it passed the system message as
   a bare string, and 5.x's `apply_chat_template` indexes every message's `content` for
   `content["type"]`, so a string raises `TypeError: string indices must be integers`. **Fixed** in
   place with typed parts, which both template generations accept, so rung 23's scored numbers are
   unaffected.
2. 🟡 **Unsloth's merge writes `processor_config.json`, not `preprocessor_config.json`.**
   `AutoProcessor` coped here; the shipping container may not. Check at packaging time.

📌 **443 prompt tokens** for one image at `max_pixels` 1280×720 — the figure to budget latency with.

## What is NOT done, in the order that can kill the rung

1. 🔴 **The FP8 delivery path.** A 27B only fits the eval GPU quantised, and we have never
   quantised a merged checkpoint. New work on the shipping route, three weeks from the deadline.
   UNAM is sm_89 like the L40S, so it can be tested there — the old dev card (sm_120) could not
   even build the kernels.
3. ⬜ V5, and the real arm.

## Files

- `_tools/v2_probe.py` — the architecture-table probe. Imports `MODEL_ARCH_MAPPING` from the
  **installed** ms-swift and raises rather than substituting a hard-coded copy; carries a
  `qwen3_vl` control so a FAIL cannot be confused with a broken probe.
- `_tools/unsloth_data.py` — converts our ms-swift `train.jsonl` to Unsloth's typed-parts format,
  and builds smoke data from public frames. **Raises on an `<image>`/path count mismatch**: the
  formats are incompatible and the failure is otherwise silent — a bare string `content` trains on
  the text and drops the image without warning.
- `_tools/smoke_unsloth.py` — the six-stage pipeline smoke.

## Open confounds, recorded before any number exists

- The A2 recipe is **not known-good on another backbone**; `lr 2e-4` is the optimum found for the
  8B over these 14,415 rows. A screen at A2-verbatim measures a **floor**.
- Unsloth arms carry a **framework** confound against the ms-swift ladder — fine for a candidate
  search, invalid for attribution ([[unsloth-is-the-route-to-gen35]]).
- The smallest gen-3.6 is 27B ⇒ generation and size move together. **Qwen3.5-9B would be nearly
  size-matched to our 8B and serves in bf16 with no FP8 step**, and the one competitor known to
  beat us runs a gen-3.5 **4B**. Recorded as the cheaper alternative arm; not chosen.
