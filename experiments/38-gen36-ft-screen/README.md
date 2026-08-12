# 38 — gen-3.6 fine-tuned screen

**Ladder:** rung 37 (dead, `G-BOUNDARY` failed) → **38** → 39 connector (queued).

**Status 2026-08-12: `G-VIABILITY` done except the arm. Pipeline, eval path and arm all BUILT and
validated; the arm has not been run.** Everything below cost **zero training GPU** — one env build,
two probes, and a 2-step smoke on a 2B. The arm runs on the pod, not here.

## 🔑 The strongest reason, and it was already in the repo

**Rank 1 on the leaderboard is an entry named `Qwen3.6 Finetuned` (user `wxyi088`), scoring
0.6235** — it moved up from 0.5653 on 2026-08-09. We are rank 11 at 0.5288, **+0.0947 behind**.
🔑 **It beats us in all four buckets and the OOD gaps are 2–4× the ID gaps** (agg_OOD +0.1223,
obj_OOD +0.1387 vs agg_ID +0.0362, obj_ID +0.0816) — the shape a stronger backbone produces, landing
hardest on `object_recognition_OOD`, our worst bucket. 🔑 **And latency bounds nobody**: 42.1 s per
20-question batch against a 220 s allowance = **19 % of budget**. Full payload in
[[local-eval-vs-judge-calibration]].

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
| **V5** | `enable_thinking=False` | ✅ **cleared** as a by-product of the eval verification — no CoT, non-empty answer. This is what scored rung 23 a 0.0000 |

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

## What is NOT done

1. ⬜ **The arm has not been run.** It needs the challenge data, so it runs on the pod.
2. ⬜ **The subject is not finally settled** — 27B (chosen) vs 9B (cheaper, no FP8). One line apart
   in `_models/unsloth_sft.py`.

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

## ✅ FP8: the route is VALIDATED end-to-end

`RESULTS_fp8_llmcompressor.json` — the **official** recipe, not transformers':

```python
IGNORE_LAYERS = ["re:.*lm_head", "re:.*embed_tokens$", "re:.*visual.*",
                 "re:.*model.visual.*", "re:.*linear_attn.*"]
oneshot(model, recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC",
                                           ignore=IGNORE_LAYERS))
model.save_pretrained(dst, save_compressed=True)
```

| | |
|---|---|
| pipeline | `DataFreePipeline` — **no calibration data**, so this step touches no challenge data and can run anywhere |
| quantise | 2.0 s (2B) |
| size | 4.25 → **3.22 GiB** (76 %) |
| reload + generate | VRAM 3.2 GiB, answer `'0'`, 0.704 s |
| **verdict** | **PASS** |

The 76 % is not a defect: the vision tower and `linear_attn` stay in bf16 and weigh heavily in a 2B.
Qwen's own 27B ratio is the one that matters — **51.8 GiB bf16 → 28.8 GiB FP8 (56 %)**, comfortably
under the 48 GB eval GPU.

⚠️ **Two envs, and the reason is measured.** `llmcompressor` requires `transformers>=5.9.0` while
Unsloth caps at `<=5.5.0`. Installing it into the training env silently upgraded transformers and
broke the cap. Split:

| env | transformers | for |
|---|---|---|
| `orena-unsloth` | **5.5.0** (pinned) | training |
| `orena-quant` | 5.14.1 + llmcompressor | quantisation |

### What failed first, and why it is worth remembering

`RESULTS_fp8_probe.json` — transformers' `FineGrainedFP8Config` **fails** on Ada, for an
architectural reason rather than ours:

```
ValueError: Matrix dimensions (16, 2048) must be divisible by block sizes (128, 128)
for model.language_model.layers.22.linear_attn.in_proj_a.weight
```

`linear_attn` is Qwen3.5's Gated DeltaNet (the 3:1 hybrid attention stack). Its `in_proj_a/b`
are **16×2048**, and `FineGrainedFP8Config` quantises in 128×128 blocks. The same merged
checkpoint loads and generates fine in bf16, so **the merge is sound — the scheme was wrong.**

**Qwen's own `Qwen3.6-27B-FP8` config settles it:**

```json
{"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
 "modules_to_not_convert": [ …871 modules… ]}
```

| excluded category | count |
|---|---|
| **`linear_attn` (DeltaNet)** | **336** — including the exact `in_proj_a/b` that failed |
| `visual` (tower **and merger**) | 246 — **the vision path is not quantised at all** |
| other (`embed_tokens`, layernorms) | 289 |

📌 **The lesson is about tool choice, not capability**: the wrong quantiser produced a hard error
that looked like a dead end. `llm-compressor` is the tool vLLM itself ships for this.

📌 A `Qwen3.5-9B` would still need **no** FP8 step at all (~18 GB bf16) — the FP8 work exists only
because the subject is a 27B.

## The real arm

`_models/unsloth_sft.py` — runs **on the pod**, where the challenge data legitimately lives. Every
piece of its pipeline was validated on UNAM with zero challenge data.

Recipe **transferred, not re-derived**: `lr 2e-4`, rank 8 / alpha 32, cosine, `warmup 0.03`,
effective batch 16 (1×16), seed 42, 1 epoch, `finetune_vision_layers=True`.

Guards, because a 10-hour run should not fail on something checkable in a second:

- `assert_dataset_is_the_controls` — sha256 of rung 18's `train.jsonl`; a different dataset would
  make the arm two variables.
- `assert_gpu_is_free` — no scheduler anywhere, so never launch onto someone else's job.
- `HEARTBEAT.json` + `save_strategy="epoch"` — recoverable, and legible without opening a
  10-hour `train.log`.
- The LoRA module census is **recorded every run**, never assumed.
- `max_pixels` is set explicitly — **Unsloth defaults to 512** and our recipe is 1280×720, so the
  visual token count would silently change.

Known and accepted: **it does not reach the merger** ([[the-merger-is-unreachable-by-default]]) —
a limitation shared with ms-swift, not a regression.
