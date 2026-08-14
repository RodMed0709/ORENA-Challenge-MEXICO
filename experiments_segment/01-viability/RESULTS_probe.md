# Rung 01 step 0 — template probe: what ms-swift ACTUALLY builds

> Run 2026-08-14 on pod `ocbm9adl4rc0ep` (A100 80GB PCIe, sm_80), **CPU only, nothing trained**.
> Env `/workspace/envs/infer`: swift **4.4.1**, transformers **4.57.6** — our pins.
> Processor loaded from `merged/checkpoint-2703` — the artifact arm (a) actually continues.
> Source: `probe_template.py`, log `/workspace/tmp/probe.log`, json `/workspace/tmp/probe_template.json`.

This probe existed to retire the four defects that made `REVIEW.md` a NO-GO. It retires three,
sharpens one, and produces two facts nobody had.

## Q1 — tokens per frame: **480**, and the factor is **32**

Read from the processor, not derived:

| | value |
|---|---|
| `image_processor.patch_size` | **16** |
| `image_processor.merge_size` | **2** |
| ⇒ px per visual-token side | **32** |
| tokens @ 960×540 (what the cache holds) | **480** |
| tokens @ 720×576 | 396 |
| tokens @ 1280×720 | 880 |
| `max_model_len` | 262,144 |

⇒ `PLAN.md` v1's **510 is wrong** (it rounded 540/32 up instead of down) and the adversarial
review's **646 is wrong** (it inverted the direction of the 28→32 correction). The number is 480.

🔴 **And B11 is confirmed:** `image_processor.size.longest_edge = 16777216` = 16384 × 32 × 32,
i.e. **the library default**. The `MAX_PIXELS=921600` this project has exported since rung 06 was
**never applied** on Qwen3-VL. It never bound anyway (a 960×540 frame is 518k px), so no FRAME
result is affected — but the flag is not the control anyone thought it was.

## Q2 — the video budget does **not** bind at our frame size ⇒ **drop the flag**

`vision_process` defaults, as read in-process: `VIDEO_MAX_TOKEN_NUM = 768`, `FPS_MAX_FRAMES = 768`,
`FPS = 2.0`, `FRAME_FACTOR = 2`.

Our frames cost **480** tokens < the **768** default cap ⇒ **the cap never fires**, and the encode
confirms it: every row came back at exactly 480 tok/frame.

⇒ 🟢 **Review finding A3 is retired.** `PLAN.md` v1 planned
`--model_kwargs '{"video_max_token_num":128}'`, which would have trained at ~362×362 — 1/8.8 the
resolution of the checkpoint arm (a) continues from, confounding the one variable the rung exists
to measure. **The fix is to pass no cap at all.** Rung 21's frames are the same 960×540 cache
frames, so both runs see identical geometry.

⚠️ Caveat, stated because it matters: swift's env-var patcher (`patch_qwen_vl_utils`) did **not**
run in this probe — the processor came from `AutoProcessor`, not swift's model loader. So what is
measured here are the **defaults**. The conclusion holds regardless (480 < 768 under any patching),
but "the flag would have capped us to 128" is inferred from the source, not measured.

## Q3 — `temporal_patch_size = 2` halves the frames. This is the biggest cost finding.

`video_grid_thw` came back as `[T, 32, 60]` with **T = ceil(K/2)**:

| K frames in | `grid_t` | visual tokens | full sequence |
|---|---|---|---|
| 12 | 6 | 2,880 | 2,970 |
| 27 | 14 | 6,720 | 6,890 |
| 36 | 18 | 8,640 | 8,850 |

`PLAN.md` v1 budgeted **17,850** tokens for K=36. The real figure is **8,640** — less than half.
⇒ the OOM risk the plan carried is gone, and the **localisation** grid the review demanded (B3)
becomes affordable: K=71 on a 299 s clip is `grid_t` 36 = **17,280** visual tokens, which is what
v1 had wrongly budgeted for a grid half as fine.

## Q4 — `raw_fps` **does** reach the encoder. Review A2 is too strong.

Same K, same frames, only `raw_fps` changed:

| K | `raw_fps` | `video_grid_thw` | `n_input_ids` |
|---|---|---|---|
| 12 | 0.115 | [[6,32,60]] | **2,970** |
| 12 | 2.0 | [[6,32,60]] | **2,965** |
| 27 | 0.115 | [[14,32,60]] | **6,890** |
| 27 | 2.0 | [[14,32,60]] | **6,873** |
| 36 | 0.115 | [[18,32,60]] | **8,850** |
| 36 | 2.0 | [[18,32,60]] | **8,829** |

The grid is identical — as A2 predicted — but the **token count changes**, and it changes in the
direction that says the fps is feeding the per-frame timestamp text Qwen3-VL injects (a slower fps
means larger clock values means more tokens). ⇒ **`chat_template_kwargs={"raw_fps": …}` is a live
per-row channel**, not a discarded one.

⚠️ Not yet proven that the changed tokens *are* the timestamps — that needs the decoded text, which
is the one loose end of this probe. Until then the mechanism is **demonstrated but not localised**,
and A2's remedy (assert on the effective `video_metadata`, never on the JSONL) still stands.

## Q5 — the `<video>` tag count is **NOT validated**. Review A2/Q3 confirmed.

| tags emitted | result |
|---|---|
| **0** | encodes **identically to 1** — `n_input_ids` 2970, same grid. swift prepends silently. |
| 1 | 2,970 |
| **2** | `[WARNING:swift] num_media: 1, num_media_tags: 2 … We will only replace the frontmost media_tags` — **warns and proceeds**, 2,973 |

⇒ a malformed row **trains wrong instead of raising**. There is no framework-side protection.
**Our build-time gate is the only thing standing between a tag bug and a silently wrong run**, which
makes Stage-2 gate 13 load-bearing rather than belt-and-braces.

## What this changes in the plan

| review finding | status after the probe |
|---|---|
| **A3** resolution confound (D3 vs D4 disagree 3.9×) | 🟢 **RETIRED** — pass no cap; 480 < 768 default |
| **B10** token factor 28 vs 32 unconfirmed | 🟢 **RESOLVED** — 32, and tokens/frame is 480 |
| **B11** `MAX_PIXELS` a campaign-long no-op | 🟢 **CONFIRMED** |
| **A4** 16 h watchdog vs a 32 h run | 🟡 **re-priced** — sequences are half of v1's estimate, so s/it falls; the wall must still be derived from a measured s/it, not assumed |
| **B3** K is 2× short for localisation | 🟢 **now affordable** — adopt the localisation grid |
| **A2** per-row `fps` discarded | 🟡 **narrowed** — the channel is live; assert on `video_metadata`, not the JSONL |
| **A2/Q3** `<video>` tag count unvalidated | 🔴 **CONFIRMED** — our gate is the only protection |

## Adopted frame policy (from these numbers)

Router: the **literal string** `"hh:mm:ss"` / `"In %"` / `"format xx%"` in the question.
Verified pure on all 20,000 rows: **TP 7,711, FP 0, FN 0**, precision = recall = **1.0000**.
🔴 **Never a timestamp regex** — that reading scores recall 0.139 with 2,346 false positives.

`K = ceil(dur / threshold_seconds) + 1` on routed rows, `K = 4` elsewhere, where
`threshold_seconds = min(5, 1 + dur·4/360)`.

| dur | thr | K | `grid_t` | visual tokens |
|---|---|---|---|---|
| 10 s | 1.111 | 10 | 5 | 2,400 |
| 29 s | 1.322 | 23 | 12 | 5,760 |
| 119 s | 2.322 | 53 | 27 | 12,960 |
| 299 s | 4.322 | 71 | 36 | 17,280 |

Cost, measured on the pod against the real cache: **12,804 train clips → 278,566 frames requested →
257,047 unique (video, frame) pairs → 253,838 to extract** (3,209 already present and reused from
FRAME's store under the same identity key). ≈ **29.5 GB** against ~81 GB free.

## Disk, measured with `du -sx` (never `df` — it reports the whole MooseFS cluster)

571.4 GB used at the start; **12.1 GB reclaimed** by deleting 9 aborted `*.incomplete` HF download
fragments → **559.3 GB**, ~81 GB against the ~640 GB empirical wall.

🟡 **Not reclaimed, available if needed: five `merged/checkpoint-901` trees at 17 GB each = 85 GB**,
one per rung-21 arm, all regenerable from adapters that still exist. Left alone deliberately —
they are other people's arms and the run does not need them.
