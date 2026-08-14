# SEGMENT rung 01 — viability gate + harness, then the continuation screen

> 🔴 **Status: v1 REVIEWED AND REJECTED — NO-GO (2 NO-GO, 1 GO-WITH-FIXES), 2026-08-14.**
> **Read `REVIEW.md` before this file.** Nothing was launched; no GPU was booked. v1 is kept
> verbatim as the record of what was proposed and what the review found — it is **not** a plan to
> execute. Four defects each individually void the run and three are silent (`rc=0`, healthy
> `grad_norm`, falling loss, all of V1–V8 passing). The structure survives; the mechanics do not.
> **Superseded sections:** D1 (per-row `fps` is discarded downstream), D3/D4 (they disagree about
> resolution by 3.9×), D7 (drops S8's ID∧OOD conjunction), §4 sub-gate A (scored a configuration
> the launch does not create) and sub-gate B (used the floor `RULES.md:81` forbids — the compliant
> floor is **0.5803**, so the gate FAILS), §6 L5 (16 h wall against a 32.3 h run).
>
> **Author:** synthesised 2026-08-14 from three independent read-only analyses.
> **Cost so far:** zero GPU. **Nothing in this file is a result.**

---

## 0. The one-paragraph claim

SEGMENT is the same 130 videos as FRAME under the identical train/test partition, with a
different question set. Our best FRAME checkpoint (rung 21 `A2_lr` ep3, `bucket_mean` 0.6496)
is a legitimate **initialisation** for it, and "static → dynamic continuation" is a real and
publishable single variable — *if* it is run against a from-base control. But the eval harness
cannot currently read a SEGMENT number correctly, and the shipped container cannot answer a
single SEGMENT question. **Rung 01 is therefore harness + gate work at zero GPU. No GPU is
booked until `G-SEG-01` passes.**

---

## 1. What is settled, with the measurement attached

| # | Fact | Evidence |
|---|---|---|
| S1 | **The video sets are identical and there is no leak.** FRAME-train ∩ SEG-train = 92/92; FRAME-train ∩ SEG-test = **0**; SEG-test = the same 38. `experiments/splits/frame_ood_v1.csv` is reusable **unchanged**. | all 8 parquets |
| S2 | **RULES §3 carries over verbatim.** `heico` = OOD (Sigmoid Resection, absent from all training), `lapchole` = ID. `assert_ood_from_qid` passes as-is. | `metrics.py:105, :635-649` |
| S3 | **Best FRAME checkpoint = rung 21 `A2_lr` ep3**, `21_lr_2e4_v1/ckpt/v0-20260729-172404/checkpoint-2703`, `bucket_mean` **0.6496140875528028**, grad_norm 126.79 → 4.72 (real training). Beats `B_rank` 0.6478, `D_clip` 0.6463, `A_lr` 0.6305, `A3_vitlr` 0.6185. Nothing since beats it (rung 38 −0.0334 CI [−0.0642,−0.0027]; rung 39 NULL under S8). | `RESULTS_A2_lr.csv:4`, `trainer_state.json`, `adapter_config.json` |
| S4 | **The merged 17.5 GB model already exists on the volume** (`merged/checkpoint-2703`, 4 shards, complete). The continuation costs **zero merge time**. | volume listing via S3 gateway |
| S5 | **ms-swift 4.4.1 supports video on Qwen3-VL** (`requires=['transformers>=4.57','qwen_vl_utils>=0.0.14','decord']`, `tags=['vision','video']`), added in v3.9.0 — inside every pin we hold. | `swift/model/models/qwen.py:1108-1113` |
| S6 | **`time` is two semantics under one format.** `2a` temporal_localization (5,111 train) = **absolute procedure clock**, 100% inside `[start,end]`. `2b` duration_estimation (174 train) = an **elapsed span** rendered as `hh:mm:ss`, mean 44.5 s. Both map to `temporal_grounding`. | leaf × in-window cross-tab |
| S7 | **The absolute offset is a given input, not something to infer.** `Request.start_time/end_time` are on the dataclass and the design promises them in writing. | `data_models.py:48-51`; `challenge_design.txt:1595-1596` |
| S8 | **Frame count is a hard ceiling on 20% of the headline, not a latency knob.** Uniform-K reach of the `time` gold: K=8 → 0.3763, K=16 → 0.6475, **K=32 → 0.9504**, K=48 → 0.9631 (asymptote; 3.69% of golds sit outside `[start,end]` and are unreachable at any K). | simulation over 7,645 `time` rows |
| S9 | **Question text routes the two new formats perfectly.** `hh:mm:ss` in the question ⇔ `time`: 7645/7645 recall, 0 false positives. `In %`/`xx%` ⇔ `percentage`: 66/66, 0 FP. | full corpus |
| S10 | **30% of the SEGMENT headline is judge-decided on n=171.** `complex_reasoning_ID` 100% judge (n=54), `event_understanding_ID` 100% (n=94), `aggregation_ID` 82.6% (n=23). | judge-format share per bucket |
| S11 | **`Percentage.threshold_pp` is dead code.** `compare` is `isclose(rel_tol=0.0, abs_tol=1e-9)` and `format_kwargs` is populated only for `time`. Percentage is exact match to 1e-9 against 2-dp continuous golds. | `formats.py:127-145`, `base_dataset.py:176-180` |
| S12 | **`--train_type` does not exist on ms-swift 4.4.x** — it is `--tuner_type`. `CLAUDE.md`'s tech-stack section is stale; the repo's engines are correct. | `vit_lora_train.py:95` |

## 2. What is broken right now, and would produce a wrong number with `rc=0`

| # | Defect | `file:line` | Consequence |
|---|---|---|---|
| **B1** 🔴 | `track = Track.FRAME if cfg.enforce_latency else None` ⇒ **5.0 s** enforced | `run.py:275` | `evaluator.py:218-220` marks every response >5 s incorrect. A K=32 SEGMENT run scores ≈0 and exits clean. **Highest-severity item in the audit.** |
| **B2** 🔴 | Smoke = `items[:n_eval]` after a sort keyed `(dataset, video_id, …)`; `"heico" < "lapchole"` | `run.py:203-207` | The first 4,000 items are **100% heico = 100% OOD**. Smoke scores 5 buckets, all OOD, labelled `bucket_mean`. Violates RULES §8 and changes the denominator. |
| **B3** 🔴 | `bucket_mean` averages *whatever survived* `min_bucket_n` with no assertion on the **set** | `metrics.py:367-376` | Two arms can average different denominators and the ledger records both as `bucket_mean`. On SEGMENT the at-risk buckets are `agg_ID` (23), `cr_ID` (54), `eu_ID` (94). |
| **B4** 🔴 | Container reads only `manifest["layout"]["frames"]`; `IMAGE_SUFFIXES` has no video types | `inference.py:93, :220-275` | The observed SEGMENT layout is `{'plain': 'plain/<qID>.mp4', 'overlayed': 'overlayed/<qID>.mp4'}` — **no `frames` key**. The shipped container resolves to 0 images and aborts with exit 2. It cannot answer one question. |
| **B5** 🔴 | `normalize_answer` repairs `^(\d+)\s*\.$` and `^(yes\|no)\s*\.$` only | `inference.py:333-334` | `"00:10:12."` is **not** repaired → `Time.verify` raises → `evaluator.py:338-344` silently scores INCORRECT. The one repair the container performs misses the format that is 38.22% of SEGMENT. |
| **B6** 🟡 | `template_floor` compares by exact string | `metrics.py:231-244` | The SDK compares `time` within a ±1.32/2.32/4.32 s window. Our floor **understates** the real trivial floor on 38.22% of the corpus. |
| **B7** 🟡 | `count_rank_report(template_pattern=r"^How many Clips appear in this frame\?")` | `metrics.py:1042` | 0.70% template overlap ⇒ the merge returns 0 rows, `_cell` returns `r=NaN, n=0`, logged without raising. |
| **B8** 🟡 | `_FORMATS` / `_ALIASES` have no `time`, no `percentage` | `ledger.py:48-83` | Tier-1 `RESULTS.csv` silently omits the accuracy of 38.55% of the corpus. Also `ledger.py:535, :627` hardcode *"the 4 real buckets"*. |
| **B9** 🟡 | `frame.data` hardcodes the `frame` parquet path; `frame_index` is a point, not a window | `data.py:107, :117` | The loader cannot reach SEGMENT at all. |
| **B10** ⚠️ | `config.py` computes visual tokens with **factor 28** (Qwen2.5-VL's patch14×merge2); Qwen3-VL is patch16×merge2 = **32** | `config.py:37-52` (its own comment flags it as unconfirmed) | Every token/latency figure in `config.py` is off by ~1.27×. Confirm against the real `AutoProcessor` before sizing anything. |
| **B11** ⚠️ | `MAX_PIXELS` is probably a **silent no-op on Qwen3-VL for the whole campaign** | `swift/model/models/qwen.py` ~685-713; our own `merged/checkpoint-2703/preprocessor_config.json` records the library default `longest_edge: 16777216`, not the 921,600 we exported | Harmless on FRAME (never binding). **Decisive here** — the live control is `VIDEO_MAX_TOKEN_NUM` / `FPS_MAX_FRAMES`. |
| **B12** ⚠️ | `results/summary.csv` is frozen at 2026-07-27 (37 rows, top 0.5724) with **no rows for rungs 18/21/27/30/35/38/39** | `RESULTS.md:16-18` | **RULES §9 is currently violated.** Fix outside this rung; do not let a SEGMENT row land in a ledger that already lies. |

---

## 3. Decisions taken, and the alternative that was rejected

### D1 — Training schema: `"videos": [[frame paths]]`, not `"images"`

ms-swift documents the frame-list form as officially supported for Qwen2/2.5/3-VL. It takes the
**video branch** (producing `video_grid_thw` + `video_metadata` + `<|video_pad|>` with temporal
position), reuses `/workspace/frames_cache/` with zero copying, and decodes no video — avoiding
the decord training hang the Qwen3-VL best practice warns about. The multi-image path carries
**no temporal position encoding**, which for a track that is 38% "when?" throws away the signal.

🔴 **The trap:** `Qwen2VLTemplate.replace_tag` stamps the **global `FPS` (default 2.0)** onto a
frame list. Sampling 36 frames across a 299 s clip is 0.12 fps; leaving `FPS=2.0` tells the model
those frames are 0.5 s apart when they are 8.3 s apart. **`fps` must be set per row** via
`chat_template_kwargs`, and asserted post-hoc (V4).

*Rejected:* `"images": [N paths]` — determinism was the argument for it, but per-row `fps` in the
frame-list form gives the same determinism (the grid is still chosen by our code and written into
the JSONL) without discarding temporal position.

### D2 — `time`: relative target + arithmetic offset, prompt carries the window

Train `2a` on `gold − start_time`; pass `Clip [hh:mm:ss – hh:mm:ss]` in **every** prompt (all
seven formats, or its presence itself signals a `time` question); add the offset back in
`answer_postprocess`. **Never offset `2b`** — its gold is already an elapsed span.

Why: `2a` is 100.00% in-window, so the relative target lives in `[0, dur]` with `dur ∈
{10,29,119,299}` — bounded and learnable. The absolute target ranges over the whole procedure
(max gold `04:55:57`). The offset is exact arithmetic on a **given** input (S7), so **no target
is estimated**.

*Rejected for now:* burning the clock into the frames via `VideoTimestampOverlayPreprocessor`.
It requires re-encoding 253 GB, puts an artefact on frames used by all seven formats (so it is
not a single variable), and its failure mode is a **confident, well-formed, wrong** timestamp —
OCR degrading gracefully into nonsense. ⚠️ **But the platform ships an `overlayed/` variant**
(observed layout, B4), so at *inference* the overlay is free. Keep `plain + arithmetic` vs
`overlayed + OCR` as a declared A/B for a later rung; do not let a `dict.get()` decide it.

### D3 — Frame policy: duration-adaptive K, question-routed, identical at train and serve

```
K = ceil(dur / (2 · threshold_seconds(dur))) + 1     if the question contains hh:mm:ss or xx%
K = 4                                                 otherwise
```
`threshold_seconds(dur) = min(5.0, 1 + dur·4/360)`. The router is 100%-pure on question text (S9)
and `dur` comes from `Request.start_time/end_time` without touching a pixel.

| dur | thr | **K** | spacing | share of test |
|---|---|---|---|---|
| 10 s | 1.111 s | 6 | 2.22 s | 3.2% |
| 29 s | 1.322 s | 12 | 2.64 s | 24.4% |
| 119 s | 2.322 s | 27 | 4.64 s | 49.7% |
| 299 s | 4.322 s | 36 | 8.64 s | 19.4% |

Mean K ≈ 11.5, mean ≈ 5,400 visual tokens/question, p95 ≈ 13,260, max ≈ 17,850 (at 510 tok/frame,
**pending B11/B12 confirmation**). This buys the full `time` ceiling (0.963) at ~36% of the token
cost of adaptive-everywhere, because 61.9% of questions never need temporal resolution.

**Train and serve use the same policy.** `literature/preprocessing/` is explicit that
train-pixels ≠ serve-pixels is a failure mode; a resolution or K lift becomes its own arm.

*Rejected:* a fixed K=8 or K=16 grid. It puts the 119 s and 299 s clips — **68.8% of the corpus**
— structurally outside their own acceptance window, capping `temporal_grounding` at ~0.10 while
looking exactly like a learning failure.

### D4 — Arms: (c) from-base control, then (a) continuation. Both on ms-swift.

- **(c)** fresh LoRA on `Qwen3-VL-8B-Instruct` with SEGMENT data. **The declared control.**
- **(a)** fresh LoRA on `merged/checkpoint-2703` (rung 21 A2 ep3). **The single variable is the
  initialisation.** Everything else pinned byte-for-byte.

Without (c), "static → dynamic continuation" is an assertion, not a result — and it is the one
sentence the paper is built on. A confounded win still counts on the leaderboard; it does not
count in a method description.

*Rejected:* (b) resume the rung-21 adapter via `--adapters`. It is documented and would work, but
it inherits `r=8, α=32` **and the merger-excluding regex baked into `adapter_config.json`**, i.e.
rung 21's capacity ceiling with no way to grow it; and `checkpoint-2703`'s cosine has annealed to
**lr 0.0**, so a true resume learns nothing. It buys nothing (a) does not, and it is harder to
describe.

🔴 **`--load_args false` must be passed explicitly** on arm (a), or ms-swift auto-reads
`args.json` from the checkpoint dir and rung 21's hyperparameters leak in silently — destroying
the single-variable contract.

**Recipe = A2 verbatim**, both arms: `--tuner_type lora --target_modules all-linear
--lora_rank 8 --lora_alpha 32 --lora_dropout 0.1 --learning_rate 2e-4 --freeze_vit false
--freeze_aligner true --per_device_train_batch_size 1 --gradient_accumulation_steps 16
--gradient_checkpointing true --vit_gradient_checkpointing true --max_grad_norm 1.0
--weight_decay 0.1 --optim adamw_torch_fused --lr_scheduler_type cosine --warmup_ratio 0.03
--torch_dtype bfloat16 --attn_implementation sdpa --seed 42 --check_model false --load_args false`,
plus `--model_kwargs '{"video_max_token_num":128,"fps_max_frames":36}'` (as `model_kwargs`, **not**
env vars, so it lands in `args.json` and a future session can audit it).

🔴 **The connector stays excluded.** Rung 39 measured that an explicit `--target_modules` reaches
it (736 = 504 + 216 + 16 aligner, 0 orphans), but adding it here is a **second variable** on a run
whose first variable is already a new modality. It is a follow-up rung.

### D5 — Loss weighting: OFF in the baseline, pre-registered as the first arm on top

Measured gradient share (tiktoken proxy — **must be re-derived with the Qwen3-VL tokenizer
on-pod before it gates anything**, RULES §1):

| group | % rows | **% gradient** | headline weight | grad per point of headline |
|---|---|---|---|---|
| `temporal_grounding` | 39.55 | **52.95** | 20% | 2.65× |
| `object_recognition` | 45.72 | 36.48 | 20% | 1.82× |
| `aggregation` | 8.57 | 4.29 | 20% | 0.22× |
| `complex_reasoning` | 2.81 | 3.55 | 20% | 0.18× |
| `event_understanding` | 3.35 | **2.73** | 20% | **0.14×** |

**89.4% of the optimiser's attention goes to two groups worth 40% of the score.** A **19× spread**
— the FRAME analogue was 1.64× and it was enough to reframe two rungs' nulls. Ship the baseline
unweighted so it is comparable to every FRAME rung; make re-weighting the first single-variable
arm. ⚠️ `frame.loss.make_compute_loss_func`'s ms-swift adapter is **unverified** (`loss.py:288-296`)
and needs a 3-step on-pod smoke printing the received args, exactly as rung 22 did.

### D6 — Corpus: all seven formats, `percentage` kept but never targeted

`percentage` is 42 train rows, exact-match to 1e-9 against 2-dp continuous golds (S11), measured
floor 0.0000, and structurally unreachable below 1 fps. Keep the rows (they cost 0.26% of the
gradient); **never spend a lever on them.**

### D7 — S3 primary cell, declared before any run

- **Lever touches the temporal/`time` path → `temporal_grounding × ID`.** n=674, 28 videos,
  **0% judge**, quantum 0.0015, paired 95% half-width ±0.024 at 10% churn.
- **Any other lever → `object_recognition × ID`.** n=1409, 28 videos, paired ±0.0165, and the only
  SEGMENT cell inheriting a FRAME calibration point (S2 deflation ÷1.5, direction transferred 8/8).
- **Never local `bucket_mean`** (RULES §S3) — worse here than on FRAME: one `agg_ID` question moves
  the headline **61×** as much as one `obj_ID` question, and 30% of it is judge-decided.
- `agg_ID` (n=23), `cr_ID` (54), `eu_ID` (94) are **veto-only cells under S8(c)**. At n=23 the
  smallest *expressible* delta is **0.0435** — above S1's ≈0.03 action threshold — so those cells
  can never grant a win at any effect size we can produce. Always report them with n beside the
  number and with the judge share stated.

### D8 — GPU: A100 80 GB PCIe

80 GB is required by the VRAM estimate (38–55 GB peak at 4,850 tokens with a trainable ViT), it is
**the SEGMENT eval hardware** so a p99 measured there is real, `sm_80` is our best-tested compute
capability, and at $1.39/h it is the best $/FLOP in the ≥80 GB tier (312 TFLOPS bf16 vs ~105 on a
5090, ~125 on an RTX PRO 6000 at $2.09/h). **Do not use the 32 GB RTX 5090** despite $0.99 — a
coin-flip OOM eight hours in, unattended.

Steps: 13,746 rows / 16 = **860 steps/epoch**. s/it is a working band of **35–65 (centre ~45)** and
**must be measured in the first 30 steps, not estimated**. At 45 s/it: 10.8 h/epoch, 32.3 h for 3.

| stage | GPU-h | cost |
|---|---|---|
| s/it + VRAM probe, 30 steps | ~1 | $2 |
| arm (c), 1 epoch + scored subsample | ~13 | $18 |
| arm (a), 1 epoch + scored subsample | ~13 | $18 |
| winner, 3 epochs + full 6,254 eval | ~36 | $50 |
| **total** | **~63** | **≈ $88** |

---

## 4. `G-SEG-01` — the blocking zero-GPU gate

Pre-registered. A gate that fires is a finding, never an obstacle (RULES §7). It can say NO-GO.

| Sub-gate | PASS threshold (pre-registered) | Measured | Verdict |
|---|---|---|---|
| **A — reachability ceiling** | `time` reach ≥ **0.90** at a K whose p95 visual tokens ≤ 16,000 | **0.9631** at mean K 11.5, p95 13,260 | **PASS** |
| **B — headroom over the prior** | train-derived per-template modal answer scores `bucket_mean` ≤ **0.40** on test | **0.2713** (`tg_ID` 0.000, `tg_OOD` 0.002) | **PASS** |
| **C — format legality by construction** | router precision = recall = **1.000** on `time` and `percentage`; **and** the post-processor emits a `verify`-legal string on 100% of a ≥200-case adversarial suite, checked against the **real** SDK verifier objects (never a copy) | router **1.0000/1.0000**, 0 FP. Post-processor **not yet written** | **A-half PASS, B-half OPEN** |
| **D — denominator stability** | all 10 buckets present with n≥2 in the full run **and** in the smoke recipe the rung will use | full run 10/10 ✅; **smoke via the prefix slice 5/10, all OOD** ❌ | **FAIL** |
| **E — judge exposure** | disclosure, not pass/fail; caps what any rung may claim | **30% of the headline** on n=171 | **DISCLOSED** |

**Current verdict: BLOCKED on D.** No GPU is booked until B1, B2 and B3 are fixed and D passes.
All three are zero-GPU verifiable, and all three currently produce a plausible wrong number.

---

## 5. Work order

### Stage 1 — harness (zero GPU, blocking)
1. `run.py:275` → `Track.SEGMENT` when the track is SEGMENT (B1).
2. Route smokes through `frame.subsample.choose` stratified on `(video, answer_format,
   capability_group)` (B2). Proportional draw at n=200 expects **0.7** `aggregation_ID` questions
   — the stratum must be forced, not sampled.
3. Add `metrics.assert_bucket_set(report, expected)` — RAISE on mismatch (B3).
4. `metrics.template_floor(compare_fn=...)`, defaulting to `==` so the flag-off path is
   byte-identical; pass `Time(threshold_seconds=…).compare` for `time` rows (B6).
5. Add `metrics.time_report` — legality rate, |Δt| distribution, cardinality-mismatch rate,
   window-hit at the per-question threshold. `time` is 20% of the headline and `frame.metrics`
   has **no instrument for it at all** (B7 sibling).
6. `metrics.count_rank_report`: make `template_pattern` required, or raise on `n==0` (B7).
7. `ledger`: add `time`/`percentage` to `_FORMATS`/`_ALIASES`, un-hardcode "4 buckets" (B8).
8. `data.py`: `track` parameter; a clip-window item identity replacing the point `frame_index` (B9).
9. Assert `src/frame/data.py:75-79`'s copy of the acceptance-window formula equals
   `FocusDataset._parse_row`'s output on a sample — two independent copies exist and a drift makes
   our local score silently stop being the platform's.
10. Confirm the token factor (28 vs 32) against the real `AutoProcessor` (B10).

### Stage 2 — corpus export (zero GPU, blocking)
11. Build the SEGMENT `train.jsonl` per D1/D2/D3, writing `videos: [[frame paths]]` with per-row
    `fps`, into `/workspace/frames_cache` under the existing identity key.
12. **Round-trip identity gate:** for all 5,111 `2a` rows, `post(relative_target, request) ==
    original_gold` byte-for-byte, **and assert the set is non-empty** — rung 15's smoke exported 16
    rows with zero `number` rows and its gate passed on an empty set.
13. **Tag/path gate:** every row's `<video>` tag count matches `len(videos[0])`, no duplicate paths,
    and the per-duration K histogram matches `{10:6, 29:12, 119:27, 299:36}` exactly. RAISE on fail.
14. **`2a`/`2b` router:** train on the true leaf (it is in the parquet). Ship the text router for
    inference and **measure its confusion matrix on all 5,285 `time` rows before it touches a
    checkpoint** — base rate 174/5,285 = 3.29%, so a 90%-accurate router still corrupts ~10% of one
    whole leaf. [[target-noise-is-the-harmful-kind]] is binding.
15. **The one measurement that must exist before any GPU:** over the train clips, compute
    `max_over_sampled_frames` vs the `number` gold at each candidate K. `number` asks for the
    maximum *in a single frame*; sampling K of `dur·fps` frames can only **under**-observe it, so a
    fraction of `number` rows become unanswerable-but-labelled. Rung 18's L1 died with a measured
    8.4% target error; this is the same mechanism with a new cause.

### Stage 3 — container (zero GPU, not blocking for training)
16. Read `manifest["layout"]` generically — iterate its keys, never hardcode `"frames"`; log which
    variant was chosen. Glob `/input/*.zip` rather than assuming `batch-frames.zip`. Match suffixes
    case-insensitively. Accept `VIDEO_SUFFIXES`. Unlink every temp clip (B4).
17. Extend `normalize_answer` for `time` and `percentage`, with a **legal fallback** (clip midpoint)
    rather than an illegal string — an illegal string skips `compare` entirely and forfeits the
    2.9–22.2% chance of landing in the window by luck (B5).

### Stage 4 — GPU, only after `G-SEG-01` passes
18. 30-step s/it + VRAM probe. **Decide the epoch count against that number, not against 45 s/it.**
19. Arm (c), 1 epoch. Arm (a), 1 epoch. Epoch-matched (RULES §6b).
20. Winner → 3 epochs, full 6,254 eval, class-balanced macro-F1 (RULES §9b — `fo_class` is 25.0% of
    test and spans 4 of 5 groups).

---

## 6. The autonomous launch contract

Rendered text into `/workspace/tmp/`, never a committed `.sh`, never a `.py` launcher
(CONSTITUTION §VIII.1). Start from `experiments/40-gen36-recipe-connector/_tools/chain40.py`,
do not write a new renderer. Every clause is a scar.

**Blocking — launch**
- **L1** `trap stop_pod EXIT` registered before the first line that can fail, **pod_id written
  literally into the rendered text**; `render()` raises on an empty id. Never from a library
  default — rung 39's `chain.py:66` defaults to `"y6h32tbhwhgxxe"`, a dead pod from another rung.
- **L2** `stop_pod` retries 5× and **reads `desiredStatus` back**, breaking only on `EXITED`. A 200
  alone proves nothing.
- **L3** Watchdog detached with `setsid nohup bash <wd> $$` — **not** a child of the chain.
- **L4** Watchdog cond. 1: `kill -0 $CHAIN_PID` fails ⇒ 60 s grace ⇒ re-read status ⇒ stop.
  `trap … EXIT` never fires on SIGKILL, and the OOM killer uses SIGKILL.
- **L5** Watchdog cond. 2: hard wall-clock (`16*3600`), stop unconditionally.
- **L6** 🔴 **NO silence/stale-log detector.** MooseFS ops (merge, `du -sx`, 17.5 GB shard writes)
  are silent for tens of minutes. L4+L5 cover the real failures; a silence detector adds a false
  positive with no coverage gain.
- **L7** `set -uo pipefail` — **never `-e`**: a failing arm must still reach its commit and its trap.
- **L8** Commit after **each stage**, never once at the end; explicit allow-list of
  `RESULTS_*.{csv,json}` at the experiment root. Commit only, no push — the volume outlives the pod
  and results are recoverable over the S3 gateway with no GPU.
- **L9** API key read from a file at fire time then `rm -f`; never inlined. A pod's `.git/config`
  has already carried a plaintext PAT once.

**Blocking — pre-flight, before one GPU-second**
- **P1** 🔴 **`df` lies.** Use `du -sx /workspace`. The volume carries an invisible ~640 GB quota; a
  run already died mid-merge on `Disk quota exceeded` after `df` reported 314 TB free. Delete
  `merged/checkpoint-901` (17.5 GB) and unneeded `optimizer.pt` first.
- **P2** `export HF_HOME` **and assert the model dir exists inside it** — an unset `HF_HOME` does
  not error, it re-downloads 52 GB.
- **P3** Assert `swift==4.4.1`, `transformers==4.57.6`, `qwen_vl_utils==0.0.14`,
  `torch.cuda.is_available()` **and the compute capability**. The documented installer has produced
  a CPU-only torch twice.
- **P4** 🔴 **Prove SMOKE shrinks the work** — assert the emitted argv and row count, not the flag.
  `recipe_sweep_train.py:275` emits `--num_train_epochs 1` in smoke *regardless of config*. For
  SEGMENT the smoke must additionally span `heico`+`lapchole`, **all three clip durations
  (29/119/299 s), and the `time` format**, or it never reaches the code path owning 38% of the corpus.
- **P5** GPU exclusivity on **two** independent signals (`nvidia-smi --query-compute-apps` **and** a
  peer log's mtime); abort loudly rather than share. Two pods on one volume go 10.7 → 22.6 s/it each.
- **P6** Gates run under papermill; a raising gate ⇒ non-zero exit ⇒ the chain stops before spending
  a training hour.

**Blocking — post-run proof**
- **V1** 🔴 **`rc=0` is not evidence.** Parse `logging.jsonl`, assert `nonzero_grad_frac` and a
  falling loss. Shape already in the repo: `nonzero_grad_steps: 306, nonzero_grad_frac: 1.0`.
- **V2** Read the adapter and count `n_llm / n_vit / n_aligner / n_orphans`; expect **720 / 0
  orphans** at A2-verbatim. A `target_modules` **string** returns early in ms-swift's `tuner.py:93`
  and **silently ignores `freeze_vit`** — the rung would measure nothing with no error.
- **V3** Dump the effective `video_grid_thw` / frames-per-sample from the first batch. `MAX_PIXELS`
  has probably been a silent no-op all campaign (B11); do not let `VIDEO_MAX_TOKEN_NUM` be the second.
- **V4** Assert the `fps` stamped on each frame list equals the real sample rate (D1's trap).
- **V5** G-INFER: count `"Inference Error: …"` sentinels, RAISE above 1%.
- **V6** `assert_class_f1_reported` (RULES §9b).
- **V7** Every epoch evaluated; cross-rung comparisons epoch-matched (RULES §6b).
- **V8** `assert_bucket_set` — all 10, or the number is not a `bucket_mean`.

---

## 7. What this plan does **not** claim

- It does not claim the continuation will beat the from-base control. That is what arm (c) is for.
- It does not claim SEGMENT is FRAME with more frames. The video sets are identical; the **task**
  is not. `number` falls 31.8% → 6.9%; `time` arrives at 38.2% with a ±1.3–4.3 s tolerance; **9 of
  15 leaf capabilities have zero FRAME training examples**; and 0.70% of question templates overlap.
  Rung 21's checkpoint is a good initialisation, not a head start on the metric.
- It does not claim the SEGMENT latency budget is pooled. The design document and the SDK both say
  **per-question, 15 s**; the pooled reading is a property of the **FRAME** harness, evidenced only
  by the FRAME template README and one FRAME payload. Design to 15.0 s per question; treat
  `120 + B×15` as headroom that may be discovered, never budget that may be spent. Read the SEGMENT
  template the moment it is obtainable and record it as a decision note.
- It does not claim `frame.metrics` needs a fork. `stratified_report` is already track-agnostic —
  it yields 10 buckets on SEGMENT unmodified. Everything in Stage 1 is an **extension** (RULES §1).

## 8. Amendments this rung owes the brain

- **RULES §4d is FALSE for SEGMENT.** It says we hold zero training examples for
  `event_understanding` and `complex_reasoning`. SEGMENT train holds `event_understanding` 208 ID /
  253 OOD and `complex_reasoning` 177 ID / 209 OOD, and all 15 leaves appear. Needs a SEGMENT-scoped
  amendment via a decision note, edited into `RULES.md` in the same commit.
- **`CLAUDE.md`/`AGENTS.md`: `--train_type` → `--tuner_type`** (S12), both files byte-identical in
  the same commit (RULES §Cross-tool).
- **A decision note recording** the per-question latency reading (§7), `Percentage.threshold_pp` as
  dead code (S11), and `MAX_PIXELS` as a probable campaign-long no-op on Qwen3-VL (B11).
