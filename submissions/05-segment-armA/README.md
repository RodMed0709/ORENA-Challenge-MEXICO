# SEGMENT submission — arm A (`Qwen3VL-8B-FT-SEGMENT-armA-v1`)

The recipe **minus the weights** for the first **SEGMENT** track container. Everything
needed to rebuild it on the pod is here; `resources/` (~17 GB) lives outside git.

> ⚠️ **Numbering.** `submissions/` already holds a `04-rung40-conn4e5-ep23` (FRAME). This
> directory was created at the path the work order named. The two tracks are independent,
> but if the submissions ledger is meant to be a single sequence, rename this to `05-`
> **before** anything references it.

---

## 1. What it is

| | |
|---|---|
| Track | **SEGMENT** (`focus.enums.Track.SEGMENT`) |
| Backbone | Qwen3-VL-8B-Instruct, bf16 |
| Adapters | rung 21 arm A2 `checkpoint-2703` → SEGMENT arm A `checkpoint-860`, **stacked, in that order** |
| Serving | plain `transformers`, greedy, `max_new_tokens=32` |
| Latency budget | **15.0 s / question** (`focus.config.TRACK_MAX_LATENCY[SEGMENT]`) |
| Measured rate | **0.94 s / question** on an A100 80 GB over 6,254 questions — ~16× headroom |
| GPU | 80 GB (eval hardware); the model is ~17 GB bf16 + ≤36 frames of activations |
| Offline | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; all weights `COPY`-ed into the image |

The offline score this container serves is `bucket_mean` **0.4964** against a bar of
0.5118 (`context/decisions/segment-arm-a-is-one-broken-bucket.md`) — **below the bar**, and
shipped anyway on purpose: we hold **zero** SEGMENT calibration points, and on FRAME the
local eval overstated the judge by +0.12 and *inverted* the bucket ordering on OOD, a fact
discovered only by submitting. One slot buys the deflation factor for every SEGMENT number
after it. Nine of the ten buckets work; `temporal_grounding` answers `00:00:00` on 72.5 % of
its rows, which is a **training-side** defect (0 malformed outputs — the harness is clean).

### Fidelity to the scored run

`inference.py` reproduces `experiments_segment/01-viability/_tools/seg_eval.py` function for
function: the system prompt (`corpus.SYSTEM_PROMPT_HEAD` + FO definitions), the
`Clip [hh:mm:ss - hh:mm:ss] of the procedure.\n<question>` user turn, the video-branch chat
template, `max_pixels=921600`, greedy `max_new_tokens=32`, the adapter stack and its merge
order, the relative→absolute inversion, and `normalize_answer` **after** it.

The corpus helpers are **vendored, not imported** — the image is sealed and has no `src/`
on its path, the same argument recorded for `normalize_answer` in `src/frame/parsing.py`.
The copies were verified equal to `src/frame/segment/corpus.py`,
`src/frame/parsing.py` and `seg_eval.to_absolute` across every duration class, both router
branches, and the strided K≤36 grid (max drift: **1 decoded frame**, from clip-relative
rather than absolute index rounding).

### Three deliberate differences from `seg_eval.py`, each stated so nobody calls this byte-identical

1. **`time_kind` comes from the question, not from a parquet row.** `Request` carries no
   `primary_capability`, so `2a` vs `2b` is decided by `is_duration_question` — the literal
   router measured exact on all 7,645 `time` rows (261/261 `2b`, 0/7,384 `2a`).
2. **An FO-class guard runs after the repair chain.** `FOClass.verify` *raises* on any token
   outside `FOType.names()` and the Evaluator turns that raise into a silent INCORRECT. The
   guard reads the accepted set **at runtime** (never hardcoded — the SDK warns that extra
   FO types may be registered for the test phase), drops `"none"` when combined with a real
   class, and drops an unaccepted token only when a valid one survives, every part is a
   short alphabetic phrase, and the token is absent from the served FO definitions.
   Everything else is byte-identical.
3. **Decoded frames are downscaled into a 960×540 box** (aspect preserved, never upscaled).
   That is the geometry of `/workspace/frames_cache`, i.e. what the arm was trained on, and
   `MAX_PIXELS` is a **measured no-op on Qwen3-VL** so nothing else would stop a 1080p clip
   from roughly quadrupling the visual-token count. `SEG_RESIZE=0` disables it.

---

## 2. 🔴 What is UNVERIFIED about `/input` — read this before blaming the model

**No real SEGMENT payload has ever been seen by this project.** The repo's only description
of one — `plain/<qID>.mp4` + `overlayed/<qID>.mp4`, **no `frames` key** — came from a fixture
whose source does not name the track. The rung-01 review explicitly downgraded it from
*observation* to **CONJECTURE** (`experiments_segment/01-viability/REVIEW.md`, finding B2).

Consequences, and what was done about each:

| Unverified | Failure if wrong | Defence in this container |
|---|---|---|
| the layout key is `plain` | shipped FRAME container reads `layout["frames"]`, resolves **0** media, answers **zero** questions | every key of `layout` is iterated **generically**; `plain` is preferred and `overlayed` is tried last, but neither is required |
| the media is an `.mp4` clip | FRAME's `IMAGE_SUFFIXES` matches no video | video **and** image suffixes, case-insensitive on `Path.suffix.lower()`; `VIDEO_SUFFIXES` is imported from the SDK with an inline fallback |
| the delivery is a directory | a ZIP arrives instead (the FRAME *interface page* declared `batch-frames.zip` while the FRAME *template* documented a directory — they disagreed, and it cost a slot) | **every** `*.zip` under `/input` is globbed and extracted into `/tmp`, never one hardcoded name |
| one file per qID | frames arrive as `<qID>/*.png`, or `<qID>_000123.png` | four shapes indexed: `<qID>.mp4`, `<qID>/`, `<qID>.png`, `<qID>_*.png`; exact stem first, then a unique prefix match |
| `request.json` is the filename | a rename is a total zero | falls back to a glob for `*request*.json` |
| **`Request.start_time` is the real absolute clip start** | if the platform pre-cuts the clip and reports `0.0`, the relative→absolute inversion adds zero, every `2a` timestamp stays relative, `temporal_grounding` (2 of 10 buckets) scores ~0 — **and nothing else looks wrong** | there is **no recovery from inside the container**. It is detected and logged at ERROR: *"EVERY request has start_time == 0.0"*. That log line is the only thing that will make the cause findable afterwards |
| the request window equals the clip | K is computed from the wrong duration | the clip's own container duration is read back and a >10 % disagreement is logged per question; a non-positive window falls back to the container duration |

**The complete `/input` inventory is logged recursively at startup**, before the manifest,
before the media index, and long before the weight load. If this run scores zero, that log
says which layout actually arrived. If more than half the qIDs cannot be resolved, the
container **exits 2 without loading the weights** rather than writing a schema-valid
`answer.json` of empty strings — a silent zero is indistinguishable from a model that knows
nothing, and that is exactly what burned one of ten slots on FRAME
(`context/decisions/submission-01-rung06.md`).

---

## 3. Frame sampling

Identical to training, because a different grid is an unmeasured second variable sitting on
top of the number.

```
routed  = corpus.needs_time_grid(question)          # literal "hh:mm:ss" / "In %" / "format xx%"
K_fine  = 4                                         if not routed
        = ceil(dur / threshold_seconds(dur)) + 1    if routed
threshold_seconds(dur) = min(5.0, 1 + dur*4/360)    # the SDK's acceptance window
stride  = ceil(K_fine / 36)                         # 36 = MEASURED VRAM ceiling (K=71 OOMs)
grid    = [i/(K_fine-1) for i in range(K_fine)][::stride]
```

The cap is applied by **striding the fine grid, never by recomputing a coarse one**: point
*j* of a K/2 grid is exactly point *2j* of a K grid, so the strided grid is a **subset** of
what training used. Duplicate landings are **collapsed**, never nudged to a neighbour.

⚠️ The K≤36 cap means 119 s clips sample at 4.58 s spacing against a 2.32 s acceptance
window, and 299 s clips at 8.54 s against 4.32 s — **coverage grade, not localisation
grade, on 68.9 % of the corpus.** `temporal_grounding` is capped by geometry here, not only
by the model.

---

## 4. The time inversion — 38.2 % of the track

`corpus.build_row` rewrote every `2a` gold to an **offset from the clip start**, so the model
emits **relative** time; the reference is **absolute**.

```
raw    -> to_absolute(raw, request.start_time, is_duration_question(question))
       -> normalize_answer(...)
       -> canonicalize_fo_classes(...)
```

* **Order is load-bearing** (it is `seg_eval.run()`'s order): absolute **first**, format
  repair **second**, because `normalize_answer` is what makes `Time.verify` accept an answer
  carrying a trailing period.
* A **`2b` elapsed span is never offset** — that would be a category error. The router is
  the literal `"for how long" | "how much time passes"`, measured exact on 7,645 rows.
* `to_absolute` is safe on every format: `"3"`, `"Yes"`, `"Clip, Sponge"` all fail
  `ts_to_seconds` and are handed back byte-identical.
* A trailing period on `hh:mm:ss` makes `Time.verify` **raise**, and
  `Evaluator._evaluate_single` catches that raise and scores INCORRECT behind a
  `logger.debug`. Silent, on the largest format in the track.

---

## 5. Expected output format

`/output/answer.json` — a JSON **array** of `focus.Response`, one object per request, in
request order, written with `focus.save_items`:

```json
[
  {"qID": "seg_000123", "content": "00:49:31",    "latency": 0.94},
  {"qID": "seg_000124", "content": "3",           "latency": 0.88},
  {"qID": "seg_000125", "content": "Clip, Sponge","latency": 0.91}
]
```

* `content` is the final post-processed answer, capped at 300 characters (the SDK's
  `OpenEnded` / `MultipleChoice` limit).
* `latency` is wall-clock seconds for that question. The Evaluator marks any response above
  **15.0 s** incorrect regardless of content; the container logs each such response at ERROR.
* A question that raises emits a **legal** fallback rather than an empty string: for an
  `hh:mm:ss` question, the clip midpoint (offset by `start_time` unless it is a `2b`
  duration question), which keeps the chance of landing inside the acceptance window that an
  illegal string forfeits outright. Every other format falls back to `""`.

Exit codes: `0` success · `1` no requests · `2` more than half the qIDs had no media (the
weights were never loaded) · `3` more than half the questions raised. The failure rate is
also logged at ERROR above **1 %**, which on a 20-question batch means the first failure.

---

## 6. Building it (on the pod — do NOT build on the Windows box)

The Windows host has a documented Docker/disk constraint: the WSL `vhdx` on `C:` never
shrinks, a 16 GB build context kills the daemon, and the first container run costs another
17 GB. **This image is built on the pod.**

### 6.1 Assemble `resources/`

```bash
cd /workspace/repo/submissions/05-segment-armA
mkdir -p resources/adapters

# base: the Qwen3-VL-8B-Instruct snapshot ALREADY on the volume
cp -a /workspace/models/qwen3-vl-8b resources/model

# adapter 1 — rung 21 arm A2, epoch 3
cp -a /workspace/repo/experiments/21-recipe-sweep/runs/21_lr_2e4_v1/ckpt/v0-20260729-172404/checkpoint-2703 \
      resources/adapters/01-rung21-a2-checkpoint-2703

# adapter 2 — SEGMENT arm A, epoch 1  (run `seg01_a_rung21_ep1`)
cp -a /workspace/repo/experiments_segment/01-viability/runs/seg01_a_rung21_ep1/ckpt/*/checkpoint-860 \
      resources/adapters/02-segment-armA-checkpoint-860
```

⚠️ Confirm both source paths on the volume before copying — the adapter directories are the
authority, not this README. Each must contain `adapter_config.json` and
`adapter_model.safetensors`; `inference.py` raises at startup if either is absent, because an
incomplete stack is a **different model**, not a degraded one. Optimizer state
(`optimizer.pt`, `scheduler.pt`, `rng_state.pth`) is dead weight in the image — delete it
from the copies.

`resources/model/` must carry the processor bundle, not only weights:
`config.json`, `*.safetensors` + index, `tokenizer.json` / `tokenizer_config.json`,
`preprocessor_config.json`, `video_preprocessor_config.json`, `chat_template.jinja`.
`AutoProcessor.from_pretrained` is called on it and offline that failure is unrecoverable.
The build gate in `inference.py` checks `config.json` and `preprocessor_config.json`.

### 6.2 Check the disk before the build

```bash
du -sx /workspace          # `df` LIES on this volume — it reported 314 TB free on a
                           # full one and a run died mid-merge on "Disk quota exceeded"
```

### 6.3 Build and save

```bash
cd /workspace/repo/submissions/05-segment-armA
docker build --platform=linux/amd64 --tag segment-algorithm .
docker save segment-algorithm | gzip -c > segment-algorithm.tar.gz
```

With `buildah` instead of Docker (no daemon / not in the group):

```bash
buildah bud --platform=linux/amd64 --tag segment-algorithm .
buildah push segment-algorithm docker-archive:/tmp/segment-algorithm.tar
gzip -c /tmp/segment-algorithm.tar > segment-algorithm.tar.gz
```

Upload the tarball, **wait for "import completed"**, then Submit.

> No `build.sh` / `do_build.sh` exists here and none may be added: `.sh` launchers are
> forbidden by `CLAUDE.md` / `CONSTITUTION.md` §VIII. These commands are the recipe.

### 6.4 If the base image tag is unavailable

`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` is chosen to match the environment the path
was validated in. If that tag is missing on the build host, do **not** silently take an older
one — the build gate asserts `torch.__version__.startswith('2.8')` and will fail, which is
the point. Either pull the right tag or change the base **and** the assertion together, in
one commit, with the reason written down.

### 6.5 Smoke test (CPU, offline — validates wiring, not answers)

```bash
docker run --rm --network none \
  -e ALLOW_CPU=1 \
  -v /path/to/fixture/input:/input:ro \
  -v /tmp/out:/output \
  segment-algorithm
```

`ALLOW_CPU=1` is the deliberate escape hatch: an 8B bf16 VLM did not finish one question in
35 minutes on CPU, so a real batch is impossible, but the run still exercises the inventory
log, `batch.json` parsing, media indexing, decoding, and `answer.json` writing — which is
the half of this container that a wrong guess turns into a zero.

### 6.6 Environment switches

| var | default | effect |
|---|---|---|
| `SEG_VARIANT` | `plain` | promotes one `layout` key above all others. `overlayed` is the declared A/B of PLAN D2 (overlay + OCR vs plain + arithmetic) — it must be chosen here, never by a `dict.get()` |
| `SEG_RESIZE` | `1` | downscale decoded frames into the 960×540 training box |
| `ALLOW_CPU` | unset | permit a CPU run (smoke only) |

---

## 7. What is NOT verified

* **The GPU path of this image has never run.** The *code* has — over 6,254 real questions
  in `/workspace/envs/infer` — but not inside this container. Same residual risk the FRAME
  containers carry, neither larger nor smaller.
* **The `/input` layout** (§2). This is the dominant risk and the reason for every defensive
  branch in `inference.py`.
* **The SEGMENT startup allowance.** FRAME's was 120 s. SEGMENT's is unknown. This container
  loads a 17 GB base and then applies + merges two adapters, which is slower than serving a
  pre-merged checkpoint. If a startup limit turns out to bite, the fix is to merge both
  adapters at build time into `resources/model/` and ship no `adapters/` directory —
  `inference.py` already handles that case (it logs the substitution and serves the model
  directly), so it is a build change with **no code change**.
* **Whether the SEGMENT leaderboard baselines are beatable from 0.4964.** Nobody has opened
  that leaderboard. This submission is partly what buys that number.

## Sources

- `experiments_segment/01-viability/_tools/seg_eval.py` — the validated inference path
- `experiments_segment/01-viability/PLAN.md` (B1–B12, D1–D8) and `REVIEW.md` (A/B/C/G findings)
- `src/frame/segment/corpus.py`, `src/frame/parsing.py` — the vendored originals
- `context/decisions/segment-arm-a-is-one-broken-bucket.md` — the 0.4964 result
- `context/decisions/submission-01-rung06.md`, `context/decisions/container-answer-path-audit.md`
  — the two container defects that cost a FRAME slot
- `experiments/06-vit-lora/_tools/submission/` — the FRAME container this is modelled on
