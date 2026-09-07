# context/NOW_segment.md — the SEGMENT track, live state

> The tracker for the SEGMENT campaign. FRAME's live state is `context/NOW.md`; older per-rung
> detail is `experiments_segment/NOW.md`. **This file is the NOW — update it, do not append
> history.** Settled verdicts go to `context/decisions/` and get linked from here.

---

## 🎯 THE BAR: 0.5118

| | score | rank |
|---|---|---|
| rank 1 — Abulcasis V1 | 0.6065 | 1 |
| **OFFICIAL Proprietary baseline (SEGMENT)** | **0.5118** | **14** |
| OFFICIAL Fine-tuned baseline (SEGMENT) | 0.4927 | 19 |
| **us — arm A, LOCAL estimate** | **0.4964** | not submitted |

**64 participants, 13 above the proprietary baseline.** Clearing BOTH — the co-authorship
condition — means beating **0.5118**. Board + field: [[segment-leaderboard-and-the-bar]].

🔑 **Ranks 12 and 15 are `Fine tuned Qwen3 VL 8B` at 0.5155 / 0.5114** — our exact backbone with
an ordinary fine-tune, straddling the bar within ±0.004. Everything above that line is recipe.

📌 SEGMENT is ~**40 %** of the prize money; FRAME is ~20 %.

---

## WHERE WE ARE (2026-08-17)

**arm A scores 0.4964 — 0.0154 short — and the ENTIRE gap is one bucket answering a constant.**

| bucket | n | arm A | floor | margin |
|---|---|---|---|---|
| `object_recognition_OOD` | 1383 | 0.6139 | 0.5503 | **+0.064** 🟢 |
| `object_recognition_ID` | 1440 | 0.5792 | 0.5500 | **+0.029** 🟢 |
| `aggregation_OOD` | 572 | 0.4161 | 0.4056 | +0.011 |
| `event_understanding_ID` | 65 | 0.8154 | 0.8191 | −0.004 |
| `complex_reasoning_OOD` | 148 | 0.5946 | 0.7432 | −0.149 |
| `aggregation_ID` | 22 | 0.6364 | 0.9130 | −0.277 |
| `complex_reasoning_ID` | 43 | 0.6279 | 0.9074 | −0.280 |
| `event_understanding_OOD` | 145 | 0.6000 | 0.8897 | −0.290 |
| **`temporal_grounding_OOD`** | **1752** | **0.0457** | 0.3299 | **−0.284** 🔴 |
| **`temporal_grounding_ID`** | **684** | **0.0351** | 0.4392 | **−0.404** 🔴 |

`bucket_mean` **0.4964** · `acc_ID` 0.4224 · `acc_OOD` 0.3355 ·
[[segment-arm-a-is-one-broken-bucket]]

### 🔴 The one thing wrong

`temporal_grounding` is **2,436 questions, 39 % of the corpus, 2/10 of the headline**, at ~4 %.
Diagnosed on the archived predictions, zero GPU:

- **1,718 of 2,370 raw `time` outputs are literally `00:00:00`** — 72.5 %
- **0 predictions malformed** ⇒ the inversion, the format repair and the frame grid are fine.
  It is the model, not the harness.
- Mechanism: [[zero-is-format-localized]]'s attractor on a new axis. `corpus` rewrites the gold
  to an offset from clip start; that concentrated the target near zero and the model learned the
  mode instead of the value. **Training-side defect.**

🔑 **Move that bucket to nothing better than its own floor and `bucket_mean` gains +0.0688 →
0.565, past the bar, every other bucket unchanged.**

### 🟢 And the FRAME work transfers

`object_recognition` — 2,823 questions, `fo_class`-dominated, exactly what rungs 00–46 grind on
— beats its floor on **both** halves.

---

## NEXT — in order

1. 🔻 **RETRACTED 2026-08-25 — there is no zero attractor to kill.** This item read *"kill the
   zero attractor on `time`"* and named `ExportConfig.relative_time = False` as the cheapest
   candidate. Measured before spending the 12 h
   (`experiments_segment/01-viability/RESULTS_time_target_premise.json`, zero GPU): the `2a`
   relative target has **median offset 44 s**, only **1.1 %** of golds are exactly zero, and
   **0** fall outside `[0, duration]`. The concentration the item was built on does not exist.

   And `00:00:00` is not the target's mode: that constant scores **2.56 %** while arm A scores
   3.51 / 4.57 % — the arm is **one to two points above answering a constant**. The failure is
   **temporal localisation**, not the target format, so no rewrite of the answer string is
   licensed. See [[segment-arm-a-is-one-broken-bucket]] §CORRECTION.

   🎯 **The bucket is still the right target** — 39 % of the corpus, +0.0688 to reach its own
   floor, and the ceiling `corpus.py:66-70` records (~0.51 at coverage density) leaves room.
   **What it needs is a hypothesis about localisation with a mechanism**, pre-registered with the
   diagnostic that would falsify it. There is no costed rung here right now.
2. **📤 Submit arm A for calibration** — decided 2026-08-17, image built (below). We hold **zero**
   SEGMENT calibration points, and FRAME's local eval overstated the judge by **+0.12** and
   *inverted* on OOD. One of ten slots buys the deflation factor for every number after it.
3. Re-read the small buckets before treating them as failures. `aggregation_ID` is **22
   questions**; its −0.277 is six answers. They are 4/10 of the headline.

---

## 📦 THE SUBMISSION IMAGE — BUILT, NOT YET UPLOADED

```
E:\orena-build\orena-segment-arma-v1.tar.gz     17.33 GB     gzip integrity verified
image:  orena-segment-arma:v1
source: submissions/05-segment-armA/   (committed)
```

Qwen3-VL-8B base + rung-21 A2 `checkpoint-2703` + SEGMENT arm `checkpoint-860`, stacked and
merged **in order** at load. Offline, 15 s/question, 80 GB.

**Verified without a GPU:** startup, full `/input` inventory dump, `request.json` parsing,
time-grid routing (1/1 routed, 0 `2b`), clip resolution from `plain/`, the 10 FO class names read
from the SDK **at runtime** (`RULES §8b`), and a clean refusal to start on CPU instead of hanging.

🔴 **NOT verified: a real `generate()` inside this image** — the build host has no GPU. The
inference path is validated on 6,254 real questions, but that ran outside the container. **This
is the residual risk on the slot.**

⚠️ **The `/input` layout is still CONJECTURE** — the repo's only description came from a fixture
that does not name its track. The resolver guesses nothing: it logs the complete inventory first,
then tries declared manifest keys (`plain` preferred, `overlayed` last, via `SEG_VARIANT`),
conventional dirs from `focus.config`, every `*.zip`, then a full sweep. Tested against **11
synthetic deliveries**; all pass or fail loudly.

🔴 **Two alarms it will shout, because nothing else would look wrong:** every
`request.start_time == 0.0` (the clip is pre-cut ⇒ every absolute time answer unrecoverable), and
an error rate above 1 %.

---

## THE TRACK IS NOT FRAME — the differences that bite

| | FRAME | SEGMENT |
|---|---|---|
| latency | 5 s, pooled per batch | **15 s PER QUESTION** |
| GPU | 48 GB | **80 GB** |
| scored buckets | 4 | **10** (5 groups × ID/OOD) |
| clip duration | 0.0 s in all 20,000 rows | median **119 s**, max 300 |
| dominant format | `fo_class` 45 % | **`time` 38.2 %** |
| template overlap with FRAME | — | **0.70 %** |
| leaves with zero FRAME examples | — | **9 of 15** |

🔑 **Latency is a non-issue, and that changes what we may spend.** Measured **0.94 s/question** at
K=27 frames — **16× inside the budget**, ~60 GB of VRAM spare. The 27B in bf16 is 52.6 GiB and did
**not** fit FRAME's 48 GB card; **it fits here.** In FRAME we fought for milliseconds; on SEGMENT
the clock is not the constraint.

🔴 **`RULES §4, §4d and §4b-i` are FRAME-only** and would mis-instruct a SEGMENT session — §4b-i's
pooled-latency reading is contradicted here by both the design doc and the SDK. Each owes a
decision note before anything leans on it.

---

## THE INSTRUMENT — read before quoting any SEGMENT number

The headline is an **unweighted mean over 10 buckets** and several are tiny: `aggregation_ID`
**n=23 over 21 distinct templates**, `complex_reasoning_ID` n=54, `event_understanding_ID` n=94.
At that size the per-template modal answer is nearly the identity, so those buckets are close to
free — **and each is worth the same 1/10 as `temporal_grounding`, which is 39 % of the corpus and
genuinely hard.**

⇒ **You do not win by lifting the hard buckets; you lose by breaking the easy ones.** Format
compliance outweighs capability here: a verifier that raises turns a right answer into a zero, and
the formats at risk are `time` (38.2 %) and `fo_class` (25.1 %).

Floors: `experiments_segment/01-viability/RESULTS_bucket_floors.csv`. ⚠️ They key the template on
the raw question, which over-fragments and **inflates** the small buckets — an upper bound.

---

## INFRASTRUCTURE — what exists now that did not

- **`_tools/seg_eval.py`** — the SEGMENT `predict()`. Reuses `frame.segment.corpus.build_row`, so
  eval and training share one prompt, one frame grid, one clip window. Chunked, resumable.
- **`_tools/bucket_floors.py`** — the per-bucket floor table.
- **`submissions/05-segment-armA/`** — the container.
- **Two harness fixes** that were silently poisoning every SEGMENT number: `run.py` hardcoded
  `Track.FRAME` (5 s cap on a 15 s track ⇒ score ≈0 at `rc=0`), and `normalize_answer` did not
  repair a trailing period on `hh:mm:ss`, which `Time.verify` **raises** on and the Evaluator
  scores INCORRECT in silence — on 38.2 % of the track.

### Where things live
- Data: `external_data/orena-data/*/data/segment/{train,test}.parquet` — **local**, 20,000 rows.
- Frames, checkpoints, adapters: RunPod volume **`ORENA-CHALLENGE` (670 GB, EU-RO-1)**, intact.
- 🔴 **`21_lr_2e4_v1/merged/` is EMPTY on the volume** (cleaned for disk). A2 ep3 survives only as
  the **adapter** `ckpt/v0-20260729-172404/checkpoint-2703`. Anything needing A2 must stack
  base → A2 → arm, merging in order.
- Envs are ON the volume (`/workspace/envs/infer` = torch 2.8 + transformers 4.57.6 + decord +
  ms-swift 4.4.1) ⇒ **a fresh pod needs no installs.**

### Pod operations, paid for on 2026-08-17
- EU-RO-1 capacity is intermittent; a retry loop got an A100 80 GB at $1.39/h after 3 rounds.
- 🔴 **A lock read once is not a lock** — two concurrent retry loops each created a pod.
- 🔴 **RunPod pods cannot build Docker** (no daemon, no socket, no privileged caps). The image was
  built locally with a **77 KB** context (`docker cp` + `commit`), never a 17 GB one.
- 🔴 **`scp -r` exited 0 having silently skipped `config.json`, `model.safetensors.index.json`,
  `chat_template.json` and `merges.txt`.** A byte-for-byte size diff against the source caught it;
  without that check the uploaded image would not have loaded the model at all.

---

## OPEN QUESTIONS
- Does the local-vs-judge deflation on SEGMENT match FRAME's +0.12? **Unmeasured** — submission 1
  buys it.
- What does `/input` actually contain? **Unknown.** The container logs it on first contact.
- `frame.ledger` and `frame.measured` still glob `experiments/*` and **cannot see this tree**, so
  `RULES §9` is unsatisfiable for SEGMENT. Results live here meanwhile.
- Is the 27B worth it here, now that 80 GB makes bf16 fit and latency has 16× headroom?
