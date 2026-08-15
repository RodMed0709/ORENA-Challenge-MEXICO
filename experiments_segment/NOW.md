# experiments_segment / NOW — live state of the SEGMENT track

## 🔴 THREE POD-STOP LAYERS ARE ARMED — read this before touching anything

| pod | what | $/h | how it stops |
|---|---|---|---|
| `ocbm9adl4rc0ep` | SEGMENT arm A, 1 epoch | 1.39 | chain trap + on-pod watchdog (18 h wall) |
| `lyxe3bqeqeselg` | **rung 40 arm B, 3 epochs (B200)** | **6.79** | chain trap + on-pod watchdog (14 h) + **an INDEPENDENT killer on the operator's machine (15 h)** |
| `7ckucfad9ws4ui` | rung 42 | 0.99 | its own chain |

The third layer exists because layers 1 and 2 both live ON the pod and share a failure
domain: if the filesystem wedges or the shell skips its trap, both die together. The
independent killer talks only to the RunPod API. It is deliberately dumb — a hard
deadline and a stop, no log parsing, no liveness heuristics — so it cannot kill a
healthy run for a clever reason. Script: `scratchpad/killer_b200.py`.

⚠️ At $6.79/h the B200 costs **$163/day idle**. If everything else fails, stop it by hand
in the RunPod console.

## rung 40 arm B at 3 epochs (running on the B200)

Leo's ep1 measured `bucket_mean` **0.5763** and beat its own rung-38 control (+0.054 ALL,
+0.062 ID, +0.031 OOD, all three CIs excluding zero). But it is **1 epoch**, and our best
FRAME checkpoint (A2 ep3) is **0.6496** at **3 epochs** — so the two are not comparable.
Epoch-matched at ep1 the 27B leads 0.5763 vs A2's 0.5592, i.e. **+0.0171**.

This run answers the only question that comparison leaves open: what the 27B does at 3
epochs. Built by importing Leo's own `ArmConfig` and calling his `main()`, with exactly
two fields changed — `num_train_epochs=3` and `run_name="40_B_connector_ep3_v1"` (so his
ep1 artifacts cannot be overwritten). A blocking gate diffs the config against his ep1
and refuses to run on any other difference; it printed **"recipe drift: NONE"**.

🔴 **Declared deviations, both of which weaken the comparison and neither of which is
optional:**
1. It runs on a **B200 (sm_100)**; his ep1 ran on an **RTX PRO 6000 (sm_120)**. No PRO 6000
   capacity existed in EU-RO-1 at launch (both SKUs returned 500).
   `archived-results-not-bit-reproducible` measures ~0.5 % of answers changing on a GPU
   swap, so this is **not bit-comparable with his ep1**. It IS comparable against A2 ep3
   and against its own trajectory.
2. `gen36-fails-the-8b-recipe-not-the-backbone-test` says **"Do NOT scale this recipe to
   3 epochs"**, and `lr 2e-4` is the **8B's** optimum, never ported to the 27B. This run is
   taken against that standing advice, deliberately, to replace an extrapolation with a
   measurement.

📌 Disk: three rung-21 **epoch-1** merges were deleted to make room (`21_vitlr_v1`,
`21_clip10_v1`, `21_lr_1e4_v1`, 17 GB each). All are regenerable — their adapters are
intact (3 each, verified before deleting) — and none was anyone's reported result, which
is always the ep3 merge. **A2's ep3 merge was verified present after the deletion.**


> Updated 2026-08-15 ~02:40 UTC. Read this before touching anything on the SEGMENT side.
> The FRAME brain (`context/INDEX.md`, `context/RULES.md`, `context/decisions/`) stays the
> single source of settled verdicts for BOTH tracks — nothing is duplicated there.

## 🟢 RUNNING RIGHT NOW — do not launch anything on this pod

**Pod `ocbm9adl4rc0ep`** ("open_cyan_snipe"), A100 80GB PCIe, sm_80, $1.39/h, EU-RO-1,
volume `ORENA-CHALLENGE` on `/workspace`.
🔴 **sshd is dead** (refuses on the mapped port while reporting "running" inside) — drive it
through the Jupyter terminal websocket, the documented fallback. Do NOT recreate the pod.

| what | state |
|---|---|
| **arm A train** `seg01_a_rung21_ep1` | 1 epoch, 860 steps, **55.8 s/it**, ETA ~12 h from 02:40 UTC |
| chain | `/workspace/tmp/seg01_chain.sh`, log `/workspace/tmp/seg01.log` |
| watchdog | detached, 18 h wall, **no silence detector** (deliberate) |
| **the pod stops itself** | `trap stop_pod EXIT` + watchdog, pod_id literal, key read before the trap |
| test-frame extraction | 110,474 frames, 38 videos, ETA ~65 min, `nice -n 10` |

After training the chain runs V1 (grad) + V2 (coverage), writes `RESULTS_gates.json`,
commits, and stops the pod. **No human action is needed.**

## What arm A is

`merged/checkpoint-2703` (rung 21 arm A2 ep3, `bucket_mean` 0.6496 — the best FRAME
checkpoint we own) **+ a new LoRA trained on SEGMENT**. The one variable vs rung 21 is the
input modality (one still frame → a K-frame clip) plus the two answer formats FRAME lacks.

Verified from the emitted `args.json`, not assumed: lr 2e-4, rank 8, alpha 32, dropout 0.1,
`freeze_vit False`, `freeze_aligner True`, 1×16, wd 0.1, `adamw_torch_fused`, cosine,
warmup 0.03, seed 42, `target_modules ['all-linear']` — **all identical to rung 21**. Only
`epochs` (1 vs 3), `dataset` and `model` differ, which are the three that must.

📌 **Correction to the review:** a merged dir DOES carry `args.json` and swift loads it.
`--load_args false` is therefore load-bearing, not a defence against a non-threat. The diff
above is the proof it held.

## Measured facts nobody had before this rung (see `01-viability/RESULTS_probe.md`)

- a visual token is **32×32 px** (patch16 × merge2) ⇒ a 960×540 cache frame is **480** tokens
- **`temporal_patch_size = 2`** ⇒ K frames cost `ceil(K/2)` temporal positions
- `VIDEO_MAX_TOKEN_NUM` defaults to **768 > 480** ⇒ the cap never binds; **pass no cap**
- **`MAX_PIXELS` has been a silent no-op on Qwen3-VL for the whole campaign** — the merged
  checkpoint's `preprocessor_config.json` carries the library default. No FRAME result moves.
- **ms-swift does NOT validate the `<video>` tag count**: 0 tags encodes identically to 1,
  2 only warns. Our build-time gate is the only protection.
- **`use_logits_to_keep` is forced OFF for every multimodal model under transformers 4.x**
  (`swift/trainers/mixin.py:209-210`), and 5.x is barred by `CLAUDE.md`.
- probe: **58.29 s/it, 74.79 GiB peak** at K≤71 → OOM; at K≤36 it fits with headroom.

## 🔴 The ceiling this rung ships with — read it before reading the result

VRAM forced the grid to **K ≤ 36**. The two longest duration classes therefore sample at
**coverage grade, not localisation grade**:

| dur | share of corpus | spacing | acceptance window |
|---|---|---|---|
| 10 s | 3.3 % | 1.11 s | 1.11 s ✅ |
| 29 s | 24.9 % | 1.32 s | 1.32 s ✅ |
| **119 s** | **48.8 %** | **4.58 s** | 2.32 s ❌ |
| **299 s** | **20.1 %** | **8.54 s** | 4.32 s ❌ |

⇒ **`temporal_grounding` (2 of the 10 buckets) is capped by geometry, not by the model.**
A weak number there is the expected consequence of this cap and must not be read as a
capability finding without re-running at a finer grid.

## Corpus (built, all gates green)

13,746 train rows · 186,173 frames referenced, all present · both routers **FP 0, FN 0**
(time/pct TP 7,711; `2a`/`2b` separates 261/261 from 0/7,384) · time round-trip verified on
**5,125 rows**, 96 of them multi-timestamp · visual tokens mean 3,335 / max 8,640 ·
SDK acceptance-window formula matched against the vendored source.

🔴 **The router is the LITERAL string `"hh:mm:ss"`, never a timestamp regex.** The regex
reading scores recall 0.139 with 2,346 false positives. This bit once already.

## Open, in order

1. **read the result** — `RESULTS_gates.json`, then score the 6,254 test rows
2. **the SEGMENT baselines are still unknown** — nobody has opened that leaderboard. Until
   then "did we beat it" is unanswerable, and it is the one input that decides whether the
   track is worth more spend.
3. `frame.ledger` and `frame.measured` **cannot see this tree** (they glob `experiments/*`),
   and `measured._rung_key` would collapse SEGMENT `01` onto FRAME `01-ood-split`. So RULES §9
   is unsatisfiable for SEGMENT until that is fixed. Results live here meanwhile.
4. the eval path itself: a SEGMENT `predict()` (temp MP4 + `base_fps`/`fps`, not a still),
   `metrics.time_report`, and the container's `/input` layout (`plain/` + `overlayed/`, no
   `frames` key — the shipped FRAME container answers **zero** SEGMENT questions).
5. RULES §4 and §4d are FRAME-only and would mis-instruct a SEGMENT session; §4b-i's pooled
   latency reading is contradicted for SEGMENT by the design doc and the SDK (per-question,
   15 s). Each owes a decision note before anything leans on it.

## Cost so far

~$19 of A100 for this run, plus ~$4 of extraction. The corpus, the probe and every gate were
zero-GPU.
