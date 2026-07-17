# THE MAP — unified FRAME strategy (meshed from 5 competing plans)

> Synthesis of 5 independent `vlm-strategist` plans (aggressive · floor-first · Copeland-ranking ·
> data-centric · deployment) against `ATTACK_LADDER.md` / `literature/FICHAS.md` / `context/EXPERIMENT_DESIGNS.md`.
> This is the north star. It **supersedes the sequencing** in ATTACK_LADDER.md (the rung recipes there
> stay valid). Dated 2026-07-13.

## The two blocks — product vs lab (added 2026-07-15)

> **Read this before the phases.** Most of this repo is a measuring instrument, not the
> deliverable. Conflating the two is why the technical picture reads as "all evaluation".
> Sourced from the official track spec (`context/challenge/`) + the challenge overview page.

```
╔══════════════════════════════════════════════════════════════════════════╗
║  BLOCK A — THE MODEL · the product · this is ALL they ever see           ║
╚══════════════════════════════════════════════════════════════════════════╝

   RGB image ──┐
               ├─►┌─────────┐  ┌───────────────────┐  ┌───────────────┐
   question ───┘  │ prompt  │─►│ Qwen3-VL-8B       │─►│ constrained   │─► "Sponge"
   carries:       │ build   │  │ + LoRA (merged)   │  │ decoding      │   ≤300 ch
   · procedure    └─────────┘  │ bf16 · 18/48 GB   │  │ ✗ NOT BUILT   │
   · timestamp      ours       │ greedy · ≤32 tok  │  └───────────────┘
   · expected                  └───────────────────┘  "how many" →
     output                                            pure integer
   · FO classes
                    NO video · NO decode · NO frame sampling · NO judge

   OBJECTIVE (exactly one)
       max  mean Accuracy over the 4 buckets
       └─ a malformed answer is scored incorrect → format hardening IS
          accuracy, not a second goal

   CONSTRAINTS (checked, not maximised)
       < 5 s / question   p99 0.59 s measured → ~4.4 s unspent
       48 GB VRAM (L40S)  18.01 GB used → do NOT quantise the 8B
       no internet        COPY weights · pinned wheels · HF_HUB_OFFLINE=1

       speed is a BUDGET TO SPEND (resolution, detector, 32B), not a goal.
       There is no prize for answering in 0.3 s instead of 2 s.


╔══════════════════════════════════════════════════════════════════════════╗
║  BLOCK B — EVALUATION · the lab · NEVER ships · ~90% of this repo        ║
╚══════════════════════════════════════════════════════════════════════════╝

   videos + parquet              released data has no loose images →
        │                        WE must synthesise what they will hand us
        ▼
   data.py  load_frame_items()   offline door (the SDK's own path hits the
        │                        gated HF Hub → DatasetNotFoundError)
        ▼
   FrameProvider.get_frame()     simulates Block A's input image
        │                        ⚠ our sampling ≠ their frame choice
        ▼
   ╔══════════════════════════════════╗
   ║   THE SAME BLOCK A ENGINE        ║  the only piece shared by A and B
   ╚══════════════════════════════════╝
        │
        ▼
   focus.Evaluator + local judge  OUR ruler, NOT our grade — their judges
        │                         are secret (≤3 LLMs, majority vote).
        ▼                         Tuning to a judge = disqualification.
   4 buckets → mean


              ┌──────────────────── THE LOOP ─────────────────────┐
              │                                                   │
   BLOCK A ───┴──► answers ──► BLOCK B ──► "you are weak HERE" ───┘──► BLOCK A

   B is NOT parallel to A — B is A's controller. Only A ships.
   Eval driving design is correct method; the bug was a MIS-CALIBRATED ruler
   (we chased raw acc; the exam is the bucket mean).


   THE 4 BUCKETS — equal weight. Volume ≠ weight.
   ┌───────────────────────────┬───────┬──────────────────────────────────┐
   │ object_recognition × ID   │  25%  │ fo_class · binary · MC ·         │
   │ object_recognition × OOD  │  25%  │ open_ended   (all ≈0.53–0.60)    │
   ├───────────────────────────┼───────┼──────────────────────────────────┤
   │ aggregation × ID          │  25%  │ number → 0.433 ⚠                 │
   │ aggregation × OOD         │  25%  │ OUR WORST, and worth HALF the    │
   │                           │       │ exam. 1 pt here ≈ 2 pt in fo_class│
   └───────────────────────────┴───────┴──────────────────────────────────┘
   OOD = 2 of 4 = 50%, on TWO axes: unseen procedure AND unseen question
   phrasing. Our Sigmoid proxy only covers the first.


   THE TWO GATES — different rules, do not mix them
   PRE-EVAL (→ Sep 1)   mean Accuracy over buckets > BOTH baselines
                        └─► final phase + CO-AUTHORSHIP ★  ← the Core Value
   FINAL   (Sep 8)      full Copeland over buckets → podium / prize money


   ⛔ BLOCKED BY THE ORGANIZERS (not our debt)
      "Release of submission example: Soon" · template + ranking toolkit
      unreleased → the Docker's I/O contract does not exist yet.
```

**What this diagram corrects in the phases below** (they were written before the
official spec was read; left intact as provenance):

- **Phase 4's gate is wrong twice.** It says *"≥6/10 buckets vs EACH, Copeland"*. FRAME
  has **4** buckets (reframe #1 below already says so — the map contradicts itself), and
  the pre-eval gate is the **mean**, not Copeland. Copeland only rules the final phase.
- **Principle 1's implication is wrong.** `fo_class` and `number` are not a pair:
  `aggregation`(=number) is a **50%** of the score; `fo_class` is one slice of the other 50%.
- **Principle 4 invents B2's size.** The spec says *"a strong open-source VLM fine-tuned by
  the organizers"* — it never says 4B (likely bled in from the `Qwen3-4B` judge). The
  "seam" may be far narrower than assumed.
- **Multi-frame and frame-selection/pHash are dead as inference levers** — they hand us one
  image. Both only affect training data and our measurement fidelity.
- **Phase 0's *"prompt aimed at the 4 unseen reasoning buckets"* is a zombie line** — FRAME
  has no reasoning buckets (reframe #1).
- **Phase 5's SGLang "reuse image cache across a frame's questions"** is likely moot: the
  contract is one image + one question per call.

## 0 — Executive roadmap (v3, improved map · added 2026-07-14)

> Canonical, self-contained plan with the strategy-notes improvements folded in. The sections
> BELOW this one (reframes, invariants, THE UNIFIED PLAN, per-angle contributions, resolved risk)
> are the **original synthesis / rationale** and are kept intact as detail + provenance.

### Fundamental principles (the "why")
1. **FRAME only has 2 capability types, not 5.** Verified on real data: only *object recognition* and *counting (aggregation)*. Complex reasoning + events live in OTHER tracks → all effort goes to the weak formats **fo_class** and **number**. No reasoning/time/percentage.
2. **Real latency headroom is ~4×, not 8×.** 5 s limit; 0.6 s measured but on an A100. Real GPU (L40S) ~2× slower → ~4× headroom. Extra frames/resolution/bigger model eat it.
3. **Copeland is margin-blind; format is first.** Winning 90% on an already-won bucket buys no votes; one silent-0 flips a bucket → hardening format matters MORE than improving the model.
4. **Win "in the seam" between the two baselines.** *(Working hypothesis, NOT measured.)* B1 (frontier zero-shot) reasons well / perceives surgery poorly; B2 (their FT 4B) perceives ID well / fails OOD. A balanced 8B LoRA that keeps reasoning AND holds OOD beats both where each is weak. *First leaderboard submission validates this.*

### Non-negotiable invariants
- **Plain LoRA** (S2Can: r=8, α=32, frozen ViT). Instruction-FT = +16, exotic adapters +2 → skip them.
- **bf16, NOT 4-bit** on the 8B (fits 48 GB; quantizing only adds latency). 4-bit reserved for the 32B wildcard.
- **ms-swift** (native Qwen3-VL + `max_pixels`/freeze-vit). Unsloth lags on this model.
- **1–3 frames**, no stacking. FRAME is single-image.
- **Checkpoint selection by per-bucket OOD win, never mean.** Discard any ckpt that wins ID but drops OOD.
- **An always-valid zero-shot floor** kept submittable as a calendar hedge.

### Phases (improvements folded in)
- **Phase 0 — Floor, format-hardening, permits (Jul 13–15):** offline Docker v0 (network-off test) · 6-gate anti-silent-0 CI · **FSM/guided decoding in serving** (force the format *during* generation) · **submit all DUAs** (CholecT50/45, Cholec80, HeiCo) · Jul 15 submit the bare floor first · **dose the 10 submissions** (pre-eval = 10 shots over 20 videos).
- **Phase 1 — Split + harness (split ✅; 2 pending):** ✅ frozen split (Sigmoid held-out = local OOD proxy, NOT the hidden leaderboard OOD) · (a) cross-corpus pHash leak scrub · (b) measure p99 on real L40S + warm-up · **SurgCheck grounding test** on rung 00 (decides whether Phase 4 is worth it).
- **Phase 2 — LoRA v1 (wk2–3):** S2Can recipe, bf16, 3 epochs first, targeting fo_class+number; single variable = LoRA ON; select by OOD; re-measure p99 per merged ckpt · **anti "model-collapse" guardrail** (mix some structured/open answers so reasoning survives).
- **Phase 3 — Data flywheel + reserves (wk4–5):** error→mint-QA→retrain, cross-procedure balance · **visual-degradation slice** (blood-soaked sponge, corner occlusion, low contrast) · **[RESERVE] YOLO→ROI / visual prompts** (only if LoRA doesn't cross the gate on fo_class/number) · **32B FP8 wildcard** only if a bucket stays stuck and the offline path is green on L40S.
- **Phase 4 — Freeze + prove the lead (wk6–7, pre-eval closes Sep 1):** network-off dress rehearsal on L40S · bootstrap self-audit vs BOTH reproduced baselines · ship only ≥6/10 buckets vs EACH, CI-clear (prefer 7/10).
- **Phase 5 — Ship + optimal serving (Sep 1–8):** final offline Docker + method description + release annotations + open-source model · **vLLM vs SGLang RadixAttention benchmark** (reuse image cache across a frame's questions).

### Calendar
| Date | Gate | Phase |
|---|---|---|
| Jul 15 | Pre-eval opens (10 submissions / 20 videos); submit the floor | 0 |
| Sep 1 | Pre-eval closes; only teams that beat both baselines advance | 4 |
| Sep 8 | Final submission (Docker + method + model) | 5 |
| October | MICCAI 2026 announcement | — |

### Two honest caveats (must stay visible)
- The B1/B2 strength/weakness profile is a **hypothesis**, not a measured fact.
- Our "OOD" is a **local proxy** (Sigmoid held-out from released data), NOT the hidden leaderboard OOD.

### STATUS (updated 2026-07-14) — Phase 2 DONE, Phase 3 redirected

- **Phase 2 (LoRA v1) — DONE, PASS.** rung-02 (r8/α32/lr2e-5, 3 epochs, frozen ViT+aligner, 1 frame, max_pixels 1280×720). Best ckpt = **epoch 2** by acc_OOD (per-epoch 0.583/0.592/0.583 — **no OOD collapse**). Full val: **pre_eval 0.174→0.708, raw 0.262→0.566, OOD 0.269→0.592**. Per format: **fo_class 0.168→0.588 (+0.42), number 0.141→0.433 (+0.29)**, all up, OOD>ID. Merged model on volume `gf78k60nlt` at `experiments/02-lora-sft/runs/02_lora_sft_v1/merged/checkpoint-1720` (each run owns its ckpt/merged; only frames are shared, in `/workspace/frames_cache/`). See `experiments/02-lora-sft/RESULTS.csv`.
- **rung-03 (prompt engineering) — faithful negative.** Rigorous select-on-val_id / confirm-on-val_ood: no prompt arm beats baseline beyond noise on OOD (winner's curse caught). Measured: FO-grounding lever = +0.061 (already in baseline). **fo_class + number are LoRA/data wins, NOT prompt.** See `experiments/03-prompt-variants/`.
- **`number` diagnosed — NOT a data problem.** It is already 31% of train (4262q) with a well-spread answer distribution (mean 2.74, not skewed low) → oversampling won't help; the bottleneck is **perception**.
- **Phase 3 REDIRECT — CORRECTED 2026-07-16. The target is the ViT, not the aligner.** Train loss AND acc_OOD both **plateau at epoch 1**. rung-02 froze **ViT + aligner**, so the LoRA only ever touched the language side; Surgical-LVLM's contribution (**VP-LoRA**, arXiv 2405.10948) is LoRA on the *visual-perception path* — we froze what the SOTA surgical VQA model adapts. **Next experiments (single-variable vs rung-02):** #1 **LoRA on the ViT**, #2 **rank probe r=8→32** (capacity vs representation vs resolution), then resolution / multi-frame / domain-augmentation. Resolution is **demoted** to a co-lever (untested assumption feeding a frozen encoder; headroom is ~4× not 8× per §2 above).
  - 🔴 **Two corrections to the earlier version of this line, both of which weaken it. Read them before spending a run.**
  - **It said "#1 unfreeze aligner/merger" and that recommendation had lost its citation.** It rested on FineViT (2603.17326) — which is a **new encoder trained from scratch on billions of recaptions, not a freeze-vs-unfreeze ablation** — and on OpenVLA (2406.09246), which is **robotics**. The aligner was never the evidenced lever. The ViT is the corrected target, but honestly: VP-LoRA reports **+2** on the visual path, so this is **a measurement we owe ourselves, not a bet the literature backs**.
  - **"Representation ceiling" is an n=1 conclusion.** It comes from **one** training run (r8/α32/lr2e-5, 3 epochs, frozen encoder) — the project's only one. A plateau at epoch 1 under a single configuration is equally consistent with "wrong lr", "r=8 too small", "3 epochs is nothing", or "we froze the thing that needed to move". **Treat the ceiling as a hypothesis, not a finding.**
  - ⚠️ **Judge the next run with `experiments/08-data-card/`, not with the per-format numbers above.** `acc_number` averages **eight templates** with trivial floors from 0.24 to 1.00, four of them degenerate; it is **not interpretable**. On the SDK's own (hierarchical) estimator `number` sits at **0.3803, CI [0.328, 0.428]**, against a template-aware floor of **0.3840** — at the video level we cannot yet distinguish the model from a constant. **That gap is what the ViT run has to move.**
- **Process fixes (do before next run):** add a `val_dataset` to swift sft (track `eval_loss` — we currently have NO val curve, only 3 OOD points), eval every half-epoch, early-stop on acc_OOD.
- **Infra notes:** measured GPU throughput for THIS workload — RTX 5090 (7.5s/step) > RTX 4090 (11.5s) > RTX PRO 6000 (31s, Blackwell kernels immature); pick by measured s/step, not tier. `run.py` hardened for 32GB (frees the 8B before + the judge after eval, or the per-epoch merge OOMs). JupyterLab launcher at `/workspace/start_jupyter.sh` (kernel "ORENA (infer env)").

---

> **Preservation note:** everything below is the original 5-plan synthesis and deeper rationale
> from which the executive roadmap above was distilled. Kept intact.

---

## Three reframes the whole panel forced (the ladder was wrong on these)

1. **FRAME only exercises ~2 capability groups — CORRECTED 2026-07-13 from real data.** The taxonomy has
   5 groups × {ID,OOD} = 10 possible buckets, but that is the CROSS-TRACK max. Verified against the real
   FRAME parquets (all rows carry `track == "frame"`; 20000 QA): FRAME contains **object_recognition +
   aggregation** only (temporal_grounding n=3 = noise). **event_understanding + complex_reasoning are
   ABSENT from FRAME** — they live in the PROCEDURE/SEGMENT tracks. FRAME formats: **fo_class + number**
   (the dominant + weakest, baseline 0.182/0.127), then binary/open_ended/MC — **no `time`, no `percentage`**.
   So FRAME is scored on **~4 buckets** (2 groups × ID/OOD), not 10. Earlier "measured 3 of 10, blind to
   the reasoning groups" was WRONG — those groups don't exist in FRAME, so there's nothing to mint for them.
   Implication: all leverage → fo_class + number; no reasoning-QA, no time/percentage handling. OOD ≈ half
   the score still holds (via the Sigmoid-held-out partition). Detect track membership via the `track`
   column / `data/frame/` path.
2. **The "8× latency headroom" is an A100 illusion.** Rung-00 p99 0.59 s was on **A100**; the eval box is
   **L40S 48GB** (~½ the memory bandwidth) → decode ~2× slower → real headroom **~4×**, eroding under
   LoRA-merge + max_pixels + frames + 32B. **We have never run one question on the real eval hardware.**
   Resolution/multi-frame/32B are NOT free.
3. **Copeland is pairwise-majority, margin-blind.** Beat each baseline on **≥6/10 buckets** — piling points
   on a won bucket buys zero votes; one silent-0 flips a bucket; a dup-qID/adversarial-misfire zeros the
   **whole run = 10/10 lost**. Two baselines have OPPOSITE profiles: B1 (frontier zero-shot) reasons well /
   perceives poorly; B2 (their FT-**4B**) perceives ID well / reasons+OOD poorly. **We win in the seam:** a
   balanced 8B LoRA that keeps generalist reasoning, holds OOD, emits zero silent-0s.

## Governing invariants (all 5 plans agree — non-negotiable)
Plain LoRA (S2Can r=8/α=32/frozen-ViT; skip exotic adapters, +16 vs +2) · **checkpoint selection by
per-bucket Copeland margin, never mean** · OOD-selected, discard any checkpoint that flips a won bucket ·
1–3 frames (temporal stacking hurts) · no thinking-mode / no agentic loops / no vision-encoder swap
(latency-fatal) · always-valid zero-shot floor stays submittable · every number reproducible from a commit.

---

## THE UNIFIED PLAN (phased, gates resolve the sequencing conflicts)

### Phase 0 — Floor + gates + DUAs  (Jul 13–15, before leaderboard opens)
*Resolves: deployment "harness first" + floor-first "ship the floor" + Copeland "silent-0 is rung ZERO".*
- **Offline Docker v0** around the existing zero-shot HF engine (`src/frame/engine.py`) — vendored pinned
  wheels, `COPY` weights, `HF_HUB_OFFLINE=1`, **network-unplugged test** decodes a real mp4. (No Dockerfile
  exists yet — Week-0 line item.)
- **6-gate silent-0 CI (rung ZERO, independent of modeling):** G1 adversarial-output scan (superset of 12
  phrases → block+neutralize) · G2 dup-qID assert · G3 ≤300 char · G4 per-format canonicalize→**vendored**
  `verify()` (number `.isdigit`; fo_class token∈`FOType.names()` **10 classes, config-driven**, drop
  instrument tokens) · G5 p99 · G6 offline-load. Runs on every prediction artifact.
- **Prompt build (merged rungs 01+02), default-OFF flags:** reference-mirroring answer-style (dense, gold-
  term, no hedge, <300) **aimed at the 4 unseen judge/reasoning buckets** (Copeland's free flip) + FO-taxonomy
  "instruments are NOT foreign objects" exclusion (baseline's #1 failure). Zero-train.
- **Submit all DUAs TODAY** (CholecT50/45, Cholec80, HeiCo — all CC BY-NC-SA 4.0; approvals take days–weeks
  → no downside starting, data does not enter training until scrubbed). Release obligation budgeted Week 8.
- **Jul 15:** submit the **bare zero-shot floor FIRST** (prove the pipeline), then the 01+02 build. Treat the
  first leaderboard number as a **measurement instrument** to close the local-vs-real OOD-denominator gap.

### Phase 1 — Split + harness  (✅ split DONE 2026-07-13)
*Resolves: floor-first "leak scrub" + deployment "re-baseline on L40S". CORRECTED: FRAME has only 2 groups, not 10.*
- ✅ **Split frozen** = `experiments/splits/frame_ood_v1.csv` (sha256 6fd34c2c), using the **organizers' own
  train/test partition** (Sigmoid held out only in test = their OOD design). train 92vid/13748q · val_id
  28/2252 (chole=ID) · val_ood 10/4000 (Sigmoid=OOD). Zero video overlap; `(dataset,video_id)` key; hash-verified.
- ✅ **FRAME scope verified** (via `track` column): only **object_recognition + aggregation**; formats
  **fo_class + number** (baseline-weak) + binary/open_ended/MC. **No `time`, no `percentage`, no reasoning
  groups** (those are PROCEDURE/SEGMENT). → ~4 scored buckets, not 10. No reasoning-QA to mint.
- **Remaining before/around training:** (a) **Cholec80 cross-corpus pHash scrub** (videoID-intersect + frame
  fingerprints vs the challenge test) — the one leak internal splitting can't catch; (b) **re-baseline p99 on
  the REAL L40S** + warm-up forward in `load()`; (c) *optional* Engine-A synthetic fo_class/number data
  augmentation (we already have 13,748 train q, so this is a booster, not a prerequisite).

### Phase 2 — LoRA fine-tune (the primary lever)  (Jul 22–Aug 5, wk2–3)
*Resolves: aggressive "fire the primary lever" — grounded on the confirmed split.*
- **LoRA v1 (S2Can recipe, bf16, 3 epochs first)** on the `frame_ood_v1` **train** split (13,748 q), targeting
  the weak formats **fo_class + number**. `system` prompt + frame sampling byte-identical to rung 00 (single
  variable = LoRA ON). Recompute the zero-shot arm on the same val for a clean Δ.
- **Select the checkpoint by acc_OOD (val_ood = Sigmoid), never mean.** Discard any checkpoint that wins ID
  but drops OOD (chole-overfit guard). `per_bucket_report` already tags ID/OOD from the manifest.
- Every merged checkpoint **re-measured p99 on L40S** (deployment gate). Next experiment: `experiments/02-lora-sft/`.
- (No Engine B / reasoning-QA — those capability groups are not in the FRAME track.)

### Phase 3 — The data flywheel + targeted bucket flips  (Aug 5–19, wk4–5)
*Resolves: data-centric "flywheel" replaces the ladder's model-side rung-climb.*
- Read v1's per-bucket errors → mint QA into the weak/empty cells → v2 → retrain. Balance to floor
  (~≥3K/cell, **cross-procedure — never let chole dominate**, OOD is 50%). This loop compounds wins.
- Only for still-losing buckets: oversample weak cells; one `max_pixels` step (re-priced on the L40S curve,
  not A100); frame-position fix for temporal (use `[start,end]` window, not start-frame — rescues the n=1
  "end-of-procedure" 0→1).
- **32B FP8 wildcard ONLY** if a specific reasoning bucket is stuck after 8B AND the FP8 offline path is
  proven green on L40S. Margin-blind lift that flips no bucket = zero votes at real p99 risk.

### Phase 4 — Freeze + prove the lead survives  (Aug 19–Sep 1, wk6–7 · pre-eval closes Sep 1)
- Freeze the Copeland-selected checkpoint. Full **network-off dress rehearsal** on L40S.
- **Self-audit with the SDK's own bootstrap** (`evaluator.py:464-513`, resample videos→questions) + Wilcoxon
  vs **BOTH reproduced baselines**. Ship only a lead that survives: **≥6/10 buckets vs EACH baseline, every
  load-bearing winning vote CI-clear** (b=1000). A hair-thin 6/10 is a NO-GO — prefer 7/10 CI-clear.
- Submit before Sep 1 close; keep the floor as fallback.

### Phase 5 — Ship  (Sep 1–8)
- Final offline Docker + method description (RAG-Research, corpus-cited) + **release generated annotations
  CC BY-NC-SA 4.0** + open-source model. Final submission ≥ Sep 6 (2-day margin).

---

## What each angle contributed to the map
- **Aggressive** → decouple LoRA-v1 from the data-factory (train on existing labeled parquet, ~1 wk saved);
  spend headroom deliberately; probe 32B early — *tempered by* the L40S headroom correction.
- **Floor-first** → zero-shot floor submitted day-1 as a proven asset; LoRA gated behind OOD+leak trust,
  3 epochs first; NO-GO on any OOD-regressing checkpoint.
- **Copeland** → the 10-bucket geometry; 6/10 free-flip via prompt aimed at reasoning buckets; per-bucket-sign
  selection metric; silent-0 = rung ZERO; two-baseline seam.
- **Data-centric** → number/fo_class are a DATA problem (Engine A exact-match templates); Engine B is the only
  path to the 2 empty reasoning groups; error-driven data flywheel; DUAs are the critical path.
- **Deployment** → the L40S-vs-A100 headroom truth; the 6-gate submission CI; FP8-only-on-measured-need;
  vLLM conditional (batch=1 negates its edge); warm-up forward.

## ~~The single biggest un-owned risk~~ — RESOLVED 2026-07-13
Earlier flagged as "event_understanding + complex_reasoning unmeasured/untrained." **Verified against the
real FRAME parquets: those two groups are NOT in the FRAME track at all** (they're PROCEDURE/SEGMENT). So
there is nothing to mint for them — the risk was a cross-track illusion. FRAME's real surface is
**object_recognition + aggregation**, formats **fo_class + number**. The genuine remaining priority: the
cross-corpus Cholec80 pHash leak scrub (rung 03 follow-up) before training, and populating both scored
groups on the OOD (Sigmoid) side — which the organizer partition already does (val_ood covers both).
