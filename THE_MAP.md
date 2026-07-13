# THE MAP — unified FRAME strategy (meshed from 5 competing plans)

> Synthesis of 5 independent `vlm-strategist` plans (aggressive · floor-first · Copeland-ranking ·
> data-centric · deployment) against `ATTACK_LADDER.md` / `literature/FICHAS.md` / `context/EXPERIMENT_DESIGNS.md`.
> This is the north star. It **supersedes the sequencing** in ATTACK_LADDER.md (the rung recipes there
> stay valid). Dated 2026-07-13.

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
