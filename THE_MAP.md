# THE MAP — unified FRAME strategy (meshed from 5 competing plans)

> Synthesis of 5 independent `vlm-strategist` plans (aggressive · floor-first · Copeland-ranking ·
> data-centric · deployment) against `ATTACK_LADDER.md` / `literature/FICHAS.md` / `context/EXPERIMENT_DESIGNS.md`.
> This is the north star. It **supersedes the sequencing** in ATTACK_LADDER.md (the rung recipes there
> stay valid). Dated 2026-07-13.

## Three reframes the whole panel forced (the ladder was wrong on these)

1. **We are scored on 10 buckets, and we measured only 3.** Score = unweighted mean over **5 capability
   groups × {ID,OOD}** (`taxonomy.py:38-70`): object_recognition · temporal_grounding · aggregation ·
   **event_understanding** · **complex_reasoning**. Rung-00 populated **3/10** (all ID). The two reasoning
   groups (4 buckets) have **zero training signal**; the entire OOD half (5 buckets) is untested. The ladder
   optimizes weak *formats* (fo_class/number) inside 3 ID buckets — 30% of the Copeland vote, blind to 70%.
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

### Phase 1 — The 10-bucket harness + L40S truth  (Jul 15–22, wk1)  ← THE gate for every later claim
*Resolves: Copeland+data-centric "populate all 10 buckets" + deployment "re-baseline on L40S" + floor-first "leak scrub".*
- **Rung 03 EXPANDED:** leak-guarded OOD split (hold out a WHOLE procedure_type, split by `(dataset,video_id)`)
  **+ Cholec80 cross-corpus scrub** (videoID-intersect + pHash frames vs challenge `test.parquet`) **+ build a
  dev eval that POPULATES all 10 buckets** — especially event_understanding, complex_reasoning, and every OOD
  half. Recompute the zero-shot arm on this split. *You cannot flip a bucket you cannot see.*
- **Re-baseline p99 on the REAL L40S** (rent on RunPod) → build the budget curve every later lever is priced
  against. Add a warm-up forward in `load()` (move CUDA-graph capture out of the timed path).
- **Engine A online** — deterministic SSG scene-graph templates over CholecT50 triplets + HeiCo masks →
  exact-match counting/existence/fo_class QA (judge-free, gold-derived). Directly attacks number 0.127 /
  fo_class 0.182 — **these are a DATA problem, not a perception wall** (data-centric rebuts the ladder here).

### Phase 2 — Reasoning data + LoRA v1  (Jul 22–Aug 5, wk2–3)
*Resolves: aggressive "decouple LoRA from the factory" + data-centric "mint the reasoning groups first".*
- **Engine B** — LLaVA-Surg two-stage extract→reason→QA with a **local** LLM (ρ=0.94) → the ONLY source of
  Group-4/5 (event/complex-reasoning) + open_ended QA. GP-VLS diversification on open_ended only. Judge-mirror
  gate ≥0.9 per batch; drop failures, don't fix.
- **LoRA v1 (S2Can recipe, bf16, 3 epochs first)** on existing labeled parquet + Engine-A data — **decoupled
  from the full factory** (aggressive's win: LoRA v1 needs no synthetic-reasoning data to start). Select by
  **per-bucket Copeland margin** on the 10-bucket dev, not mean-OOD.
- Every merged checkpoint **re-enters the Docker, re-measured p99 on L40S** (deployment gate).

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

## The single biggest un-owned risk (flagged by 2 independent plans)
**event_understanding + complex_reasoning are entirely unmeasured and untrained.** No plan can claim a
Copeland lead until Phase-1's dev split populates those 2 groups + all OOD halves, and Phase-2's Engine B
mints training data for them. This is the highest-leverage gap and the map's first structural priority after
the floor ships.
