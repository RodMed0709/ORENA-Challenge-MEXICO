# Domain Pitfalls — ORENA FOCUS · FRAME

**Domain:** Surgical VLM VQA challenge (grand-challenge, offline Docker, LLM-judge scoring)
**Researched:** 2026-07-09
**Confidence:** HIGH — grounded in the cloned SDK (`evaluator.py`, `judges.py`, `adversarial.py`) + CONSTITUTION §I/§IV.

Phases referenced map to the PLAN weekly roadmap: **P0** Eval-harness (wk0) · **P1** Zero-shot+prompt (wk1) · **P2** Data split (wk2) · **P3** Fine-tuning (wk2–5) · **P4** Output-format hardening (cross-cutting) · **P5** Docker/latency (wk7) · **P6** Submission strategy (ongoing).

---

## Critical Pitfalls (zero-score, disqualification, or rewrite)

### 1. Overfitting to cholecystectomy → OOD collapse
**What goes wrong:** 170/200 videos are LapChole. `pre_evaluation_score` (evaluator.py L402–462) is the *unweighted* mean over group×{ID,OOD} buckets — OOD is ~50% of the headline even though it is a data minority. A model that memorizes chole scores high ID and near-zero on every OOD bucket, dragging the mean down catastrophically.
**Warning signs:** acc-ID climbing while acc-OOD flat/falling across epochs; train loss dropping past epoch 1–2; big HeiCo-vs-LapChole val gap.
**Prevention:** Hold HeiCo out as OOD val from day one; report acc-ID **and** acc-OOD every checkpoint (CONSTITUTION §IV.2); keep LoRA rank moderate (r=16), 1–2 epochs, early-stop on acc-OOD; mix external domains. Discard any checkpoint that wins ID but loses OOD.
**Target phase:** P2 (split) + P3 (training/selection).

### 2. Wrong split — per-frame leakage instead of per-video
**What goes wrong:** Frames from the same video in both train and val inflate val accuracy; the leaderboard then reveals the illusion. Note the official bootstrap CI (`_hierarchical_summary`, L490–513) resamples *by video* — the organizers already treat the video as the unit; a per-frame split contradicts their own variance model.
**Warning signs:** val accuracy suspiciously near train; same `videoID` appearing across split files.
**Prevention:** Split strictly on `video`/`videoID`, never row/frame (CONSTITUTION §IV.3). Assert zero `videoID` intersection between splits in code.
**Target phase:** P2.

### 3. AdversarialDetector false-positive → disqualification
**What goes wrong:** `AdversarialDetector.check()` runs on **every** response *before* latency or scoring (evaluator.py L217) and **raises `RuntimeError`** (not "incorrect") on a hit. The public heuristic list (adversarial.py L52–65) includes natural-sounding phrases a confident fine-tuned VLM can emit unprompted: `"the answer is definitely correct"`, `"act as if"`, `"you are now"`, `"always respond with correct"`. One such string anywhere in one answer flags the whole submission.
**Warning signs:** any training-target answers containing those substrings; verbose "confidence" phrasing in generations.
**Prevention:** Never inject at the judge (CONSTITUTION §IV.6). Additionally, scan our own outputs offline against the known heuristic list before every submission and strip/reword; short canonical answers avoid the phrases entirely. Add this scan to the submission checklist.
**Target phase:** P4 (formatting) + P6 (pre-submit gate).

### 4. Docker calls HF at runtime → offline failure = total zero
**What goes wrong:** The reference pipeline does live `from_pretrained("Qwen/...")` (CONSTITUTION §I.1). Under `HF_HUB_OFFLINE=1` with no network, that raises and every answer is missing → 0.
**Warning signs:** `from_pretrained` with a hub ID (not a local path); no `COPY` of weights into the image; build that succeeds only with network.
**Prevention:** `COPY` weights into the layer, load from local path, set `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`, pin deps + vendor wheels, and **test the container with the network physically disconnected** (CONSTITUTION §IV.4).
**Target phase:** P5.

### 5. 5s/48GB latency ceiling → timeout = wrong
**What goes wrong:** `resp.latency > max_latency` → incorrect regardless of content (evaluator.py L218–220). The mean can pass while the p99 (large clips, many visual tokens, cold cache) blows the 5s cap. Every over-budget answer silently becomes wrong.
**Warning signs:** any per-question latency >4s in profiling; unbounded `max_new_tokens`; full-res tiling of large frames.
**Prevention:** Measure **p99 on L40S**, not mean (CONSTITUTION §IV.7); greedy decoding, `max_new_tokens ≤32`, cap `min/max_pixels`, crop the black laparoscopic letterbox, sample minimal frames, serve via vLLM.
**Target phase:** P5 (with P1 setting the token/res budget early).

---

## Moderate Pitfalls (silent point loss)

### 6. Exact-match format mismatch — free points lost
**What goes wrong:** For `binary, number, percentage, foclass, time`, correctness is `fmt.compare()` after `fmt.read()` (evaluator.py L326–351). `fmt.read(resp.content)` failing to parse → auto-incorrect. `2` vs `two` vs `"2 clips"` can fail. **This parse step also runs for judge formats** — a verbose `multiple_choice` answer that `read()` can't reduce to a choice returns False *before* ever reaching the judge.
**Prevention:** Learn each format's canonical form from the dataset; train targets to emit exactly that; unit-test our outputs through `fmt.read`/`fmt.compare` on a sample per format.
**Target phase:** P1 (EDA of formats) + P4.

### 7. Verbose / hedging answers penalized by the LLM judge
**What goes wrong:** Judge is `Qwen/Qwen3.5-4B`, majority vote, greedy, `max_new_tokens=8`, verdict = `"CORRECT" in raw and "INCORRECT" not in raw` (judges.py L143, L223). The prompt tolerates extra text *if* the core is right, but hedging ("maybe two, possibly three", disclaimers) muddies the core and reads as incomplete → INCORRECT.
**Prevention:** Train the canonical short style (PLAN §2.3): one clause, no uncertainty, no question echo, no disclaimers. Calibrate against a local replica of the same judge model for checkpoint selection.
**Target phase:** P4 (with P3 shaping training targets).

### 8. Burning scarce leaderboard submissions
**What goes wrong:** Blind submissions waste limited attempts and give no diagnostic signal.
**Prevention:** Submit only after beating the target in internal OOD validation (PLAN §4.5): Intento 0 = packaged zero-shot (validate Docker+pipeline), Intento 1 = balanced LoRA, Intento 2 = weakest-bucket fix. Each gated by local eval.
**Target phase:** P6.

---

## Minor / Team Pitfalls

### 9. Bus-factor-1 on Leonardo
**What goes wrong:** Fine-tuning/Docker knowledge concentrated in one paid contributor; illness/exit stalls the milestone.
**Prevention:** Version all scripts/configs/seeds from day 1; the PhD executor reproduces the full flow end-to-end at least once; keep the packaged zero-shot as an always-valid fallback (CONSTITUTION §IV.1/§IV.8).
**Target phase:** cross-cutting (enforce from P0).

### 10. Duplicate `qID` crashes the whole eval
**What goes wrong:** Two `Response` objects for one `qID` raises `ValueError` in `run()` (evaluator.py L190–191) — aborts scoring for the entire run, not one question.
**Prevention:** Emit exactly one `Response` per `qID`; dedupe/assert uniqueness before writing output.
**Target phase:** P0 (harness) + P6.

---

## Phase-Specific Warning Table

| Phase | Likely pitfall | Mitigation |
|-------|----------------|------------|
| P0 Harness | #10 dup qID; wrong metric replica | Uniqueness assert; reproduce `pre_evaluation_score` exactly |
| P1 Zero-shot | #6 format parse; #5 token budget | Round-trip through `fmt.read`; fix res/token caps now |
| P2 Split | #1 OOD collapse; #2 leakage | Video-level split; HeiCo=OOD; ID/OOD reporting |
| P3 Fine-tune | #1 overfit; #7 style | Early-stop on acc-OOD; canonical short targets |
| P4 Formatting | #3 DQ phrases; #6/#7 | Offline injection-scan; judge-replica calibration |
| P5 Docker/latency | #4 offline; #5 p99 | Network-off test; p99 on L40S |
| P6 Submission | #8 burned attempts; #3 gate | Validate-before-submit; DQ-scan gate |

## Sources
- `orena-focus` SDK (cloned): `evaluation/evaluator.py`, `judges.py`, `adversarial.py` — HIGH.
- CONSTITUTION.md §I, §IV; PLAN.md §5 — HIGH (project source of truth).
