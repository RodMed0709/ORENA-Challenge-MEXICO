# Experiment Design Specs — distilled from the design-agent pass (2026-07-13)

> Per-rung engineering specs produced by parallel `vlm-specialist` agents, grounded file:line in our
> actual code + the vendored SDK. Companion to `ATTACK_LADDER.md` (the plan) and `literature/FICHAS.md`
> (the evidence). Next session: split these into proper `experiments/<id>/` + `context/<id>/CONTEXT.md`.

## ⚠️ Ground-truth code corrections (override earlier assumptions)

1. **`fo_class` is CASE-INSENSITIVE.** `vendor/orena-focus/.../formats.py:179-194` builds `lower_map`
   and canonicalizes on read — Title-Case is NOT required. The real gate: each comma-separated token
   must be a **known class name** (or `none`), else `verify()` raises → auto-wrong. The lever is
   **suppressing non-class tokens** (LigaSure, "instrument", "clip applier"), not casing.
2. **The FO registry has 10 classes, not 8.** `FOType.names()` = the 8 lapchole/heico subset **+ Mesh
   + Absorbable Hemostatic Agent**; header warns more may arrive via metadata. Keep the class list
   **config-driven**, never hard-coded, or a real Mesh answer scores 0.
3. **AdversarialDetector scans the model OUTPUT, not the prompt** (`adversarial.py:26,49`), runs
   **inside the eval future and aborts the ENTIRE run** on a hit (not one question). Public list = 12
   phrases incl. innocent misfires ("the answer is definitely correct", "you are now", "act as if").
   Real detector is private/swapped-in → be over-conservative. Mandatory offline pre-submission scan.
4. **`number` gate** = `text.strip().isdigit()` → bare non-negative int only (no sign/decimal/units).
5. **Judge** = `TransformersJudge` default `Qwen/Qwen3.5-4B` (**dead HF id** → we use `Qwen/Qwen3-4B`),
   greedy, `max_new_tokens=8`, `enable_thinking=False`, verdict = `"CORRECT" in raw and "INCORRECT"
   not in raw`, majority_vote.

---

## Rung 01 — Prompt / FO-taxonomy grounding  (cheap, zero-train)
- **ONE variable:** `grounded_prompt: bool = False` in engine cfg; OFF → returns unchanged
  `SYSTEM_PROMPT` object → byte-identical to rung 00.
- **Where:** add `GROUNDED_SYSTEM_PROMPT` in `src/frame/engine.py` next to `SYSTEM_PROMPT`; select in
  `_messages` (`engine.py:53-55`). Baseline already appends `FO_DEFINITIONS` (`engine.py:20-27`), so the
  variable is: enumerated class list + **named-instrument exclusion scoped to FO questions** + per-format
  output contract (digits-only / class-names / <300 char / no hedging).
- **Clip-applier rule:** a clip counts only once in the abdomen, not while loaded in the applier
  (`foreign_objects.py:196-203` — the baseline miscalls this).
- **Adversarial-clean:** verified draft trips 0/12 signals ("You are an expert" ≠ "you are now"; avoided
  "act as", "respond with correct", "definitely").
- **Expected:** fo_class (0.182) biggest gain, object_recognition secondary, **number marginal**
  (miscount is perception, prompt can't fix — faithful ~0 delta is valid). Watch binary/open_ended/MC for
  regression from over-terse contract.

## Rung 02 — Answer canonicalizer + offline judge-mirror + adversarial guard  (zero-train insurance)
- **`src/frame/canonicalize.py`** — one entry `canonicalize(raw, answer_format, *, enabled=False, char_cap=300)`,
  routes per format; each branch ends by calling the **vendored** `get_format_class(fmt).verify()`
  (gate = source of truth, no drift). number→first int token; fo_class→alias-resolve to `FOType.names()`,
  **drop instrument tokens**, `None` sentinel; open_ended/MC→strip `<think>`/markdown/hedging, front-load
  the gold-matching term (reference-guided grading), truncate ≤300 at sentence boundary. **Never fabricate**
  a digit/answer — on failure return raw stripped so the gate fails honestly. Default OFF = passthrough.
  Wire at `engine.py:94` (replaces blunt `out[:char_cap]`); needs `answer_format` passed into `predict`.
- **`experiments/<id>/_tools/judge_mirror.py`** — reuse `focus.evaluation.judges.TransformersJudge`
  (don't re-implement). Judge is greedy/deterministic → vary the **judge model set**
  (`Qwen/Qwen3-4B` ± `Qwen/Qwen2.5-3B-Instruct`) and report a **band** `[all-agree-CORRECT, any-CORRECT]`.
  A/B two phrasings pre-submission, keep the one maximizing the **lower** bound.
- **`src/frame/adversarial_guard.py`** — superset of the 12 vendored phrases + paraphrase families;
  `assert_clean(predictions)` is a **hard pre-submission gate** (one hit = disqualification → block);
  `neutralize()` falls back to bare extractive answer, then re-scans.
- **Harness:** EXTRACT `pre_evaluation_score` from the single official `Evaluator.run` (`run.py:131-137`,
  `summary.csv` row `pre_evaluation/SCORE`) — the guard/mirror are read-only linters over saved artifacts,
  never a rival scoring pass.

## Rung 03 — train / validation split + leak-guard  (harness gate for all LoRA claims)
> **Corrected 2026-07-13 (with Leo): train + validation ONLY, no local test.** The real test is the
> leaderboard, so a reserved local test only wastes trainable data. Hold out ONE procedure_type as the
> OOD validation slice (NOT all of HeiCo) so the other surgery types stay in train — the model must learn
> cross-procedure since OOD is half the score. For the FINAL submission model, fold validation back into
> train and retrain on all. (Supersedes the "HeiCo held out whole" assumption in `.planning/phases/03-*`.)
- **Split key = `(dataset, video_id)` tuple, never a row/frame** (heico & lapchole id ranges overlap →
  qID prefixed `ds__` at `data.py:43`; `frame_index = round(start*base_fps)` at `data.py:91` is why
  row-splitting leaks). Two validation slices: `val_id` (held-out videos of trained procedure_types) +
  `val_ood` (one **whole held-out procedure_type**, HeiCo Stage-3 style; keep lapchole + the other heico
  procedures in train).
- **`src/frame/split.py`** (built): `SplitConfig`, `build_split` (seeded `default_rng(42)`),
  `write_manifest`/`load_manifest` (CSV + `.sha256` sidecar = source of truth, verified reload — never
  re-derive), `apply_split`, `assert_no_leak`, `assert_all_matched`, `per_bucket_report`. Marking
  `ref.ood=True` for `val_ood` at EVAL time is deferred to the eval experiment (rung 05). 8 unit tests green.
- **Report:** per capability_group × {ID,OOD} acc with `n` printed; **headline the worst bucket**, refuse
  to headline any bucket with `n<~10` (baseline n=1 temporal swung macro-mean ⅓). Optimize worst-case
  tail, not mean. Discard any checkpoint that wins ID but drops any OOD bucket.
- **🔴 BIGGEST LEAK TRAP (cross-corpus):** our lapchole train (~170 Cholec80-family videos) is public and
  the organizers' hidden lapchole test is likely the **same Cholec80/CholecT lineage** → real-leaderboard
  leak that internal video-disjoint splitting can't catch. **Guard:** (a) intersect train `(dataset,
  video_id)` vs challenge `test.parquet` videoIDs, drop matches; (b) **pHash** sampled frames per video
  (via `FrameProvider.get_frame` `data.py:127`), drop fingerprint collisions; record hashes in manifest.

## Rung 05 — PRIMARY: LoRA instruction-tune Qwen3-VL-8B (ms-swift)  [promoted per Leo; see note]
> The strategist ladder numbers this 05; the LoRA design agent wrote it as "rung 01" under Leo's earlier
> "LoRA is primary" framing. **Canonical = ATTACK_LADDER rung 05**; the cheap prompt/judge/split rungs
> (01-04) land first as its prerequisites. Recipe + integrity rules below are authoritative either way.
- **ONE variable:** `LoRA ON`. Frame/prompt/judge/gen/`max_pixels`/`max_new_tokens=64` all held identical
  to rung 00. **Recompute the zero-shot arm on the held-out split** (can't reuse 0.174 — different
  denominator now that train videos are excluded). Δ = FT − zeroshot, both on held-out. `mark_heico_ood=True`.
- **Recipe (S2Can):** `swift sft --train_type lora --torch_dtype bfloat16 --freeze_vit true
  --freeze_aligner true --lora_rank 8 --lora_alpha 32 --lora_dropout 0.1 --target_modules all-linear
  --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.03 --num_train_epochs 5
  --save_strategy epoch --per_device_train_batch_size 2 --gradient_accumulation_steps 8
  --gradient_checkpointing true --attn_impl flash_attn --seed 42`, `MAX_PIXELS=921600` (==`config.py:32`,
  train tokens == serve tokens). **Stay plain** (Surgical-LVLM: IFT +16, exotic +2 → no DoRA/vision-unfreeze
  in the primary rung). bf16 not NF4 (fits one 80GB dev GPU). 8B not 4B (beat org's FT-4B with FT-8B).
- **Data:** ShareGPT multimodal JSONL; `system` = `engine.SYSTEM_PROMPT` **verbatim** (reject Phase-3's
  different string — it confounds the variable); assistant = `reference.answer` verbatim; frames via the
  **same** `FrameProvider.get_frame` path (pixel parity). Carve the labeled parquet by video (train = a
  subset of lapchole videos; heico never in train = OOD proxy).
- **Engine wrapper (spec-compliant):** `experiments/<id>/_models/lora_sft_train.py` exposes `main(cfg,
  stage=...)` calling ms-swift Python API `from swift.llm import sft_main`; notebook imports it. Stages:
  build_split → export → train (headless `nbconvert` for detached full run) → merge (`swift export
  --merge_lora`) → eval (two arms on held-out).
- **#1 failure = OOD collapse (chole overfit).** Guard: OOD arm baked into eval; **select checkpoint by
  held-out OOD, never ID**; NO-GO / faithful-negative if FT doesn't beat zero-shot on OOD; if OOD drops,
  next move is *fewer* epochs / lower rank, not more.
- **Sibling rungs:** 01a lr=1e-4 · 01b rank=16 · 01c epoch sweep · 01d vision-unfreeze · 01e balanced
  oversampling (rung 06 lever). Merge → validate vLLM greedy == HF greedy on 50 items + **p99 on L40S**.

## Rung 08 — Question-conditioned frame selection  (MED, after-LoRA adjunct)
- Current sampler `data.py:91` takes 1 frame at **start** timestamp, **ignores `end_time`** — collapses
  the `[start,end]` window to its first frame (the real weakness).
- **Cheaper probe FIRST:** A/B a heuristic timestamp fix (use the `[start,end]` window / mid/end), zero
  new weights — isolates whether frame *position* even matters. Only if it lifts, promote to SigLIP.
- **SigLIP rung:** `google/siglip-so400m-patch14-384` (Apache-2.0, ~0.86 GB), sample `N_cand=12` in
  `[start,end]`, one batched encode, cosine-sim vs question, top-`k=1` (keep k=1 vs uniform-k=1 as the
  single variable; 1→3 frames is a SEPARATE rung). Added latency ~50-100 ms → new p99 ~0.7 s (fits 8×
  headroom). Bundle offline (`COPY`, `HF_HUB_OFFLINE=1`), +1 GB VRAM, load once. Moves temporal_grounding
  (+ the n=1 "end-of-procedure" 0→1 flip); will NOT move number/fo_class (perception = LoRA's job).
- **Honest call:** after-LoRA adjunct, not a pre-FT lever — baseline errors are perception, a better
  frame doesn't fix "genuinely wrong."

## Notion command-center (second brain)
Full ready-to-paste markdown drafted (mission / hard constraints / rung-00 status / attack ladder /
literature levers / guardrails / team+workflow / next actions) — leads every `0.174` mention with the
robust signals per the baseline trust-caveats. Source saved for pasting/creation into Notion next.
