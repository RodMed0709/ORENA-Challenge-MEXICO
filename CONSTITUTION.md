# PROJECT CONSTITUTION — ORENA FOCUS · FRAME Track

> **Source of truth for hard facts and non-negotiable rules.** Every spec, plan, or piece of code must respect this. Verified against the official repo `IMSY-DKFZ/orena-focus` (pip `orena-focus`, module `focus`) + the challenge PDFs. If anything here clashes with an assumption in another doc, **this constitution wins**.

---

## I. Verified facts from the official repo (do NOT make things up on top of this)

### I.1 Nature of the repo
- It is an **SDK/toolkit**, NOT a submission skeleton. Pip package: `orena-focus`. Import: `import focus`.
- It **does NOT ship** a Dockerfile, grand-challenge config, or submission template. Those come from the platform (challenge website), not from the repo.
- The reference pipeline downloads weights live from HF (`Qwen/Qwen3-VL-4B-Instruct`). For a real submission you must **package it offline** (see §IV).

### I.2 Submission contract (the interface we implement)
- There is no base `Model` class to inherit. The de facto contract (from `examples/inference.py`):
  ```python
  class <OurEngine>:
      def load(self): ...
      def predict(self, sample: VideoSample) -> str: ...
  ```
- **Input:** `sample.request.question` (str) + `sample.video_path` (temporary mp4 on disk) + `sample.fps`.
  - ⚠️ **The visual input is VIDEO, not a single image.** FRAME slices a short clip; we still receive a video path + fps. Our code samples frame(s) from the clip.
- **Output:** the raw model text, collected as:
  ```python
  Response(qID=req.qID, content=prediction, latency=latency)
  ```
  Fields: `qID: str`, `content: str` (raw VLM text), `latency: float` (seconds).

### I.3 Data schema (VERBATIM — real field names)
Raw HuggingFace row (parsed in `base_dataset._parse_row`):
```
id, video, timestamp_start, timestamp_end, procedure_type,
question, primary_capability, secondary_capabilities (list),
answer_format, answer, clinical_relevance (bool)
```
- Timestamps = strings `"hh:mm:ss"`.
- Parsed into two dataclasses:
  - **`Request`**: `qID:str, videoID:str, start_time:float, end_time:float, procedure_type:str, question:str`
  - **`Reference`**: `qID:str, primary:Capability, _format:str, answer:str, format_kwargs:dict, secondaries:tuple, ood:bool=False, clinical:bool`
- ❌ The fields `image`, `question_type`, and `metadata` do NOT exist. Do not assume them.

### I.4 Answer formats (8) — exact vs judge (VERIFIED in `formats.py`/`evaluator.py`)
`answer_format` ∈ `{binary, number, percentage, fo_class, open_ended, matching, multiple_choice, time}`. ⚠️ it is **`fo_class`** (not `foclass`).
- **EXACT match** (`fmt.compare()`): `binary, number, percentage, fo_class, time`.
  - `number`: **only `str.isdigit()`**. → `"two"`, `"2 clips"`, `"2.0"`, `"-1"` **all fail**. Pure numeric output.
  - `percentage`: `isclose(abs_tol=1e-9)` = exact (the `threshold_pp` param is defined but **not used**).
  - `fo_class`: **set-equality case-INSENSITIVE** against the valid names (`formats.py` uses `lower_map`). They are **10 config-driven classes** from `FOType.names()`: Sponge, Clip, Specimen Bag, Silicone Loop, External Drain, Needle, Gallstone, Specimen, Mesh, Absorbable Hemostatic Agent (NOT 8). Order, duplicates, and case do not matter. ⚠️ Never hardcode the list in a prompt/rule: keep it config-driven or a real `Mesh`/`Absorbable Hemostatic Agent` scores 0.
  - `time`: the number of timestamps must match, with pairwise tolerance of ~5s.
- **LLM-as-judge** (`JUDGE_FORMATS`): `open_ended, matching, multiple_choice`. The judge emits `CORRECT`/`INCORRECT`; verdict = `"CORRECT" in raw and "INCORRECT" not in raw`. Default judge: `TransformersJudge` with **`Qwen/Qwen3.5-4B`**, or `APIJudge`. Majority vote.
  - `matching` is gated by a `fullmatch` regex **before** the judge, even though it is judge-routed.

### I.4-bis SILENT gates that give you a 0 (read from the code — CRITICAL)
1. **Universal parse gate:** `fmt.read(prediction)` runs on **EVERY** response BEFORE the judge/scoring. If it raises `ValueError` → incorrect. Applies **also** to judge formats.
2. **>300 characters → auto-incorrect.** Brevity is NOT a preference, it is a hard gate.
3. **`AdversarialDetector.check()` raises `RuntimeError`** on phrases from its heuristic list → **disqualifies the whole submission**. And it **misfires on innocent phrases**: `"the answer is definitely correct"`, `"you are now"`, `"act as if"`, `"always respond with correct"`. → Scan our own outputs offline against that list before submitting.
4. **Duplicate `qID` → `ValueError` aborts the ENTIRE eval run** (not one question, the whole score). Guarantee unique qIDs.
5. **Latency kills at p99, not the mean.** Timeout = incorrect even if the content is correct.

### I.5 Metric
- Headline = **`pre_evaluation_score`**: **unweighted** mean over up to **10 buckets** = 5 capability groups × {in-distribution, out-of-distribution}. Flat per-question accuracy within each bucket.
- Output row: `summary_df` with `level="pre_evaluation", name="SCORE"`. Writes `results.csv`/`summary.csv` if `output_dir` is passed.
- Missing or timed-out answers = **incorrect**.
- `ood` is a `Reference` field but the public code leaves it at `False`; the private test split populates it.

### I.6 Base dependencies (min. pinned)
Python `>=3.10`. `datasets>=2.14, decord>=0.6, huggingface-hub>=0.17, opencv-python>=4.8, pandas>=2.0, numpy>=1.23, torch>=2.0, torchvision>=0.15, transformers>=4.30, tiktoken>=0.5, progiter, matplotlib, pillow, requests`. Inference extras: `qwen-vl-utils, accelerate`.

**Headless execution (dev, does NOT go into Docker):** `papermill` + **`ipykernel`** — the canonical way to run a long notebook in the background without exporting it to `.py` (see `EXPERIMENT_REPO_STRUCTURE_SPEC.md` §5b). 🔴 **`papermill` alone is not enough: it needs a registered kernel** (`python -m ipykernel install --user --name <env>`), or it fails with `NoSuchKernel`. Verified with papermill 2.7.0 on 2026-07-16.

**HARD pins for Qwen3-VL (verified):** `transformers==4.57.*` (below that the `qwen3_vl` arch does not load; do NOT jump to 5.x) + `qwen-vl-utils>=0.0.14`. Fine-tune: **ms-swift `>=4.2`** (native Qwen3-VL support, `--max_pixels`, `--freeze_vit/--freeze_aligner`). Serving: **vLLM `>=0.11`** (verified pair 0.11.2 + transformers 4.57). **Two separate Python envs** (train: ms-swift+bitsandbytes+flash-attn; serve: vLLM) — torch/flash-attn pins clash; the handoff artifact = already-merged LoRA weights. Quant: **bf16 LoRA** for the 8B (fits in 80GB, no NF4 loss); QLoRA NF4 only for the 32B wildcard; on the L40S (Ada CC 8.9) the lever is **FP8 w8a8** if p99 gets tight.

---

## II. Platform constraints (from the PDFs — not from the repo)

- **Evaluation hardware (THEY run it):** 1× NVIDIA **L40S 48GB**.
- **FRAME latency = 5.0 s/question** (`TRACK_MAX_LATENCY = {FRAME:5.0, SEGMENT:15.0, PROCEDURE:30.0}`). Going over = incorrect answer.
- **Docker OFFLINE**: no internet at inference. Weights and deps packaged in the container.
- **Submission** via grand-challenge (DKFZ clone).
- External public data + public pretrained models are allowed, **documenting** and **releasing** the extra annotations.
- **The final model must be released open-source** to be eligible for a prize.

---

## III. Objective and competitive red line

- **Floor (what matters):** beat **the 2 baselines** → **Nature BME** co-authorship. Baseline #1 = frontier zero-shot; **baseline #2 = Qwen3-VL-4B fine-tuned by them**.
- **Do not overfit to cholecystectomy.** OOD weighs the same as ID in the score. A model that shines ID and collapses OOD loses.
- **Be even across the 5 capabilities.** Unweighted mean over buckets: being terrible in one bucket sinks the overall score.

---

## IV. Hard engineering rules (non-negotiable)

1. **There is always a valid submission > 0.** The packaged zero-shot is the floor; never break it while chasing improvements.
2. **Every improvement is validated against OOD, not just ID.** Report acc-ID and acc-OOD at every checkpoint. Discard the one that wins ID but loses OOD.
3. **Split by `video`/`videoID`, NEVER by frame.** Frames from the same video = data leakage.
4. **Truly offline Docker:** `COPY` weights into the layer, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, pinned versions, vendored wheels. Test the container **with the network disconnected** before uploading.
5. **Canonical output format.** For exact-match formats, the answer must fit the dataset format. For judge formats: short, no hedging, no repeating the question, no disclaimers.
6. **NEVER prompt-inject the judge.** There is an `AdversarialDetector` → detection = immediate disqualification.
7. **Measure p99 latency on L40S**, not the mean, before freezing. Low `max_new_tokens`, greedy (no beam), cap on resolution/visual tokens.
8. **Reproducibility:** fixed seeds, versioned configs, `requirements` pinned from day 1. Bus factor > 1: anyone on the team must be able to run the pipeline.

---

## V. Collaborative work rules (git / sync)

Private repo: `RodMed0709/ORENA-Challenge-MEXICO`. Syncs **local + RunPod + the friend's machine**.

1. **`main` always functional.** Work on branches (`feat/`, `exp/`, `fix/`). Merge to `main` only what runs.
2. **`git pull` before starting, `push` when done.** Avoid divergence between the 3 machines.
3. **NEVER commit:** `.secrets.env`, tokens, raw data, weights, checkpoints, videos. (Already blocked in `.gitignore`.)
4. **Data and weights live in RunPod / HF**, not in git. The repo carries **code, specs, configs, docs**.
5. **Atomic commits** with a clear message. One experiment = one branch + its versioned config.
6. **Specs before code** (SDD): every feature starts from a spec in `specs/`, not from typing directly.

---

## VI. Team rules

- 3 Nature authors: you (lead), Leonardo (AI Scientist, Xantolo AI Lab), 3rd technical executor. Gilberto = advisor (acknowledgments).
- Only **Nature BME** exists as a publication. There are no LNCS proceedings or workshop.
- Leonardo's money and prize split: see `PLAN.md` §1 and `PROPUESTA-LEO.docx`.

## VII. Evidence and literature rules

1. **Corpus:** the literature lives in `literature/` — `INDEX.md` ranked by tiers (versioned), PDFs in `literature/pdfs/` (local, gitignored). ~30 papers indexed; the paywalled ones are provided by the user.
2. **Recommendations = with a citation.** When a technical/method recommendation is requested, do NOT opine off the cuff: search the **corpus + web**, cite what each piece of research did and whether it worked, and mark confidence. Engine: MCP **`RAG-Research`** (`verify_claim_against_corpus`, `tier_papers`, `build_bibliography`, `draft_methods`, `draft_section`, `export_docx`).
3. **Paper writing** is a project deliverable. Mandatory minimum: the submission's **method description** (`SUB-01`). Optional: our own method paper (authors at the lead's discretion), written with the RAG-Research MCP against the corpus.
4. **Traceability:** every claim that enters the paper must be verified against the corpus (`verify_claim_against_corpus`) before being included.

## VIII. Repo structure and experiment discipline (BINDING)

**The complete standard is `EXPERIMENT_REPO_STRUCTURE_SPEC.md` (repo root). It is mandatory for all experiment development.** Summary of the non-negotiables:

1. **The ONE rule:** **notebooks generate runs**; `.py` files are **importable libraries, NEVER launchers**. No `run_*.py`/`main.py` run by hand, no `.sh` chains, no notebook that does `subprocess` to a `.py`. Config goes **inline in the notebook** and calls `engine.main(cfg)`. The only "runnable" `.py`: `report.py` (imported from a cell). We work in **Jupyter**.
2. **A single canonical `src/`.** Today it is `src/frame/` — that is THE library, every experiment imports from there. A second first-party `src/` is forbidden. Experiment-specific glue → `experiments/<id>/_tools/` (folder-private).
3. **Two-part store:** `experiments/<id>/` (artifacts: notebooks `NN_<slug>.ipynb`, `_models/` engines only, `report.py`, `RESULTS.csv`, a README that **opens with the ladder**) + `context/<id>/CONTEXT.md` (curated context, outside the artifacts dir: Objective/Setup-config/Decisions/Results/Next).
4. **Single-variable A/B:** an experiment changes **exactly ONE thing** vs a named baseline; each lever is a flag **default OFF**; with everything OFF the run is byte-identical to the baseline.
5. **Honest, leak-guarded eval:** the eval split = the leak-guard split (by `video`, not by frame — see §IV.3); report the primary metric as **Δ vs baseline in the target's units** (challenge accuracy).
6. **build → smoke → review → full:** build the engine → smoke (`SMOKE` toggle, small pass) → **independent read-only review (GO/GO-WITH-FIXES/NO-GO with file:line)** BEFORE any full run → one clean full run. Faithful negatives (no improvement/regression) are valid results, recorded in the ladder, not re-rolled.
7. **Gitignored:** `experiments/*/runs/` (regenerable checkpoints/predictions/logs) and `external_data/` (datasets — verify md5 when available). `docs/` deliverables only.

## IX. Cleanup discipline (user rule — BINDING)

**Do not create files or folders haphazardly.** Consistency > convenience.

1. **Temporary files** (smoke tests, test scripts, scratch): use them, and **as soon as they are NOT needed, DELETE them**. Briefly log that it was deleted and why it is no longer needed.
2. Temporaries go to the **session scratchpad**, not the repo, unless they are an artifact of permanent value.
3. **Every new file/folder in the repo is justified** against the §VIII structure. If it does not fit, it is not created.
4. **Progressive** cleanup: do not let junk pile up "for the end" — clean as things are freed up.
5. `_models/` = engines only; delete the moment `smoke_*.py`, `run_*.py`, `.sh` chains, `resume_*.py` show up.

> **Reconciliation note with the current GSD plans:** Phases 1-3 defined `scripts/run_baseline.py`, `scripts/evaluate.py`, `scripts/export_jsonl.py`. When **executing** the experiments layer, those launchers become **notebook cells** (`experiments/<id>/NN_<slug>.ipynb` importing engines from `src/frame`), NOT run by hand. `src/frame` (the library + Phase 1 pytest tests) stays as the only `src/`. Adjust the plans in execution to respect §VIII.

---
*Constitution v1 — grounded in the official repo + PDFs + literature corpus + EXPERIMENT_REPO_STRUCTURE_SPEC. Updated if the repo changes or the pre-evaluation opens (Jul 15).*
