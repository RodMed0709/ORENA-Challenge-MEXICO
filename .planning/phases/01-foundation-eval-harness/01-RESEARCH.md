# Phase 1: Foundation & Eval Harness - Research

**Researched:** 2026-07-09
**Domain:** Reproducible dev foundation + a GPU-free local scorer that reproduces the `orena-focus` SDK's `pre_evaluation_score` (FRAME track)
**Confidence:** HIGH — grounded in the cloned SDK source at git tag **v0.3.4** (`9b32641`), read line-by-line; external facts (PyPI, judge model) web-verified.

## Summary

Phase 1 builds no model and touches no GPU. It stands up (a) a pinned, reproducible environment that installs `orena-focus` and runs its example eval path end-to-end, (b) a 3-machine private-repo sync + experiment-tracking convention, (c) SDK-native data loading + an EDA report, and (d) a thin **eval harness** that scores a precomputed `responses.json` and reproduces the SDK headline metric.

The single most important discovery: **the harness is a thin wrapper, not a reimplementation.** `focus.Evaluator().run(requests, references, responses, track=Track.FRAME)` already computes `pre_evaluation_score`, already returns the per-`group × {ID,OOD}` bucket breakdown (EVAL-03), already raises `ValueError` on duplicate `qID` (EVAL-03 abort), and already enforces the 5.0 s FRAME latency gate. Our code extracts the `level="pre_evaluation", name="SCORE"` row and the `buckets_df`, adds a friendly pre-check + CSV/summary artifact, and — crucially — makes the LLM judge **swappable** so the harness runs GPU-free.

The one real design decision is the judge. To score with **no model loaded** (EVAL-02), pass `Evaluator(judges=[])`: every judge-routed format (`open_ended`, `matching`, `multiple_choice`) is forced INCORRECT. That does **not** reproduce the true score on judge formats, so the harness needs three modes: (1) `judges=[]` exact-match-only (pure CI, GPU-free); (2) an injectable `MockJudge` for testing the judge-routing path deterministically GPU-free; (3) the real `TransformersJudge('Qwen/Qwen3.5-4B')` for full reproduction (CPU-capable but slow; realistically run on the GPU box). `pre_evaluation_score` itself is pure arithmetic — reproduction is **exact (tolerance 0)**, and the bootstrap CIs are seeded (`seed=42`) hence deterministic too.

**Primary recommendation:** Wrap `focus.Evaluator`; never touch scoring math. Build the harness against a hand-written stub `responses.json` and a synthetic reference fixture that includes `ood=True` rows (public data is 100% `ood=False`, so the OOD bucket path is otherwise untestable). Pin `transformers==4.57.*` even for this GPU-free env because the SDK judge needs the Qwen3 chat template.

## User Constraints

No `CONTEXT.md` exists for this phase (no `/gsd:discuss-phase` was run). There are therefore no user-locked decisions, discretion areas, or deferred ideas to copy verbatim. The authoritative constraints for this phase come from `CONSTITUTION.md` and the project `CLAUDE.md` (see "Project Constraints" below) and must be treated with the same authority as locked decisions.

## Project Constraints (from CLAUDE.md + CONSTITUTION.md)

These are hard, non-negotiable directives the plan MUST honor:

- **All repo content in English.** Specs before code (SDD). (`CLAUDE.md` Process)
- **NEVER add Claude as a git contributor** — no `Co-Authored-By` trailers, no Claude in commit authorship. (`CLAUDE.md` Process) — overrides the global CLAUDE.md default.
- **Never commit** secrets (`.secrets.env`, tokens), raw data, videos, frames, weights, checkpoints, or temp clips. Data + weights live on RunPod/HF, not git. (`CONSTITUTION.md` §V.3/§V.4)
- **`main` always runnable.** Work on branches (`feat/`, `exp/`, `fix/`); `git pull` before, `push` after; one experiment = one branch + one versioned config. (`CONSTITUTION.md` §V.1/§V.2/§V.5)
- **Reproducibility from day 1:** fixed seeds, versioned configs, pinned requirements; bus factor > 1 — any teammate can run the pipeline. (`CONSTITUTION.md` §IV.8)
- **Do NOT reinvent SDK I/O** (loader, evaluator, frame extraction, judge, scoring). Reusing `focus` prevents silent drift from the official score. (`REQUIREMENTS.md` Out of Scope; `ARCHITECTURE.md`)
- **Two separate Python envs** (train: ms-swift; serve: vLLM). Do not merge them — torch/flash-attn pins conflict. (`STACK.md`, `CONSTITUTION.md` §I.6)
- **`transformers==4.57.*`** hard floor for Qwen3-VL and the Qwen3 judge; never jump to 5.x. Python `>=3.10,<3.13`. (`STACK.md`)
- **GPU-free phase:** the deliverable must score a precomputed `responses.json` with no model loaded. (Phase objective, EVAL-02)

<phase_requirements>
## Phase Requirements

| ID | Description (from REQUIREMENTS.md) | Research Support (what enables it) |
|----|-----------------------------------|-------------------------------------|
| **FND-01** | Install `orena-focus` SDK and reproduce its example inference/eval path end-to-end on one machine | Cloned source = tag **v0.3.4** (PyPI shows only 0.1.1 → pin to v0.3.4 via git/tag). `examples/evaluation.py` is the GPU-free eval path; `examples/data_preparation.py` + `examples/inference.py` are the full path. Install extras `qwen-vl-utils transformers accelerate`. |
| **FND-02** | Private repo synced across local + RunPod + teammate; secrets/data never committed | Repo `RodMed0709/ORENA-Challenge-MEXICO`. `.gitignore` must block `.secrets.env`, `data/`, `*.mp4`, frames, `*.safetensors`/checkpoints, `/tmp` clips, HF cache. `FOCUS_ROOT_DIR` env var points each machine at its own data root (out of git). |
| **FND-03** | Pinned two-env dependency setup reproducible from a lockfile | Env A (train): `ms-swift>=4.2 transformers==4.57.* torch>=2.5 qwen-vl-utils>=0.0.14 accelerate>=1.0 bitsandbytes>=0.45 deepspeed peft flash-attn`. Env B (serve): `vllm>=0.11 transformers==4.57.* qwen-vl-utils>=0.0.14 orena-focus`. Env C (harness/EDA, this phase): minimal — `orena-focus==0.3.4 transformers==4.57.* pandas numpy datasets`. Use uv/pip-tools lockfiles per env. |
| **FND-04** | Experiment-tracking convention usable by all 3 members | `experiments/runs/<id>/{config.yaml, summary.csv, checkpoint_uri.txt}` committed (tiny); weights on HF. One branch + one versioned YAML per experiment. Config carries seed, model_id, split hash, sampling policy. |
| **DATA-01** | Load `heico-focus-vqa` + `lapchole-focus-vqa` via SDK loader into `Request`/`Reference` | `FocusDataset("heico"\|"lapchole", DatasetSplit.TEST, Track.FRAME)`. Repo IDs: `orena-dkfz/{heico,lapchole}-focus-vqa`. Annotations stream from HF — **no root_dir, no videos, no GPU needed** to get `.requests` / `.references`. |
| **DATA-03** | EDA report: answer_format dist, capability-group dist, per-format canonical answer shapes | Iterate `dataset.references`: `_format`, `primary.group.value`, and `answer` string. 8 formats, 5 groups (15 leaves). Report canonical shapes by running representative `answer`s through `fmt.read`. Commit as markdown + CSV. |
| **EVAL-01** | Local harness wrapping `Evaluator().run(..., track=Track.FRAME)` that reproduces `pre_evaluation_score` | `run()` appends a `level="pre_evaluation", name="SCORE"` row to `summary_df`; also `Evaluator().pre_evaluation_score(results_df) -> (score, buckets_df)`. Reproduction is EXACT (pure means; no tolerance needed). |
| **EVAL-02** | Harness scores a precomputed `responses.json` with no model loaded | `load_responses(path)` → `list[Response]`; feed to `run()`. **`Evaluator(judges=[])`** = no model at all (judge formats forced INCORRECT). `responses.json` schema: `[{"qID","content","latency"}]`. |
| **EVAL-03** | Report accuracy per capability-group × {ID,OOD} bucket (not just headline); abort on duplicate qID | `pre_evaluation_score()` returns `buckets_df` with `group,ood,accuracy,count`. Duplicate `qID` already raises `ValueError` inside `run()`. Public data is all `ood=False` → build a synthetic `ood=True` fixture to exercise/verify the OOD path. |
| **EVAL-04** | Local LLM judge (`Qwen/Qwen3.5-4B` via `TransformersJudge`) wired for judge-routed formats | `Evaluator(judges=[TransformersJudge(model_name="Qwen/Qwen3.5-4B", device=...)])` or `Evaluator(judge_kwargs={"device": ...})`. `device="cpu"` works (GPU-free) but 4B on CPU is slow → keep judge optional/mockable for local runs; run real judge on the GPU box. |
</phase_requirements>

## Standard Stack

### Core (this phase's env — "Env C" harness/EDA)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `orena-focus` | **==0.3.4** (git tag `v0.3.4`) | Data loader, dataclasses, Evaluator, judge, formats | The official SDK — the only source of the true metric. PyPI currently publishes 0.1.1; **the cloned source we must match is 0.3.4**, so pin to the git tag or the matching PyPI release when it appears. |
| `transformers` | **==4.57.*** | Qwen3 chat template for the judge; Qwen3-VL arch downstream | Hard floor for `Qwen/Qwen3.5-4B` judge (`enable_thinking=False` kwarg) and Qwen3-VL; SDK only requires `>=4.30` but that is too low for us. |
| `datasets` | >=2.14 | Streams HF annotation parquet inside `FocusDataset` | SDK dependency; the loader calls `load_dataset`. |
| `pandas` | >=2.0 | `results_df` / `summary_df` handling | SDK returns pandas DataFrames. |
| `numpy` | >=1.23 | Bootstrap CI math inside SDK | SDK dependency. |
| Python | >=3.10,<3.13 | Runtime | SDK requires ≥3.10; vLLM/ms-swift envs cap <3.13. |

### Supporting (needed only when the full example path / real judge runs)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `qwen-vl-utils` | >=0.0.14 | `process_vision_info` for the inference example | FND-01 full path only (not needed for GPU-free scoring). |
| `accelerate` | >=1.0 | `device_map` for non-CPU judge/model | Only when judge/model runs on GPU. |
| `decord` / `opencv-python` | >=0.6 / >=4.8 | Video clip + frame extraction | Only when downloading videos / building clips (Phase 2). |
| `huggingface-hub` | >=0.17 | `snapshot_download` of videos | Only when downloading data. |

### Two-env lockfiles (FND-03) — set up now, exercised later
| Env | Key pins | Phase 1 obligation |
|-----|----------|--------------------|
| **Train (A)** | `ms-swift>=4.2 transformers==4.57.* torch>=2.5 qwen-vl-utils>=0.0.14 accelerate>=1.0 bitsandbytes>=0.45 deepspeed>=0.15 flash-attn>=2.6 peft>=0.13` | Prove it resolves + installs from a lockfile on RunPod. |
| **Serve (B)** | `vllm>=0.11 (target 0.11.2) transformers==4.57.* qwen-vl-utils>=0.0.14 orena-focus decord opencv-python` | Prove it resolves + installs from a lockfile. |
| **Harness (C)** | `orena-focus==0.3.4 transformers==4.57.* datasets pandas numpy` (+ `accelerate` if judge on GPU) | The env the Phase 1 harness/EDA actually run in. |

**Installation (Env C, this phase):**
```bash
# pin to the exact cloned source revision
pip install "git+https://github.com/IMSY-DKFZ/orena-focus.git@v0.3.4"
pip install "transformers==4.57.*" datasets pandas numpy
# add for the real judge on GPU: pip install accelerate
```

**Version verification (do at plan/execute time — training data may be stale):**
```bash
pip index versions orena-focus        # confirm 0.3.4 is on PyPI, else install from git tag
python -c "import transformers, focus; print(transformers.__version__, focus.__version__)"
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('Qwen/Qwen3.5-4B').id)"  # judge exists (verified 2026-07)
```

## Architecture Patterns

### Recommended module structure (this phase)
```
src/frame/
├── data/
│   └── loading.py       # thin: build_dataset(name, track=FRAME) -> FocusDataset
├── eval/
│   ├── harness.py       # wrap focus.Evaluator; extract SCORE + buckets_df; CSV artifact
│   └── judges.py        # MockJudge (subclass focus…Judge) for GPU-free judge-path tests
scripts/
├── run_eda.py           # DATA-03 report generator
└── evaluate.py          # CLI: --responses responses.json --dataset heico [--no-judge]
configs/                 # experiment YAMLs (seed, dataset, judge mode)
experiments/runs/<id>/   # committed config.yaml + summary.csv
tests/
└── test_harness.py      # golden pre_evaluation_score, dup-qID abort, ID/OOD buckets
```

### Pattern 1: Thin harness over `Evaluator`
**What:** Call the SDK; extract the two things we report.
**When:** EVAL-01/02/03.
```python
# Source: focus/examples/evaluation.py + evaluation/evaluator.py (v0.3.4)
from focus import FocusDataset, DatasetSplit, Track, Evaluator, load_responses

def score(dataset_name: str, responses_path: str, judges=None):
    ds = FocusDataset(dataset_name, DatasetSplit.TEST, Track.FRAME)  # annotations stream from HF
    responses = load_responses(responses_path)                       # [{qID,content,latency}]
    ev = Evaluator(judges=judges)          # judges=[] => GPU-free, judge formats forced INCORRECT
    results_df, summary_df = ev.run(
        requests=ds.requests, references=ds.references, responses=responses,
        track=Track.FRAME,                 # enforces 5.0s FRAME latency gate
    )
    pre = summary_df.loc[summary_df["level"] == "pre_evaluation", "accuracy"].iloc[0]
    _, buckets_df = ev.pre_evaluation_score(results_df)   # group × {ID,OOD} breakdown (EVAL-03)
    return pre, buckets_df, results_df, summary_df
```

### Pattern 2: Swappable judge for GPU-free reproduction of the judge path
**What:** A deterministic in-memory judge so the judge-routing branch is testable with no model.
**When:** EVAL-04 wiring + tests; lets the harness validate judge-format handling GPU-free.
```python
# Source: focus/evaluation/judges.py — Judge ABC has one method: judge(request, reference, candidate) -> bool
from focus.evaluation.judges import Judge

class MockJudge(Judge):
    def __init__(self, verdict=True): self._v = verdict
    def judge(self, request, reference, candidate) -> bool:
        return self._v                      # or exact-match heuristic for fixtures

# GPU-free judge-path test:  Evaluator(judges=[MockJudge(True)])
# Real reproduction (GPU box): Evaluator(judges=[TransformersJudge("Qwen/Qwen3.5-4B", device="cuda:0")])
# No model at all:             Evaluator(judges=[])   # judge formats -> INCORRECT
```
Note: `majority_vote([], ...)` returns `False` (n=0, no vote reaches threshold) — that is exactly why `judges=[]` forces judge formats incorrect.

### Pattern 3: EDA straight off references (no data download)
```python
# Source: focus/data/base_dataset.py — FocusDataset needs no root_dir/videos for annotations
from collections import Counter
from focus import FocusDataset, DatasetSplit, Track
ds = FocusDataset("heico", DatasetSplit.ALL, Track.FRAME)
fmt_dist   = Counter(r._format for r in ds.references)
group_dist = Counter(r.primary.group.value for r in ds.references)
# canonical shape per format: sample answers, run through ref.format.read(ref.answer)
```

### Anti-Patterns to Avoid
- **Reimplementing scoring / bucketing / judging.** Any divergence silently lies about the leaderboard number. Extract from `summary_df`; never recompute means.
- **Assuming a tolerance on `pre_evaluation_score`.** It is pure `mean()` over buckets — reproduction must be **exact**. Bootstrap CIs are seeded (`seed=42`) → also deterministic. A golden test should assert equality, not `approx` (except for float repr).
- **Testing OOD buckets on public data.** Public `Reference.ood` is always `False` (populated only by the private test split). Without a synthetic `ood=True` fixture, the OOD branch of EVAL-03 is never exercised.
- **Constructing `FocusConfig` for EDA/scoring.** `FocusConfig.__init__` raises if `root_dir` doesn't exist. Loading annotations + scoring need no config at all — only video/frame steps do.
- **Committing the `responses.json` stub with real data / the temp `/tmp` clips.** Keep stubs tiny + synthetic; `.gitignore` the data root.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parse HF rows → objects | Custom parser | `FocusDataset` (`_parse_row`) | Handles `id→qID`, `hh:mm:ss→seconds`, capability code resolution, `time` `threshold_seconds` auto-calc. |
| Score answers | Custom compare logic | `ref.format.read()` + `.compare()` via `Evaluator` | 8 formats, each with exact gates (`number`=`isdigit`, `fo_class`=set-equality, `time`=pairwise tolerance). |
| Headline metric | Custom bucket mean | `Evaluator.pre_evaluation_score()` | Exact SDK definition: unweighted mean over ≤10 `group×{ID,OOD}` buckets, empty skipped. |
| ID/OOD × group report | Custom groupby | `pre_evaluation_score()[1]` (`buckets_df`) | Already the exact breakdown EVAL-03 asks for. |
| Duplicate-qID guard | Custom dedupe | `Evaluator.run()` (raises `ValueError`) | Enforced upstream; SDK aborts the whole run — mirror this, don't suppress it. |
| Latency gate | Custom timeout check | `run(track=Track.FRAME)` | Applies `TRACK_MAX_LATENCY[FRAME]=5.0`; `>` limit → incorrect + `timed_out`. |
| Response (de)serialization | Custom JSON | `save_items` / `load_responses` | Canonical `responses.json` schema the whole team shares. |
| Judge prompt / verdict | Custom LLM call | `TransformersJudge` / `APIJudge` | Fixed system+user template, `enable_thinking=False`, `"CORRECT" in raw and "INCORRECT" not in raw`, majority vote. |

**Key insight:** Every line of scoring we write is a line that can drift from the organizers' number. Phase 1's value is *fidelity*, not features — wrap, extract, and prove equality with a golden test.

## Runtime State Inventory

Phase 1 is greenfield (new repo scaffolding, no rename/migration), but two "runtime state" concerns matter for a trustworthy foundation:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | HF dataset cache (`~/.cache/huggingface`) + `FOCUS_ROOT_DIR` video/frame trees | Keep out of git; document each machine's `FOCUS_ROOT_DIR`. |
| Live service config | HF Hub auth token (for gated repos, if `orena-dkfz/*` is gated) | Store in `.secrets.env` (gitignored); each machine logs in via `huggingface-cli login`. Verify repos are reachable pre-Jul-15. |
| OS-registered state | None — no daemons/tasks registered this phase | None. |
| Secrets/env vars | `FOCUS_ROOT_DIR` (per-machine path), `HF_TOKEN` | Set via shell profile / `.secrets.env`, never committed. |
| Build artifacts | `orena-focus` installed from git tag → an `.egg-info`/site-packages entry | None to migrate; pin the tag so all 3 machines match. |

## Common Pitfalls

### Pitfall 1: "No model loaded" silently under-scores judge formats
**What goes wrong:** `Evaluator(judges=[])` marks every `open_ended`/`matching`/`multiple_choice` question INCORRECT, so the GPU-free score is a **lower bound**, not the true `pre_evaluation_score`.
**Why:** `majority_vote([], …)` returns `False`; the parse gate still runs first.
**How to avoid:** Treat `judges=[]` as "exact-match-only sanity score"; label harness output with the judge mode used. For full reproduction, run the real `TransformersJudge` (GPU box) or a `MockJudge` in tests.
**Warning signs:** Judge-format buckets at exactly 0.0; overall score far below expectation.

### Pitfall 2: OOD path untested because public data has no OOD
**What goes wrong:** EVAL-03 "reports ID/OOD buckets" passes on public data but only ever populates the `ood=False` half → the OOD branch is never exercised; a bug there ships silently to Phase 4.
**Why:** `Reference.ood` defaults `False` and is populated only by the private test split.
**How to avoid:** Add a synthetic fixture (see `tests/conftest.py` `make_reference(ood=True)`) with both ID and OOD rows across ≥2 groups; assert the bucket count and per-bucket accuracy (mirrors SDK `TestPreEvaluationScore`).

### Pitfall 3: Duplicate qID aborts the *entire* run, not one question
**What goes wrong:** Two `Response` for one `qID` → `ValueError` in `run()` → no score at all.
**Why:** Intentional SDK guard (`resp_map` dedupe check).
**How to avoid:** Pre-assert uniqueness before writing `responses.json` and surface a clear error; never silence the SDK's exception. Add a golden test asserting the raise (matches SDK `test_duplicate_response_raises`).

### Pitfall 4: Wrong track / wrong latency gate
**What goes wrong:** Examples ship with `Track.SEGMENT` (15 s). Using it for us applies the wrong 15 s gate instead of FRAME's 5.0 s.
**Why:** Copy-paste from `examples/evaluation.py`.
**How to avoid:** Hard-code `Track.FRAME` in the harness; a test asserts a 6 s response is `timed_out` under FRAME (mirrors SDK `test_track_resolves_limit`).

### Pitfall 5: `transformers` too low breaks the judge
**What goes wrong:** SDK only pins `transformers>=4.30`; a resolver could install an old 4.x that lacks Qwen3's `enable_thinking` chat-template kwarg → judge crashes.
**How to avoid:** Pin `==4.57.*` in Env C's lockfile even though this phase is GPU-free.

### Pitfall 6: `Qwen/Qwen3.5-4B` reachability / size
**What goes wrong:** The judge model is ~9.3 GB; first `TransformersJudge()` call downloads it. On a metered/offline box this stalls, and if the repo were gated it would 401.
**How to avoid:** Pre-download on the GPU box; keep the judge optional in CI. Verified 2026-07 that `Qwen/Qwen3.5-4B` exists on HF.

## Code Examples

### Reproduce the SDK headline score, GPU-free (exact-match-only)
```python
# Source: evaluator.py run() + pre_evaluation_score() (v0.3.4)
from focus import FocusDataset, DatasetSplit, Track, Evaluator, load_responses
ds = FocusDataset("heico", DatasetSplit.TEST, Track.FRAME)
resp = load_responses("responses.json")
results_df, summary_df = Evaluator(judges=[]).run(
    ds.requests, ds.references, resp, track=Track.FRAME, output_dir="experiments/runs/eda-smoke")
score = summary_df.loc[summary_df.level == "pre_evaluation", "accuracy"].iloc[0]
```

### The `responses.json` seam (EVAL-02 decoupling)
```json
[
  {"qID": "q001", "content": "2",   "latency": 1.23},
  {"qID": "q002", "content": "yes", "latency": 0.98}
]
```
`content` is the raw VLM text; the harness never needs the engine. Produced by `save_items(responses, path)`.

### Golden reproduction test (assert exact, not approx)
```python
# Mirrors SDK tests/test_evaluator.py::TestPreEvaluationScore
# Two buckets: object_recognition/ID = 0.5, aggregation/OOD = 1.0 -> mean 0.75
score, buckets = Evaluator().pre_evaluation_score(results_df)
assert len(buckets) == 2
assert score == 0.75
```

## State of the Art

| Old assumption | Current reality (v0.3.4) | Impact |
|----------------|--------------------------|--------|
| PyPI `orena-focus` is current | PyPI shows **0.1.1**; cloned/authoritative source is **0.3.4** | Pin to git tag `v0.3.4` (or the matching PyPI release once published) so all 3 machines + the metric match. |
| Judge is `Qwen3-4B` | SDK hardcodes `Qwen/Qwen3.5-4B` (`DEFAULT_JUDGE_MODEL`) | Use the exact string; it exists on HF (verified). |
| Format literal `foclass` | It is **`fo_class`** | Already corrected in CONSTITUTION/FEATURES; EDA must key on `fo_class`. |

**Deprecated/outdated:** none for this phase. Do not use `transformers` 5.x (breaks Qwen3 tooling + ms-swift cap).

## Open Questions

1. **True score reproduction needs the judge — where does it run in "GPU-free"?**
   - Known: `TransformersJudge(device="cpu")` works but Qwen3.5-4B on CPU is very slow; `judges=[]` under-scores judge formats.
   - Recommendation: harness supports 3 modes (`no-judge`, `mock`, `real`); the *canonical* reproduction number is produced with the real judge on the GPU box (Phase 2+), while Phase 1 CI uses `no-judge` + `mock`. Document the mode alongside every committed score.

2. **Are `orena-dkfz/heico-focus-vqa` / `lapchole-focus-vqa` public and stable before Jul 15?**
   - Known: repo IDs are hardcoded in `config.py`; annotations stream via `datasets`.
   - Unclear: gating status and whether a `revision`/tag should be pinned for reproducibility.
   - Recommendation: verify reachability now; pin `FocusDataset(..., revision="<tag>")` once a dataset tag exists (constructor + `download` both accept `revision`).

3. **What actually changes when pre-eval opens (Jul 15)?**
   - Known: public `ood` is all `False`; the private test split populates `ood=True` and may add FO classes via metadata (`FOType.names()` is dynamic).
   - Recommendation: confirm at Jul 15 (a) that FRAME `test` split exists with real `ood` labels, (b) the exact `pre_evaluation_score` on the leaderboard equals our local harness on the same responses (fidelity check), (c) whether the FRAME config name is exactly `"frame"`.

4. **Does the leaderboard's `pre_evaluation_score` == our local number bit-for-bit?**
   - Recommendation: the first real submission is a fidelity probe — submit a known `responses.json`, compare leaderboard vs local. Any gap means a wrong track, judge mode, or dataset revision.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10–3.12 | all envs | verify | — | pyenv/conda install |
| `orena-focus` v0.3.4 | FND-01, all EVAL/DATA | install | 0.3.4 (git tag) | pip from git if not on PyPI |
| `transformers==4.57.*` | judge, downstream | install | 4.57.* | — (hard pin) |
| HF Hub network | DATA-01 annotation streaming | needed at load | — | pre-cache dataset; offline later |
| `Qwen/Qwen3.5-4B` (~9.3 GB) | EVAL-04 real judge | download-on-first-use | — | `MockJudge` / `judges=[]` for GPU-free |
| GPU | **not required this phase** | — | — | CPU judge or mock |
| RunPod + teammate machines | FND-02/03 sync | provision | — | — |

**Missing dependencies with no fallback:** none block Phase 1 — it is deliberately GPU-free and data-download-free for the harness/EDA path.
**Missing dependencies with fallback:** real judge (GPU) → `MockJudge`/`judges=[]`; downloaded videos (Phase 2) not needed here.

## Validation Architecture

`workflow.nyquist_validation` is `true` → this section applies.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest>=7.0` (SDK already uses it; project inherits) |
| Config file | none yet — add `pyproject.toml [tool.pytest]` or `pytest.ini` in Wave 0 |
| Quick run command | `pytest tests/test_harness.py -x -q` |
| Full suite command | `pytest -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | `pre_evaluation_score` reproduced exactly on a golden fixture | unit | `pytest tests/test_harness.py::test_golden_pre_eval_score -x` | ❌ Wave 0 |
| EVAL-02 | Scores stub `responses.json` with `judges=[]` (no model) | unit | `pytest tests/test_harness.py::test_score_precomputed_no_model -x` | ❌ Wave 0 |
| EVAL-03 | `buckets_df` has correct group×{ID,OOD} rows on synthetic OOD fixture | unit | `pytest tests/test_harness.py::test_id_ood_buckets -x` | ❌ Wave 0 |
| EVAL-03 | Duplicate `qID` raises `ValueError` | unit | `pytest tests/test_harness.py::test_duplicate_qid_aborts -x` | ❌ Wave 0 |
| EVAL-04 | Judge-routed format handled via injected `MockJudge` (GPU-free) | unit | `pytest tests/test_harness.py::test_judge_path_with_mock -x` | ❌ Wave 0 |
| EVAL-01 | FRAME latency gate marks a 6 s response `timed_out` | unit | `pytest tests/test_harness.py::test_frame_latency_gate -x` | ❌ Wave 0 |
| DATA-01 | `FocusDataset("heico"/"lapchole", …, Track.FRAME)` yields `Request`/`Reference` | integration (network) | `pytest tests/test_data_loading.py::test_load_frame_split -x` (marked `@pytest.mark.network`) | ❌ Wave 0 |
| DATA-03 | EDA emits format + group distributions + per-format canonical shape | integration | `pytest tests/test_eda.py::test_eda_report_shapes -x` | ❌ Wave 0 |
| FND-01 | Example eval path runs end-to-end on the stub | smoke | `python scripts/evaluate.py --responses fixtures/stub_responses.json --no-judge` | ❌ Wave 0 |
| FND-02/03/04 | Repo sync, lockfile install, tracking convention | manual + CI | `pip install -r requirements/harness.lock && python -c "import focus"` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_harness.py -x -q` (pure-unit, GPU-free, < 5 s).
- **Per wave merge:** `pytest -q` (unit + non-network integration; mark `@network`/`@gpu` tests to skip in CI).
- **Phase gate:** full suite green + one manual FND-01 smoke run on a real machine before `/gsd:verify-work`.

### Wave 0 Gaps
- [ ] `pytest.ini` / `[tool.pytest]` with markers `network`, `gpu` — so CI stays GPU-free/offline-safe.
- [ ] `tests/conftest.py` — reuse SDK-style `make_request`/`make_reference`/`make_response` fixtures incl. `ood=True` rows.
- [ ] `tests/fixtures/stub_responses.json` — hand-written precomputed responses (EVAL-02 seam).
- [ ] `tests/test_harness.py`, `tests/test_data_loading.py`, `tests/test_eda.py` — cover the map above.
- [ ] `src/frame/eval/judges.py::MockJudge` — deterministic GPU-free judge.
- [ ] Framework install: `pip install pytest` in Env C lockfile.

## Sources

### Primary (HIGH confidence — read line-by-line, cloned `orena-focus` @ v0.3.4)
- `src/focus/evaluation/evaluator.py` — `Evaluator.run` signature, dup-qID raise (L190), adversarial+latency order (L217-220), `_evaluate_single` parse gate (L328-351), `pre_evaluation_score` bucket math (L402-462), seeded bootstrap (L488).
- `src/focus/data/data_models.py` — `Request`/`Reference`/`Response` fields; `save_items`/`load_responses` schema.
- `src/focus/evaluation/judges.py` — `Judge` ABC, `TransformersJudge` (`DEFAULT_JUDGE_MODEL="Qwen/Qwen3.5-4B"`, `device="cpu"`), `majority_vote`, `JUDGE_FORMATS` routing.
- `src/focus/data/{base_dataset,video_dataset,frame_dataset}.py` — `FocusDataset` (annotation-only, no config), `_parse_row`, `FocusVideoDataset(stride)`/`VideoSample`, `FocusFrameDataset`/`FrameSample`, `DATASET_BASE_FPS`.
- `src/focus/data/formats.py` — 8 formats + gates; `JUDGE_FORMATS`. `src/focus/config.py` — `FOCUS_DATASETS`, `TRACK_MAX_LATENCY`, `FocusConfig` (root_dir must exist). `src/focus/enums.py`, `taxonomy.py`.
- `examples/{evaluation,inference,data_preparation}.py` — the FND-01 paths. `tests/{test_evaluator,conftest}.py` — golden-test patterns + fixtures.
- `pyproject.toml` — base deps + `requires-python>=3.10`. `git describe` → `v0.3.4` (`9b32641`).

### Secondary (MEDIUM — web-verified 2026-07)
- PyPI/libraries.io `orena-focus` (shows 0.1.1) — [libraries.io](https://libraries.io/pypi/orena-focus), [GitHub](https://github.com/IMSY-DKFZ/orena-focus).
- Judge model existence — [Qwen/Qwen3.5-4B on HF](https://huggingface.co/Qwen/Qwen3.5-4B).
- Transformers 4.57 Qwen3-VL floor — [HF Qwen3-VL docs](https://huggingface.co/docs/transformers/v4.57.3/model_doc/qwen3_vl).

### Project source of truth (HIGH)
- `CONSTITUTION.md` §I–V, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`, `.planning/research/{ARCHITECTURE,STACK,FEATURES,PITFALLS}.md`, `CLAUDE.md`.

## Metadata

**Confidence breakdown:**
- SDK API surface (signatures, fields, scoring): **HIGH** — read directly from v0.3.4 source + its own tests.
- Standard stack / version pins: **HIGH** for the 4.57/0.3.4 floors; **MEDIUM** for whether 0.3.4 is on PyPI (pin to git tag to be safe).
- Judge behavior / GPU-free modes: **HIGH** — `majority_vote([])→False` and `device="cpu"` confirmed in source.
- Dataset availability + Jul-15 changes: **MEDIUM/LOW** — repo IDs known; gating + real OOD labels unverifiable until pre-eval opens (see Open Questions).

**Research date:** 2026-07-09
**Valid until:** ~2026-08-09 for the SDK facts (stable, pinned tag); re-verify dataset reachability + leaderboard fidelity at **Jul 15** pre-eval open.
