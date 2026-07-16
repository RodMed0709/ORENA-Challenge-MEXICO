# Experiment Repo Structure Spec (drop-in)

> Self-contained standard for structuring a machine-learning / research
> experiment repo. Point a fresh Claude Code session at THIS single file and it
> can standardize a brand-new repo (training OR eval/inference — e.g. a VLM
> VQA-over-video-frames project) the same way, with no access to the repo this
> was distilled from.

## 0. Purpose + how to use

Philosophy in one breath: **notebooks are the runnable surface that generate
runs; `.py` files are importable libraries, never launchers; every experiment
changes exactly ONE thing versus a named baseline; every result is honest,
leak-guarded, and reviewed before it is trusted.**

**Two vocabulary terms used throughout:**
- **Rung** = one single-variable run in a numbered sequence (`03`, `03a`,
  `03b`, `04b` …). One rung changes exactly one thing vs a named baseline.
- **Ladder** = the ordered table of an experiment's rungs (notebook ↔ rung ↔
  metric ↔ verdict), best-so-far marked ⭐.

**An experiment is any single-variable run — training OR eval/inference.**
Most mechanics below read "training-flavored" (stages, checkpoints,
warm-start) but the spec fits both shapes:
- **Training experiment:** the engine is a trainer (`*_train_*.py`); a run
  produces a checkpoint + metrics.
- **Eval / inference experiment** (e.g. a prompt × model × frame-sampling
  sweep against a hosted VLM API — no training): the engine is an
  eval/inference harness (`eval_*.py`); `runs/<tag>/` holds predictions +
  metrics + `config_snapshot.yaml`; stage / checkpoint / warm-start language
  simply **does not apply**. The single-variable, leak-guard,
  build→smoke→review→full, one-`src`, and notebook-provenance rules apply
  **unchanged**.

## 0a. Sync discipline — GitHub is the code bus (BINDING)

**Code and text move through GitHub, never by direct local→pod copy.** The flow
is one-directional and non-negotiable:

```
   local (edit + commit)  ──push──▶  GitHub  ──pull/reset──▶  pod /workspace/repo
```

- **Code, notebooks, configs, docs, specs:** edit locally, `commit`, `push` to
  GitHub; on the pod `git pull` (or `git fetch && git reset --hard origin/<branch>`
  — gitignored `runs/`, `external_data/` survive a hard reset). **Never** `scp`/
  `rsync` source onto the pod, and **never** edit source directly on the pod.
  The pod's working tree is disposable and always reconstructable from a commit.
- **Binary artifacts only** (frames, images, run outputs, weights, `*.parquet`):
  a direct line is allowed — `scp` or the volume's S3 API — because they are
  gitignored and too big/opaque for git. Pull them **down** from the pod to view;
  do not push them into the repo.
- **Rationale:** every experiment is reproducible from a commit hash; there is no
  "it only exists on the pod" state; provenance stays in git history.
- The pod holds a clone at `/workspace/repo`; treat it as a checkout, not a
  source of truth. A cheap CPU pod is sufficient to run the `git pull` that lands
  new code on the volume between GPU sessions.

How a session uses this file:
1. Read the whole spec once.
2. Inventory the target repo (where source lives, how experiments are
   organized today, what launchers/wrappers exist).
3. Apply Section 3 (top-level skeleton) + Section 4 (experiment layout).
4. Convert launchers into notebooks (Sec 1, 5); reduce `_models/` to engines
   only (Sec 6); rename to the convention (Sec 7).
5. Enforce the discipline (Sec 8) and the one-`src` rule (Sec 9).
6. Run the post-restructure review (Sec 10); require GO.
7. Walk the checklist (Sec 11) per folder.

Replace metric placeholders with your own ("your primary eval metric — VQA
accuracy, exact-match, F1, …", "your named baseline", "the deliverable
artifact").

---

## 1. The one rule

> **Notebooks generate runs. `.py` files are libraries, never launchers.**

- The human runs **notebooks**, top-to-bottom. The human does **not** run
  `.py` files by hand.
- A `run_*.py` / `main.py` driver you launch from a shell is **WRONG**. Its
  orchestration belongs in a notebook cell.
- Engines (`*_train_*.py`, `eval_*.py`, model/harness factory) stay
  **importable**. The notebook imports the engine and calls `engine.main(...)`.
- No `.sh` chains. No wrapper notebook that `subprocess`-calls a `.py`. That is
  the anti-pattern this rule exists to kill — put config inline in the notebook
  and call the engine function directly.
- **Sanctioned exceptions (the only runnable `.py`):** `report.py` (Sec 4) is
  a reporting library — call it from a reporting notebook cell (`import report`),
  never a hand-run script. If your environment forces a detached long run
  (Sec 8), the detached entrypoint is still the **notebook executed
  headlessly** (e.g. `jupyter nbconvert --execute`), not a hand-written
  launcher.

Why: a notebook is the unit of provenance. Reading one notebook top-to-bottom
tells you exactly what produced a run — the variable changed, the config, the
call, the result — with nothing hidden in shell history.

---

## 2. Two-part store

Every experiment is split across **two** locations:

| Location | Holds | Committed? |
|---|---|---|
| `experiments/<id>/` | Real artifacts: notebooks, engines, configs, results, deliverable | yes (except `runs/`) |
| `context/<id>/CONTEXT.md` | The single curated, in-scope context — **outside** the artifact dir | yes |

`CONTEXT.md` is the short, always-current narrative for one experiment. It
lives **outside** `experiments/<id>/` so context stays skimmable and a session
can load exactly one context file without dragging in checkpoints and logs.
Five headings: **Objective** (the question, 1–2 lines) / **Setup-config**
(data split, base model, key hyperparameters, the ONE variable) / **Decisions**
(choices + why, tight) / **Results** (ladder outcome: numbers + verdicts) /
**Next** (immediate next move).

Load ONLY the `CONTEXT.md` for the experiment in scope. Do not bulk-read all
context files.

> Context vs `_docs/`: `context/<id>/CONTEXT.md` is the ONE curated context.
> `experiments/<id>/_docs/` holds supporting long-form notes **referenced by**
> CONTEXT.md (a design memo, a roadmap) — never a second competing summary.

---

## 3. Repo top-level skeleton

```
repo/
├── src/                # THE canonical first-party package. Exactly one.
├── experiments/        # one subdir per experiment (artifacts)
│   └── <id>/
├── context/            # one CONTEXT.md per experiment (curated context)
│   └── <id>/CONTEXT.md
├── external_data/      # datasets & external assets — GITIGNORED
├── docs/               # deliverables only (reports, papers, exported docs)
└── README.md
```

- **Exactly ONE canonical `src/` package.** Every experiment imports from it.
  No second first-party source tree anywhere (full rule in Sec 9).
- `experiments/` and `context/` mirror each other by `<id>`.
- `external_data/` is **gitignored** (large, download-regenerable). When a
  dataset ships a checksum, **verify md5** on setup and record it.
- `docs/` holds only deliverables (the `.docx`/`.pdf`/paper you hand off).
  Planning/narrative `.md` do **not** live here.

---

## 4. Experiment folder layout

```
experiments/<id>/
├── README.md               # OPENS with the ladder table (see below)
├── report.py               # reporting library (imported from a cell, not run)
├── RESULTS.csv             # scorecard: one row per rung
├── results_*.csv           # optional auxiliary scorecards
├── 00_<slug>.ipynb         # trainer/eval notebooks at root, run order 0-based
├── 01_<slug>.ipynb
├── <id>_report.docx        # deliverable export (optional)
├── _models/                # ENGINES ONLY (+ its own README index)
│   └── README.md
├── _tools/                 # folder-private helpers (optional)
├── _docs/                  # supporting narrative (optional)
├── .gitignore
└── runs/                   # GITIGNORED — generated outputs
    └── <tag>/
```

- **README opens with the LADDER** — the first thing in the file:

  | Notebook | Rung | Metric (primary) | Verdict |
  |---|---|---|---|
  | `00_baseline.ipynb` | 00 | 0.612 acc | baseline |
  | `01_frame_sampling.ipynb` | 01 | 0.640 acc | GO ⭐ |
  | `02_prompt_variant.ipynb` | 02 | 0.605 acc | NO-GO (faithful negative) |

- **`report.py`** — the single reporting library (A/B generator). Imported from
  a reporting notebook cell (`import report; report.render(...)`), never run by
  hand.
- **`RESULTS.csv`** — machine-readable scorecard, one row per rung.
- **`NN_<slug>.ipynb`** — trainer/eval notebooks at the folder root, 0-based,
  top-to-bottom.
- **`_models/`** — real engines only (Sec 6), with its own `README.md` index.
- **`_tools/`** — folder-private helpers (Sec 9).
- **`_docs/`** — supporting long-form narrative referenced by CONTEXT.md.
- **`runs/<tag>/`** — gitignored: checkpoints/predictions, logs,
  `config_snapshot.yaml` — all regenerable.

---

## 5. Notebook template

Every trainer/eval notebook follows the same cell order.

**(a) Markdown title** — the rung + the ONE variable vs the named baseline.
```markdown
# 01 — frame sampling rate
Rung 01. Baseline = `00_baseline`. The ONE variable: frames-per-clip 8 → 16.
Everything else identical to baseline.
```

**(b) Bootstrap** — walk up to the repo root by a guaranteed marker (bounded so
it can never run past `/`), then put the experiment's `_models` (if present)
and the canonical `src/` on `sys.path`. Single-sourced: `REPO` comes from the
same walk, so it works at any nesting depth.
```python
import sys
from pathlib import Path

# Walk up to the repo root: the dir that has `.git` or a top-level `src/`.
EXP_DIR = Path.cwd()
REPO = EXP_DIR
while REPO != REPO.parent and not ((REPO / ".git").exists() or (REPO / "src").is_dir()):
    REPO = REPO.parent
# EXP_DIR = the experiments/<id> dir we started in (has notebooks/_models).
for p in (EXP_DIR / "_models", REPO / "src"):
    if p.is_dir():
        sys.path.insert(0, str(p))
```
> No `_models/`? (pure eval/inference repo) The loop is fine — it only adds
> `_models` to the path *if it exists*, and never depends on it to find REPO.

**(c) Markdown "sibling rungs"** — how to get closely-related rungs by changing
ONE value here, instead of spawning a near-duplicate notebook.
```markdown
## Sibling rungs (change ONE value, rerun, bump the tag)
- 01a: frames = 24
- 01b: frames = 16 + stride 2
Minor tweaks live here as prose + a new runs/<tag>/ + a RESULTS.csv row —
NOT a new notebook.
```

**(d) Config INLINE + guards + import engine + snapshot.** Config lives in the
notebook, not a launcher. `SMOKE` toggle → tiny fast pass, no separate smoke
script.
```python
import yaml
import eval_engine as engine        # trainer OR eval/inference harness

SMOKE = False                       # True => tiny fast pass

cfg = dict(
    variable_under_test = 16,       # THE one lever
    epochs      = 1 if SMOKE else 30,   # (training only; omit for pure eval)
    limit_items = 4 if SMOKE else None,
    out_dir     = EXP_DIR / "runs" / "01_frame_sampling_v1_0709",
)
cfg["out_dir"].mkdir(parents=True, exist_ok=True)
with open(cfg["out_dir"] / "config_snapshot.yaml", "w") as f:
    yaml.safe_dump({k: str(v) for k, v in cfg.items()}, f)
```

**(e) Stage cell(s)** — call `engine.main(...)`. A **stage** is any sequential
sub-run; multi-stage is **optional**. Use one stage when that is all it needs.
Examples of multi-stage in ONE notebook: `train → eval`; `retrieve → answer →
score`; or (fine-tuning only) a constrained pass then a full pass. Do not force
a two-stage shape where the experiment doesn't have one.
```python
engine.main(cfg)                    # single stage
# or, when the experiment has phases:
# engine.main(cfg, stage=1); engine.main(cfg, stage=2)
```

**(f) Markdown result** — the headline number, where the run landed, and how to
evaluate it (which `report` call, which split).

**Only meaningful rungs get their own notebook.** A minor parameter tweak is
prose in the sibling-rungs cell + a `runs/<tag>/config_snapshot.yaml` + a
`_models/README.md` line + a `RESULTS.csv` row. Never a wrapper notebook that
shells out to a `.py`.

---

## 5b. Long / headless runs — `papermill` is the canonical path (BINDING)

**The problem this closes.** "Notebooks generate runs" (§1) collides with reality when a run takes
hours: you cannot hold a Jupyter session open for it, and the pod session ends. Rung 05 hit this and
solved it by exporting the notebook to `05_bottleneck_audit.py`, `sed`-ing a flag, and `nohup`-ing the
script — which violates §1 (`.py` are libraries, never launchers) and made provenance a manual
`sha256` + diff exercise. It survived only because the export happened to differ from the notebook by
exactly one line **and that could be proven after the fact**. **A training run gets no such second
chance: if provenance breaks, the weights are not auditable.**

**`papermill` executes the notebook itself, headless, with parameters injected — and writes an output
notebook that IS the provenance record.** No export, no `sed`, no launcher, and the source notebook is
never modified.

### The `parameters` cell (convention)

The **first code cell** of any notebook meant to run headless carries the `parameters` tag and holds
**only** the knobs — no logic:

```python
# cell metadata: {"tags": ["parameters"]}
SMOKE = True          # default: the cheap path
```

papermill inserts a second `# Parameters` cell immediately after it, overriding the defaults. **The
tagged cell must stay a plain assignment block**, or the override lands in the wrong place.

### Invocation

```bash
papermill  <notebook>.ipynb  runs/<tag>/executed.ipynb  -p SMOKE False  -k <kernel>
```

Run it under `nohup`/`&` for long jobs. It is a **command**, not a launcher file — no `.sh`, no
`run_*.py`. Config still lives in the notebook (§1 holds).

### What you get (measured, not assumed — verified 2026-07-16, papermill 2.7.0)

| | |
|---|---|
| **Source notebook** | **untouched** — the flag is never edited on disk |
| `runs/<tag>/executed.ipynb` | the exact code that ran **+ every output**, cell by cell |
| `metadata.papermill.parameters` | **what was actually injected** (`{'SMOKE': False}`) |
| `metadata.papermill.duration` / `.exception` / `.input_path` | run record |
| per-cell `metadata.papermill.status` | `completed` / `failed` / `pending` |

**On failure** — the case that matters for a long run: papermill **exits non-zero** (detectable by a
monitor), records `exception: True`, marks the offending cell `failed` and everything after it
`pending`, **and still writes the output notebook with the expensive work already done preserved**. A
gate that fires (e.g. a trainable-parameter check) stops the run, leaves the reason on disk, and does
not execute what follows.

> **The output notebook replaces the `sha256` dance.** It carries the code, the parameters, the outputs
> and the failure state in one artifact. **`runs/` is gitignored**, so it stays a local/pod artifact —
> `scp` it down with the rest.

### Requirements (both env and pod)

```bash
pip install papermill ipykernel
python -m ipykernel install --user --name <env> --display-name "<env>"
```

🔴 **`papermill` alone is not enough — it needs a registered kernel** (`NoSuchKernel` otherwise).
Installing the package and forgetting `ipykernel install` is the first way this fails.

## 6. `_models/` — engines only

`_models/` contains **engines and only engines**: trainer(s) (`*_train_*.py`),
evaluator/inference harness(es) (`eval_*.py`), and the model/harness factory.

**Delete on sight** (their logic moves into notebooks): `smoke_*.py` (→ the
`SMOKE` toggle), `run_*.py` launchers, `.sh` chains, `resume_*.py` and similar
orchestration.

**`_models/README.md` is an index** — engine ↔ rung ↔ notebook ↔ run:

| Engine | Used by rung | Notebook | Latest run |
|---|---|---|---|
| `eval_engine.py` | 00, 01 | `00_…`, `01_…` | `01_…_v1_0709` |

**Do NOT rename shared engine files.** An engine imported by other experiment
folders is a shared contract — `grep` the whole repo before renaming/moving. If
anything outside the folder imports it, leave the name and add a `DO NOT
rename — imported by <X>` note in `_models/README.md`.

---

## 7. Naming

- **Rung number = experiment number**, letter suffixes for close variants:
  `03`, `03a`, `04b`. Notebook and its run share the rung.
- **Notebook:** `NN_<slug>.ipynb`, 0-based, run order top-to-bottom.
- **Run tag:** `<NB>_<slug>_v<N>_<MMDD>` — e.g. `01_frame_sampling_v1_0709`.
  Intentional, never a bare date; bump `v<N>` on every rerun; the tag alone
  identifies which notebook / version / day produced the run.
- **Do not rename run tags referenced by other folders** — grep first.

---

## 8. Experiment discipline (non-negotiables)

**Single-variable A/B contract.** One experiment changes **exactly ONE thing**
vs a **named baseline**. Every lever is a **toggleable flag, default OFF**, and
with all flags OFF the run is **byte-identical to the baseline**. Must bundle
levers? Mark the run **confounded** and keep the flags **individually
separable**.

**Honest, leak-guarded eval.** The **eval split MUST equal the leak-guard
split** — evaluate on exactly the data your leak guard protected; nothing that
could have leaked into training/prompt-tuning. Report your primary metric as a
**single-variable Δ vs the named baseline** (e.g. "+2.8 pts VQA accuracy").
**Report in the target's units:** if your internal metric/units differ from the
leaderboard/deliverable target (resolution, frame count, tokenization, scale),
convert before reporting — never compare across unit spaces.

**Build → smoke → review → full.** (1) Build the engine. (2) Smoke — tiny pass
via the `SMOKE` toggle. (3) Review — an **independent, read-only reviewer**
returns **GO / GO-WITH-FIXES / NO-GO** with **file:line evidence**, BEFORE any
full run. Never launch an un-reviewed full run. (4) Full — one clean run.

**Detached, contention-aware runs (for expensive runs — training or heavy
inference).** A long run must **survive session death** — the detached
entrypoint is the **notebook executed headlessly** (e.g. `nbconvert --execute`),
not a hand-written launcher. One clean run per experiment. **Check for
expensive-resource contention first** (a shared GPU, an API rate limit / quota,
a paid endpoint) and wait your turn. *Not applicable to cheap local eval.*

**Reuse the baseline result** *(training experiments)*. If a lever is
byte-identical to the baseline with its flag OFF, do **not** re-run the baseline
arm — reuse the baseline's existing checkpoint/cached outputs and run only the
candidate arm.

**Faithful negatives are valid results.** A correctly-run experiment showing "no
improvement" / "regression" is a real outcome. Record it in the ladder as NO-GO
with the number; do not bury it or re-roll until it looks good.

---

## 9. The ONE-`src` rule (strict) + `_tools` private

- **Exactly ONE shared first-party `src/` package** at the repo root; every
  experiment imports it via the bootstrap cell (Sec 5b).
- **No per-experiment `src/`.** Find one → fold it into the top-level `src/`
  and repair imports.
- **Per-experiment repetitive wiring → that folder's `_tools/`**, which is
  **folder-private**: nothing outside `experiments/<id>/` may import it. It is
  for glue specific to one experiment, not worth promoting to `src/`.
- **Escape hatch (documented):** vendored **third-party external** code may live
  in its own dir with an explicit note (what it is, where it came from). A
  **second first-party `src/` is never allowed.**

Decision rule: reusable across experiments → `src/`. Specific to one experiment
→ that experiment's `_tools/`. Foreign code you didn't write → vendored dir with
a note.

---

## 10. Post-restructure review (required)

Restructuring is high-risk (cross-folder imports, run-tag references, path
depth). After **any** restructure, dispatch a **read-only review agent** and
require a **GO** before done. It must confirm:

- [ ] `py_compile` succeeds on **every** `.py`.
- [ ] Every notebook **validates** (parses/loads cleanly).
- [ ] **No dangling references** to deleted files (launchers, smoke scripts,
      `.sh` chains, old paths).
- [ ] **Cross-folder importers still compile** — anything importing a
      moved/renamed engine still resolves.
- [ ] Baseline / warm-start artifacts (checkpoints or cached outputs) **exist**
      at the paths the notebooks expect *(for experiments that have them)*.

Mechanics: `runs/` stays gitignored; moves are plain `mv`, kept within the
folder where possible; **do not commit** unless explicitly asked.

---

## 11. Quick checklist to standardize a folder

- [ ] Exactly ONE top-level `src/`; no per-experiment first-party `src/`.
- [ ] `experiments/<id>/` for artifacts; `context/<id>/CONTEXT.md` outside it
      (Objective / Setup-config / Decisions / Results / Next).
- [ ] README **opens with the ladder** (notebook ↔ rung ↔ metric ↔ verdict,
      leader ⭐).
- [ ] Trainer/eval notebooks `NN_<slug>.ipynb` at folder root, 0-based.
- [ ] Every notebook: title → bootstrap (marker-anchored, depth-agnostic) →
      sibling-rungs → inline config + `SMOKE` toggle + `import engine` +
      `config_snapshot.yaml` → stage(s) → result.
- [ ] `_models/` = **engines only**; `smoke_*`/`run_*`/`.sh`/`resume_*` deleted;
      `_models/README.md` indexes engine ↔ rung ↔ notebook ↔ run.
- [ ] Shared engine files NOT renamed (grep first; note "DO NOT rename").
- [ ] Folder-specific repetitive wiring in `_tools/` (folder-private).
- [ ] `report.py` imported from a reporting cell (not run); `RESULTS.csv`
      scorecard present.
- [ ] Run tags `<NB>_<slug>_v<N>_<MMDD>`, intentional, `v<N>` bumped on rerun;
      cross-referenced tags not renamed.
- [ ] Single-variable A/B: one lever vs named baseline, flags default OFF =
      byte-identical; bundled levers marked confounded but separable.
- [ ] Honest eval: eval split = leak-guard split; primary metric Δ vs baseline;
      reported in the target's units.
- [ ] build → smoke → independent review (GO/GO-WITH-FIXES/NO-GO + file:line)
      → full; detached run = headless notebook, survives session death;
      resource contention checked.
- [ ] `runs/` gitignored; `external_data/` gitignored + md5 verified.
- [ ] Post-restructure review agent returned **GO**.

---

### Appendix — environment-specific notes (may not apply to your project)

- Some agent harnesses block a subagent from writing files literally named
  `README.md` / `REPORT.md` / `CONTEXT.md`. Workaround: the subagent writes the
  content to a `.txt` sibling and the parent session places the final `.md`.
  Ignore if your environment doesn't have this limitation.
