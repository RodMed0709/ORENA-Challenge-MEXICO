"""Emit `16a_zero_probe.ipynb`. Folder-private tooling — run it, do not import it.

The notebook is the run generator (EXPERIMENT_REPO_STRUCTURE_SPEC §5): title -> bootstrap ->
inline config + SMOKE toggle -> import the engine from `_models/` -> run -> report. This
script exists so the notebook's cells stay reviewable as plain text in a diff.

    python experiments/16-count-probes/_tools/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "16a_zero_probe.ipynb"

MD_TITLE = """# 16a — the zero probe: can the model emit 0, and did our SFT take it away?

**Question.** Our training set contains **no negative example** in any numeric or class
format (0 of 2,495 per-class counts, 0 of 2,628 total counts, 0 of 8,969 `fo_class` golds).
Before minting synthetic zeros — expensive, and carrying ~8.4% one-directional label noise
pointed *toward* zero on a model already biased -0.66 — measure whether the capability is
missing at all.

**External reference.** HoloCount (arXiv:2607.06420, Table 3, Null-Target Prompting)
reports **Qwen3-VL-8B at 96.4%** on absent-object questions, ahead of Gemini-3.1-Pro
(55.2%). If the base has it and our checkpoint does not, the cure is regularisation, not
data.

**Design.** Paired, same frames, same questions, two models:
`base` = `/workspace/models/qwen3-vl-8b` vs `ft` = rung-06 **ep3** (`checkpoint-2580`,
`bucket_mean` 0.5724 — the current best, and the epoch-matched control that closed rungs
14 and 15). Two arms — ABSENT and PRESENT — and two formats — `number` and `binary`.

**Read `emit_rate`, not `acc`.** `emit_rate` is a property of the model's output and is
noise-free. `acc` compares against a closure-derived label that is wrong ~8.4% of the time,
always in the same direction. The PRESENT arm is the control that stops "always answer 0"
from looking like a win; the harmonic mean is the two-axis score.

**Pre-registered readings (write them down BEFORE looking):**
- If `base` emit_rate_absent is high (say >0.5) and `ft` is near 0 -> **our SFT destroyed it**.
  Next move is regularisation / answer-prior repair, and the minting plan is demoted.
- If BOTH are near 0 -> the capability is absent in this domain; minting becomes the
  candidate, dosed 3:1-2:1 and adversarially sampled.
- If BOTH are high -> there is no zero problem to fix; drop the whole branch.
- If `ft` emit_rate_present is also high, `ft` has a zero attractor, not a zero capability —
  read the harmonic mean, never `acc_absent` alone.
"""

CODE_BOOTSTRAP = '''\
# --- bootstrap -----------------------------------------------------------------
import logging, os, sys, json
from pathlib import Path

REPO = Path("/workspace/repo")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "vendor" / "orena-focus" / "src"))
sys.path.insert(0, str(Path.cwd() / "_models"))          # experiment-private glue

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
'''

# 🔴 PARAMETERS CELL — RAW LITERALS ONLY, no derived values.
# papermill injects its overrides in a NEW cell placed AFTER this one. Anything COMPUTED
# here is computed from the pre-injection values and is never recomputed, so
# `-p SMOKE False` would silently leave a derived `N_FRAMES_PER_CELL = 8` in place and the
# "full" run would quietly re-run the smoke. That happened once here; the derive cell below
# and its gate exist so it cannot happen again.
CODE_CONFIG = '''\
# --- parameters (papermill injects overrides directly below this cell) ----------
SMOKE = True          # flip to False (or -p SMOKE False) for the full probe
SEED  = 0

RUN            = "16a_zero_probe_v1"
DATA_ROOT      = "/workspace/orena-data"
FRAMES_CACHE   = "/workspace/frames_cache"
MODEL_BASE     = "/workspace/models/qwen3-vl-8b"
MODEL_FT       = "/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/merged/checkpoint-2580"

N_FRAMES_SMOKE = 8    # per distribution (ID, OOD)
N_FRAMES_FULL  = 120
MAX_NEW_TOKENS = 8    # "0" / "yes" need almost nothing
'''

CODE_DERIVE = '''\
# --- derived (MUST live below the parameters cell — see the note in _tools) ------
DATA_ROOT    = Path(DATA_ROOT)
FRAMES_CACHE = Path(FRAMES_CACHE)
RUN_DIR      = Path.cwd() / "runs" / RUN
MODELS       = {"base": Path(MODEL_BASE), "ft": Path(MODEL_FT)}

N_FRAMES_PER_CELL = N_FRAMES_SMOKE if SMOKE else N_FRAMES_FULL
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("run dir:", RUN_DIR, "| SMOKE:", SMOKE, "| frames/cell:", N_FRAMES_PER_CELL)
'''

CODE_BUILD = '''\
# --- build the probe set (gold only — no model has been loaded yet) -------------
from frame.config import BaselineConfig
from frame.data import load_frame_items
import zero_probe as zp

cfg = BaselineConfig(data_root=DATA_ROOT)
items = load_frame_items(cfg, splits=("test",))     # val half — the distribution we are scored on
probes = zp.build_probe_set(items, n_frames_per_cell=N_FRAMES_PER_CELL, seed=SEED)

import collections
print("questions:", len(probes), "| frames:", len({p.frame_key for p in probes}))
print(collections.Counter((p.distribution, p.arm, p.fmt) for p in probes))

# RAISING gates — an empty or lopsided probe set must abort BEFORE a model is loaded.
# The first smoke run built 0 questions (wrong Reference attribute) and still spent a full
# 17 GB checkpoint load discovering nothing. Gates raise, never warn (RULES: gates RAISE).
assert probes, "probe set is EMPTY — check zero_probe.answer_format against the SDK schema"
# The gate that makes a silent smoke-as-full impossible: the realized frame count must match
# the mode we think we are in. Without it, `-p SMOKE False` failing to take effect looks
# exactly like a successful full run.
_frames = len({p.frame_key for p in probes})
_expect = 2 * N_FRAMES_PER_CELL   # two distributions
assert _frames == _expect, (
    f"SMOKE={SMOKE} implies {_expect} frames but the probe set has {_frames} — "
    "a papermill parameter almost certainly did not take effect"
)
_cells = collections.Counter((p.distribution, p.arm) for p in probes)
assert len(_cells) == 4, f"expected ID/OOD x absent/present = 4 cells, got {dict(_cells)}"
_n_abs = sum(1 for p in probes if p.arm == "absent")
_n_prs = sum(1 for p in probes if p.arm == "present")
assert _n_abs == _n_prs, f"arms must be balanced: {_n_abs} absent vs {_n_prs} present"

missing = [p.frame_path for p in probes if not Path(p.frame_path).exists()]
assert not missing, f"{len(missing)} probe frames missing from the cache, e.g. {missing[:3]}"
print("gates passed: non-empty, 4 cells, arms balanced, all frames present")
'''

CODE_RUN = '''\
# --- run both models over the SAME questions -----------------------------------
import gc, time
from frame.engine import QwenFrameEngine

rows = []
for tag, path in MODELS.items():
    t0 = time.time()
    mcfg = BaselineConfig(data_root=DATA_ROOT, model_path=path, max_new_tokens=MAX_NEW_TOKENS)
    eng = QwenFrameEngine(mcfg)
    eng.load()
    rows += zp.run_probe(eng, probes, model_tag=tag)
    print(f"{tag}: {len(probes)} questions in {time.time()-t0:.0f}s")
    del eng; gc.collect(); torch.cuda.empty_cache()   # 32 GB card — free before the next load
'''

CODE_REPORT = '''\
# --- score + persist -----------------------------------------------------------
import pandas as pd

df = pd.DataFrame(rows)
df.to_csv(RUN_DIR / "probe_rows.csv", index=False)

res = zp.score(rows)
(RUN_DIR / "score.json").write_text(json.dumps(res, indent=2))

print("=== POOLED (read emit_rate_absent first — it is noise-free) ===")
print(pd.DataFrame(res["pooled"]).T.round(4).to_string())
print()
print("=== PER CELL (model | distribution | format) ===")
print(pd.DataFrame(res["cells"]).T.round(4).to_string())
'''

CODE_DELTA = '''\
# --- the paired comparison that answers the question ---------------------------
# Same frame, same question, base vs ft. A paired read, because the two models saw
# identical inputs — two independent rates would discard that pairing.
piv = (df.pivot_table(index=["frame_key", "cls", "arm", "fmt", "distribution"],
                      columns="model", values="is_zero", aggfunc="first")
         .dropna().reset_index())
ab = piv[piv.arm == "absent"]
print(f"ABSENT, paired n={len(ab)}")
print(f"  base emits 0/no : {ab['base'].mean():.4f}")
print(f"  ft   emits 0/no : {ab['ft'].mean():.4f}")
print(f"  delta (ft-base) : {ab['ft'].mean() - ab['base'].mean():+.4f}")
print(f"  HoloCount reference for Qwen3-VL-8B on Null-Target Prompting: 0.9640")
print()
pr = piv[piv.arm == "present"]
print(f"PRESENT (control — high here means a ZERO ATTRACTOR, not a capability), n={len(pr)}")
print(f"  base emits 0/no : {pr['base'].mean():.4f}")
print(f"  ft   emits 0/no : {pr['ft'].mean():.4f}")
piv.to_csv(RUN_DIR / "paired.csv", index=False)
print("\\nwrote", RUN_DIR / "paired.csv")
'''

CODE_INSPECT = '''\
# --- eyeball: what did each model actually SAY? (user rule: error examples every run) ---
show = (df[df.arm == "absent"]
        .groupby("model")
        .head(12)[["model", "distribution", "cls", "fmt", "named", "raw"]])
print(show.to_string(index=False))
'''


def cell_md(src: str, cid: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def cell_code(src: str, cid: str, tags: list[str] | None = None) -> dict:
    # papermill injects overrides UNDER the cell tagged `parameters`; without that tag
    # `-p SMOKE False` is silently ignored and the notebook keeps its inline default.
    return {
        "cell_type": "code",
        "id": cid,
        "execution_count": None,
        "metadata": {"tags": tags} if tags else {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


OUT_16C = HERE.parent / "16c_len_points.ipynb"

MD_16C = """# 16c — derive the count as `len(points)` instead of verbalising it

**The finding.** Alghisi et al. 2026 (*Getting to the Point*, arXiv:2603.21746), Qwen2.5-VL-7B
+ LoRA r=32/α=64, exact-match count accuracy on their OOD split (count-range extrapolation —
structurally our shape): direct count **23.41%**, point-then-count **verbalised** 14.04%,
**`#Coord` = `len(points)` 94.96%**. The lever is not pointing. It is **refusing to let the
model state the number** — taking the count as the length of its own coordinate list.

**Why rung 15 did not test this.** Rung 15 replicated Gautam's structured count *format* and
returned a null. And Gautam's Table I (re-verified from the repo PDF, p.5) shows the *joint*
count+point objective **costs** counting — counting-only 0.26 vs count+point 1.52. So the
structured target is the trick and the joint objective is the tax. `len(points)` is a third
thing: a decoding-side derivation.

⚠️ **The convention is a trap.** Qwen2-VL used `[0,1000]` → **Qwen2.5-VL switched to absolute
pixels** → **Qwen3-VL switched back to `[0,1000]`, points as `point_2d` in JSON**. Gautam and
Alghisi both ran Qwen2.5-VL, so *their exact serialisation is wrong for our backbone*. Arm
`a1` matches our native convention on purpose.

**Arms** (single variable = the answer target; image and question identical):
`a0` bare integer (control) · `a1` `point_2d` JSON, count = `len()` · `a2` numbered points
(Molmo), count = last index.

**Scored on** the *Clips* template — 681 val questions, 12 distinct true values, margin over
the template-aware floor only **+0.026**: the single largest hole in the exam. Read **margin**,
never raw accuracy.

🔴 **A null here is NOT decisive.** Alghisi report that prompt-only point-then-count is weak
without fine-tuning, and that a black-screen substitution costs <2% — the count is read off the
model's own text, not re-derived from pixels. This probe bounds the *prompt-only* path and
measures p99 at the longer `max_new_tokens`; it cannot bound the trained one.
"""

CODE_16C_CONFIG = '''\
# --- parameters (RAW LITERALS ONLY; papermill injects overrides directly below) --
SMOKE = True
SEED  = 0

RUN        = "16c_len_points_v1"
DATA_ROOT  = "/workspace/orena-data"
# rung 06 ep3 — the current best checkpoint and the epoch-matched control (bucket_mean 0.5724)
MODEL_PATH = "/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/merged/checkpoint-2580"

TEMPLATE      = r"how many\\s+clips"   # the 681-question hole; widen only with a reason
ARMS          = ("a0", "a1", "a2")
N_ITEMS_SMOKE = 24
'''

CODE_16C_DERIVE = '''\
# --- derived (MUST live below the parameters cell) -------------------------------
DATA_ROOT  = Path(DATA_ROOT)
MODEL_PATH = Path(MODEL_PATH)
RUN_DIR    = Path.cwd() / "runs" / RUN
N_ITEMS    = N_ITEMS_SMOKE if SMOKE else None   # None = the whole template
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("run dir:", RUN_DIR, "| SMOKE:", SMOKE, "| arms:", ARMS, "| n_items:", N_ITEMS or "ALL")
'''

CODE_16C_BUILD = '''\
# --- select the scored items (gold only; no model loaded yet) --------------------
from frame.config import BaselineConfig
from frame.data import load_frame_items
import len_points as lp

cfg   = BaselineConfig(data_root=DATA_ROOT)
items = load_frame_items(cfg, splits=("test",))
sel   = lp.select_number_items(items, template_re=TEMPLATE)
if N_ITEMS:
    sel = sel[:N_ITEMS]

import collections
golds = [g for _, g in sel]
assert sel, f"no items matched {TEMPLATE!r} — check the template regex against the corpus"
print("items:", len(sel), "| distinct golds:", sorted(set(golds)))
print("distribution:", collections.Counter("OOD" if it.dataset == "heico" else "ID" for it, _ in sel))
print("trivial floor (always answer the mode):", round(lp.template_floor(golds), 4))
'''

CODE_16C_RUN = '''\
# --- run the three arms on ONE model load ---------------------------------------
# All three arms differ only in the prompt suffix and max_new_tokens, so a single load
# serves them all — and keeps the image path byte-identical across arms.
import time, gc
from frame.engine import QwenFrameEngine

rows = []
for arm in ARMS:
    mcfg = BaselineConfig(data_root=DATA_ROOT, model_path=MODEL_PATH,
                          max_new_tokens=lp.ARM_MAX_NEW_TOKENS[arm])
    eng = QwenFrameEngine(mcfg); eng.load()
    t0 = time.time()
    rows += lp.run_arm(eng, sel, arm)
    dt = time.time() - t0
    print(f"arm {arm}: {len(sel)} questions in {dt:.0f}s  ({dt/max(len(sel),1):.3f} s/q)")
    del eng; gc.collect(); torch.cuda.empty_cache()
'''

CODE_16C_REPORT = '''\
# --- score: MARGIN over the template floor, never raw accuracy -------------------
import pandas as pd, json

df = pd.DataFrame(rows); df.to_csv(RUN_DIR / "rows.csv", index=False)
res = lp.score(rows); (RUN_DIR / "score.json").write_text(json.dumps(res, indent=2))
print(pd.DataFrame(res).T.round(4).to_string())
print()
print("parse methods:", df.groupby(["arm", "method"]).size().to_dict())
print("Alghisi reference (Qwen2.5-VL-7B, their OOD): direct 0.2341 | verbalised 0.1404 | len() 0.9496")
'''

CODE_16C_INSPECT = '''\
# --- eyeball: what did each arm actually emit? (user rule: error examples every run) ---
for arm in ARMS:
    sub = df[df.arm == arm].head(4)
    print(f"--- {arm} ---")
    for r in sub.itertuples():
        print(f"  gold={r.gold} pred={r.value} ok={r.correct} | {str(r.raw)[:110]!r}")
'''


OUT_16B = HERE.parent / "16b_detector_vs_gold.ipynb"

MD_16B = """# 16b — is the gold count something ANY visual system can see in the frame?

**This is the probe that bounds the other two.** Rung 06 ep3 is our best checkpoint and its
**`number_margin_OOD` is 0.0 — exactly the trivial floor.** Three explanations license
opposite work: the model cannot count; the model was never taught to; or **the label is not a
count of frame-visible instances at all**.

The third is not idle. `context/ERROR_ANATOMY.md` records the gold moving **±0.86 between
frames ≤1 s apart**, "total instances" contradicting the sum of per-class counts on **≥11%** of
frames, and a **blind human scoring r = −0.17** against the gold. Our own model scores
**r = +0.43**. If a frozen open-vocabulary detector also fails to correlate, the quantity is not
recoverable from the pixels — and rung 15's pre-registered annotation-ceiling question is
answered.

**Instrument.** Frozen **OWLv2** (`google/owlv2-base-patch16-ensemble`, Apache-2.0, on-pod —
🔒 no challenge frame leaves the machine, per the DUA) over the **681 val `Clips`** questions.

## 🔴 Pre-registered thresholds — fixed before any number is seen

| outcome | verdict |
|---|---|
| **r ≥ 0.5 and MAE < 1.0** | the gold IS frame-visible → pointing learnable → proceed to a rung |
| **r ≈ 0.2–0.5** | ambiguous → ~100 clinician-adjudicated frames before any training spend |
| **r < 0.2** | not frame-visible → **NO-GO on pointing**, re-aim every `number` lever |

## ⚠️ The confound, and the two mandatory mitigations

A detector failure is also a detector **domain** failure — OWLv2 has never seen laparoscopic
tissue. That is the trap rung 12 fell into three times. So: (1) a **positive control** query
(`"a surgical instrument"`) the detector must be able to do — if that collapses too, the
instrument is blind here and the honest result is **NOT MEASURABLE**, not a NO-GO; and (2) an
eyeball subset, exported so a human can check whether the detector finds clips a person can see.
"""

CODE_16B_CONFIG = '''\
# --- parameters (RAW LITERALS ONLY; papermill injects overrides directly below) --
SMOKE = True
SEED  = 0

RUN           = "16b_detector_vs_gold_v1"
DATA_ROOT     = "/workspace/orena-data"
DETECTOR_DIR  = "/workspace/models/detectors/owlv2-base"
TEMPLATE      = r"how many\\s+clips"
THRESHOLD     = 0.15          # pre-registered; do NOT tune it after seeing r
N_ITEMS_SMOKE = 24
'''

CODE_16B_DERIVE = '''\
# --- derived (MUST live below the parameters cell) -------------------------------
DATA_ROOT = Path(DATA_ROOT)
RUN_DIR   = Path.cwd() / "runs" / RUN
N_ITEMS   = N_ITEMS_SMOKE if SMOKE else None
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("run dir:", RUN_DIR, "| SMOKE:", SMOKE, "| threshold:", THRESHOLD,
      "| n_items:", N_ITEMS or "ALL")
'''

CODE_16B_RUN = '''\
# --- select items, run the frozen detector ---------------------------------------
from frame.config import BaselineConfig
from frame.data import load_frame_items
import len_points as lp                 # reuse the template selector — one definition
import detector_vs_gold as dvg

items = load_frame_items(BaselineConfig(data_root=DATA_ROOT), splits=("test",))
sel   = lp.select_number_items(items, template_re=TEMPLATE)
if N_ITEMS:
    sel = sel[:N_ITEMS]
assert sel, f"no items matched {TEMPLATE!r}"
print("items:", len(sel),
      "| videos:", len({it.video_id for it, _ in sel}),
      "| distinct golds:", sorted({g for _, g in sel}))

det = dvg.Owlv2Counter(DETECTOR_DIR); det.load()
rows = dvg.run(det, sel, threshold=THRESHOLD)
'''

CODE_16B_REPORT = '''\
# --- score against the PRE-REGISTERED thresholds ---------------------------------
import pandas as pd, json

df = pd.DataFrame(rows); df.to_csv(RUN_DIR / "rows.csv", index=False)
res  = dvg.correlate(rows, "n_clip")
ctrl = dvg.correlate(rows, "n_control")
(RUN_DIR / "score.json").write_text(json.dumps({"clip": res, "control": ctrl}, indent=2))

print("=== CLIP vs gold ==="); print(pd.DataFrame(res).T.round(4).to_string())
print(); print("=== CONTROL (surgical instrument) vs gold — sanity, not a hypothesis ===")
print(pd.DataFrame(ctrl).T.round(4).to_string())
print(); print("=== detector mean per gold value (a FLAT line means no signal) ===")
print(pd.DataFrame(dvg.by_gold_value(rows)).T.to_string())

r   = res["ALL"]["spearman_r"]; mae = res["ALL"].get("mae", float("nan"))
zr  = res["ALL"].get("det_zero_rate", float("nan"))
cz  = ctrl["ALL"].get("det_zero_rate", float("nan"))
print()
print(dvg.VERDICT)
print()
print(f"MEASURED: spearman r = {r:.4f} | MAE = {mae:.4f} | detector-zero rate = {zr:.3f}")
print(f"reference points already in the repo: blind human r = -0.17 | our model r = +0.43")
if cz > 0.9:
    print("🔴 CONTROL COLLAPSED (detector finds no instrument on >90% of frames) -> "
          "the instrument is blind in this domain. Report NOT MEASURABLE, not NO-GO.")
elif r >= 0.5 and mae < 1.0:
    print("VERDICT: gold IS frame-visible -> pointing is learnable")
elif r < 0.2:
    print("VERDICT: gold is NOT frame-visible -> NO-GO on pointing; re-aim every number lever")
else:
    print("VERDICT: ambiguous -> adjudicate ~100 frames before spending training compute")
'''

CODE_16B_INSPECT = '''\
# --- eyeball export so a human can check the detector, not just its correlation ---
eye = df.sample(min(40, len(df)), random_state=SEED)[
    ["frame_key", "distribution", "gold", "n_clip", "maxscore_clip", "n_control"]
]
eye["frame_path"] = "/workspace/frames_cache/" + eye["frame_key"]
eye.to_csv(RUN_DIR / "eyeball_40.csv", index=False)
print(eye.head(15).to_string(index=False))
print("\\nwrote", RUN_DIR / "eyeball_40.csv",
      "— open the frames and judge whether the detector missed clips a person can see")
'''


OUT_16D = HERE.parent / "16d_format_audit.ipynb"

MD_16D = """# 16d — the systematic finder: where does OUR fine-tuning emit SDK-illegal answers?

**Where this came from.** 16a asked an off-template question and the fine-tuned checkpoint
answered `"1."` — trailing period — on **87%** of `number` questions. `Number.verify` gates on
`str.strip().isdigit()`, so all of those are **auto-incorrect**. Base: **0.0000**. We created it.
That was luck. This notebook makes it a sweep.

🟢 **It needs no ground truth.** "Would the SDK's verifier accept this string?" is a property of
the model's OUTPUT alone, so any phrasing can be audited without annotating anything.

🔴 **And that is why the defect was invisible.** Our scored eval only ever asks the corpus's own
templates, so a fragility living outside them cannot appear in any number we report — by
construction. The hidden test is the organizers' generator, not ours.

**Design.** Real corpus questions, asked under meaning-preserving surface variants (`v0` is
verbatim = the control). Per (model × format × variant): the illegal-answer rate and *which*
violation occurred. The product is `regressions()` — the list of places where `ft` is more
illegal than `base`, or more illegal than its own `v0`.

⚠️ **This measures format legality, not correctness.** Every illegal answer is scored wrong
regardless of whether the model knew the answer, so the rate is a **floor** on lost points.
"""

CODE_16D_CONFIG = '''\
# --- parameters (RAW LITERALS ONLY; papermill injects overrides directly below) --
SMOKE = True
SEED  = 0

RUN        = "16d_format_audit_v1"
DATA_ROOT  = "/workspace/orena-data"
MODEL_BASE = "/workspace/models/qwen3-vl-8b"
MODEL_FT   = "/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/merged/checkpoint-2580"

FORMATS      = ("number", "binary", "fo_class")   # the exact-match formats; judge formats have no gate to mirror
N_PER_FORMAT_SMOKE = 6
N_PER_FORMAT_FULL  = 40
MAX_NEW_TOKENS     = 32
'''

CODE_16D_DERIVE = '''\
# --- derived (MUST live below the parameters cell) -------------------------------
DATA_ROOT = Path(DATA_ROOT)
RUN_DIR   = Path.cwd() / "runs" / RUN
MODELS    = {"base": Path(MODEL_BASE), "ft": Path(MODEL_FT)}
N_PER_FORMAT = N_PER_FORMAT_SMOKE if SMOKE else N_PER_FORMAT_FULL
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("run dir:", RUN_DIR, "| SMOKE:", SMOKE, "| n/format:", N_PER_FORMAT)
'''

CODE_16D_BUILD = '''\
# --- select real corpus questions per format (gold not needed — see the header) ---
from frame.config import BaselineConfig
from frame.data import load_frame_items
import format_audit as fa

items = load_frame_items(BaselineConfig(data_root=DATA_ROOT), splits=("test",))
items_by_fmt = {f: fa.select_by_format(items, f, N_PER_FORMAT, seed=SEED) for f in FORMATS}

for f, v in items_by_fmt.items():
    assert v, f"no items selected for format {f!r}"
    print(f"{f:10s} n={len(v):3d} videos={len({i.video_id for i in v}):3d}")
n_calls = sum(len(v) for v in items_by_fmt.values()) * len(fa.VARIANTS) * len(MODELS)
print("variants:", list(fa.VARIANTS), "| total model calls:", n_calls)

# show the variants on one real question so the transformation is reviewable, not implicit
_q = str(items_by_fmt["number"][0].request.question)
for name, fn in fa.VARIANTS.items():
    print(f"  {name:14s} {fn(_q)[:90]}")
'''

CODE_16D_RUN = '''\
# --- run both models over every (question x variant) -----------------------------
import gc, time
from frame.engine import QwenFrameEngine

rows = []
for tag, path in MODELS.items():
    t0 = time.time()
    eng = QwenFrameEngine(BaselineConfig(data_root=DATA_ROOT, model_path=path,
                                         max_new_tokens=MAX_NEW_TOKENS))
    eng.load()
    rows += fa.run_audit(eng, items_by_fmt, model_tag=tag)
    print(f"{tag}: done in {time.time()-t0:.0f}s")
    del eng; gc.collect(); torch.cuda.empty_cache()
'''

CODE_16D_REPORT = '''\
# --- the product: where fine-tuning made us MORE illegal --------------------------
import pandas as pd, json

df = pd.DataFrame(rows); df.to_csv(RUN_DIR / "rows.csv", index=False)
res = fa.score(rows); (RUN_DIR / "score.json").write_text(json.dumps(res, indent=2))

tbl = pd.DataFrame(res).T[["n", "illegal_rate", "top_violation"]]
tbl.index = pd.MultiIndex.from_tuples([tuple(k.split("|")) for k in tbl.index],
                                      names=["model", "fmt", "variant"])
print("=== illegal-answer rate (lower is better; this is a FLOOR on lost points) ===")
print(tbl.unstack("model")["illegal_rate"].round(4).to_string())
print()
print("=== REGRESSIONS: ft more illegal than base, or than its own v0 ===")
regs = fa.regressions(res)
if regs:
    print(pd.DataFrame(regs).to_string(index=False))
    pd.DataFrame(regs).to_csv(RUN_DIR / "regressions.csv", index=False)
else:
    print("none above the 0.10 threshold — a faithful negative, and worth recording as one")
print()
print("=== violation histogram (ft only) ===")
print(df[(df.model == "ft") & (~df.sdk_legal)].violation.value_counts().to_string())
'''

CODE_16D_INSPECT = '''\
# --- the fixable list: one real example per (fmt, violation) ----------------------
bad = df[(df.model == "ft") & (~df.sdk_legal)]
for (f, v), g in bad.groupby(["fmt", "violation"]):
    r = g.iloc[0]
    print(f"[{f} / {v}]  n={len(g)}")
    print(f"    Q: {r.question[:100]}")
    print(f"    A: {r.raw!r}")
'''


def main() -> None:
    nb = {
        "cells": [
            cell_md(MD_TITLE, "t-title"),
            cell_code(CODE_BOOTSTRAP, "c-bootstrap"),
            cell_code(CODE_CONFIG, "c-config", tags=["parameters"]),
            cell_code(CODE_DERIVE, "c-derive"),
            cell_code(CODE_BUILD, "c-build"),
            cell_code(CODE_RUN, "c-run"),
            cell_code(CODE_REPORT, "c-report"),
            cell_code(CODE_DELTA, "c-delta"),
            cell_code(CODE_INSPECT, "c-inspect"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", OUT)

    nb16c = {
        "cells": [
            cell_md(MD_16C, "t-title-16c"),
            cell_code(CODE_BOOTSTRAP, "c-bootstrap-16c"),
            cell_code(CODE_16C_CONFIG, "c-config-16c", tags=["parameters"]),
            cell_code(CODE_16C_DERIVE, "c-derive-16c"),
            cell_code(CODE_16C_BUILD, "c-build-16c"),
            cell_code(CODE_16C_RUN, "c-run-16c"),
            cell_code(CODE_16C_REPORT, "c-report-16c"),
            cell_code(CODE_16C_INSPECT, "c-inspect-16c"),
        ],
        "metadata": nb["metadata"],
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_16C.write_text(json.dumps(nb16c, indent=1), encoding="utf-8")
    print("wrote", OUT_16C)

    nb16b = {
        "cells": [
            cell_md(MD_16B, "t-title-16b"),
            cell_code(CODE_BOOTSTRAP, "c-bootstrap-16b"),
            cell_code(CODE_16B_CONFIG, "c-config-16b", tags=["parameters"]),
            cell_code(CODE_16B_DERIVE, "c-derive-16b"),
            cell_code(CODE_16B_RUN, "c-run-16b"),
            cell_code(CODE_16B_REPORT, "c-report-16b"),
            cell_code(CODE_16B_INSPECT, "c-inspect-16b"),
        ],
        "metadata": nb["metadata"],
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_16B.write_text(json.dumps(nb16b, indent=1), encoding="utf-8")
    print("wrote", OUT_16B)

    nb16d = {
        "cells": [
            cell_md(MD_16D, "t-title-16d"),
            cell_code(CODE_BOOTSTRAP, "c-bootstrap-16d"),
            cell_code(CODE_16D_CONFIG, "c-config-16d", tags=["parameters"]),
            cell_code(CODE_16D_DERIVE, "c-derive-16d"),
            cell_code(CODE_16D_BUILD, "c-build-16d"),
            cell_code(CODE_16D_RUN, "c-run-16d"),
            cell_code(CODE_16D_REPORT, "c-report-16d"),
            cell_code(CODE_16D_INSPECT, "c-inspect-16d"),
        ],
        "metadata": nb["metadata"],
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_16D.write_text(json.dumps(nb16d, indent=1), encoding="utf-8")
    print("wrote", OUT_16D)


if __name__ == "__main__":
    main()
