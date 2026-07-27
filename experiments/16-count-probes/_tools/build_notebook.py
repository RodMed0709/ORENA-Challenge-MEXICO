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

CODE_CONFIG = '''\
# --- config (inline, per EXPERIMENT_REPO_STRUCTURE_SPEC — no launcher scripts) ---
SMOKE = True          # flip to False for the full probe
SEED  = 0

RUN          = "16a_zero_probe_v1"
RUN_DIR      = Path.cwd() / "runs" / RUN
DATA_ROOT    = Path("/workspace/orena-data")
FRAMES_CACHE = Path("/workspace/frames_cache")

MODELS = {
    "base": Path("/workspace/models/qwen3-vl-8b"),
    "ft":   Path("/workspace/repo/experiments/06-vit-lora/runs/06_vit_lora_v1/merged/checkpoint-2580"),
}

N_FRAMES_PER_CELL = 8 if SMOKE else 120   # per distribution (ID, OOD)
MAX_NEW_TOKENS    = 8                     # "0" / "yes" need almost nothing
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

missing = [p.frame_path for p in probes if not Path(p.frame_path).exists()]
assert not missing, f"{len(missing)} probe frames missing from the cache, e.g. {missing[:3]}"
print("all probe frames present in the shared cache")
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


def main() -> None:
    nb = {
        "cells": [
            cell_md(MD_TITLE, "t-title"),
            cell_code(CODE_BOOTSTRAP, "c-bootstrap"),
            cell_code(CODE_CONFIG, "c-config", tags=["parameters"]),
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


if __name__ == "__main__":
    main()
