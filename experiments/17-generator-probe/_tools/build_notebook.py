"""Regenerate 17_generator_probe.ipynb from source strings. Diffable; never runs the rung."""

import json
from pathlib import Path

CELLS: list[tuple[str, str, list[str]]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src, []))


def code(src: str, tags: list[str] | None = None) -> None:
    CELLS.append(("code", src, tags or []))


md(r"""# rung 17 — can the CoA generator actually see?

Inference only. No training, no checkpoint, nothing merged.

The pre-registration is `context/17-generator-probe/CONTEXT.md` and it is fixed. This
notebook produces the number; **it does not decide the verdict** — `probe.verdict` applies the
rule that was written before the number existed.

| reference | `bucket_mean` | margin ID / OOD |
|---|---|---|
| 00-baseline (our 8B, zero-shot) | 0.2557 | −0.088 / −0.191 — **below floor everywhere** |
| 06-vit-lora (our fine-tuned 8B) | 0.5667 | +0.207 / +0.148 |
| **17 — Qwen3-VL-32B as a perceiver** | _this run_ | _this run_ |
""")

code(r"""SMOKE = True      # True -> 8 questions: proves the chain, decides nothing
PROBE_N = 200     # the pre-registered probe size
""", tags=["parameters"])

code(r"""import logging, sys, time
from pathlib import Path

for p in (Path.cwd() / "_models", Path.cwd().parents[1] / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S")

import probe as P

cfg = P.ProbeConfig(smoke=SMOKE, probe_n=PROBE_N)
print(f"model    : {cfg.model_id}")
print(f"questions: {cfg.n}   (SMOKE={SMOKE})")
print(f"run_dir  : {cfg.run_dir}")
""")

md(r"""## 🔴 GPU headroom gate

The 32B is 63 GB in bf16. Loading it beside an 8B LoRA training run OOMs **the training** —
the cheap job kills the expensive one. This raises rather than warns, because the damage
lands on somebody else's multi-hour run and is not recoverable.""")

code(r"""hw = P.assert_gpu_headroom(cfg)
print("GPU headroom OK:", hw)
""")

md(r"""## Generate — the generator answers the FRAME question directly

No gold. No scaffold. No sibling fact-sheet. Pure perception, through rung 09's own driver and
model load, so nothing here can differ from the generation this probe is vouching for.""")

code(r"""t0 = time.perf_counter()
csv_path = P.generate(cfg)
print(f"generated in {(time.perf_counter()-t0)/60:.1f} min -> {csv_path}")
""")

md(r"""## Score canonically

`frame.metrics.stratified_report` only (RULES §1), read as **margin over the template-aware
floor**. Raw accuracy is unreadable here: the floor is 0.337 ID / 0.460 OOD, so a model can
look respectable and be sitting on the trivial constant.""")

code(r"""from frame import ledger, metrics

gold = ledger.gold_from_frame_parquets(cfg.data_root, split="train")
rep = P.score_canonically(cfg, csv_path, gold)

print(f"bucket_mean {rep['bucket_mean']:.4f}")
print(f"acc    ID {rep['acc_ID']:.4f}   OOD {rep['acc_OOD']:.4f}")
print(f"floor  ID {rep['floor_ID']:.4f}   OOD {rep['floor_OOD']:.4f}")
print(f"MARGIN ID {rep['margin_ID']:+.4f}   OOD {rep['margin_OOD']:+.4f}")
""")

code(r"""import pandas as pd
print(pd.DataFrame(rep["by_format"])[["answer_format", "n", "accuracy", "floor", "margin"]]
      .to_string(index=False))
""")

md(r"""## The pre-registered verdict

Applied by code, from the rule written before the number existed. A human reads it against
`context/17-generator-probe/CONTEXT.md` — this cell does not get to reinterpret it.""")

code(r"""print(P.verdict(rep))
print()
print("Compare — our own 8B zero-shot: bucket_mean 0.2557, margin -0.088 ID / -0.191 OOD")
print("⚠️ Asymmetric reading, pre-registered: a general VLM also loses points to task format,")
print("   so a weak score UNDER-states perception. Negative = decisive. Positive = directional.")
""")

code(r"""if not SMOKE:
    ledger.register_run(rep, run_dir=cfg.run_dir, experiment="17-generator-probe",
                        run=cfg.run_name, model=str(cfg.model_id))
    print("registered in the ledger ->", cfg.run_dir / "stratified.json")
else:
    print("SMOKE: not registered — an 8-question score is not a result")
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": ({"tags": tags} if tags else {}),
         "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s, tags in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "python3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "17_generator_probe.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
