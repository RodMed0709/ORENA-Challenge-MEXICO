"""Assemble the Stage-1 eyeball CSV from the two backend completion files.

Folder-private glue for experiment 09 (not a library, not a launcher chain — a thin
reproducible step the notebook / orchestrator calls). The actual scaffold generation is
done OUTSIDE python (deepseek via the MCP, Claude via the orchestrator); this reads the
resulting `{qID: scaffold}` JSON maps back in, rebuilds the exact same stratified sample
(deterministic seed) via the engine, validates every scaffold, and writes the side-by-side
`sample_eyeball.csv` for human review.

Usage (from repo root):
    python experiments/09-coa-sft/_tools/build_sample_eyeball.py \
        <deepseek_completions.json> <claude_completions.json>

Both JSON files map qID -> scaffold string. Missing qIDs are left blank (flags all False).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_models"))
import coa_scaffold_gen as g  # noqa: E402

logger = logging.getLogger(__name__)


def build(deepseek_path: Path, claude_path: Path, cfg: g.GenConfig | None = None) -> Path:
    cfg = cfg or g.GenConfig()
    df = g.load_train_rows(cfg)
    factsheets = g.build_factsheets(df)
    sample = g.stratified_sample(cfg, df)  # deterministic — same rows as generation

    ds_comp = json.loads(Path(deepseek_path).read_text(encoding="utf-8"))
    cl_comp = json.loads(Path(claude_path).read_text(encoding="utf-8"))
    completions = {
        qid: {"deepseek": ds_comp.get(qid, ""), "claude": cl_comp.get(qid, "")}
        for qid in sample["qID"]
    }
    rows = g.build_eyeball_rows(cfg, sample, factsheets, completions)
    out = g.write_eyeball_csv(cfg, rows)

    # also stash the raw completions next to the CSV so it reproduces from run artifacts
    (cfg.run_dir / "deepseek_completions.json").write_text(
        json.dumps(ds_comp, ensure_ascii=False, indent=1), encoding="utf-8")
    (cfg.run_dir / "claude_completions.json").write_text(
        json.dumps(cl_comp, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def flag_summary(csv_path: Path) -> None:
    import pandas as pd
    d = pd.read_csv(csv_path)
    n = len(d)
    print(f"\n=== eyeball flag summary (n={n}) ===")
    for b in ("deepseek", "claude"):
        valid = int(d[f"{b}__valid"].sum())
        leak = int(d[f"{b}__answer_leak"].sum())
        match = int(d[f"{b}__answer_matches_gold"].sum())
        tags = int(d[f"{b}__has_all_tags"].sum())
        coh_after = int(d[f"{b}__nothing_after_answer"].sum())
        wc = d[f"{b}__word_count"].mean()
        print(f"  {b:9s}: valid {valid}/{n} | answer_matches_gold {match}/{n} | "
              f"answer_leak {leak}/{n} | has_all_tags {tags}/{n} | "
              f"nothing_after {coh_after}/{n} | mean_words {wc:.0f}")
    print("\n=== answer_leak by answer_format (both backends must reason to the answer) ===")
    for b in ("deepseek", "claude"):
        by = d.groupby("answer_format")[f"{b}__answer_leak"].mean().round(2).to_dict()
        print(f"  {b:9s}: {by}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ds, cl = Path(sys.argv[1]), Path(sys.argv[2])
    out = build(ds, cl)
    flag_summary(out)
    print(f"\nwrote -> {out}")
