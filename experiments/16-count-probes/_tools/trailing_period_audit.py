"""16a follow-up — does the trailing-period habit cost anything where we are scored?

Folder-private glue. Importable; a notebook calls ``corpus_census(...)`` and ``coverage(...)``.
Not a launcher.

## The question this closes

Rung 30's subset builder found 32 rows of ``train.jsonl`` with counting intent and a non-digit
gold (``'2.'``, ``'Two.'``, ``'Intestine: 1.'``), all under the phrasing *"Please provide a single
integer"*, which appears nowhere else in the corpus — a candidate source of the trailing-period
habit probe 16a measured at 86.7% ID / 87.5% OOD against a base-model rate of 0.0000.

The question was never *"is the habit real"* (16a measured that) but *"does it cost points on the
scored path"*. Two halves, both cheap, neither previously run:

* ``corpus_census`` — where does the phrasing live: which split, which ``answer_format``,
  which ID/OOD side, and what do its golds look like.
* ``coverage`` — on a run's **raw** model strings, how many ``number`` answers are clean digits,
  how many the container's repair would fix, and how many it would miss.

``predictions.json`` holds the raw output: ``src/frame/metrics.py:1392`` — *"``run.run_baseline``
persists the raw model strings"*. Nothing normalises before it is written, so the second half is
not circular. Checked before it was relied on.

🔑 **Write the computation down, not only the number.** The ``±0.86`` label-noise figure of
``context/ERROR_ANATOMY.md`` reached 14 files and closed levers while its computation was never
committed; when it was finally re-derived it did not reproduce. A corpus-quality claim without a
runnable derivation is not a record.
"""

from __future__ import annotations

import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

#: The organizers' phrasing under which every malformed counting gold sits.
SINGLE_INTEGER = "single integer"

#: The shipped container's repair, verbatim: ``submissions/02-rung21-a2/inference.py:333``.
#: Note the ``\s*`` — ``experiments/30-grpo-number/_models/grpo_reward.py:43`` drops it.
CONTAINER_REPAIR = re.compile(r"^(\d+)\s*\.$")
CLEAN_DIGITS = re.compile(r"^\d+$")

CORPUS_GLOB = "external_data/orena-data/*/data/frame/*.parquet"


def load_corpus(repo: Path = Path(".")) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(repo / CORPUS_GLOB))):
        ds = path.split("orena-data/")[1].split("/")[0]
        df = pd.read_parquet(path)
        df["ds"] = ds
        df["split"] = Path(path).stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def corpus_census(repo: Path = Path(".")) -> dict:
    """Where the `single integer` phrasing lives, and what its golds look like."""
    df = load_corpus(repo)
    hit = df.question.str.contains(SINGLE_INTEGER, case=False, na=False)
    sub = df[hit]

    scored = df[df.split == "test"]
    n_scored_number = int((scored.answer_format == "number").sum())
    return {
        "corpus_rows": int(len(df)),
        "single_integer_rows": int(hit.sum()),
        "by_split_ds_format": {
            "/".join(map(str, k)): int(v)
            for k, v in sub.groupby(["split", "ds", "answer_format"]).size().items()
        },
        "ood": {str(k): int(v) for k, v in sub.ood.value_counts().items()},
        "golds": {str(k): int(v) for k, v in sub.answer.value_counts().items()},
        "golds_all_end_in_period": bool(
            sub.answer.astype(str).str.strip().str.endswith(".").all()
        ),
        # the population where the habit would actually be scored
        "scored_number_questions": n_scored_number,
        "scored_number_using_this_phrasing": int(
            (
                scored.answer_format.eq("number")
                & scored.question.str.contains(SINGLE_INTEGER, case=False, na=False)
            ).sum()
        ),
    }


def coverage(run_dir: Path) -> dict:
    """On a run's RAW model strings, how far the container's repair actually reaches."""
    run_dir = Path(run_dir)
    preds = {
        p["qID"]: p["content"]
        for p in json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    }
    with open(run_dir / "results.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["answer_format"] == "number"]

    cats: Counter[str] = Counter()
    uncovered: list[tuple[str, str]] = []
    for r in rows:
        out = (preds.get(r["qID"]) or "").strip()
        if CLEAN_DIGITS.match(out):
            cats["clean_digits"] += 1
        elif CONTAINER_REPAIR.match(out):
            cats["repaired_by_container"] += 1
        else:
            cats["not_covered"] += 1
            uncovered.append((r["ood"], out[:60]))

    n = max(len(rows), 1)
    return {
        "run_dir": str(run_dir),
        "number_questions": len(rows),
        "counts": dict(cats),
        "fractions": {k: round(v / n, 4) for k, v in cats.items()},
        "uncovered_examples": Counter(o for _, o in uncovered).most_common(15),
        "uncovered_by_ood": dict(Counter(o for o, _ in uncovered)),
    }
