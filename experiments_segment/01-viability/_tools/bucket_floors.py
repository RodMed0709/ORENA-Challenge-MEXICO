"""The SEGMENT per-bucket trivial-floor table, split ID/OOD. Zero GPU, library only.

WHY THIS EXISTS AND WHY IT COMES FIRST
`RULES §10` says judge by MARGIN over the template-aware floor, never by raw accuracy — and
`experiments_segment/01-viability/REVIEW.md:36` measured the SEGMENT trivial `bucket_mean` at
**0.5803**, with 6 of 10 buckets above 0.40. Until the floor is known per bucket, an arm's
`bucket_mean` cannot be read at all: on `aggregation_ID` (n=23 over 21 distinct templates) the
per-template modal answer is nearly the identity, so a high accuracy there means nothing.

The floor is computed with `frame.metrics.template_floor` — the canonical implementation
(`RULES §1`: extend the module, never re-derive beside it) — against the **eval** set, never
the train prior (`RULES §5`; the first attempt at this gate used the train prior and REVIEW.md
B1 failed it for exactly that).

ID/OOD comes from the qID prefix (`RULES §3`): `heico` = OOD, `lapchole` = ID. The `ood`
column in the parquet is all-False on public data and must never be trusted.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from frame.metrics import _load_capability, template_floor

# `focus/__init__.py` eagerly imports transformers/datasets, so `from focus.taxonomy import
# Capability` fails on a machine without torch — which is exactly where a zero-GPU floor table
# should be runnable. `frame.metrics._load_capability` is the repo's own solution to that and
# falls back to loading the stdlib-only `taxonomy.py` directly. Reusing it rather than copying
# it is `RULES §1`: extend the module, never re-derive beside it.
Capability = _load_capability()


def load_segment_test(data_root: str | Path) -> pd.DataFrame:
    """The 6,254 SEGMENT test rows, both datasets, with the fields the floor needs."""
    root = Path(data_root)
    frames = []
    for ds in ("heico", "lapchole"):
        df = pd.read_parquet(root / ds / "data" / "segment" / "test.parquet")
        df["qID"] = f"{ds}__" + df["id"].astype(str)
        df["dataset"] = ds
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # RULES §3 — never `df["ood"]`, it is all-False on the public split.
    out["split"] = out["qID"].str.split("__").str[0].map({"heico": "OOD", "lapchole": "ID"})
    # RULES §2 — leaf -> group ALWAYS via Capability.group, never by filtering the leaf column.
    out["group"] = out["primary_capability"].map(lambda c: Capability.from_any(c).group.name)
    return out


def bucket_floor_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (group × split): n, trivial floor, and how guessable the bucket is.

    `n_templates` and `n_per_template` are reported beside the floor because they are what
    makes it readable: a floor of 0.91 on 23 questions spread over 21 templates is an artefact
    of the slice being smaller than its own template vocabulary, not a property of the task.
    """
    rows = []
    for (grp, split), sub in df.groupby(["group", "split"], sort=True):
        # `template_floor` takes the slice and the column names — the template IS the raw
        # question string here (the corpus has no separate template id on SEGMENT rows).
        floor = template_floor(sub.assign(template=sub["question"]),
                               answer_col="answer", template_col="template")
        n_tpl = sub["question"].nunique()
        rows.append({
            "bucket": f"{grp}_{split}",
            "group": grp, "split": split, "n": len(sub),
            "n_templates": n_tpl,
            "q_per_template": round(len(sub) / max(1, n_tpl), 2),
            "floor": round(float(floor), 4),
            "formats": ", ".join(f"{k} {v}" for k, v in
                                 sub["answer_format"].value_counts().head(3).items()),
        })
    t = pd.DataFrame(rows).sort_values(["group", "split"]).reset_index(drop=True)
    return t


def summarise(t: pd.DataFrame) -> dict:
    """`bucket_mean` of the trivial floor — the number an arm must beat to have done anything."""
    return {
        "n_buckets": int(len(t)),
        "trivial_bucket_mean": round(float(t["floor"].mean()), 4),
        "trivial_bucket_mean_ID": round(float(t.loc[t["split"] == "ID", "floor"].mean()), 4),
        "trivial_bucket_mean_OOD": round(float(t.loc[t["split"] == "OOD", "floor"].mean()), 4),
        "n_questions": int(t["n"].sum()),
        "buckets_above_0.40": int((t["floor"] > 0.40).sum()),
        "smallest_bucket": t.loc[t["n"].idxmin(), "bucket"],
        "smallest_n": int(t["n"].min()),
    }
