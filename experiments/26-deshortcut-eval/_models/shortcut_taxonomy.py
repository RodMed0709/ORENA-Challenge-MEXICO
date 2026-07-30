"""Which of our val questions hand the model part of the answer, and what that is worth.

Engine for rung 26 (roadmap phase 0.4). Importable; a notebook calls ``exposure(...)`` and
``strata_report(...)``. Zero GPU: question text, gold, and an already-scored ``results.csv``.

## The question

Our headline is a MARGIN over a template-aware trivial floor: +0.207 ID / +0.148 OOD at rung 06.
That floor answers *"what does a constant that knows the answer distribution score?"* — it does
NOT answer *"what does a model score that reads the QUESTION but not the FRAME?"*. SurgCheck
(`literature/vlm-techniques/FICHAS.md` §v11) shows those are different quantities across five
surgical VQA models: question phrasing implicitly constrains the answer space, and reported
accuracy can reflect that rather than visual understanding.

Our own data carries the construct openly. Two mechanisms, both read off the question text:

  * **cardinality given** — *"There is one surgical foreign object visible in the frame. What
    surgical foreign object is visible…"* (673 `fo_class` rows). The count is handed over, which
    collapses an exact-SET task into a 1-of-10 pick.
  * **entity named** — *"How many **Clips** appear in this frame?"*, *"Do **Clips** and
    **Sponges** co-occur…"*, *"Where is the center of the **External drain**…"*. The class is
    given, so identity never has to be recovered — and the positional template additionally
    *presupposes* the object is present.

## What this module does and does not settle

It measures **exposure** (how much of the eval set carries a shortcut) and the **margin gap**
between the shortcut-carrying and shortcut-free cells of the same format×distribution. That is
the pre-registration evidence for whether the paired de-named eval is worth a GPU run.

🔴 It does NOT prove exploitation. The two cells are DIFFERENT QUESTIONS, not a paired
manipulation — a higher margin on shortcut-carrying questions is equally consistent with those
questions simply being easier in a way the template floor does not price. Only SurgCheck's actual
design settles that: the SAME frame and the SAME gold, asked with and without the entity name,
kept well-defined by one of four grounding cues (bounding box, arrow, spatial position,
periphrasis). That needs inference and is the GPU half — see ``PLAN.md``.

🔴 And the aggregate is a TRAP, measured: pooled over formats the ID gap reads +0.0493, which is
almost entirely a **format-mix artefact** (`multiple_choice` is 100% shortcut, `binary` 55%,
`number` 39.5%, `fo_class` 25%, and the formats have very different intrinsic difficulty). Within
format the sign is not even constant. Never quote the pooled number; ``strata_report`` therefore
refuses to produce one.
"""

from __future__ import annotations

import re

import pandas as pd

# ``there is one`` / ``there are two`` — the count, asserted as a premise of the question.
CARDINALITY_RE = re.compile(r"\bthere (?:is|are) (?:one|two|three|four|five|six|\d+)\b", re.I)

# Minimum rows for a cell to be reported at all. Below this a margin is an anecdote, and the
# template floor itself becomes unstable (RULES §13 — effective n is videos, not questions).
MIN_CELL_N = 30


def _named_entity_re() -> re.Pattern:
    """Any canonical FO class name appearing in the question text.

    Read from ``FOType.names()`` at runtime, never hard-coded — RULES §8b, because the
    organizers' in-prompt list and the scoring registry disagree on their tenth element.
    """
    from focus.foreign_objects import FOType  # noqa: PLC0415 — keeps import cost off module load

    names = sorted(FOType.names(), key=len, reverse=True)  # longest first: "Specimen Bag" > "Specimen"
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")s?\b", re.I)


def classify(questions: pd.Series) -> pd.DataFrame:
    """Per-question shortcut flags: ``leak_cardinality``, ``leak_named_entity``, ``shortcut``.

    Deterministic and purely lexical — no model, no gold. Declared before any margin is read so
    the strata cannot be tuned to the answer.
    """
    named = _named_entity_re()
    q = questions.astype(str)
    out = pd.DataFrame(index=questions.index)
    out["leak_cardinality"] = q.str.contains(CARDINALITY_RE, regex=True)
    out["leak_named_entity"] = q.str.contains(named, regex=True)
    out["shortcut"] = out["leak_cardinality"] | out["leak_named_entity"]
    return out


def exposure(gold: pd.DataFrame) -> pd.DataFrame:
    """How much of the eval set carries each shortcut, by ``answer_format``.

    ``gold`` needs ``question`` and ``answer_format``. Returns counts and shares; a `distribution`
    column, if present, is reported too so exposure can be checked for ID/OOD confounding (it is
    balanced on our val set: 0.393 ID vs 0.413 OOD, which is why the ID/OOD read below is not
    itself a shortcut artefact).
    """
    df = gold.join(classify(gold["question"]))
    keys = ["answer_format"] + (["distribution"] if "distribution" in df.columns else [])
    agg = df.groupby(keys)[["leak_cardinality", "leak_named_entity", "shortcut"]].agg(["sum", "mean"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.join(df.groupby(keys).size().rename("n")).reset_index()


def strata_report(scored: pd.DataFrame, *, answer_col: str = "answer") -> pd.DataFrame:
    """Accuracy, template-aware floor and MARGIN for each format × distribution × shortcut cell.

    ``scored`` is a per-question frame carrying ``question``, ``answer_format``, ``distribution``,
    ``correctness`` and the gold ``answer`` — i.e. a ``results.csv`` merged with the gold parquet.

    The floor is recomputed WITHIN each cell via ``frame.metrics.template_floor`` (RULES §1: never
    re-derived here). Cells below ``MIN_CELL_N`` are dropped rather than reported small.

    Returns one row per cell plus a ``gap`` column on the shortcut rows (margin_shortcut −
    margin_free) where both cells of a format×distribution survived. **No pooled row is produced,
    deliberately** — see the module docstring.
    """
    from frame.metrics import template_floor, template_of  # noqa: PLC0415

    need = {"question", "answer_format", "distribution", "correctness", answer_col}
    if missing := need - set(scored.columns):
        raise KeyError(f"strata_report needs {sorted(missing)}")

    df = scored.join(classify(scored["question"]))
    df = df.assign(_tmpl=df["question"].map(template_of))

    rows = []
    for (fmt, dist, sc), cell in df.groupby(["answer_format", "distribution", "shortcut"]):
        if len(cell) < MIN_CELL_N:
            continue
        acc = float(cell["correctness"].mean())
        floor = template_floor(cell, answer_col=answer_col, template_col="_tmpl")
        rows.append({
            "answer_format": fmt, "distribution": dist, "shortcut": bool(sc), "n": int(len(cell)),
            "accuracy": acc, "floor": floor, "margin": acc - floor,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["gap"] = float("nan")
    for (fmt, dist), g in out.groupby(["answer_format", "distribution"]):
        if set(g["shortcut"]) == {True, False}:
            sc_m = float(g.loc[g["shortcut"], "margin"].iloc[0])
            fr_m = float(g.loc[~g["shortcut"], "margin"].iloc[0])
            out.loc[(out.answer_format == fmt) & (out.distribution == dist)
                    & out.shortcut, "gap"] = sc_m - fr_m
    return out.sort_values(["answer_format", "distribution", "shortcut"]).reset_index(drop=True)


__all__ = ["CARDINALITY_RE", "MIN_CELL_N", "classify", "exposure", "strata_report"]
