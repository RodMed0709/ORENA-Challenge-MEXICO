"""Rung 50 arm A — rewrite `fo_class` training targets to state the set SIZE first.

`"Clip, Sponge"` becomes `"2: Clip, Sponge"`. Nothing else in the corpus moves: same rows, same
order, same images, same prompts, and every non-`fo_class` answer byte-identical.

🔴 **Which rows are `fo_class` is decided by the QUESTION TEMPLATE, never by guessing from the
answer.** `metrics.template_of` on the 6,252-question eval yields 188 templates and **zero** that
map to more than one `answer_format`, so the map is a function. Applied to the 14,415-row corpus
it covers 75.3 % of rows; the uncovered remainder is paraphrased and minted `How many …`
questions, and **not one of them has an answer that parses as a class set** — measured, so the
map cannot silently convert a `number` row. A heuristic on the answer would have: an
`open_ended` answer of exactly `"Sponge"` reads as a class set.

At inference `strip_prefix` removes the `N: ` and the container emits the string shape the SDK
already scores. A prefix the model gets WRONG is never used to truncate or pad the list — the
disagreement between the stated N and the list length is counted and reported instead, and it is
the most informative number this arm can produce.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PREFIX_RE = re.compile(r"^\s*(\d+)\s*:\s*")


def format_map(inspect_csv: Path, metrics) -> dict[str, str]:
    """`template_of(question)` → `answer_format`, from an eval that carries the label.

    RAISES if any template maps to more than one format: a map that is not a function would
    convert rows of the wrong kind and the arm would be measuring the converter.
    """
    import pandas as pd

    ev = pd.read_csv(inspect_csv)
    seen: dict[str, set] = {}
    for q, f in zip(ev.question, ev.answer_format):
        seen.setdefault(metrics.template_of(q), set()).add(f)
    ambiguous = {t: v for t, v in seen.items() if len(v) > 1}
    if ambiguous:
        raise AssertionError(
            f"{len(ambiguous)} template(s) map to several answer_formats, e.g. "
            f"{list(ambiguous.items())[:2]}"
        )
    return {t: next(iter(v)) for t, v in seen.items()}


def add_prefix(answer: str, n: int) -> str:
    return f"{n}: {answer}"


def strip_prefix(text: str) -> str:
    """Remove a leading `N: `. Idempotent, and a no-op on a string that has none."""
    return PREFIX_RE.sub("", str(text), count=1).strip()


def stated_n(text: str) -> int | None:
    m = PREFIX_RE.match(str(text))
    return int(m.group(1)) if m else None


def build(src: Path, dst: Path, fmt_map: dict, metrics, valid_lower: dict) -> dict:
    """Write the arm-A corpus and run gates A-G1 … A-G3. RAISES on any failure."""
    rows = [json.loads(line) for line in Path(src).open(encoding="utf-8")]
    out, card, n_fo, unconverted = [], Counter(), 0, 0
    risky = 0

    for r in rows:
        q = r["messages"][-2]["content"].replace("<image>", "").strip()
        a = r["messages"][-1]["content"]
        fmt = fmt_map.get(metrics.template_of(q))
        gold = metrics.read_fo_class(a, valid_lower)

        # A-G2 — an uncovered row whose answer LOOKS like a class set is the failure mode the
        # template map exists to prevent. Count it; a non-zero count kills the build.
        if fmt is None and gold is not None:
            risky += 1

        if fmt == "fo_class":
            n_fo += 1
            if gold is None:
                unconverted += 1
            elif PREFIX_RE.match(a):
                raise AssertionError(
                    f"answer already carries an N: prefix, stripping would corrupt it: {a!r}"
                )
            else:
                n = 0 if gold == frozenset({"None"}) else len(gold)
                card[n] += 1
                r = json.loads(json.dumps(r))  # never mutate the source row in place
                r["messages"][-1]["content"] = add_prefix(a, n)
        out.append(r)

    if risky:
        raise AssertionError(
            f"A-G2 FAILED: {risky} uncovered rows have answers that parse as a class set"
        )
    if unconverted:
        raise AssertionError(
            f"A-G2 FAILED: {unconverted} fo_class rows whose gold does not parse"
        )

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with Path(dst).open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    gates = round_trip(src, dst, fmt_map, metrics, valid_lower)
    gates.update({"n_rows": len(out), "n_fo_class": n_fo,
                  "cardinality": dict(sorted(card.items()))})
    return gates


def round_trip(src: Path, dst: Path, fmt_map: dict, metrics, valid_lower: dict) -> dict:
    """A-G1 (round-trip identity) and A-G3 (nothing else moved). RAISES on any mismatch."""
    a_rows = [json.loads(line) for line in Path(src).open(encoding="utf-8")]
    b_rows = [json.loads(line) for line in Path(dst).open(encoding="utf-8")]
    if len(a_rows) != len(b_rows):
        raise AssertionError(f"A-G3 FAILED: {len(a_rows)} rows in, {len(b_rows)} out")

    changed = bad_round_trip = other_changed = 0
    for a, b in zip(a_rows, b_rows):
        aa, bb = a["messages"][-1]["content"], b["messages"][-1]["content"]
        if a.get("images") != b.get("images") or a["messages"][:-1] != b["messages"][:-1]:
            other_changed += 1
            continue
        if aa == bb:
            continue
        changed += 1
        # A-G1 — the ONLY licensed difference is a prefix that strips back to the original set.
        if strip_prefix(bb) != aa.strip():
            bad_round_trip += 1
            continue
        if metrics.read_fo_class(strip_prefix(bb), valid_lower) != metrics.read_fo_class(
                aa, valid_lower):
            bad_round_trip += 1

    if other_changed:
        raise AssertionError(f"A-G3 FAILED: {other_changed} rows changed outside the answer")
    if bad_round_trip:
        raise AssertionError(f"A-G1 FAILED: {bad_round_trip} rows do not round-trip")
    return {"n_changed": changed, "n_bad_round_trip": 0, "n_rows_touched_elsewhere": 0}


def prefix_disagreement(answers) -> dict:
    """Eval-time diagnostic: does the model's stated N match the list it then produces?

    Not a scoring quantity — `strip_prefix` is what feeds the scorer, and the prefix never
    truncates or pads. This exists because "the model says 3 and lists 2" is the single most
    direct evidence about whether the cardinality commitment took.
    """
    stated, agree, missing = Counter(), 0, 0
    total = 0
    for raw in answers:
        n = stated_n(raw)
        if n is None:
            missing += 1
            continue
        total += 1
        stated[n] += 1
        listed = len([p for p in strip_prefix(raw).split(",") if p.strip()])
        agree += int(listed == n)
    return {"n_with_prefix": total, "n_without_prefix": missing,
            "agreement": round(agree / total, 4) if total else None,
            "stated_hist": dict(sorted(stated.items()))}
