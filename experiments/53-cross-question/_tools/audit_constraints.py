"""Rung 53 — size the cross-question constraint, and audit whether the golds even agree.

[[cross-question-constraints]] (Leo, 2026-07-28, never used) measured that the organizers ask
several questions about ONE frame and that those answers constrain each other: 979 frames carry
both a `number` and an `fo_class` question, 495 pair a per-class count with the total.

Before minting a single training row this asks two questions the note did not:

1. **How much of that survives inside the corpus we actually train on** (rung 47's 14,415 rows),
   as opposed to across all 20,000 public questions?
2. 🔴 **Do the golds AGREE?** Where a frame carries `fo_class` and *"how many different foreign
   object classes"*, the gold set's size and the gold count are the same quantity written twice.
   Any disagreement is **label noise we can see for free**, and it would also be an upper bound
   on what a consistency objective could ever be trained to satisfy. Training a model to satisfy
   a constraint its own labels break is how a rung produces a confident null.

Zero GPU. Formats are identified by `metrics.template_of` against the eval's own labels — the
same map rung 50 arm A used, which was verified to convert nothing it should not.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RE_CLASSES = re.compile(r"different foreign object class", re.I)
RE_INSTANCES = re.compile(r"different foreign object instance|how many foreign object", re.I)
RE_PERCLASS = re.compile(r"how many (\w+)", re.I)


def build_format_map(inspect_csv: Path, metrics) -> dict:
    import pandas as pd

    ev = pd.read_csv(inspect_csv)
    seen: dict[str, set] = {}
    for q, f in zip(ev.question, ev.answer_format):
        seen.setdefault(metrics.template_of(q), set()).add(f)
    bad = {t: v for t, v in seen.items() if len(v) > 1}
    if bad:
        raise AssertionError(f"template map is not a function: {list(bad)[:2]}")
    return {t: next(iter(v)) for t, v in seen.items()}


def index_frames(corpus: Path, fmt_map: dict, metrics) -> dict:
    """frame image path -> what the corpus says about it, across all its questions."""
    frames: dict[str, dict] = defaultdict(dict)
    for line in Path(corpus).open(encoding="utf-8"):
        r = json.loads(line)
        q = r["messages"][-2]["content"].replace("<image>", "").strip()
        a = r["messages"][-1]["content"].strip()
        imgs = r.get("images") or []
        if not imgs:
            continue
        img = imgs[0]
        fmt = fmt_map.get(metrics.template_of(q))
        if fmt == "fo_class":
            frames[img]["fo"] = a
        elif a.isdigit():
            n = int(a)
            if RE_CLASSES.search(q):
                frames[img]["n_class"] = n
            elif RE_INSTANCES.search(q):
                frames[img]["n_inst"] = n
            else:
                m = RE_PERCLASS.search(q)
                if m:
                    frames[img].setdefault("perclass", {})[m.group(1).lower()] = n
    return frames


def audit(frames: dict, metrics, valid_lower: dict) -> dict:
    have_fo = {k: v for k, v in frames.items() if "fo" in v}
    pair_class = {k: v for k, v in have_fo.items() if "n_class" in v}
    pair_inst = {k: v for k, v in have_fo.items() if "n_inst" in v}
    pair_pc = {k: v for k, v in frames.items() if "perclass" in v and "n_inst" in v}

    agree = disagree = unparsed = 0
    examples, deltas = [], Counter()
    for v in pair_class.values():
        s = metrics.read_fo_class(v["fo"], valid_lower)
        if s is None:
            unparsed += 1
            continue
        n = 0 if s == frozenset({"None"}) else len(s)
        if n == v["n_class"]:
            agree += 1
        else:
            disagree += 1
            deltas[n - v["n_class"]] += 1
            if len(examples) < 6:
                examples.append({"fo_gold": v["fo"], "set_size": n,
                                 "n_class_gold": v["n_class"]})

    # per-class counts must not exceed the total instances on the same frame
    pc_ok = pc_bad = 0
    for v in pair_pc.values():
        if sum(v["perclass"].values()) <= v["n_inst"]:
            pc_ok += 1
        else:
            pc_bad += 1

    return {
        "frames_in_corpus": len(frames),
        "frames_with_fo_class": len(have_fo),
        "fo_class_AND_n_classes": len(pair_class),
        "fo_class_AND_n_instances": len(pair_inst),
        "perclass_AND_n_instances": len(pair_pc),
        "gold_consistent": agree,
        "gold_inconsistent": disagree,
        "gold_unparsed": unparsed,
        "inconsistency_rate": round(disagree / max(agree + disagree, 1), 4),
        "disagreement_sizes": dict(sorted(deltas.items())),
        "examples": examples,
        "perclass_within_total": pc_ok,
        "perclass_exceeds_total": pc_bad,
    }


def run(corpus: Path, inspect_csv: Path, out: Path, metrics) -> dict:
    fmt = build_format_map(inspect_csv, metrics)
    valid = {n.lower(): n for n in tuple(metrics._load_fotype().names())}
    frames = index_frames(corpus, fmt, metrics)
    res = audit(frames, metrics, valid)
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS_constraint_audit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res
