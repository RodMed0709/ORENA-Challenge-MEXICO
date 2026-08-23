"""Rung 53b — open the 24 discordant frames and ask WHY the two golds disagree.

The audit found 24 of 218 frames where the `fo_class` gold names fewer classes than the
*"how many different foreign object classes"* gold counts — and **all 24 in that direction**.
Noise scatters; this does not. Two explanations survive and they have opposite consequences:

* **SCOPE** — the two templates ask different things (e.g. one restricted to a class list the
  other is not). Then nothing is wrong with the labels and the constraint is simply not `=`.
* **UNDER-LISTING** — the `fo_class` gold omits classes the counting gold counts. Then the
  target for **42.8 % of the eval** is systematically short, and every `fo_class` number this
  campaign has produced is measured against an incomplete gold.

This dumps the exact question pair, both golds, and the frame path, so the question can be
settled by reading the templates and then by opening the images — not by argument.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from audit_constraints import (RE_CLASSES, RE_INSTANCES, RE_PERCLASS,  # noqa: F401
                               build_format_map)


def collect(corpus: Path, fmt_map: dict, metrics, valid_lower: dict) -> list[dict]:
    """Every frame carrying both templates, with the raw questions kept verbatim."""
    per_frame: dict[str, dict] = {}
    for line in Path(corpus).open(encoding="utf-8"):
        r = json.loads(line)
        q = r["messages"][-2]["content"].replace("<image>", "").strip()
        a = r["messages"][-1]["content"].strip()
        imgs = r.get("images") or []
        if not imgs:
            continue
        d = per_frame.setdefault(imgs[0], {"frame": imgs[0]})
        fmt = fmt_map.get(metrics.template_of(q))
        if fmt == "fo_class":
            d["fo_q"], d["fo_a"] = q, a
        elif a.isdigit() and RE_CLASSES.search(q):
            d["cnt_q"], d["cnt_a"] = q, int(a)

    out = []
    for d in per_frame.values():
        if "fo_a" not in d or "cnt_a" not in d:
            continue
        s = metrics.read_fo_class(d["fo_a"], valid_lower)
        if s is None:
            continue
        n = 0 if s == frozenset({"None"}) else len(s)
        d["set_size"] = n
        d["delta"] = n - d["cnt_a"]
        out.append(d)
    return out


def run(corpus: Path, inspect_csv: Path, out: Path, metrics) -> dict:
    fmt = build_format_map(inspect_csv, metrics)
    valid = {n.lower(): n for n in tuple(metrics._load_fotype().names())}
    rows = collect(corpus, fmt, metrics, valid)
    bad = [r for r in rows if r["delta"] != 0]
    good = [r for r in rows if r["delta"] == 0]

    # Does the DISAGREEMENT track the template wording? If the discordant frames use a
    # different fo_class template than the concordant ones, it is SCOPE, not under-listing.
    def tmpl_mix(rs, key):
        return Counter(metrics.template_of(r[key]) for r in rs).most_common(4)

    res = {
        "n_pairs": len(rows), "n_discordant": len(bad),
        "fo_templates_in_DISCORDANT": tmpl_mix(bad, "fo_q"),
        "fo_templates_in_CONCORDANT": tmpl_mix(good, "fo_q"),
        "cnt_templates_in_DISCORDANT": tmpl_mix(bad, "cnt_q"),
        "cnt_templates_in_CONCORDANT": tmpl_mix(good, "cnt_q"),
        "rows": [{k: r[k] for k in ("fo_q", "fo_a", "cnt_q", "cnt_a", "set_size", "delta",
                                    "frame")} for r in bad],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS_discordant_pairs.json").write_text(json.dumps(res, indent=1))

    print(f"pairs={len(rows)}  discordant={len(bad)}")
    print("\nfo_class TEMPLATES — discordant vs concordant")
    for t, n in res["fo_templates_in_DISCORDANT"]:
        print(f"  DIS  {n:4d}  {t[:88]}")
    for t, n in res["fo_templates_in_CONCORDANT"]:
        print(f"  CON  {n:4d}  {t[:88]}")
    print("\ncount TEMPLATES — discordant vs concordant")
    for t, n in res["cnt_templates_in_DISCORDANT"]:
        print(f"  DIS  {n:4d}  {t[:88]}")
    for t, n in res["cnt_templates_in_CONCORDANT"]:
        print(f"  CON  {n:4d}  {t[:88]}")
    print("\nfirst 6 discordant pairs")
    for r in bad[:6]:
        print(f"  fo  : {r['fo_a']!r} (size {r['set_size']})")
        print(f"  cnt : {r['cnt_a']}   Q: {r['cnt_q'][:80]}")
        print(f"  img : {Path(r['frame']).name}")
        print()
    return res
