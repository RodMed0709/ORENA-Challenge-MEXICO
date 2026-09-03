"""The other half of the gate: WHY does deleting the spurious Clip not fix the answer?

`clip_fp_buys_esm.py` measures the ceiling. This asks what the residue is made of, because
that is the part that tells us where the mass actually sits. For every fo_class answer that
carries a spurious `Clip` and is STILL not an exact match once `Clip` is deleted, classify
the remaining disagreement between the stripped prediction and the gold.
"""
import csv, collections
from pathlib import Path
from focus.data.formats import FOClass

FMT, CLIP, HERE = FOClass(), "Clip", Path(__file__).resolve().parent


def rd(t):
    try: return FMT.read(t or "")
    except Exception: return None


agg = collections.Counter(); tot = 0
per_arm = {}
for f in sorted(HERE.glob("r*.csv")):
    c = collections.Counter()
    for r in csv.DictReader(f.open()):
        if r["answer_format"] != "fo_class":
            continue
        g, p = rd(r["ground_truth"]), rd(r["our_answer"])
        if g is None or p is None or CLIP not in p or CLIP in g:
            continue
        s = frozenset(x for x in p if x != CLIP)
        if s == g:
            c["FIXED by removing Clip"] += 1; continue
        if not s:
            c["Clip was the WHOLE answer, gold is not empty"] += 1
        elif s < g:
            c[f"still MISSING {len(g - s)} class(es)"] += 1
        elif s > g:
            c["still has EXTRA class(es) besides Clip"] += 1
        else:
            c["disjoint / swapped classes"] += 1
    per_arm[f.stem] = c; agg += c; tot += sum(c.values())

print(f"{'residue of every spurious-Clip answer, all 8 arms pooled':52} {'n':>7} {'share':>7}")
print("-" * 68)
for k, v in agg.most_common():
    print(f"  {k:50} {v:7d} {v/tot:6.1%}")
print(f"  {'TOTAL spurious-Clip answers':50} {tot:7d}")
