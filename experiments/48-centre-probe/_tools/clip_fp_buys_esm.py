"""THE PENDING GATE: does suppressing spurious `Clip` buy exact-set-match?

`context/decisions/clip-is-inferred-from-phase.md` closes by demanding this test *before*
any training run or hour of annotation is spent on the clip front, and it has never run:

    "n=3 concludes nothing, but it is enough to demand a cheap test -- does fewer clip FPs
     buy exact-set-match? -- before spending a training run or an hour of human annotation"

The reason it is demanded: on the three platform anchors clip aggressiveness runs the WRONG
way. A2 is the LEAST aggressive with Clip (FP 0.870) and rung 42 beats it by 0.052.

This measures the CEILING, per question, not a cross-arm correlation over n=3. For every
`fo_class` question, take the model's own answer and delete `Clip` wherever `Clip` is in the
prediction and NOT in the gold. Then ask how many predictions become exact set matches that
were not. That number is the most a perfect clip-suppressor could ever buy on this corpus --
an oracle no trained model can beat, because it is handed the gold to decide what to delete.

Set equality is computed through the SDK's own `FOClass.verify`, never a hand-rolled
comparison (RULES §8b: the accepted vocabulary is the scorer's, not ours).
"""
from __future__ import annotations
import csv, importlib.util, sys, types
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The SDK's OWN parser and comparison, not a hand-rolled one. `FOClass.read()` returns a
# frozenset of canonical names and its docstring states "comparison is exact set equality",
# so exact-set-match here is literally what the scorer computes (RULES §8b).
from focus.data.formats import FOClass

FMT = FOClass()
CLIP = "Clip"


def as_set(text: str) -> frozenset[str] | None:
    """The SDK's canonical set, or None when `verify()` would raise (not comparable)."""
    try:
        return FMT.read(text or "")
    except Exception:
        return None


def main() -> int:
    print(f"{'arm':12} {'n_fo':>6} {'ESM now':>9} {'clipFP':>8} {'ESM oracle':>11} {'gain':>8} "
          f"{'of the FPs':>12}")
    print("-" * 76)
    rows_out = []
    for f in sorted(HERE.glob("r*.csv")):
        arm = f.stem
        n = esm = fp = fixed = fp_already = unparsable = 0
        for r in csv.DictReader(f.open()):
            if r["answer_format"] != "fo_class":
                continue
            n += 1
            g, p = as_set(r["ground_truth"]), as_set(r["our_answer"])
            if g is None or p is None:
                unparsable += 1
                continue
            match = (g == p)
            esm += match
            spurious = (CLIP in p) and (CLIP not in g)
            if not spurious:
                continue
            fp += 1
            if match:
                fp_already += 1          # cannot happen, but assert it rather than assume
            stripped = frozenset(x for x in p if x != CLIP)
            if stripped and stripped == g and not match:
                fixed += 1
        assert fp_already == 0, "a spurious Clip cannot coexist with an exact match"
        gain = fixed / n if n else 0.0
        print(f"{arm:12} {n:6d} {esm/n:9.4f} {fp/n:8.4f} {(esm+fixed)/n:11.4f} "
              f"{gain:+8.4f} {fixed:5d}/{fp:<6d}")
        rows_out.append(dict(arm=arm, n_fo_class=n, esm_now=round(esm/n, 4),
                             clip_fp_rate=round(fp/n, 4), esm_oracle=round((esm+fixed)/n, 4),
                             oracle_gain=round(gain, 4), fixed=fixed, clip_fps=fp,
                             unparsable=unparsable))
    with (HERE / "RESULTS_clip_fp_buys_esm.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    print("\nESM now      = exact set match on fo_class today")
    print("clipFP       = share of fo_class answers carrying a Clip the gold does not have")
    print("ESM oracle   = after deleting every spurious Clip, gold in hand")
    print("of the FPs   = how many of those spurious-Clip answers become exact once Clip goes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
