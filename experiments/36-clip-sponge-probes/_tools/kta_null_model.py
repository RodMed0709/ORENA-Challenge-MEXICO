"""Rung 36b, KT-A re-read: what is this probe's chance level, and does 0.5917 clear it?

Folder-private glue. Importable; a notebook cell calls ``main()``.

🔴 **This does not re-run KT-A. It re-reads the rows KT-A already wrote**
(``RESULTS_kta_rerank_full.csv``, n=529), against a null model that the original
pre-registration never declared. Zero GPU, zero model load.

## Why the original verdict is not readable

``RESULTS_kt_verdict_full.json`` reports ``gold_in_top2 = 0.5917`` against
``kill_line = 0.55``, and imports ``rung33_number_analogue = 0.456`` as the
reference. The threshold does not transfer between the two designs:

* **Rung 33** measured ``P(gold in top-2)`` over the model's **full token
  distribution**. Chance there is ~0 — the gold is one value among thousands.
* **KT-A** measures the same quantity over a **hand-built candidate list of 3-5
  options that contains the gold by construction** (``kt_probes.py``:
  ``BASE_CANDIDATES + [greedy, gold]``). Chance there is not 0. It is 2/n_cands.

So 0.55 means "well above chance" in rung 33 and something else entirely here.
Importing it verbatim compared a number to a threshold calibrated on a different
reference class.

## The two defensible nulls, and they disagree

**N1, unconditional.** The gold sits at a uniformly random rank among n_cands
candidates. ``P(rank <= 2) = 2/n_cands`` → 0.667 / 0.500 / 0.400 for lists of
3 / 4 / 5.

**N2, conditional on the greedy answer holding rank 1.** Measured, not assumed:
the arm's own greedy answer takes rank 1 in **456 of 529 rows (86.2 %)**, and on
these rows that answer is wrong by construction — they are A2's errors. If rank 1
is effectively spoken for, "gold in top-2" collapses to "gold leads the remaining
n-1", so ``P = 1/(n_cands - 1)`` → 0.500 / 0.333 / 0.250.

N1 is the conservative reading and N2 the generous one. The verdict flips between
them, which is the finding: **the probe was never calibrated, so its number cannot
be adjudicated in the direction the file claims.**

## What is reported

Per-stratum and pooled rates against both nulls, bootstrap CIs on the paired
delta (each row carries its own chance, so the delta is computed per row and then
resampled), and the stratum breakdown that the pooled figure hides.

⚠️ This module reports. It does not adjudicate — the same discipline
``kt_probes.py`` set for itself. The verdict field it writes names what the number
clears and what it does not, and stops there.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
ROWS_CSV = EXP / "RESULTS_kta_rerank_full.csv"
OUT_JSON = EXP / "RESULTS_kta_null.json"

#: The line KT-A pre-registered, kept so the re-read is against the original claim.
KILL_LINE = 0.55
#: What rung 33 measured, kept only to state that it is NOT comparable.
RUNG33_ANALOGUE = 0.456
BOOTSTRAP = 20_000
SEED = 20260813


def _load() -> list[dict]:
    with open(ROWS_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["gold_rank"] = int(r["gold_rank"])
        r["n_cands"] = int(r["n_cands"])
        if not 1 <= r["gold_rank"] <= r["n_cands"]:
            raise AssertionError(
                f"gold outside its own candidate list ({r['qID']}): rank "
                f"{r['gold_rank']} of {r['n_cands']}. The null model assumes the "
                "list contains the gold by construction, which is what kt_probes.py "
                "does; if that ever stops holding, every number below is wrong."
            )
    return rows


def _boot(deltas: list[float], seed: int = SEED, n: int = BOOTSTRAP) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(deltas)
    draws = sorted(sum(deltas[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return draws[int(0.025 * n)], draws[int(0.975 * n)]


def _rate(rows: list[dict], hit, chance) -> dict:
    """Observed rate, its per-row chance, and a bootstrap CI on the paired delta."""
    obs = sum(1.0 if hit(r) else 0.0 for r in rows) / len(rows)
    exp = sum(chance(r) for r in rows) / len(rows)
    deltas = [(1.0 if hit(r) else 0.0) - chance(r) for r in rows]
    lo, hi = _boot(deltas)
    return {"n": len(rows), "observed": obs, "chance": exp, "delta": obs - exp,
            "delta_ci95": [lo, hi], "delta_excludes_zero": lo > 0 or hi < 0}


def main() -> dict:
    rows = _load()
    top2 = lambda r: r["gold_rank"] <= 2                       # noqa: E731
    rank1 = lambda r: r["gold_rank"] == 1                      # noqa: E731
    n1 = lambda r: 2.0 / r["n_cands"]                          # noqa: E731
    n2 = lambda r: 1.0 / (r["n_cands"] - 1)                    # noqa: E731

    greedy_rank1 = sum(1 for r in rows if r["top1"].strip() == r["a2"].strip())

    strata = defaultdict(list)
    for r in rows:
        strata[r["n_cands"]].append(r)

    result = {
        "source": str(ROWS_CSV.relative_to(EXP.parent.parent)),
        "n": len(rows),
        "reruns_the_probe": False,
        "kill_line_as_preregistered": KILL_LINE,
        "rung33_analogue": RUNG33_ANALOGUE,
        "greedy_holds_rank1": {"n": greedy_rank1, "frac": greedy_rank1 / len(rows)},
        "gold_rank_histogram": dict(sorted(Counter(r["gold_rank"] for r in rows).items())),
        "pooled": {
            "top2_vs_N1_unconditional": _rate(rows, top2, n1),
            "top2_vs_N2_conditional": _rate(rows, top2, n2),
            "rank1_vs_uniform": _rate(rows, rank1, lambda r: 1.0 / r["n_cands"]),
        },
        "by_n_cands": {
            str(k): {
                "n": len(v),
                "top2_observed": sum(1 for r in v if top2(r)) / len(v),
                "chance_N1": 2.0 / k,
                "chance_N2": 1.0 / (k - 1),
                "delta_N1": sum(1 for r in v if top2(r)) / len(v) - 2.0 / k,
            }
            for k, v in sorted(strata.items())
        },
    }

    p = result["pooled"]
    below_n1 = [k for k, v in result["by_n_cands"].items() if v["delta_N1"] < 0]
    result["reading"] = {
        "clears_kill_line_on_its_face": p["top2_vs_N1_unconditional"]["observed"] > KILL_LINE,
        "beats_N1_unconditional": p["top2_vs_N1_unconditional"]["delta_excludes_zero"]
        and p["top2_vs_N1_unconditional"]["delta"] > 0,
        "beats_N2_conditional": p["top2_vs_N2_conditional"]["delta_excludes_zero"]
        and p["top2_vs_N2_conditional"]["delta"] > 0,
        "strata_below_N1_chance": below_n1,
        "rank1_below_chance": p["rank1_vs_uniform"]["delta"] < 0,
        "verdict": (
            "NOT READABLE AS PRE-REGISTERED. The 0.55 kill line was imported from rung 33, "
            "whose chance level is ~0 over the full token distribution; here the gold is in "
            "the candidate list by construction, so chance is 2/n_cands. Against the "
            "conservative null N1 the margin is small and at least one stratum sits BELOW "
            "chance; against the generous null N2 the effect is real and large. The two "
            "nulls disagree about the same number, and neither was declared before the run. "
            "The original 'LICENSED' verdict does not follow from these rows."
        ),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def report(result: dict | None = None) -> None:
    r = result or main()
    p = r["pooled"]
    print(f"KT-A re-read - n={r['n']}, no GPU, no re-run\n")
    print(f"greedy answer holds rank 1 in {r['greedy_holds_rank1']['n']}/{r['n']} rows "
          f"({r['greedy_holds_rank1']['frac']:.1%}) - this is what makes N2 arguable")
    print(f"gold rank histogram: {r['gold_rank_histogram']}\n")
    print(f"{'null':<28}{'observed':>10}{'chance':>10}{'delta':>10}   CI95")
    for key, label in (("top2_vs_N1_unconditional", "top-2 vs N1 (2/n)"),
                       ("top2_vs_N2_conditional", "top-2 vs N2 (1/(n-1))"),
                       ("rank1_vs_uniform", "rank-1 vs uniform (1/n)")):
        d = p[key]
        print(f"{label:<28}{d['observed']:>10.4f}{d['chance']:>10.4f}{d['delta']:>+10.4f}"
              f"   [{d['delta_ci95'][0]:+.4f}, {d['delta_ci95'][1]:+.4f}]")
    print(f"\n{'n_cands':>8}{'rows':>7}{'top-2':>9}{'N1':>8}{'N2':>8}{'d vs N1':>10}")
    for k, v in r["by_n_cands"].items():
        print(f"{k:>8}{v['n']:>7}{v['top2_observed']:>9.4f}{v['chance_N1']:>8.4f}"
              f"{v['chance_N2']:>8.4f}{v['delta_N1']:>+10.4f}")
    print(f"\nVERDICT: {r['reading']['verdict']}")


if __name__ == "__main__":
    report()
