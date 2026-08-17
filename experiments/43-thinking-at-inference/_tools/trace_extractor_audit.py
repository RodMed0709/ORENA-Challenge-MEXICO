#!/usr/bin/env python3
"""
TASK 1 (blocking, zero GPU) — audit the "82.9 % of failed `number` traces contain the gold"
claim by supplying the denominator it is missing.

Runs ENTIRELY on UNAM. Reads only local files. Writes one aggregate JSON.
No challenge text is emitted — only counts, rates and integers.

Pre-registered verdict (written before the numbers, see session prompt):
  - median >= 2 candidates AND trivial rules near 1/k  -> DEAD BRANCH
  - 1 dominant candidate AND trivial rules high        -> proceed to task 2
"""
import json, os, re, sys, statistics
from collections import Counter

BASE = os.path.expanduser("~/storage/rung43")
THINK = os.path.join(BASE, "r43_ep23_think_uncapped")
NOTHINK = os.path.join(BASE, "r43_ep23_nothink")
GOLD = os.path.expanduser("~/storage/orena-data/heico/data/frame/test.parquet")
OUT = os.path.expanduser("~/storage/tmp/rod_RESULTS_trace_extractor_audit.json")

import pandas as pd

# ---------------------------------------------------------------- load
g = pd.read_parquet(GOLD)
g["qID"] = "heico__" + g["id"].astype(str)
gold = dict(zip(g["qID"], g["answer"]))
fmt = dict(zip(g["qID"], g["answer_format"]))

res = pd.read_csv(os.path.join(THINK, "results.csv"))
correct = dict(zip(res["qID"], res["correctness"].astype(bool)))

preds = json.load(open(os.path.join(THINK, "predictions.json")))
nothink = {r["qID"]: r["content"] for r in json.load(open(os.path.join(NOTHINK, "predictions.json")))}
res_nt = pd.read_csv(os.path.join(NOTHINK, "results.csv"))
correct_nt = dict(zip(res_nt["qID"], res_nt["correctness"].astype(bool)))

INT_RE = re.compile(r"(?<![\w.])(\d{1,3})(?![\w.])")
WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
WORD_RE = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.I)


def numbers_in(text, plausible_max):
    """STRICT: ordered integer tokens (standalone digits + number words), decimals excluded"""
    out = []
    for m in re.finditer(r"(?<![\w.])(\d{1,3})(?![\w.])|\b(" + "|".join(WORDS) + r")\b", text, re.I):
        if m.group(1) is not None:
            v = int(m.group(1))
        else:
            v = WORDS[m.group(2).lower()]
        if 0 <= v <= plausible_max:
            out.append(v)
    return out


def numbers_loose(text, plausible_max):
    """LOOSE: any digit run <= plausible_max, anywhere, decimals included.
    This is the extraction that reproduces the reported 82.9 % oracle recall."""
    return [int(x) for x in re.findall(r"\d+", text) if int(x) <= plausible_max]


def split_trace(content):
    """returns (trace_part, closed)"""
    if "</think>" in content:
        return content.split("</think>")[0], True
    return content, False


# gold vocabulary for `number` (defines "plausible")
num_qids = [q for q in gold if fmt.get(q) == "number"]
gold_vals = []
for q in num_qids:
    try:
        gold_vals.append(int(str(gold[q]).strip()))
    except ValueError:
        pass
PLAUSIBLE_MAX = max(gold_vals)
gold_hist = Counter(gold_vals)

rows = []
for r in preds:
    q = r["qID"]
    if fmt.get(q) != "number":
        continue
    try:
        gv = int(str(gold[q]).strip())
    except ValueError:
        continue
    trace, closed = split_trace(r["content"])
    seq = numbers_in(trace, PLAUSIBLE_MAX)
    loose_seq = numbers_loose(r["content"], PLAUSIBLE_MAX)
    distinct = sorted(set(seq))
    loose_distinct = sorted(set(loose_seq))
    cnt = Counter(seq)
    rule_last = seq[-1] if seq else None
    rule_mode = None
    if cnt:
        top = max(cnt.values())
        # tie-break: latest occurring among the tied modes
        tied = [v for v, c in cnt.items() if c == top]
        rule_mode = max(tied, key=lambda v: len(seq) - 1 - seq[::-1].index(v))
    rows.append(dict(
        qID=q, gold=gv, correct=bool(correct.get(q, False)), closed=closed,
        n_tok=len(seq), distinct=distinct, k=len(distinct),
        gold_in_trace=gv in distinct,
        k_loose=len(loose_distinct), gold_in_trace_loose=gv in loose_distinct,
        # a neighbour of the gold also sitting in the trace is what makes the
        # candidate set ambiguous rather than merely large ([[counting-has-two-failure-modes]])
        neighbour_in_trace=((gv - 1) in distinct or (gv + 1) in distinct),
        rule_last=rule_last, rule_mode=rule_mode,
        parsed=r.get("answer"), n_gen=r.get("n_gen_tokens"),
        correct_nt=bool(correct_nt.get(q, False)),
    ))

N = len(rows)
fail = [r for r in rows if not r["correct"]]
ok = [r for r in rows if r["correct"]]


def pct(a, b):
    return round(a / b, 4) if b else None


def qs(vals):
    vals = sorted(vals)
    if not vals:
        return {}
    n = len(vals)
    return dict(min=vals[0], p25=vals[int(.25 * n)], median=statistics.median(vals),
                p75=vals[int(.75 * n)], p90=vals[int(.90 * n)], max=vals[-1],
                mean=round(sum(vals) / n, 3))


def rule_acc(subset, key):
    hit = sum(1 for r in subset if r[key] is not None and r[key] == r["gold"])
    return dict(n=len(subset), hits=hit, acc=pct(hit, len(subset)))


def random_pick_acc(subset):
    """expected accuracy of an extractor choosing uniformly among the distinct candidates"""
    tot = 0.0
    for r in subset:
        if r["k"] and r["gold_in_trace"]:
            tot += 1.0 / r["k"]
    return round(tot / len(subset), 4) if subset else None


out = {
    "what": "Task 1 — denominator audit of the trace-extractor claim. Zero GPU. "
            "Source: rung 43 ep2/3 thinking arm, uncapped, 4000 heico-test questions, "
            "restricted to answer_format == number.",
    "source_files": {
        "traces": os.path.join(THINK, "predictions.json"),
        "judge": os.path.join(THINK, "results.csv"),
        "control_nothink": os.path.join(NOTHINK, "results.csv"),
        "gold": GOLD,
    },
    "population": {
        "n_number_questions": N,
        "n_failed_think": len(fail),
        "n_correct_think": len(ok),
        "acc_think_number": pct(len(ok), N),
        "acc_nothink_number": pct(sum(1 for r in rows if r["correct_nt"]), N),
        "gold_vocab_max": PLAUSIBLE_MAX,
        "gold_hist": dict(sorted(gold_hist.items())),
        "closed_think_tag": pct(sum(1 for r in rows if r["closed"]), N),
    },

    # (0) does the 82.9 % reproduce, and under which extraction?
    "A0_oracle_recall": {
        "strict_pre_close_plus_words": {
            "on_failed": dict(n=len(fail), hits=sum(1 for r in fail if r["gold_in_trace"]),
                              recall=pct(sum(1 for r in fail if r["gold_in_trace"]), len(fail))),
            "on_correct": dict(n=len(ok), hits=sum(1 for r in ok if r["gold_in_trace"]),
                               recall=pct(sum(1 for r in ok if r["gold_in_trace"]), len(ok))),
            "on_all": dict(n=N, hits=sum(1 for r in rows if r["gold_in_trace"]),
                           recall=pct(sum(1 for r in rows if r["gold_in_trace"]), N)),
        },
        "loose_any_digit_full_content": {
            "on_failed": dict(n=len(fail), hits=sum(1 for r in fail if r["gold_in_trace_loose"]),
                              recall=pct(sum(1 for r in fail if r["gold_in_trace_loose"]), len(fail))),
            "on_all": dict(n=N, hits=sum(1 for r in rows if r["gold_in_trace_loose"]),
                           recall=pct(sum(1 for r in rows if r["gold_in_trace_loose"]), N)),
            "median_k": statistics.median([r["k_loose"] for r in fail]),
            "note": "THIS is the extraction that reproduces the reported ~83 % — and it is also "
                    "the extraction with the LARGEST candidate set. The recall and the ambiguity "
                    "are the same number seen from two sides.",
        },
        "note": "recall on ALL number questions is the hard CEILING of any trace extractor",
        "neighbour_of_gold_also_in_trace_failed": pct(
            sum(1 for r in fail if r["neighbour_in_trace"]), len(fail)),
    },

    # (a) how many plausible candidates coexist
    "Aa_candidates_per_trace": {
        "failed": qs([r["k"] for r in fail]),
        "correct": qs([r["k"] for r in ok]),
        "all": qs([r["k"] for r in rows]),
        "n_tokens_failed": qs([r["n_tok"] for r in fail]),
        "k_histogram_failed": dict(sorted(Counter(r["k"] for r in fail).items())),
        "random_pick_acc_failed": random_pick_acc(fail),
        "random_pick_acc_all": random_pick_acc(rows),
    },

    # (b) trivial rules on the same failures
    "Ab_trivial_rules": {
        "failed": {"last_before_close": rule_acc(fail, "rule_last"),
                   "most_frequent": rule_acc(fail, "rule_mode")},
        "all_number": {"last_before_close": rule_acc(rows, "rule_last"),
                       "most_frequent": rule_acc(rows, "rule_mode")},
        "comparators": {
            "think_arm_as_shipped": pct(len(ok), N),
            "nothink_arm_control": pct(sum(1 for r in rows if r["correct_nt"]), N),
        },
    },

    # (c) damage on what the model ALREADY got right
    "Ac_flip_rate_on_correct": {
        "last_before_close": dict(
            n=len(ok),
            breaks=sum(1 for r in ok if r["rule_last"] != r["gold"]),
            break_rate=pct(sum(1 for r in ok if r["rule_last"] != r["gold"]), len(ok))),
        "most_frequent": dict(
            n=len(ok),
            breaks=sum(1 for r in ok if r["rule_mode"] != r["gold"]),
            break_rate=pct(sum(1 for r in ok if r["rule_mode"] != r["gold"]), len(ok))),
    },
}

# net effect table: rule applied to every number question vs each arm
for name, key in (("last_before_close", "rule_last"), ("most_frequent", "rule_mode")):
    fixed = sum(1 for r in fail if r[key] == r["gold"])
    broken = sum(1 for r in ok if r[key] != r["gold"])
    out["Ab_trivial_rules"].setdefault("net", {})[name] = dict(
        fixed=fixed, broken=broken, net=fixed - broken,
        delta_vs_think=round((fixed - broken) / N, 4))

json.dump(out, open(OUT, "w"), indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
print("\nWROTE", OUT, file=sys.stderr)
