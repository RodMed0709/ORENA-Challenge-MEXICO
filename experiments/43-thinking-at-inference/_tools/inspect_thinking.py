"""Rung 43 — ON-POD triage of a thinking run. Prints AGGREGATES ONLY.

🔴 Read this before changing what it prints.

`context/decisions/no-external-api-for-challenge-data.md` is SETTLED and BINDING under
the ORENA FOCUS DUA: challenge frames and **annotations** (questions, gold answers) never
leave the pod, "analysis" is named inside its scope, and the note is explicit that *the
PURPOSE is irrelevant — the act of transmission is the violation*. Model generations are
derived from the frames and are only meaningful read against the gold, so they are
treated the same way.

⇒ This script exists so the question *"is the thinking output any good?"* can be answered
**without a single generation leaving the pod**. Every line it prints is a count, a rate,
or a mean length. Its output is paste-safe by construction; a generation's text is not,
and must not be added here for convenience.

Run it after a probe arm finishes:

    python inspect_thinking.py <run_dir>        # dir holding predictions.json

What it answers, in the order the questions actually matter:

  1. Did `</think>` survive the decode at all?  (if not, nothing else means anything)
  2. How many traces were truncated before reaching it?
  3. Does parsing change what a scorer would see?  (chars before vs after)
  4. How much of the 5.0 s FRAME budget did the traces eat?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from think_parse import THINK_CLOSE, parse_stats, split_thinking  # noqa: E402


def main(run_dir: str) -> int:
    rd = Path(run_dir)
    pred = rd / "predictions.json"
    if not pred.exists():
        print(f"FATAL: no predictions.json in {rd}")
        return 1

    items = json.loads(pred.read_text())
    contents = [it.get("content", "") for it in items]
    lat = [float(it["latency"]) for it in items if it.get("latency") is not None]
    st = parse_stats(contents)

    print(f"== rung 43 thinking triage — {rd.name} ==")
    print(f"n generations                {st['n']}")
    print()
    print("-- 1. did the tag survive the decode? --")
    n_with_tag = sum(1 for c in contents if THINK_CLOSE in c)
    print(f"contain {THINK_CLOSE!r:11}        {n_with_tag}")
    if n_with_tag == 0:
        print("  🔴 ZERO. Either every trace overran the budget, or the tokenizer treats")
        print("     </think> as SPECIAL and skip_special_tokens=True deleted it. Those need")
        print("     different fixes. DO NOT score this run.")
        return 2

    print()
    print("-- 2. truncation --")
    print(f"truncated (no close tag)     {st['n_truncated']}  ({100 * st['truncated_rate']:.1f} %)")
    print(f"closed but empty answer      {st['n_closed_but_empty']}")

    print()
    print("-- 3. does parsing change what is scored? --")
    print(f"mean chars, RAW              {sum(len(c) for c in contents) / max(1, st['n']):.0f}")
    print(f"mean chars, PARSED answer    {st['mean_answer_chars']:.0f}")
    print(f"mean trace chars             {st['mean_trace_chars']:.0f}   (max {st['max_trace_chars']})")
    print("  ^ if RAW is large and PARSED is small, the unparsed reading is scoring the")
    print("    trace. That is the ~0 artifact, not a result about reasoning.")

    if lat:
        lat_sorted = sorted(lat)
        p99 = lat_sorted[min(len(lat_sorted) - 1, int(0.99 * len(lat_sorted)))]
        over = sum(1 for x in lat if x > 5.0)
        print()
        print("-- 4. the 5.0 s FRAME budget --")
        print(f"p99 latency                  {p99:.3f} s")
        print(f"max latency                  {max(lat):.3f} s")
        print(f"over 5.0 s (scored WRONG)    {over}  ({100 * over / len(lat):.1f} %)")
        if over:
            print("  ⚠️ Those are scored INCORRECT by the harness (enforce_latency=True).")
            print("     A loss here may be latency, not reasoning — see PLAN.md 'CONFOUNDED")
            print("     BY LATENCY'.")

    # Shape-only sanity: how many answers are short enough to plausibly BE an answer.
    # Still a count, still no text.
    parsed = [split_thinking(c).answer for c in contents]
    print()
    print(f"parsed answers under 40 chars {sum(1 for a in parsed if a and len(a) < 40)}"
          f" / {sum(1 for a in parsed if a)}")
    print("  ^ FRAME answers are short. A low share means the split is landing in the")
    print("    wrong place even though the tag was found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
