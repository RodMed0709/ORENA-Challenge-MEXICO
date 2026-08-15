"""Rung 43 — split a thinking generation into its trace and its ANSWER.

🔴 Why this file has to exist, and why it is not a second variable.

With `enable_thinking=True` the chat template leaves the `<think>` block OPEN, so the
model emits `…reasoning…</think>\\n\\nanswer`. Nothing in the pipeline splits that:
`screen_engine.predict_samples` decodes the generation and returns
`out[: answer_char_cap]` — the raw string. So the scorer receives the *reasoning* and
the answer is never in what gets judged.

That is not a small penalty, it is a FLOOR: rung 23a's first bf16 smoke scored **0/24**
this exact way, with every answer looking like *"The user wants me to identify… 1.
**Analyze the image:**"*, and the engine's own docstring still carries that scar. Its
fix was to SUPPRESS thinking. This rung turns it back on, so it inherits the defect
unless the answer is extracted.

⇒ **Extraction is constitutive of the arm, not a treatment applied to it.** Without it
`enable_thinking=True` does not measure reasoning; it measures whether the first 300
characters of a chain of thought happen to collide with the gold string, which is ~0.
Scoring that would produce a confident, CI-excludes-zero NO-GO for a parser we never
wrote — a format failure read as incapacity, which is precisely what G-INFER exists to
prevent.

📌 `eval_arm.build_baseline_config` RAISES on a non-None `answer_postprocess`, and that
gate is right for rung 40, where a post-processor would have been a second variable. We
do NOT fight it: the run saves the RAW generation to `predictions.json`, and both
readings are derived from that file OFFLINE. One generation, two scorings, no GPU spent
twice and no gate bypassed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"


@dataclass(frozen=True)
class ThinkSplit:
    """One generation, taken apart. `answer` is "" exactly when `closed` is False."""

    trace: str
    answer: str
    closed: bool


def split_thinking(text: str) -> ThinkSplit:
    """Split on the LAST `</think>`; everything after it is the answer.

    🔻 Last and not first, and the first draft of this function had it backwards. The
    reasoning for `find` was that a trace quoting the tag would make a right-split
    swallow the answer. A unit case shows the opposite: on
    `"…I should end with </think> now.</think>\\n\\nscissors"`, `find` yields
    `"now.</think>\\n\\nscissors"` and `rfind` yields `"scissors"`.

    `rfind` is only wrong if the ANSWER itself contains the tag. FRAME answers are `"2"`,
    `"scissors"`, `"yes"` — a closing think tag inside one is not a case this corpus has.
    A trace that mentions the tag while reasoning is the plausible case, and `rfind`
    survives it.

    A generation with no `</think>` ran out of `max_new_tokens` mid-trace and therefore
    never emitted an answer at all. That is returned as `closed=False` with an EMPTY
    answer — deliberately, so it scores as wrong (it IS wrong: nothing was answered) and
    is counted separately as truncation rather than blamed on reasoning.
    """
    if not isinstance(text, str):
        return ThinkSplit(trace="", answer="", closed=False)
    # An "Inference Error: …" sentinel is not a trace and must survive untouched, or
    # G-INFER (frame.run:243) can no longer see the failures it exists to catch.
    if text.startswith("Inference Error:"):
        return ThinkSplit(trace="", answer=text, closed=True)
    idx = text.rfind(THINK_CLOSE)
    if idx == -1:
        return ThinkSplit(trace=text, answer="", closed=False)
    return ThinkSplit(
        trace=text[:idx].removeprefix(THINK_OPEN).strip(),
        answer=text[idx + len(THINK_CLOSE):].strip(),
        closed=True,
    )


def parse_stats(contents: list[str]) -> dict:
    """AGGREGATES ONLY — counts and lengths, never a generation's text.

    🔴 This shape is not incidental. `context/decisions/no-external-api-for-challenge-data.md`
    is SETTLED and BINDING: challenge frames AND annotations stay on-pod, and "analysis"
    is named in its scope. Model outputs are derived from the frames and are read against
    the gold, so they do not leave the pod either. Numbers do. Every field below is a
    count, a rate or a mean length — paste-safe by construction.
    """
    splits = [split_thinking(c) for c in contents]
    n = len(splits)
    closed = [s for s in splits if s.closed]
    empty_answers = sum(1 for s in closed if not s.answer)
    return {
        "n": n,
        "n_closed": len(closed),
        "n_truncated": n - len(closed),          # never reached `</think>`
        "truncated_rate": (n - len(closed)) / n if n else float("nan"),
        "n_closed_but_empty": empty_answers,     # closed the block, said nothing after
        "mean_trace_chars": (sum(len(s.trace) for s in splits) / n) if n else float("nan"),
        "max_trace_chars": max((len(s.trace) for s in splits), default=0),
        "mean_answer_chars": (sum(len(s.answer) for s in closed) / len(closed))
        if closed else float("nan"),
    }


def assert_tag_survived(contents: list[str]) -> None:
    """RAISE if not one generation contains `</think>` (RULES §7: gates raise).

    Two very different failures produce zero closes and they must not be confused:

    1. Every trace overran `max_new_tokens` — a budget problem, fix by raising it.
    2. `</think>` is a SPECIAL token in this tokenizer, so `skip_special_tokens=True`
       in `screen_engine.predict_samples` deleted it from the decoded string. Then the
       split point does not exist in the text at all and no budget fixes it — the engine
       has to decode with `skip_special_tokens=False`, or split on the token ids.

    Unverified as of writing: it needs the real tokenizer, which lives on the pod. This
    gate is how we find out on 20 smoke questions instead of after a full arm.
    """
    if contents and not any(THINK_CLOSE in c for c in contents):
        raise AssertionError(
            f"not one of {len(contents)} generations contains {THINK_CLOSE!r}. "
            "Either every trace was truncated (raise max_new_tokens / answer_char_cap), "
            "or the tokenizer treats it as a special token and skip_special_tokens=True "
            "stripped it (decode with skip_special_tokens=False). These need different "
            "fixes — do not score this run."
        )


def load_contents(predictions_json: str | Path) -> tuple[list[str], list[str]]:
    """Return (qIDs, raw contents) from a run's `predictions.json`, in file order."""
    items = json.loads(Path(predictions_json).read_text())
    return [it["qID"] for it in items], [it.get("content", "") for it in items]


def write_parsed_predictions(src_json: str | Path, dst_json: str | Path) -> dict:
    """Copy `predictions.json` with every `content` replaced by its extracted answer.

    Returns the aggregate stats. The destination is scored by the SAME evaluator the raw
    file is (RULES §EVAL rule 1 — extend the module, never reimplement scoring beside
    it); nothing here computes a metric.
    """
    items = json.loads(Path(src_json).read_text())
    contents = [it.get("content", "") for it in items]
    assert_tag_survived(contents)
    for it in items:
        it["content"] = split_thinking(it.get("content", "")).answer
    Path(dst_json).parent.mkdir(parents=True, exist_ok=True)
    Path(dst_json).write_text(json.dumps(items, indent=2))
    stats = parse_stats(contents)
    log.info("parsed %d predictions -> %s | truncated %d (%.1f%%)",
             stats["n"], dst_json, stats["n_truncated"], 100 * stats["truncated_rate"])
    return stats
