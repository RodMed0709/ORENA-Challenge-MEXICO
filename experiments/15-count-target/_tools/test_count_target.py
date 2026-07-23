"""Offline proof of rung 15's parser and rewrite gates. No GPU, no swift, no decord.

Run it from the experiment dir (or from a notebook cell):

    python _tools/test_count_target.py

Why this file exists at all (CONSTITUTION §VIII.2 / §IX.3): the parser is the gate this
rung stands on. If it does not round-trip 100 % of the training targets, every `number`
number the rung produces is meaningless — and that property is fully checkable without a
GPU, so it must be checked before one is paid for. It lives in this experiment's
folder-private ``_tools/`` because it is tied to this step; there is no top-level
``tests/`` in this repo, ever (user rule, BINDING).

The template inventory is not hand-written: it is read from the data card's committed
tables (``experiments/08-data-card/tables/templates_{train,val}.csv``), so the regex is
tested against the REAL question strings — including the open-ended trap
``"How many radiopaque clips are visible in this video? Please provide a single integer."``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
for p in (EXP / "_models", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

from count_target import (  # noqa: E402
    SAFE_ANSWER,
    STATUS_BARE_INT,
    STATUS_MALFORMED,
    STATUS_PASSTHROUGH,
    STATUS_SALVAGED,
    STATUS_STRUCTURED,
    RowSpec,
    assert_nonnumber_lines_identical,
    assert_parser_never_raises,
    assert_targets_round_trip,
    count_label,
    is_count_question,
    parse_count,
    render_count_target,
    rewrite_train_lines,
)

TABLES = REPO / "experiments" / "08-data-card" / "tables"
# Golds observed in the data: the `number` templates carry up to 12 distinct answers
# (templates_val.csv n_distinct_answers = 11 and 12), and 0 is exercised too because
# Number.verify accepts it and a per-class count of zero is representable.
GOLDS = list(range(0, 13))


def _templates(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def test_regex_matches_exactly_the_number_templates() -> None:
    for name in ("templates_train.csv", "templates_val.csv"):
        d = _templates(name)
        d["hit"] = d["template"].map(is_count_question)
        missed = d[(d.answer_format == "number") & ~d.hit]
        extra = d[(d.answer_format != "number") & d.hit]
        assert missed.empty, f"{name}: `number` templates not matched:\n{missed[['template']]}"
        assert extra.empty, f"{name}: non-`number` templates matched:\n{extra[['answer_format','template']]}"
        n = int((d.answer_format == "number").sum())
        print(f"  {name}: {n}/{n} number templates matched, 0/{len(d)-n} others — OK")

    trap = "How many radiopaque clips are visible in this video? Please provide a single integer."
    assert not is_count_question(trap), "the open_ended 'How many …' trap was matched"
    print("  open_ended 'How many radiopaque clips … in this video' trap NOT matched — OK")


def test_round_trip_over_every_real_template_and_gold() -> None:
    """🔴 The load-bearing property: 100 % round-trip, exhaustively.

    Every real `number` template (train ∪ val) × every gold 0..12 — the full cross,
    not a sample.
    """
    tpls = sorted(
        set(_templates("templates_train.csv").query("answer_format=='number'").template)
        | set(_templates("templates_val.csv").query("answer_format=='number'").template)
    )
    targets = []
    for i, t in enumerate(tpls):
        label = count_label(t)
        assert label, f"no label extracted from {t!r}"
        for g in GOLDS:
            targets.append({"qID": f"synthetic__{i}_{g}", "question": t,
                            "label": label, "gold": g,
                            "target": render_count_target(label, g)})
    stats = assert_targets_round_trip(targets)
    assert stats["n"] == len(tpls) * len(GOLDS)
    assert stats["by_status"] == {STATUS_STRUCTURED: stats["n"]}
    print(f"  round-trip: {stats['n']}/{stats['n']} "
          f"({len(tpls)} real templates x golds {GOLDS[0]}..{GOLDS[-1]}) — 100 % — OK")
    print(f"  example target: {targets[0]['target']}")
    print(f"  example target: {targets[len(GOLDS)*1 + 3]['target']}")


def test_malformed_inputs_degrade_and_never_raise() -> None:
    """Perek 2026 (v02) measures 4.76 % malformation for vanilla CoT-SFT — this is the
    behaviour under exactly that failure, enumerated."""
    cases: dict[str, tuple[str, str]] = {
        # raw                                        -> (expected answer, expected status)
        '{"label": "Clips", "counts": 3}':            ("3", STATUS_STRUCTURED),
        '{"label": "Clips", "counts": 3":':           ("3", STATUS_STRUCTURED),
        "{'label': 'Clips', 'counts': 12}":           ("12", STATUS_STRUCTURED),
        '```json\n{"label": "Clips", "counts": 4}\n```': ("4", STATUS_STRUCTURED),
        '{"counts": 2, "label": "Clips"}':            ("2", STATUS_STRUCTURED),
        '{"label": "Clips", "counts": "5"}':          ("5", STATUS_STRUCTURED),
        '{"label": "Clips", "COUNTS": 6}':            ("6", STATUS_STRUCTURED),
        '  {"label":"Clips","counts":0}  ':           ("0", STATUS_STRUCTURED),
        "3":                                          ("3", STATUS_BARE_INT),
        " 07 ":                                       ("7", STATUS_BARE_INT),
        "There are 3 clips.":                         ("3", STATUS_SALVAGED),
        '{"label": "Clips", "cou':                    (SAFE_ANSWER, STATUS_MALFORMED),
        '{"label": "Clips"}':                         (SAFE_ANSWER, STATUS_MALFORMED),
        "":                                           (SAFE_ANSWER, STATUS_MALFORMED),
        "   ":                                        (SAFE_ANSWER, STATUS_MALFORMED),
        "I cannot determine the number.":             (SAFE_ANSWER, STATUS_MALFORMED),
        "Inference Error: CUDA out of memory":         (SAFE_ANSWER, STATUS_MALFORMED),
        "between 2 and 4":                            (SAFE_ANSWER, STATUS_MALFORMED),
        "{}":                                         (SAFE_ANSWER, STATUS_MALFORMED),
        "counts":                                     (SAFE_ANSWER, STATUS_MALFORMED),
        "-2":                                         ("2", STATUS_SALVAGED),
        "3.5":                                        (SAFE_ANSWER, STATUS_MALFORMED),
    }
    for raw, (exp_ans, exp_status) in cases.items():
        r = parse_count(raw, is_count=True)
        assert (r.answer, r.status) == (exp_ans, exp_status), (
            f"parse_count({raw!r}) -> ({r.answer!r}, {r.status!r}), expected "
            f"({exp_ans!r}, {exp_status!r})"
        )
    # None and non-str must not throw either
    for weird in (None, 7, 3.5, b"3", ["3"], {"counts": 3}):
        r = parse_count(weird, is_count=True)
        assert r.answer.isdigit(), f"parse_count({weird!r}) -> {r.answer!r} is not Number-valid"
    assert_parser_never_raises(list(cases) + ["\x00\x01", "counts: ", "9" * 40])
    print(f"  malformed battery: {len(cases)} enumerated cases + 6 non-str + 3 junk — "
          "all degraded to a Number-valid answer, none raised — OK")


def test_passthrough_is_byte_identical() -> None:
    """Non-count questions must come back untouched — that is the guarantee that this
    rung changes ONE format and not five."""
    for s in ("Clip, Sponge", "yes", "top/left", "", "the sponge is grasped by a grasper"):
        r = parse_count(s, is_count=False)
        assert r.answer == s and r.status == STATUS_PASSTHROUGH, (s, r)
    print("  passthrough: 5 non-count answers returned byte-identical — OK")


def _line(qid: str, answer: str, img: str) -> str:
    return json.dumps({
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "<image>Q"},
            {"role": "assistant", "content": answer},
        ],
        "images": [img],
    }, ensure_ascii=False) + "\n"


def test_rewrite_touches_only_number_rows(tmp: Path) -> None:
    q_count = "How many Clips appear in this frame? Please provide a number."
    q_fo = "List all foreign objects that are visible in this video frame. Please provide the class names."
    specs = [
        RowSpec("d__1", "number", q_count, "3", "/f/a.jpg"),
        RowSpec("d__2", "fo_class", q_fo, "Clip, Sponge", "/f/b.jpg"),
        RowSpec("d__3", "number", q_count, "0", "/f/c.jpg"),
        RowSpec("d__4", "binary", "Do Clips and Sponges co-occur in this frame? Please answer with yes or no.", "yes", "/f/d.jpg"),
    ]
    base = [_line(s.qID, s.answer, s.image_path) for s in specs]
    new, targets = rewrite_train_lines(list(base), specs)
    assert len(targets) == 2
    assert json.loads(new[0])["messages"][2]["content"] == '{"label": "Clips", "counts": 3}'
    assert json.loads(new[2])["messages"][2]["content"] == '{"label": "Clips", "counts": 0}'
    assert new[1] == base[1] and new[3] == base[3], "a non-`number` line was re-serialised"

    bp, np_ = tmp / "base.jsonl", tmp / "new.jsonl"
    bp.write_text("".join(base), encoding="utf-8")
    np_.write_text("".join(new), encoding="utf-8")
    got = assert_nonnumber_lines_identical(bp, np_, specs)
    assert got == {"n_rows": 4, "n_rewritten": 2, "n_untouched": 2}
    assert_targets_round_trip(targets)
    print("  rewrite: 2/4 rows rewritten, 2/2 other rows byte-identical, targets round-trip — OK")


def test_misalignment_raises(tmp: Path) -> None:
    """A silent off-by-one would attach counts to the wrong questions. It must RAISE."""
    q = "How many Clips appear in this frame? Please provide a number."
    specs = [RowSpec("d__1", "number", q, "3", "/f/a.jpg")]
    for bad, why in (
        ([_line("d__1", "3", "/f/WRONG.jpg")], "image path"),
        ([_line("d__1", "9", "/f/a.jpg")], "gold answer"),
        ([_line("d__1", "3", "/f/a.jpg"), _line("d__2", "1", "/f/b.jpg")], "row count"),
    ):
        try:
            rewrite_train_lines(bad, specs)
        except AssertionError:
            continue
        raise SystemExit(f"FAIL: misaligned {why} did not raise")
    # a `number` row whose question is not a known counting template must RAISE, not
    # silently pass through — that is how a new template would go unnoticed.
    try:
        rewrite_train_lines([_line("d__1", "3", "/f/a.jpg")],
                            [RowSpec("d__1", "number", "How many clips?", "3", "/f/a.jpg")])
    except AssertionError:
        pass
    else:
        raise SystemExit("FAIL: unknown `number` template did not raise")
    print("  misalignment: 3 alignment faults + 1 unknown template all RAISED — OK")


def test_safe_answer_is_never_modal() -> None:
    """SAFE_ANSWER must cost us, not borrow the floor. Checked on the committed
    per-template modal answers of BOTH splits."""
    for name in ("templates_train.csv", "templates_val.csv"):
        d = _templates(name).query("answer_format=='number'")
        modal = sorted({str(a).strip() for a in d.modal_answer})
        assert SAFE_ANSWER not in modal, f"{name}: SAFE_ANSWER {SAFE_ANSWER!r} is modal"
        print(f"  {name}: modal answers {modal}; SAFE_ANSWER={SAFE_ANSWER!r} not among them — OK")


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("test_regex_matches_exactly_the_number_templates"); test_regex_matches_exactly_the_number_templates()
        print("test_round_trip_over_every_real_template_and_gold"); test_round_trip_over_every_real_template_and_gold()
        print("test_malformed_inputs_degrade_and_never_raise"); test_malformed_inputs_degrade_and_never_raise()
        print("test_passthrough_is_byte_identical"); test_passthrough_is_byte_identical()
        print("test_rewrite_touches_only_number_rows"); test_rewrite_touches_only_number_rows(tmp)
        print("test_misalignment_raises"); test_misalignment_raises(tmp)
        print("test_safe_answer_is_never_modal"); test_safe_answer_is_never_modal()
    print("\nALL GATES PASS (offline: no GPU, no swift, no decord)")


if __name__ == "__main__":
    main()
