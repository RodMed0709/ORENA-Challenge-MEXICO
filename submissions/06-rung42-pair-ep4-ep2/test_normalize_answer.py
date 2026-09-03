"""Test the container's answer boundary against the SDK's ACTUAL verifiers.
Folder-private; run it directly.

    python submissions/03-rung42-connector-ood/test_normalize_answer.py

Two functions are under test — `normalize_answer` (submission 02) and `clamp_class_tokens`
(new in 03, RULES §8b). The file keeps its old name because
`context/decisions/container-answer-path-audit.md` cites it by path.

Two properties are asserted, and the second matters more than the first:

1. The repair works — strings the SDK rejects become strings it accepts.
2. **Everything else is byte-identical.** This runs inside the shipped container, so a
   normalisation that touched anything beyond its target would silently change answers that
   were already correct. The identity half of this test is the safety gate.

The expected values are not hand-written: they are checked against `focus.data.formats`
themselves, so the test cannot drift from the SDK it is mirroring.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

def _find_repo() -> Path:
    """Walk up until `vendor/orena-focus` appears.

    Was `parents[4]`, hardcoded to this file living in
    `experiments/06-vit-lora/_tools/submission/`. Moving the package to
    `submissions/02-rung21-a2/` silently pointed it at a directory OUTSIDE the repo and the
    test died with a FileNotFoundError. A depth count is not a repo root.
    """
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "vendor" / "orena-focus" / "src").is_dir():
            return cand
    raise RuntimeError(
        f"no repo root above {here} contains vendor/orena-focus/src — "
        "this test needs the vendored SDK to check our repair against the REAL verifiers"
    )


REPO = _find_repo()
sys.path.insert(0, str(REPO / "vendor" / "orena-focus" / "src"))

# Import the function under test without importing the whole container (which needs torch).
_SRC = (Path(__file__).parent / "inference.py").read_text(encoding="utf-8")
_ns: dict = {"re": re}
_start = _SRC.index("_TRAILING_DOT_INT")
_end = _SRC.index("@torch.no_grad()", _start)
exec(compile(_SRC[_start:_end], "inference.py:normalize_answer", "exec"), _ns)  # noqa: S102
normalize_answer = _ns["normalize_answer"]
clamp_class_tokens = _ns["clamp_class_tokens"]


def _load_sdk_formats():
    """Load `focus/data/formats.py` BY PATH.

    Importing `focus.data.formats` normally drags in `focus/__init__` -> `evaluation` ->
    `judges` -> `transformers`, so the test would only run where the full inference stack is
    installed. This test must run anywhere — it guards a container, and the point is to check
    our repair against the SDK's REAL verifiers rather than a copy of them.
    """
    import importlib.util
    import types

    # `formats.py` imports `focus.foreign_objects`, which re-enters `focus/__init__` and
    # ultimately `transformers`. We only need the pure-Python verifiers, so stub the heavy
    # leaf out. Test-only: it never touches the container or the eval path.
    if "transformers" not in sys.modules:
        stub = types.ModuleType("transformers")
        stub.AutoModelForCausalLM = stub.AutoTokenizer = object
        sys.modules["transformers"] = stub

    path = REPO / "vendor" / "orena-focus" / "src" / "focus" / "data" / "formats.py"
    spec = importlib.util.spec_from_file_location("_focus_formats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FMT = _load_sdk_formats()


def _sdk_accepts(text: str, fmt: str) -> bool:
    verifier = {"number": _FMT.Number(), "binary": _FMT.Binary()}[fmt]
    try:
        verifier.verify(text)
        return True
    except Exception:
        return False


REPAIRS = [  # (raw, fmt) — SDK rejects raw, must accept the normalised form
    ("1.", "number"),
    ("0.", "number"),
    ("12.", "number"),
    ("3 .", "number"),
    ("Yes.", "binary"),
    ("no.", "binary"),
    ("NO .", "binary"),
]

UNTOUCHED = [  # must come back byte-identical
    "1", "0", "12", "yes", "no", "Yes", "3.5", "1.2.3", "about 3", "3 clips",
    "Clip, Sponge", "Clip, Sponge.", "none", "", "   ", "1..", ".", "12:03",
    "The specimen bag is visible.", "2 or 3", "-1", "1,000",
]


def _fo_verifier():
    """The REAL `FOClass` verifier, with the REAL registry — never a hand-copied name list.

    This is the whole point of RULES §8b: the prompt's 10-item list and the scorer's 10-item
    list disagree on their 10th element, so a test that hard-codes either one proves nothing
    about the scorer.
    """
    return _FMT.FOClass()


def _fo_accepts(text: str) -> bool:
    try:
        _fo_verifier().verify(text)
        return True
    except Exception:
        return False


def _check_clamp() -> list[str]:
    """`clamp_class_tokens` must (a) rescue answers the SDK RAISES on and (b) never touch
    anything else. (b) is the safety gate: this runs on every answer in the container."""
    failures: list[str] = []
    legal = {n.lower(): n for n in _FMT.FOType.names()}
    real = list(_FMT.FOType.names())
    assert len(real) >= 2, "the SDK registry is too small to build a two-class case"
    a, b = real[0], real[1]

    # (a) rescues — the SDK rejects the raw string and must accept the clamped one
    rescues = [
        f"{a}, NotAThing",
        f"NotAThing, {a}",
        f"{a}, {b}, Bandaid",
        f"none, {a}",              # 'none' combined with a class also RAISES
    ]
    for raw in rescues:
        got = clamp_class_tokens(raw, legal)
        if _fo_accepts(raw):
            failures.append(f"premise broken: SDK already accepts {raw!r} as fo_class")
            continue
        if not _fo_accepts(got):
            failures.append(f"clamp failed: {raw!r} -> {got!r}, still rejected as fo_class")

    # (b) byte-identical — legal class answers, and every non-class answer shape
    untouched = [
        a, f"{a}, {b}", "none", "None", a.lower(), a.upper(),
        "1", "0", "12", "yes", "no", "Yes", "3.5", "",
        "NotAThing", "NotAThing, AlsoNot",     # nothing legal to salvage: leave it alone
        "The specimen bag is visible.", "2 or 3", "1,000", "12:03",
    ]
    for raw in untouched:
        got = clamp_class_tokens(raw, legal)
        if got != raw:
            failures.append(f"clamp NOT byte-identical: {raw!r} -> {got!r}")
    return failures


def main() -> int:
    failures: list[str] = []

    for raw, fmt in REPAIRS:
        got = normalize_answer(raw)
        if _sdk_accepts(raw.strip(), fmt):
            failures.append(f"premise broken: SDK already accepts {raw!r} as {fmt}")
            continue
        if not _sdk_accepts(got, fmt):
            failures.append(f"repair failed: {raw!r} -> {got!r}, still rejected as {fmt}")

    for raw in UNTOUCHED:
        got = normalize_answer(raw)
        if got != raw.strip():
            failures.append(f"NOT byte-identical: {raw!r} -> {got!r} (expected {raw.strip()!r})")

    # The exact defect probe 16a measured, end to end.
    assert normalize_answer("1.") == "1", "the 16a case must repair"
    assert _sdk_accepts("1", "number") and not _sdk_accepts("1.", "number")

    failures += _check_clamp()

    # RULES §8b, asserted rather than trusted: an unrecognised token does not score 0, it
    # RAISES. If that ever stops being true this test should be the thing that notices.
    _bad = f"{_FMT.FOType.names()[0]}, NotAThing"
    assert not _fo_accepts(_bad), (
        "premise gone: FOClass.verify no longer rejects an unknown class name — re-read "
        "RULES §8b before keeping the clamp"
    )

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"PASS — {len(REPAIRS)} repairs verified against the real SDK verifiers, "
          f"{len(UNTOUCHED)} inputs byte-identical, and the fo_class clamp checked against "
          f"the SDK's own registry ({len(_FMT.FOType.names())} names read at runtime)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
