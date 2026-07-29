"""Test `normalize_answer` against the SDK's ACTUAL verifiers. Folder-private; run it directly.

    python experiments/06-vit-lora/_tools/submission/test_normalize_answer.py

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

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "vendor" / "orena-focus" / "src"))

# Import the function under test without importing the whole container (which needs torch).
_SRC = (Path(__file__).parent / "inference.py").read_text(encoding="utf-8")
_ns: dict = {"re": re}
_start = _SRC.index("_TRAILING_DOT_INT")
_end = _SRC.index("@torch.no_grad()", _start)
exec(compile(_SRC[_start:_end], "inference.py:normalize_answer", "exec"), _ns)  # noqa: S102
normalize_answer = _ns["normalize_answer"]


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

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"PASS — {len(REPAIRS)} repairs verified against the real SDK verifiers, "
          f"{len(UNTOUCHED)} inputs byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
