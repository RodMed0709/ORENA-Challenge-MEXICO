"""Replay the MEASURED predictions through the container's OWN arbitrate() and check it
reproduces the numbers the arm was approved on. No GPU, no model, no network.

This is the check the 27B container never had: the thing measured on the box and the thing
that ships are now provably the same function. If `arbitrate` is edited and these numbers
move, the edit changed the arm, not just the code.
"""
import csv, os, re, sys
from pathlib import Path
# 🔴 Load the two functions from the SHIPPED source text without importing the module:
# inference.py imports torch/transformers/focus at module level and none of them exist
# on the build box. Extracting by AST means this test reads the exact bytes that ship --
# a stub-based import could pass against a module that never loads in the container.
import ast, types
_SRC = (Path(__file__).parent / "inference.py").read_text()
_TREE = ast.parse(_SRC)
_WANT = {"as_class_set", "arbitrate"}
_NS: dict = {"re": re, "frozenset": frozenset}
_found = set()
for _node in _TREE.body:
    if isinstance(_node, ast.FunctionDef) and _node.name in _WANT:
        exec(compile(ast.Module(body=[_node], type_ignores=[]), "inference.py", "exec"), _NS)
        _found.add(_node.name)
assert _found == _WANT, f"missing from inference.py: {_WANT - _found}"
I = types.SimpleNamespace(**{k: _NS[k] for k in _WANT})

LEGAL = {n.lower(): n for n in (
    "Clip", "Sponge", "Specimen bag", "Silicone loop", "External drain", "Needle",
    "Gallstone", "Specimen", "Mesh", "Absorbable Hemostatic Agent")}

def load_pairs(d: Path, a: str, b: str):
    rows_a = list(csv.DictReader((d / f"{a}.csv").open()))
    rows_b = list(csv.DictReader((d / f"{b}.csv").open()))
    assert len(rows_a) == len(rows_b), "prediction files disagree in length"
    return rows_a, rows_b

def norm(s):
    v = I.as_class_set(s, LEGAL)
    return v if v is not None else None

def run(d: Path, a: str, b: str, label: str, expect_base: float, expect_arm: float):
    ra, rb = load_pairs(d, a, b)
    ok_a = ok_arm = 0
    for x, y in zip(ra, rb):
        g = norm(x["gold"])
        merged, _ = I.arbitrate(x["pred"], y["pred"], LEGAL)
        ok_a += norm(x["pred"]) == g
        ok_arm += norm(merged) == g
    n = len(ra)
    got_base, got_arm = ok_a / n, ok_arm / n
    print(f"{label}: n={n}  base {got_base:.4f} (esperado {expect_base:.4f})  "
          f"arm {got_arm:.4f} (esperado {expect_arm:.4f})  delta {got_arm-got_base:+.4f}")
    assert abs(got_base - expect_base) < 5e-4, f"{label}: base moved"
    assert abs(got_arm - expect_arm) < 5e-4, f"{label}: ARM MOVED -- arbitrate() is not the function that was measured"

if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    run(d, "r42", "r42_ep2", "ID  (490 fo_class, 8 videos held-out)", 0.8408, 0.8367)
    print("\nOK -- arbitrate() reproduce las cifras con las que se aprobo el brazo.")
