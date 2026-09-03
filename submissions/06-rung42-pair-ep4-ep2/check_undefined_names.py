#!/usr/bin/env python3
"""Fail the build if `inference.py` reads a name it never defines.

🔴 Why this file exists. This container shipped TWO NameErrors in eleven days, both in
`inference.py`, both invisible to every test we had:

  * 2026-08-25 — `dev = getattr(model, ...)`: `model` is never assigned. Killed the run
    immediately after loading 33 GB, on the GPU path too.
  * 2026-08-27 — `fixed = normalize_answer(raw)`: the function had been dropped along with
    its own `test_normalize_answer.py`. Killed every batch AFTER the vLLM call returned,
    so the platform charged us ~3.8 h of compute and reported "failed on one or more cases"
    for all 100 of them, with no logs.

Neither is reachable by the CPU wiring smoke: `answer_batch` returns at `if llm is None`
thirty-odd lines before the second one. Neither is reachable by a model test either — the
model answered 4,000 questions natively through the same vLLM path. The only thing that
catches this class of defect cheaply is reading the file's own name bindings, which is
what this does. It costs milliseconds and needs neither a GPU nor the weights.

Deliberately simple: scope-insensitive, so a name bound ANYWHERE in the module (def, class,
import, assignment, function argument, comprehension target, `with`/`except`/`for` binding)
counts as defined. That under-reports — it will not catch a genuine use-before-assignment
across scopes — and it never cries wolf, which is what makes it usable as a build gate.
"""
import ast
import builtins
import sys
from pathlib import Path

TARGET = Path(__file__).with_name("inference.py")


def bound_names(tree: ast.AST) -> set[str]:
    names: set[str] = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names |= {(a.asname or a.name.split(".")[0]) for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {(a.asname or a.name) for a in node.names}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arguments):
            for a in (*node.posonlyargs, *node.args, *node.kwonlyargs):
                names.add(a.arg)
            for a in (node.vararg, node.kwarg):
                if a is not None:
                    names.add(a.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global):
            names |= set(node.names)
    return names


def main() -> int:
    tree = ast.parse(TARGET.read_text(encoding="utf-8"), filename=str(TARGET))
    known = bound_names(tree)
    unbound: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in known:
            unbound.setdefault(node.id, []).append(node.lineno)

    if not unbound:
        print(f"OK: every name read in {TARGET.name} is bound somewhere in the module.")
        return 0

    print(f"FAIL: {TARGET.name} reads {len(unbound)} name(s) it never binds.", file=sys.stderr)
    for name, lines in sorted(unbound.items()):
        print(f"  {name!r}  read at line(s) {lines}  — 0 bindings", file=sys.stderr)
    print("\nThis is a NameError at runtime. Do NOT build or submit this image.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
