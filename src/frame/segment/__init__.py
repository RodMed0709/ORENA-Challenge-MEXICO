"""SEGMENT-track library.

A sub-package of the ONE canonical `frame` package (a second first-party `src/`
is forbidden — `EXPERIMENT_REPO_STRUCTURE_SPEC.md` §9). The `frame` name is a
historical misnomer kept because renaming it would break imports across 42 rungs.

Kept import-light on purpose: `frame/__init__.py` promises that importing a
pure-pandas submodule does not drag in torch/transformers, and this package
must not break that promise. Import submodules explicitly.
"""
