# Vendored third-party code

> Lives at repo root `vendor/` (moved out of `experiments/00-baseline/` — vendored
> third-party code must not sit inside an experiment dir, per spec §9).

`orena-focus/` is a **vendored copy** of the official challenge SDK
(`https://github.com/IMSY-DKFZ/orena-focus`, tag **v0.3.4**), kept here as
read-only reference for building the zero-shot baseline (see `examples/inference.py`
and `examples/evaluation.py`).

- **Do NOT edit it.** It is not our first-party code (CONSTITUTION §VIII.2 / spec §9 escape hatch).
- For actual use, install the package: `pip install "orena-focus @ git+https://github.com/IMSY-DKFZ/orena-focus.git@v0.3.4"` (import as `focus`).
- Our baseline notebook imports the installed `focus` package + our `src/frame`, not this folder.
