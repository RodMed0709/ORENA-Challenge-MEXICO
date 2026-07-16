# Vendored third-party code

> ## ⚠️ THIS COPY IS INCOMPLETE — do not read it to verify an API
>
> **`src/focus/data/` was never vendored.** Missing: `data_models.py` (`Request`, `Reference`,
> `Response`, `save_items`, `load_responses`, `load_references`, `load_requests`), `formats.py`
> (`ts_to_seconds`), and the dataset loader. Only 33 files are tracked here, and
> `tests/test_data_models.py` tests a module that isn't present — the vendoring was partial.
>
> **This matters because our first-party code imports exactly what is missing:**
> `src/frame/run.py:16` and `src/frame/data.py:20-21` import from `focus.data.data_models` and
> `focus.data.formats`. **Reading this folder to check those APIs returns nothing and looks like the
> symbol does not exist.**
>
> **Verify against the installed package instead** — which is what actually gets imported anyway
> (see below):
>
> ```bash
> python -c "import focus.data.data_models as dm; print(dm.__file__, hasattr(dm, 'load_responses'))"
> python -c "import inspect, focus.data.data_models as dm; print(inspect.signature(dm.save_items))"
> ```
>
> **Cost of not knowing this:** on 2026-07-16 it burned review rounds on `experiments/05-bottleneck-audit`
> — APIs were asserted because they could not be checked, and a real API (`load_responses`) was
> challenged as invented. The gap, not anyone's discipline, was the root cause.
>
> **Not completing the copy on purpose.** The installed package is the import path and the source of
> truth; a second, partial copy is what caused the confusion. This warning is the fix.
>
> *(Note: `focus/data/` is a **Python module**, a few KB. It is unrelated to the **FOCUS dataset** —
> the HuggingFace videos/annotations that live on the pod's data volume.)*

> Lives at repo root `vendor/` (moved out of `experiments/00-baseline/` — vendored
> third-party code must not sit inside an experiment dir, per spec §9).

`orena-focus/` is a **vendored copy** of the official challenge SDK
(`https://github.com/IMSY-DKFZ/orena-focus`, tag **v0.3.4**), kept here as
read-only reference for building the zero-shot baseline (see `examples/inference.py`
and `examples/evaluation.py`).

- **Do NOT edit it.** It is not our first-party code (CONSTITUTION §VIII.2 / spec §9 escape hatch).
- For actual use, install the package: `pip install "orena-focus @ git+https://github.com/IMSY-DKFZ/orena-focus.git@v0.3.4"` (import as `focus`).
- Our baseline notebook imports the installed `focus` package + our `src/frame`, not this folder.
