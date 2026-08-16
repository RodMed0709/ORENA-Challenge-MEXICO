# Vendored third-party code

> ## ✅ THIS COPY IS COMPLETE — and it is the only one that exists
>
> **Read it to verify an API.** `src/focus/data/` is here and tracked: `data_models.py`
> (`Request`, `Reference`, `Response`, `save_items`, `load_responses`, `load_references`,
> `load_requests`), `formats.py` (`ts_to_seconds`), plus `base_dataset.py`, `frame_dataset.py`,
> `video_dataset.py` and `download.py`. **39 files tracked**, all eight symbols above present.
>
> **There is no installed package to check against.** `focus` is not pip-installed on any of our
> machines — not locally, not in any of UNAM's four environments. Everything that runs it puts
> `<repo>/vendor/orena-focus/src` on `sys.path` instead (see the rung-43 job scripts), so **this
> folder IS the import path**.
>
> ```bash
> python -c "import sys; sys.path.insert(0,'vendor/orena-focus/src'); \
>   import focus.data.data_models as dm; print(dm.__file__, hasattr(dm,'load_responses'))"
> ```
>
> 🔻 **Corrected 2026-08-16. This block used to say the opposite** — "THIS COPY IS INCOMPLETE …
> `src/focus/data/` was never vendored … verify against the installed package instead". That was
> true when written: a `data/` pattern in `.gitignore` had silently eaten the directory. It was
> fixed on **2026-07-18** by `1a8e9ff` ("track focus SDK data/ source — gitignore data/ pattern
> nuked it") and the warning was never updated, so for a month the repo told readers to distrust
> the only copy there is and to check it against a package nobody has installed.
>
> **The original cost is worth keeping:** on 2026-07-16 the real gap burned review rounds on
> `experiments/05-bottleneck-audit` — APIs were asserted because they could not be checked, and a
> real one (`load_responses`) was challenged as invented. A stale fix-notice costs the same way.
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
- **Installing it is still the cleaner path where you can**:
  `pip install "orena-focus @ git+https://github.com/IMSY-DKFZ/orena-focus.git@v0.3.4"` (import as
  `focus`), and `requirements/harness.txt` pins exactly that.
- 🔻 **But nothing we run today does that.** No environment we own has `focus` installed; every job
  puts `<repo>/vendor/orena-focus/src` on `sys.path`. That is why the copy must stay complete, and
  why UNAM needs a repo checkout at all even though nobody edits code there.
