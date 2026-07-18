"""frame — FRAME-track baseline harness for the ORENA SAVE FOCUS challenge.

Importable library only (no launchers). A notebook supplies an inline config
and calls :func:`run_baseline`. All pipeline logic lives here.

Top-level names are exposed LAZILY (PEP 562): importing ``frame`` — or a
pure-pandas submodule like ``frame.metrics`` / ``frame.ledger`` — must NOT pull
in the heavy inference/eval stack (torch, transformers, focus.evaluation). Those
are loaded only when ``BaselineConfig`` / ``run_baseline`` are actually accessed.
"""

from __future__ import annotations

__all__ = ["BaselineConfig", "run_baseline"]


def __getattr__(name: str):
    if name == "BaselineConfig":
        from frame.config import BaselineConfig

        return BaselineConfig
    if name == "run_baseline":
        from frame.run import run_baseline

        return run_baseline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
