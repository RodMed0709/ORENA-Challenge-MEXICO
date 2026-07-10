"""frame — FRAME-track baseline harness for the ORENA SAVE FOCUS challenge.

Importable library only (no launchers). A notebook supplies an inline config
and calls :func:`run_baseline`. All pipeline logic lives here.
"""

from frame.config import BaselineConfig
from frame.run import run_baseline

__all__ = ["BaselineConfig", "run_baseline"]
