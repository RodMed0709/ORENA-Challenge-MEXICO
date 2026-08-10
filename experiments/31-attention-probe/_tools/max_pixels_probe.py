"""Does ``max_pixels`` do anything? CPU only, no weights, no GPU, no pod.

## Why this lives here

`attention_probe.py:174-182` records that ``AutoProcessor.from_pretrained(..., max_pixels=N)`` is a
**no-op on the qwen-vl-utils path**, because ``process_vision_info`` does the resize. That is one of
the seven silent no-ops of 2026-08-08, and `src/frame/engine.py:43` does exactly the same call —
while `CLAUDE.md` describes ``max_pixels`` as what protects the 5 s budget. Rung 31 is the rung that
found it, builds four processors in one process, and is therefore its owner.

This probe separates claims that were being argued past each other.

## What it measured, 2026-08-09 (transformers 4.57.6)

**Q1 — the image-processor class DOES honour it.** One process per cell, so no instance can
contaminate another:

| frame | pixels | no cap | cap = 921,600 | cap = 65,536 |
|---|---|---|---|---|
| 1280×720 | 921,600 | 1196 tok | **1125 tok** | 72 tok |
| 960×540 | 518,400 | 646 | 646 (unchanged) | 72 |
| 720×576 | 414,720 | 546 | 546 (unchanged) | 80 |

🟢 **1,125 is exactly what `src/frame/config.py:40-50` predicts** — 1280×720 rounds to 1288×728 =
937,664 px, just over the 921,600 cap, resized to 1260×700. That comment is the only one of three
competing accounts that was right.

🔴 **But it does not follow that the cap acts in the container.** `engine.py:161` resizes through
``qwen_vl_utils.process_vision_info`` and `engine.py:100`'s ``_messages()`` carries **no
``max_pixels`` key**, so no-op #4 stands on the path inference actually uses.
⚠️ **Untested link:** whether ``AutoProcessor.from_pretrained(..., max_pixels=N)`` forwards the
kwarg to the image processor at all. That needs the checkpoint's config JSONs (~50 KB, no GPU) and
is the one measurement that would close this.

**Q2 — 🔴 an upstream bug in ``Qwen2VLImageProcessorFast``.** Its ``__init__`` does
``size = self.size if size is None else size`` and then mutates that dict in place — and
``self.size`` is the **class attribute**. So building one processor with ``max_pixels`` changes the
default for every processor built afterwards **in the same process**:

```
[1] fresh, no max_pixels        : longest_edge = 1,003,520
[2] built with max_pixels=65536 : longest_edge =    65,536
[3] fresh again, no max_pixels  : longest_edge =    65,536   ← leaked
    1280x720 through [3]        : 72 tokens instead of 1196
```

The slow processor does **not** have it — its ``__init__`` builds a fresh dict literal.
⚠️ Rung 31 builds four processors in one process (`attention_probe.py:287` inside the arm loop at
:325) with the **same** ``max_pixels`` in all four, so its between-arm comparison holds; only an
absolute level would be at risk, and rung 31 already declares its levels non-transferable.

## Running it

``python max_pixels_probe.py`` runs the whole thing. The grid re-execs one subprocess per cell on
purpose: a single process cannot measure Q1 and Q2 at once, because Q2 corrupts Q1. The first
version of this probe did exactly that and produced nonsense (960×540 "uncapped" → 72 tokens),
which is how the leak was found.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

CAP = 1280 * 720
FRAMES = [(1280, 720), (960, 540), (720, 576)]


def _cls(fast: bool):
    if fast:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
            Qwen2VLImageProcessorFast as C,
        )
    else:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor as C,
        )
    return C


def tokens(proc, w: int, h: int) -> tuple[int, int, int]:
    """Visual tokens after the merger for one WxH RGB frame."""
    out = proc(images=[np.zeros((h, w, 3), dtype=np.uint8)], return_tensors="pt")
    g = out["image_grid_thw"][0].tolist()
    return g[1], g[2], (g[1] * g[2]) // (proc.merge_size ** 2)


def cell(fast: bool, max_pixels: int | None, w: int, h: int) -> dict:
    """ONE measurement in a clean interpreter — see Q2 for why this is not a loop."""
    r = subprocess.run(
        [sys.executable, __file__, "--cell", "fast" if fast else "slow",
         "none" if max_pixels is None else str(max_pixels), str(w), str(h)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(r.stdout.strip().splitlines()[-1])


def leak(fast: bool) -> dict:
    """Does a capped instance change the default of the next uncapped one?"""
    C = _cls(fast)
    before = C().size["longest_edge"]
    capped = C(max_pixels=256 * 256).size["longest_edge"]
    after = C().size["longest_edge"]
    return {"fresh_before": before, "capped": capped, "fresh_after": after,
            "leaks": after != before}


def main() -> None:
    for fast in (True, False):
        name = "FAST (the transformers default)" if fast else "SLOW"
        print(f"\n=== Q1 — does max_pixels change visual tokens?  [{name}] ===")
        for w, h in FRAMES:
            cells = [cell(fast, m, w, h) for m in (None, CAP, 256 * 256)]
            got = "  |  ".join(f"{c['gh']}x{c['gw']} -> {c['tok']:>4} tok" for c in cells)
            print(f"  {w}x{h:<5} {w * h:>9,} px  |  {got}")
        print(f"=== Q2 — does it leak between instances?  [{name}] ===")
        print(f"  {leak(fast)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cell":
        _fast = sys.argv[2] == "fast"
        _mp = None if sys.argv[3] == "none" else int(sys.argv[3])
        _w, _h = int(sys.argv[4]), int(sys.argv[5])
        _p = _cls(_fast)() if _mp is None else _cls(_fast)(max_pixels=_mp)
        _gh, _gw, _tok = tokens(_p, _w, _h)
        print(json.dumps({"gh": _gh, "gw": _gw, "tok": _tok}))
    else:
        main()
