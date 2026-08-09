"""Step 6 — turn the downloaded mask package into a viewer.

Reporting library. Imported from a notebook cell; ``build(pkg, out_html)``.

Runs entirely off ``_tools/export_masks.py``'s tarball: plain numpy + Pillow + base64, no GPU
and no volume. The controls are already adjudicated — C1 separation **6.475** against 0.5,
C2 persistence **0.9167** against 0.90 — so this exists for the question the numbers cannot
answer: **what is SAM 2 actually finding on our frames?**

⚠️ **Instances, not classes.** A colour is one segmented object, never "a Clip". The C2 panels
colour each instance by whether its IoU survived the adjacent frame, which is the persistence
statistic made visible rather than a new measurement.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

#: distinguishable hues; index 0 is background and stays transparent
PALETTE = np.array([
    [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
    [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200],
    [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128],
], np.uint8)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _overlay(labels: np.ndarray, survived: list[bool] | None, alpha: int = 150) -> str:
    """Label map -> RGBA PNG. If `survived` is given, colour by track survival instead."""
    from PIL import Image

    h, w = labels.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    n = int(labels.max())
    for k in range(1, n + 1):
        m = labels == k
        if not m.any():
            continue
        if survived is not None:
            ok = survived[k - 1] if k - 1 < len(survived) else False
            rgba[m, :3] = np.array([60, 180, 75] if ok else [230, 25, 75], np.uint8)
        else:
            rgba[m, :3] = PALETTE[1 + (k - 1) % (len(PALETTE) - 1)]
        rgba[m, 3] = alpha
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return _b64(buf.getvalue())


def build(pkg: Path, out_html: Path, iou_survive: float = 0.5) -> Path:
    pkg = Path(pkg)
    meta = json.loads((pkg / "meta.json").read_text(encoding="utf-8"))

    def labels(tag: str) -> np.ndarray:
        return np.load(pkg / "labels" / f"{tag}.npz")["labels"]

    def frame(tag: str) -> str:
        return _b64((pkg / "frames" / f"{tag}.jpg").read_bytes())

    c1 = [m for m in meta if m["tag"].startswith("c1_")]
    c2 = [m for m in meta if m["tag"].startswith("c2_")]

    def stack(img_b64: str, ov_b64: str) -> str:
        return (f'<div class="stack"><img src="data:image/jpeg;base64,{img_b64}" alt="">'
                f'<img class="ov" src="data:image/png;base64,{ov_b64}" alt=""></div>')

    c1_cards = []
    for m in c1:
        kind = "gold ≥ 5" if "high" in m["tag"] else "gold == 1"
        ov = _overlay(labels(f"{m['tag']}_a"), None)
        c1_cards.append(
            f'<figure><figcaption><b>{kind}</b>'
            f'<span class="n">{m["ds"]} · gold {m["gold"]} · <b>{m["n_masks"]}</b> instancias</span>'
            f'</figcaption>{stack(frame(f"{m['tag']}_a"), ov)}</figure>'
        )

    c2_cards = []
    for m in c2:
        p = m.get("pair")
        if not p:
            continue
        surv = [i >= iou_survive for i in m["ious"]]
        a = stack(frame(f"{m['tag']}_a"), _overlay(labels(f"{m['tag']}_a"), surv))
        b = stack(frame(f"{m['tag']}_b"), _overlay(labels(f"{m['tag']}_b"), surv))
        c2_cards.append(
            f'<section class="pair"><h3>{m["ds"]} · gold {m["gold"]} · '
            f'persistencia <b>{p["persistence"]:.3f}</b> '
            f'<span class="n">({p["n_survived"]}/{p["n_seed"]} sobreviven · '
            f'IoU medio {p["mean_iou"]:.3f})</span></h3>'
            f'<div class="two"><div><span class="lbl">frame A — sembrado</span>{a}</div>'
            f'<div><span class="lbl">frame B — propagado (~1/fps después)</span>{b}</div></div>'
            f'</section>'
        )

    html = f"""<!doctype html><meta charset="utf-8">
<title>Paso 6 — qué ve SAM 2</title>
<style>
:root{{--bg:#faf9f7;--fg:#1c1b1a;--mut:#6b6864;--line:#e2ded9;--card:#fff}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15140f;--fg:#eae7e1;--mut:#9a958c;--line:#2e2b26;--card:#1d1c17}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:1.5rem;margin:0 0 .3rem}} h2{{font-size:1.1rem;margin:2.2rem 0 .2rem}}
.sub{{color:var(--mut);margin:0 0 1.4rem}}
.warn{{background:var(--card);border:1px solid var(--line);border-left:3px solid #c2410c;
 padding:.85rem 1rem;border-radius:6px;margin:0 0 1.6rem;color:var(--mut);font-size:.92rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:.7rem}}
figcaption{{display:flex;justify-content:space-between;align-items:baseline;
 font-size:.85rem;margin-bottom:.45rem;gap:.5rem}}
.n{{color:var(--mut);font-variant-numeric:tabular-nums;font-size:.82rem}}
.stack{{position:relative;line-height:0;border-radius:5px;overflow:hidden;border:1px solid var(--line)}}
.stack img{{width:100%;height:auto;display:block}} .stack img.ov{{position:absolute;inset:0}}
section.pair{{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:.9rem 1rem;margin-bottom:1.2rem}}
section.pair h3{{margin:0 0 .6rem;font-size:.95rem;font-weight:600}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}}
.lbl{{display:block;font-size:.78rem;color:var(--mut);margin-bottom:.3rem}}
.key{{display:flex;gap:1.2rem;font-size:.85rem;color:var(--mut);margin:.2rem 0 1rem}}
.key i{{display:inline-block;width:.8rem;height:.8rem;border-radius:2px;margin-right:.35rem;vertical-align:-1px}}
@media(max-width:700px){{.two{{grid-template-columns:1fr}}}}
</style>
<div class="wrap">
<h1>Paso 6 — qué ve SAM 2 en nuestros frames</h1>
<p class="sub">Los dos controles bloqueantes pasaron: <b>C1</b> separación <b>6.475</b> contra 0.5 ·
<b>C2</b> persistencia <b>0.9167</b> contra 0.90. Esto es lo que hay detrás de esos números.</p>

<div class="warn"><b>Instancias, no clases.</b> SAM 2 es agnóstico de clase: un color es
<i>un objeto segmentado</i>, nunca «un Clip». El tope de área es lo que impide que el tejido de
fondo salga como instancia #1 — un punto central en un frame de <code>heico</code> devuelve el
90&nbsp;% de la imagen. Y esto <b>no reabre el paso 6</b>: su cierre no dependía de SAM.</div>

<h2>C1 — ¿el número de instancias separa <code>gold ≥ 5</code> de <code>gold == 1</code>?</h2>
<p class="sub">Un color por instancia sembrada.</p>
<div class="grid">{"".join(c1_cards)}</div>

<h2>C2 — ¿aguanta el track entre frames adyacentes?</h2>
<div class="key"><span><i style="background:#3cb44b"></i>sobrevive (IoU ≥ {iou_survive})</span>
<span><i style="background:#e6194b"></i>se pierde</span></div>
{"".join(c2_cards)}
</div>"""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"viewer -> {out_html} ({out_html.stat().st_size / 1e6:.2f} MB)")
    return out_html
