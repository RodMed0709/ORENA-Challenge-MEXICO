"""Rung 31 — turn a downloaded run package into the tracked artifacts.

Reporting library. Imported from a notebook cell; ``build(pkg, exp)`` writes:

* ``RESULTS_attention.csv``            — one row per arm, the comparable table
* ``RESULTS_attention_per_question.csv`` — arm x question
* ``RESULTS_attention_curve.csv``     — arm x layer, the visual-mass curve over depth
* ``docs/viewers/attention_viewer.html`` — self-contained, no external assets

🔴 **Runs off the DOWNLOADED package, never off the pod.** `runs/` is gitignored, so the
numbers that reach `main` have to be re-derived here from `_tools/export_run.py`'s tarball.
Everything below is plain numpy + base64; nothing calls a GPU and nothing needs the volume.

The heatmap is ``n_image_tokens`` values over the merged patch grid. At 512x512 that is 231
tokens = **21 x 11**, and the factorisation is asserted rather than assumed — a silently
wrong reshape would produce a plausible-looking but meaningless picture, which is the one
failure a viewer cannot show you.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path

import numpy as np

ARM_LABEL = {
    "base": "base — Qwen3-VL-8B, no fine-tuning",
    "rung02": "rung 02 — first LoRA SFT",
    "rung06": "rung 06 — ViT-LoRA (freeze_vit=false)",
    "a2": "A2 — 21_lr_2e4_v1 ep3, shipped as submission 02",
}
ARM_ORDER = ["base", "rung02", "rung06", "a2"]


def grid_shape(n: int, width_px: int, height_px: int) -> tuple[int, int]:
    """(rows, cols) for ``n`` merged patches over a ``width x height`` image.

    Asserted, not guessed: an aspect-consistent factorisation must exist and multiply back
    to ``n``. A wrong reshape draws a convincing picture of nothing.
    """
    for cols in range(1, n + 1):
        if n % cols:
            continue
        rows = n // cols
        if abs((cols / rows) - (width_px / height_px)) < 0.12:
            return rows, cols
    raise AssertionError(
        f"no aspect-consistent factorisation of {n} tokens for a {width_px}x{height_px} image"
    )


def _png(arr: np.ndarray) -> str:
    from PIL import Image

    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _heat_png(heat: np.ndarray, rows: int, cols: int, size: tuple[int, int]) -> str:
    """Heat as an RGBA overlay: alpha carries the mass, hue is a fixed warm ramp."""
    from PIL import Image

    g = heat.reshape(rows, cols).astype(np.float64)
    g = (g - g.min()) / max(g.max() - g.min(), 1e-12)
    rgba = np.zeros((rows, cols, 4), np.uint8)
    rgba[..., 0] = (255 * np.clip(g * 1.6, 0, 1)).astype(np.uint8)
    rgba[..., 1] = (255 * np.clip(g * 1.6 - 0.6, 0, 1)).astype(np.uint8)
    rgba[..., 2] = (255 * np.clip(g * 1.6 - 1.3, 0, 1)).astype(np.uint8)
    rgba[..., 3] = (215 * g).astype(np.uint8)
    im = Image.fromarray(rgba, "RGBA").resize(size, Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _jpg_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


#: a patch is "black margin" below this mean luminance. Laparoscopic frames are a circular
#: scope image letterboxed into a rectangle, so the corners are dead pixels carrying no scene.
MARGIN_LUM = 20


def margin_analysis(pkg: Path, exp: Path, arms: dict) -> dict:
    """Split the attention mass into black margin vs scene content, and measure concentration.

    🔑 **The observation is legokna's, not the tool's.** Reading the 12 frames by eye they
    wrote, in q00/q01/q02/q03/q04/q06/q08/q11, that every arm puts attention on *"los margenes
    negros"* — and separately that A2 finds one zone strongly where rung 06 finds both
    (*"a2 solo detecta una zona, rung06 si que detecta ambas"*). Source:
    ``local/fuentes/analisis mascaras de atencion rung 31.md``, 2026-08-08. Both are measured
    here because a described pattern that is never quantified is the ±0.86 failure again.

    Margin fraction matters twice over: it is a finding on its own, and it corrects the
    headline. ``visual_mass`` counts attention on image tokens, and a token on the letterbox
    is an image token carrying no scene, so the raw number overstates every arm — unevenly.
    """
    from PIL import Image

    meta = json.loads((pkg / "questions.json").read_text(encoding="utf-8"))
    n_tok = arms[next(iter(arms))]["agg"]["n_image_tokens"]
    rows_g, cols_g = grid_shape(n_tok, meta[0]["width"], meta[0]["height"])

    def patch_lum(i: int) -> np.ndarray:
        im = np.asarray(Image.open(pkg / "frames" / f"q{i:02d}.jpg").convert("L"), float)
        h, w = im.shape
        ph, pw = h // rows_g, w // cols_g
        return np.array([[im[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw].mean()
                          for c in range(cols_g)] for r in range(rows_g)])

    lum = {m["i"]: patch_lum(m["i"]) for m in meta}
    area = float(np.mean([(v < MARGIN_LUM).mean() for v in lum.values()]))

    out = {"margin_area_fraction": area, "arms": {}}
    for arm, d in arms.items():
        marg, top5, ent, hot = [], [], [], []
        for m in meta:
            h = np.load(pkg / "heat" / f"{arm}_q{m['i']:02d}.npy").reshape(rows_g, cols_g)
            h = h / h.sum()
            bg = lum[m["i"]] < MARGIN_LUM
            marg.append(float(h[bg].sum()))
            v = h[~bg]
            v = v / v.sum()
            top5.append(float(np.sort(v)[-5:].sum()))
            ent.append(float(-(v * np.log(v + 1e-12)).sum() / np.log(len(v))))
            hot.append(int((v > 2 * v.mean()).sum()))
        raw = d["agg"]["visual_mass_mean"]
        content = 1.0 - float(np.mean(marg))
        out["arms"][arm] = {
            "margin_fraction": float(np.mean(marg)), "content_fraction": content,
            "visual_mass_raw": raw, "visual_mass_content": raw * content,
            "top5_content": float(np.mean(top5)), "entropy_content": float(np.mean(ent)),
            "hot_patches": float(np.mean(hot)),
        }

    base = out["arms"].get("base", {}).get("visual_mass_content") or 1.0
    with open(exp / "RESULTS_attention_margin.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "margin_fraction", "content_fraction", "visual_mass_raw",
                    "visual_mass_content", "vs_base", "top5_content", "entropy_content",
                    "hot_patches", "margin_area_fraction"])
        for arm, v in out["arms"].items():
            w.writerow([arm, round(v["margin_fraction"], 4), round(v["content_fraction"], 4),
                        round(v["visual_mass_raw"], 6), round(v["visual_mass_content"], 6),
                        round(v["visual_mass_content"] / base, 3), round(v["top5_content"], 4),
                        round(v["entropy_content"], 4), round(v["hot_patches"], 1),
                        round(area, 4)])
    return out


def write_csvs(pkg: Path, exp: Path) -> dict:
    arms = {}
    for a in ARM_ORDER:
        p = pkg / f"RESULTS_{a}.json"
        if p.exists():
            arms[a] = json.loads(p.read_text(encoding="utf-8"))

    with open(exp / "RESULTS_attention.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "label", "n_questions", "n_image_tokens", "visual_mass_mean",
                    "visual_mass_last_layer", "emb_norm_visual", "emb_norm_text",
                    "emb_cos_visual_to_text", "nearest_vocab_tokens"])
        for a, d in arms.items():
            g = d["agg"]
            w.writerow([a, ARM_LABEL[a], len(d["per_question"]), g["n_image_tokens"],
                        round(g["visual_mass_mean"], 6), round(g["visual_mass_last_layer"], 6),
                        round(g["emb_norm_visual"], 4), round(g["emb_norm_text"], 4),
                        round(g["emb_cos_visual_to_text"], 6),
                        " ".join(t.strip() or "␠" for t in g["nearest_vocab_tokens"])])

    with open(exp / "RESULTS_attention_per_question.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "q", "dataset", "ood", "gold", "n_image_tokens",
                    "visual_mass_mean", "visual_mass_last_layer",
                    "emb_norm_visual", "emb_norm_text"])
        for a, d in arms.items():
            for q in d["per_question"]:
                w.writerow([a, q["i"], q["ds"], q["ood"], q["gold"], q["n_image_tokens"],
                            round(q["visual_mass_mean"], 6),
                            round(q["visual_mass_last_layer"], 6),
                            round(q["emb_norm_visual"], 4), round(q["emb_norm_text"], 4)])

    with open(exp / "RESULTS_attention_curve.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "layer", "visual_mass"])
        for a, d in arms.items():
            for li, v in enumerate(d["agg"]["visual_mass_curve"]):
                w.writerow([a, li, round(v, 6)])
    return arms


def build(pkg: Path, exp: Path, out_html: Path) -> Path:
    pkg, exp = Path(pkg), Path(exp)
    arms = write_csvs(pkg, exp)
    marg = margin_analysis(pkg, exp, arms)
    questions = json.loads((pkg / "questions.json").read_text(encoding="utf-8"))
    present = [a for a in ARM_ORDER if a in arms]

    n_tok = arms[present[0]]["agg"]["n_image_tokens"]
    rows_g, cols_g = grid_shape(n_tok, questions[0]["width"], questions[0]["height"])

    curves = {a: arms[a]["agg"]["visual_mass_curve"] for a in present}
    n_layers = len(curves[present[0]])
    ymax = max(max(c) for c in curves.values()) * 1.15

    def spark(vals: list[float]) -> str:
        pts = " ".join(
            f"{i / (n_layers - 1) * 300:.1f},{80 - v / ymax * 76:.1f}"
            for i, v in enumerate(vals)
        )
        return f'<polyline points="{pts}" fill="none" stroke="currentColor" stroke-width="2"/>'

    cards = []
    for q in questions:
        i = q["i"]
        frame = _jpg_b64(pkg / "frames" / f"q{i:02d}.jpg")
        tiles = []
        for a in present:
            heat = np.load(pkg / "heat" / f"{a}_q{i:02d}.npy")
            overlay = _heat_png(heat, rows_g, cols_g, (q["width"], q["height"]))
            vm = next(x for x in arms[a]["per_question"] if x["i"] == i)
            tiles.append(
                f'<figure><figcaption><b>{a}</b>'
                f'<span class="n">mass {vm["visual_mass_mean"]:.4f}</span></figcaption>'
                f'<div class="stack"><img src="data:image/jpeg;base64,{frame}" alt="">'
                f'<img class="ov" src="data:image/png;base64,{overlay}" alt=""></div></figure>'
            )
        cards.append(
            f'<section class="q"><h3>q{i:02d} · {q["ds"]} · '
            f'{"OOD" if q["ood"] else "ID"} · gold <b>{q["gold"]}</b></h3>'
            f'<p class="qt">{q["question"]}</p>'
            f'<div class="tiles">{"".join(tiles)}</div></section>'
        )

    trows = "".join(
        f'<tr><td><code>{a}</code></td><td>{ARM_LABEL[a]}</td>'
        f'<td class="num">{arms[a]["agg"]["visual_mass_mean"]:.4f}</td>'
        f'<td class="num">{arms[a]["agg"]["visual_mass_last_layer"]:.4f}</td>'
        f'<td class="num">{arms[a]["agg"]["emb_norm_visual"]:.2f}</td>'
        f'<td class="num">{arms[a]["agg"]["emb_norm_text"]:.2f}</td>'
        f'<td class="num">{arms[a]["agg"]["emb_cos_visual_to_text"]:.4f}</td></tr>'
        for a in present
    )
    _mb = marg["arms"].get("base", {}).get("visual_mass_content") or 1.0
    mrows = "".join(
        f'<tr><td><code>{a}</code></td>'
        f'<td class="num">{marg["arms"][a]["margin_fraction"]*100:.1f}%</td>'
        f'<td class="num">{marg["arms"][a]["content_fraction"]*100:.1f}%</td>'
        f'<td class="num">{marg["arms"][a]["visual_mass_raw"]:.4f}</td>'
        f'<td class="num"><b>{marg["arms"][a]["visual_mass_content"]:.4f}</b></td>'
        f'<td class="num">{marg["arms"][a]["visual_mass_content"]/_mb:.2f}x</td>'
        f'<td class="num">{marg["arms"][a]["top5_content"]*100:.1f}%</td>'
        f'<td class="num">{marg["arms"][a]["entropy_content"]:.3f}</td></tr>'
        for a in present
    )
    sparks = "".join(
        f'<div class="sp"><span>{a}</span>'
        f'<svg viewBox="0 0 300 84" preserveAspectRatio="none">{spark(curves[a])}</svg></div>'
        for a in present
    )

    html = f"""<!doctype html><meta charset="utf-8">
<title>Rung 31 — attention probe</title>
<style>
:root{{--bg:#faf9f7;--fg:#1c1b1a;--mut:#6b6864;--line:#e2ded9;--card:#fff}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15140f;--fg:#eae7e1;--mut:#9a958c;--line:#2e2b26;--card:#1d1c17}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:1.5rem;margin:0 0 .3rem}}
.sub{{color:var(--mut);margin:0 0 1.6rem}}
.warn{{background:var(--card);border:1px solid var(--line);border-left:3px solid #c2410c;
 padding:.85rem 1rem;border-radius:6px;margin:0 0 1.6rem;color:var(--mut);font-size:.92rem}}
table{{width:100%;border-collapse:collapse;margin:.5rem 0 2rem;font-size:.93rem}}
th,td{{border-bottom:1px solid var(--line);padding:.5rem .6rem;text-align:left}}
th{{color:var(--mut);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
code{{background:var(--line);padding:.08em .35em;border-radius:3px;font-size:.9em}}
.curves{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1rem;margin-bottom:2.5rem}}
.sp{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.7rem .8rem;color:#c2410c}}
.sp span{{display:block;color:var(--fg);font-weight:600;font-size:.85rem;margin-bottom:.3rem}}
.sp svg{{width:100%;height:74px;display:block}}
section.q{{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:1rem 1.1rem;margin-bottom:1.4rem}}
section.q h3{{margin:0 0 .2rem;font-size:1rem}}
.qt{{margin:0 0 .9rem;color:var(--mut);font-size:.88rem}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.9rem}}
figure{{margin:0}}
figcaption{{display:flex;justify-content:space-between;align-items:baseline;
 font-size:.82rem;margin-bottom:.3rem}}
figcaption .n{{color:var(--mut);font-variant-numeric:tabular-nums}}
.stack{{position:relative;line-height:0;border-radius:6px;overflow:hidden;border:1px solid var(--line)}}
.stack img{{width:100%;height:auto;display:block}}
.stack img.ov{{position:absolute;inset:0;mix-blend-mode:screen}}
</style>
<div class="wrap">
<h1>Rung 31 — where the model looks</h1>
<p class="sub">{len(questions)} questions, paired across {len(present)} checkpoints ·
{n_tok} image tokens on a {cols_g}×{rows_g} patch grid · readout position = last prompt token</p>

<div class="warn"><b>Read with the limits.</b> Raw attention is a weak explanation
(attention×gradient or rollout is stronger). The readout position is one convention among
several. <code>max_pixels</code> is 512×512, <b>not</b> the eval's 1280×720 — the between-arm
comparison holds because every arm uses the same value, but the absolute level does not
transfer to the deployed configuration. This describes internals; it is not a causal claim
about score.</div>

<table><thead><tr><th>arm</th><th>checkpoint</th><th>visual mass</th><th>last layer</th>
<th>‖visual‖</th><th>‖text‖</th><th>cos v↔t</th></tr></thead><tbody>{trows}</tbody></table>

<h2 style="font-size:1.05rem;margin:0 0 .2rem">Black margin vs scene content</h2>
<p class="sub" style="margin-bottom:.6rem">Observation by <b>legokna</b>, measured here —
<code>local/fuentes/analisis mascaras de atencion rung 31.md</code>. The letterbox is
<b>{marg["margin_area_fraction"]*100:.1f}%</b> of the frame, so that is the chance level.</p>
<table><thead><tr><th>arm</th><th>on margin</th><th>on content</th><th>visual mass (raw)</th>
<th>corrected</th><th>vs base</th><th>top-5 patches</th><th>entropy</th></tr></thead><tbody>
{mrows}</tbody></table>

<h2 style="font-size:1.05rem;margin:0 0 .2rem">Visual mass over depth</h2>
<p class="sub" style="margin-bottom:.8rem">layer 0 → {n_layers - 1}, shared y-axis</p>
<div class="curves">{sparks}</div>

{"".join(cards)}
</div>"""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"viewer -> {out_html}  ({out_html.stat().st_size / 1e6:.2f} MB)")
    return out_html
