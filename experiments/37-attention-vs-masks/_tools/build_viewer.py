"""Rung 37 — build the self-contained adjudication surface for G-BOUNDARY.

Folder-private glue. Importable; a notebook cell calls ``build(Config(...))``.

One HTML file, images embedded as base64, no server and nothing to install — the same contract as
`docs/viewers/`. Three panels per frame, the same pixels in all three:

* **raw** — what the model saw in arm A
* **masks** — SAM's instances, toggleable, opacity slider, so *covered* and *clean* can be judged
* **overlay** — exactly the image arm B was fed, so the pilot's 0.40 agreement can be judged by
  eye rather than argued: is the drop occlusion, or is it the masks?

🔴 **The adjudication records THREE marks, not two.** `PLAN.md`'s gate has `covered` and `clean`
only, and that is what the dry run showed to be broken: on a metallic clip the adjudicator can see
neither the object nor its absence, so a forced binary silently becomes either a `covered=0` that
kills the gate on the instrument, or a skipped frame that inflates it. **`unadjudicable` is the
third state**, and it is exported so the rate is a measurement rather than a hidden bias.

⚠️ This tool does not decide the gate. It records marks and their counts; the thresholds stay in
`PLAN.md`, still unamended, and `RESULTS_gboundary.json` is written by hand from the export — the
number that moves a decision does not get to live only in a browser tab.
"""

from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


@dataclass
class Config:
    run_dir: Path = Path("experiments/37-attention-vs-masks/runs/37_masks_v1")
    ab_dir: Path = Path("experiments/37-attention-vs-masks/runs/37_ab_overlay_v1")
    out: Path = Path("experiments/37-attention-vs-masks/runs/gboundary_viewer.html")
    #: downscale the embedded PNGs; the masks are judged on shape, not on texture
    max_width: int = 960


def _b64(img, fmt: str = "JPEG", quality: int = 80) -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def build(cfg: Config) -> Path:
    from PIL import Image

    run_dir, ab_dir = Path(cfg.run_dir), Path(cfg.ab_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    cards = []
    for r in meta:
        tag = r["tag"]
        img = Image.open(run_dir / "frames" / f"{tag}_a.jpg").convert("RGB")
        if img.width > cfg.max_width:
            img = img.resize((cfg.max_width, round(img.height * cfg.max_width / img.width)))
        labels = np.load(run_dir / "labels" / f"{tag}_a.npz")["labels"]

        # the mask layer as one RGBA PNG: colour per instance, transparent background
        lab = np.asarray(Image.fromarray(labels).resize(img.size, Image.NEAREST))
        rgba = np.zeros((*lab.shape, 4), np.uint8)
        for k in range(1, int(lab.max()) + 1):
            m = lab == k
            if m.any():
                rgba[m, :3] = _PALETTE[(k - 1) % len(_PALETTE)]
                rgba[m, 3] = 255

        over_p = ab_dir / "overlays" / f"{tag}.jpg"
        over = None
        if over_p.exists():
            o = Image.open(over_p).convert("RGB")
            if o.width > cfg.max_width:
                o = o.resize((cfg.max_width, round(o.height * cfg.max_width / o.width)))
            over = _b64(o)

        cards.append({
            "tag": tag, "slice": r.get("slice", ""), "cls": r.get("asked_class", ""),
            "ds": r["ds"], "n_masks": r["n_masks"],
            "question": r.get("question", ""), "gold": r.get("gold", ""),
            "pair": r.get("pair", {}).get("persistence"),
            "raw": _b64(img), "mask": _b64(Image.fromarray(rgba), "PNG"), "over": over,
        })

    payload = html.escape(json.dumps(cards), quote=False).replace("</", "<\\/")
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print(f"{len(cards)} cards -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


_TEMPLATE = r"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>rung 37 — G-BOUNDARY</title><style>
:root{--bg:#12151a;--fg:#e8ecf1;--mut:#96a0ad;--line:#262c36;--ok:#3fb950;--no:#f85149;--unk:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:20;background:#0d1117ee;backdrop-filter:blur(8px);
border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:650}
.pill{background:#1c2230;border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12px;color:var(--mut)}
button{background:#1c2230;color:var(--fg);border:1px solid var(--line);border-radius:7px;
padding:5px 11px;cursor:pointer;font-size:13px}button:hover{border-color:#3d4756}
button.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
.card{border-bottom:1px solid var(--line);padding:16px}
.hd{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:9px}
.tag{font-family:ui-monospace,monospace;font-weight:650}
.q{color:var(--mut);font-size:13px;flex:1 1 100%}
.q b{color:#dbe3ec;font-weight:600}
.panes{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.pane{position:relative}
.pane h4{margin:0 0 5px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);font-weight:600}
.stack{position:relative;line-height:0;background:#000;border-radius:7px;overflow:hidden}
.stack img{width:100%;display:block}
.stack img.ov{position:absolute;inset:0}
.marks{display:flex;gap:7px;margin-top:10px;flex-wrap:wrap;align-items:center}
.marks span{font-size:11px;color:var(--mut);margin-right:2px}
button.cov.on{background:var(--ok);border-color:var(--ok);color:#04180a}
button.cln.on{background:#1f6feb;border-color:#1f6feb}
button.unk.on{background:var(--unk);border-color:var(--unk);color:#231a02}
button.non.on{background:var(--no);border-color:var(--no)}
#sum{padding:14px 16px;background:#0d1117;border-top:1px solid var(--line);position:sticky;bottom:0}
pre{background:#0b0f14;border:1px solid var(--line);border-radius:7px;padding:11px;
overflow:auto;font-size:12px;max-height:220px;margin:8px 0 0}
label{font-size:12px;color:var(--mut);display:flex;gap:6px;align-items:center}
input[type=range]{width:110px}
</style></head><body>
<header>
  <h1>rung 37 · G-BOUNDARY</h1>
  <span class="pill" id="cnt"></span>
  <label>opacidad <input type="range" id="op" min="0" max="100" value="55"></label>
  <button id="tgl" class="on">máscaras</button>
  <button id="tov" class="on">overlay del A/B</button>
  <button id="flt">solo sin marcar</button>
  <button id="exp">exportar veredicto</button>
</header>
<div id="list"></div>
<div id="sum"></div>
<script id="d" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('d').textContent);
const M = {};
const el = (h)=>{const t=document.createElement('template');t.innerHTML=h.trim();return t.content.firstChild};

function card(c,i){
  const n = el(`<div class="card" data-i="${i}">
    <div class="hd"><span class="tag">${c.tag}</span>
      <span class="pill">${c.cls||c.slice}</span><span class="pill">${c.ds}</span>
      <span class="pill">${c.n_masks} instancias</span>
      ${c.pair!=null?`<span class="pill">persistencia ${c.pair.toFixed(3)}</span>`:''}
      <div class="q"><b>P:</b> ${c.question||'—'} &nbsp; <b>gold:</b> ${c.gold||'—'}</div></div>
    <div class="panes">
      <div class="pane"><h4>cruda</h4><div class="stack"><img src="${c.raw}"></div></div>
      <div class="pane"><h4>máscaras SAM</h4><div class="stack"><img src="${c.raw}">
        <img class="ov mk" src="${c.mask}"></div></div>
      ${c.over?`<div class="pane tov"><h4>overlay (lo que vio el brazo B)</h4>
        <div class="stack"><img src="${c.over}"></div></div>`:''}
    </div>
    <div class="marks">
      <span>covered</span>
      <button class="cov" data-v="1">sí</button><button class="non" data-v="0">no</button>
      <span style="margin-left:12px">clean</span>
      <button class="cln" data-v="1">sí</button><button class="non cl0" data-v="0">no</button>
      <button class="unk" style="margin-left:12px">no adjudicable</button>
    </div></div>`);
  const set=(k,v)=>{M[c.tag]=M[c.tag]||{};M[c.tag][k]=v;paint();sum();};
  const bs=n.querySelectorAll('.marks button');
  bs[0].onclick=()=>set('covered',1); bs[1].onclick=()=>set('covered',0);
  bs[2].onclick=()=>set('clean',1);   bs[3].onclick=()=>set('clean',0);
  bs[4].onclick=()=>{const m=M[c.tag]||{};set('unadjudicable',m.unadjudicable?0:1);};
  return n;
}
function paint(){
  document.querySelectorAll('.card').forEach(n=>{
    const c=D[+n.dataset.i], m=M[c.tag]||{}, b=n.querySelectorAll('.marks button');
    b[0].classList.toggle('on',m.covered===1); b[1].classList.toggle('on',m.covered===0);
    b[2].classList.toggle('on',m.clean===1);   b[3].classList.toggle('on',m.clean===0);
    b[4].classList.toggle('on',!!m.unadjudicable);
  });
}
function sum(){
  const g=D.filter(c=>c.slice==='g_boundary');
  const mk=c=>M[c.tag]||{};
  const un=g.filter(c=>mk(c).unadjudicable).length;
  const dec=g.filter(c=>!mk(c).unadjudicable&&mk(c).covered!=null);
  const cov=dec.filter(c=>mk(c).covered===1);
  const cl=cov.filter(c=>mk(c).clean!=null);
  const b1=dec.length?cov.length/dec.length:NaN;
  const b2=cl.length?cl.filter(c=>mk(c).clean===1).length/cl.length:NaN;
  const f=x=>isNaN(x)?'—':x.toFixed(3);
  document.getElementById('cnt').textContent =
    `${g.filter(c=>mk(c).covered!=null||mk(c).unadjudicable).length}/${g.length} marcados`;
  document.getElementById('sum').innerHTML =
    `<b>B1</b> covered ${f(b1)} <span style="color:var(--mut)">(${cov.length}/${dec.length} adjudicables)</span> ·
     <b>B2</b> clean|cov ${f(b2)} <span style="color:var(--mut)">(n=${cl.length})</span> ·
     <b style="color:var(--unk)">no adjudicables ${un}/${g.length}</b>
     <span style="color:var(--mut)"> — recuentos, NO el veredicto. Los umbrales viven en PLAN.md y siguen sin enmendar.</span>`;
}
const L=document.getElementById('list');
D.forEach((c,i)=>L.appendChild(card(c,i)));
document.getElementById('op').oninput=e=>
  document.querySelectorAll('.mk').forEach(x=>x.style.opacity=e.target.value/100);
document.getElementById('op').dispatchEvent(new Event('input'));
document.getElementById('tgl').onclick=e=>{e.target.classList.toggle('on');
  document.querySelectorAll('.mk').forEach(x=>x.style.display=e.target.classList.contains('on')?'':'none');};
document.getElementById('tov').onclick=e=>{e.target.classList.toggle('on');
  document.querySelectorAll('.tov').forEach(x=>x.style.display=e.target.classList.contains('on')?'':'none');};
document.getElementById('flt').onclick=e=>{e.target.classList.toggle('on');
  const on=e.target.classList.contains('on');
  document.querySelectorAll('.card').forEach(n=>{const c=D[+n.dataset.i],m=M[c.tag]||{};
    n.style.display=(on&&(m.covered!=null||m.unadjudicable))?'none':'';});};
document.getElementById('exp').onclick=()=>{
  const rows=D.map(c=>({tag:c.tag,slice:c.slice,cls:c.cls,...(M[c.tag]||{})}));
  const pre=el('<pre></pre>'); pre.textContent=JSON.stringify(rows,null,1);
  document.getElementById('sum').appendChild(pre); pre.scrollIntoView({block:'nearest'});};
sum();
</script>
</body></html>
"""
