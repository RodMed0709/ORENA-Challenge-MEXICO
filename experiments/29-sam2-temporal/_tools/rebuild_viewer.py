"""Rebuild the SAM 2 mask viewer with a review UI, reusing the images already embedded.

🔴 **This is a RECOVERY tool, and it is currently the only source of the review UI.**
`report.py` is the canonical generator, but it builds from the mask package
(``meta.json`` + ``labels/`` + ``frames/``) and **that package no longer exists** — not on disk and
not on the S3 volume, where only the built HTML survives. So this reads the 32 data URIs and their
captions back out of the page and re-emits them around an interface built for ONE question:

    C1 counted masks. It never checked WHERE they land. Do they land on foreign objects?

That interface — overlay toggle, opacity, zoom to 14x, an A/B blink comparator for C2, a per-frame
verdict and Markdown export — is what made the 2026-08-09 eye pass possible, and that eye pass is
what killed C1 as a clip-count control (`local/fuentes/analisis-mascaras-sam2.md`).

⚠️ **Known gap, recorded not fixed:** ``report.py`` still emits the OLD read-only layout. When the
package is next re-exported (rung 36's `G-BOUNDARY` needs 40 fresh frames), the UI below should be
ported into ``report.py`` and this file deleted. Until then, deleting this file loses the UI.

⚠️ Input and output are the same path. The original is recoverable from git
(``git show 08d3f06^:docs/viewers/sam2_masks_viewer.html``) but re-running this against its own
output will not re-extract — the structure changes. Read the source page first, then write.
"""
import re
import json
from pathlib import Path

SRC = Path("docs/viewers/sam2_masks_viewer.html")
OUT = Path("docs/viewers/sam2_masks_viewer.html")

html = SRC.read_text(encoding="utf-8")


def imgs(block: str) -> list[str]:
    return re.findall(r'src="(data:image/[a-z]+;base64,[A-Za-z0-9+/=]+)"', block)


def txt(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


# ── C1 ────────────────────────────────────────────────────────────────────────
c1 = []
for i, f in enumerate(re.findall(r"<figure>.*?</figure>", html, re.S), 1):
    cap = txt(re.search(r"<figcaption>(.*?)</figcaption>", f, re.S).group(1))
    group, _, rest = cap.partition("lapchole") if "lapchole" in cap else cap.partition("heico")
    ds = "lapchole" if "lapchole" in cap else "heico"
    gold = re.search(r"gold (\d+)", rest or cap)
    n = re.search(r"(\d+) instancias", cap)
    a, b = imgs(f)
    c1.append({"id": f"C1-{i:02d}", "arm": "gold ≥ 5" if "≥" in group else "gold == 1",
               "ds": ds, "gold": gold.group(1) if gold else "?",
               "n": n.group(1) if n else "?", "base": a, "ov": b})

# ── C2 ────────────────────────────────────────────────────────────────────────
c2 = []
for i, s in enumerate(re.findall(r'<section class="pair">.*?</section>', html, re.S), 1):
    h = txt(re.search(r"<h3>(.*?)</h3>", s, re.S).group(1))
    ds = "lapchole" if "lapchole" in h else "heico"
    gold = re.search(r"gold (\d+)", h)
    pers = re.search(r"persistencia ([\d.]+)", h)
    surv = re.search(r"\((\d+/\d+) sobreviven", h)
    iou = re.search(r"IoU medio ([\d.]+)", h)
    a, ao, b, bo = imgs(s)
    c2.append({"id": f"C2-{i:02d}", "ds": ds, "gold": gold.group(1) if gold else "?",
               "pers": pers.group(1) if pers else "?", "surv": surv.group(1) if surv else "?",
               "iou": iou.group(1) if iou else "?",
               "a": a, "ao": ao, "b": b, "bo": bo})

print(f"extracted: {len(c1)} C1 figures, {len(c2)} C2 pairs")

VERDICTS = [("obj", "sobre objetos foráneos"), ("mix", "mixto"),
            ("tis", "sobre tejido / anatomía"), ("idk", "no distingo")]


def radios(cid: str) -> str:
    return "".join(
        f'<label class="v"><input type="radio" name="v_{cid}" value="{k}">{lab}</label>'
        for k, lab in VERDICTS)


c1_cards = "".join(f"""
<figure class="card" data-id="{d['id']}">
  <figcaption>
    <span class="id">{d['id']}</span>
    <span class="arm {'hi' if '≥' in d['arm'] else 'lo'}">{d['arm']}</span>
    <span class="n">{d['ds']} · gold {d['gold']} · <b>{d['n']}</b> instancias</span>
  </figcaption>
  <div class="stack" data-zoom>
    <img src="{d['base']}" alt="">
    <img class="ov" src="{d['ov']}" alt="">
  </div>
  <div class="verdict">{radios(d['id'])}</div>
  <textarea placeholder="qué ves — ¿las manchas caen sobre instrumentos, clips, gasas… o sobre tejido?"
            data-note="{d['id']}"></textarea>
</figure>""" for d in c1)

c2_cards = "".join(f"""
<section class="card pair" data-id="{d['id']}">
  <h3><span class="id">{d['id']}</span> {d['ds']} · gold {d['gold']}
      <span class="n">persistencia <b>{d['pers']}</b> · {d['surv']} sobreviven · IoU medio {d['iou']}</span></h3>
  <div class="blinkwrap">
    <div class="blink stack" data-zoom data-blink>
      <img class="fa" src="{d['a']}" alt=""><img class="ov fa" src="{d['ao']}" alt="">
      <img class="fb" src="{d['b']}" alt=""><img class="ov fb" src="{d['bo']}" alt="">
      <span class="tag">A</span>
    </div>
    <div class="two">
      <div><span class="lbl">A — sembrado</span>
        <div class="stack" data-zoom><img src="{d['a']}" alt=""><img class="ov" src="{d['ao']}" alt=""></div></div>
      <div><span class="lbl">B — propagado (~1/fps después)</span>
        <div class="stack" data-zoom><img src="{d['b']}" alt=""><img class="ov" src="{d['bo']}" alt=""></div></div>
    </div>
  </div>
  <div class="verdict">{radios(d['id'])}</div>
  <textarea placeholder="¿la silueta sigue al MISMO objeto, o se queda pegada a una zona de tejido?"
            data-note="{d['id']}"></textarea>
</section>""" for d in c2)

CSS = """
:root{--bg:#faf9f7;--fg:#1c1b1a;--mut:#6b6864;--line:#e2ded9;--card:#fff;--acc:#0f766e;--warn:#c2410c}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15140f;--fg:#eae7e1;--mut:#9a958c;--line:#2e2b26;--card:#1d1c17;--acc:#5eead4;--warn:#fb923c}}
:root[data-theme=dark]{--bg:#15140f;--fg:#eae7e1;--mut:#9a958c;--line:#2e2b26;--card:#1d1c17;--acc:#5eead4;--warn:#fb923c}
*{box-sizing:border-box}
body{margin:0;padding:0 0 5rem;background:var(--bg);color:var(--fg);
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:1.5rem 1.25rem}
h1{font-size:1.5rem;margin:0 0 .3rem}
h2{font-size:1.15rem;margin:2.4rem 0 .3rem;padding-top:.6rem;border-top:1px solid var(--line)}
.sub{color:var(--mut);margin:0 0 1.2rem}
.task{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);
 padding:.9rem 1rem;border-radius:6px;margin:0 0 1rem}
.task b{color:var(--fg)}
.warn{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
 padding:.85rem 1rem;border-radius:6px;margin:0 0 1.4rem;color:var(--mut);font-size:.92rem}
.classes{display:flex;flex-wrap:wrap;gap:.35rem;margin:.6rem 0 0}
.classes span{background:var(--bg);border:1px solid var(--line);border-radius:99px;
 padding:.1rem .55rem;font-size:.8rem;font-variant-numeric:tabular-nums}
/* sticky toolbar */
.bar{position:sticky;top:0;z-index:50;background:var(--card);border-bottom:1px solid var(--line);
 padding:.55rem 1.25rem;display:flex;gap:1.1rem;align-items:center;flex-wrap:wrap;font-size:.86rem}
.bar label{display:flex;align-items:center;gap:.4rem;color:var(--mut)}
.bar input[type=range]{width:130px}
button{font:inherit;font-size:.86rem;padding:.32rem .7rem;border:1px solid var(--line);
 border-radius:6px;background:var(--bg);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--acc)}
button.pri{background:var(--acc);color:var(--bg);border-color:var(--acc);font-weight:600}
kbd{font:inherit;font-size:.78rem;background:var(--bg);border:1px solid var(--line);
 border-radius:4px;padding:0 .3rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}
.card{margin:0 0 1.1rem;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:.75rem}
.card.done{border-color:var(--acc)}
figcaption,h3{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;
 font-size:.86rem;margin:0 0 .5rem;font-weight:600}
h3{font-size:.95rem}
.id{font-variant-numeric:tabular-nums;color:var(--mut);font-weight:400;margin-right:.4rem}
.arm{padding:.05rem .45rem;border-radius:99px;font-size:.78rem}
.arm.hi{background:#0f766e22;color:var(--acc)} .arm.lo{background:#c2410c22;color:var(--warn)}
.n{color:var(--mut);font-variant-numeric:tabular-nums;font-size:.82rem;font-weight:400}
.stack{position:relative;line-height:0;border-radius:5px;overflow:hidden;border:1px solid var(--line);cursor:zoom-in}
.stack img{width:100%;height:auto;display:block}
.stack img.ov{position:absolute;inset:0;opacity:var(--ovop,.85);transition:opacity .12s}
body.noov .stack img.ov{opacity:0}
.blink{position:relative}
.blink img.fb{position:absolute;inset:0;opacity:0}
.blink.b img.fa{opacity:0} .blink.b img.fb{opacity:1}
.blink.b img.ov.fb{opacity:var(--ovop,.85)}
.blink .tag{position:absolute;top:.4rem;left:.4rem;background:#000a;color:#fff;
 font-size:.75rem;padding:.1rem .45rem;border-radius:4px;line-height:1.4}
.two{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-top:.7rem}
.lbl{display:block;font-size:.78rem;color:var(--mut);margin-bottom:.25rem}
.verdict{display:flex;flex-wrap:wrap;gap:.3rem;margin:.6rem 0 .45rem}
.v{font-size:.8rem;border:1px solid var(--line);border-radius:99px;padding:.15rem .55rem;
 cursor:pointer;color:var(--mut);display:flex;align-items:center;gap:.3rem}
.v:has(input:checked){border-color:var(--acc);color:var(--acc);font-weight:600}
.v input{margin:0}
textarea{width:100%;min-height:2.6rem;font:inherit;font-size:.85rem;padding:.4rem .5rem;
 border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);resize:vertical}
textarea:focus{outline:2px solid var(--acc);outline-offset:-1px}
/* lightbox */
#lb{position:fixed;inset:0;z-index:200;background:#000d;display:none;place-items:center}
#lb.on{display:grid}
#lbi{position:relative;line-height:0;max-width:96vw;max-height:92vh;overflow:hidden;cursor:grab}
#lbi img{max-width:96vw;max-height:92vh;transform-origin:0 0}
#lbi img.ov{position:absolute;inset:0;opacity:var(--ovop,.85)}
body.noov #lbi img.ov{opacity:0}
#lbbar{position:fixed;bottom:1rem;left:50%;transform:translateX(-50%);background:var(--card);
 border:1px solid var(--line);border-radius:8px;padding:.4rem .8rem;font-size:.82rem;color:var(--mut);
 display:flex;gap:.9rem;align-items:center}
@media(max-width:760px){.two{grid-template-columns:1fr}}
"""

JS = """
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const KEY='sam2-review-notes-v1';

/* ── persistence ─────────────────────────────────────────────── */
const load=()=>{try{return JSON.parse(localStorage.getItem(KEY))||{}}catch(e){return{}}};
const save=d=>localStorage.setItem(KEY,JSON.stringify(d));
let notes=load();
function mark(card){
  const id=card.dataset.id, d=notes[id]||{};
  card.classList.toggle('done', !!(d.v||(d.t||'').trim()));
}
$$('textarea[data-note]').forEach(t=>{
  const id=t.dataset.note;
  if(notes[id]?.t) t.value=notes[id].t;
  t.addEventListener('input',()=>{notes[id]={...notes[id],t:t.value};save(notes);mark(t.closest('.card'))});
});
$$('.verdict input').forEach(r=>{
  const id=r.name.slice(2);
  if(notes[id]?.v===r.value) r.checked=true;
  r.addEventListener('change',()=>{notes[id]={...notes[id],v:r.value};save(notes);mark(r.closest('.card'))});
});
$$('.card').forEach(mark);

/* ── overlay opacity + toggle ────────────────────────────────── */
const op=$('#op');
op.addEventListener('input',()=>document.documentElement.style.setProperty('--ovop',op.value/100));
const toggleOv=()=>document.body.classList.toggle('noov');
$('#ovbtn').addEventListener('click',toggleOv);

/* ── blink comparator (C2) ───────────────────────────────────── */
$$('[data-blink]').forEach(b=>{
  const tag=$('.tag',b);
  const flip=()=>{b.classList.toggle('b');tag.textContent=b.classList.contains('b')?'B':'A'};
  b.addEventListener('mouseenter',flip); b.addEventListener('mouseleave',flip);
});

/* ── lightbox with wheel-zoom + drag-pan ─────────────────────── */
const lb=$('#lb'), lbi=$('#lbi');
let z=1,x=0,y=0,drag=null;
function apply(){$$('#lbi img').forEach(i=>i.style.transform=`translate(${x}px,${y}px) scale(${z})`)}
function open(stack){
  lbi.innerHTML='';
  $$('img',stack).forEach(i=>{const c=i.cloneNode();c.style.transform='';lbi.appendChild(c)});
  z=1;x=0;y=0;apply();lb.classList.add('on');
}
$$('[data-zoom]').forEach(s=>s.addEventListener('click',e=>{if(!e.target.closest('input,textarea'))open(s)}));
lb.addEventListener('click',e=>{if(e.target===lb)lb.classList.remove('on')});
lbi.addEventListener('wheel',e=>{e.preventDefault();
  const f=e.deltaY<0?1.15:1/1.15, r=lbi.getBoundingClientRect();
  const mx=e.clientX-r.left, my=e.clientY-r.top;
  x=mx-(mx-x)*f; y=my-(my-y)*f; z=Math.min(14,Math.max(.4,z*f)); apply();},{passive:false});
lbi.addEventListener('pointerdown',e=>{drag={x:e.clientX-x,y:e.clientY-y};lbi.setPointerCapture(e.pointerId);lbi.style.cursor='grabbing'});
lbi.addEventListener('pointermove',e=>{if(!drag)return;x=e.clientX-drag.x;y=e.clientY-drag.y;apply()});
lbi.addEventListener('pointerup',()=>{drag=null;lbi.style.cursor='grab'});

/* ── keyboard ────────────────────────────────────────────────── */
addEventListener('keydown',e=>{
  if(e.target.matches('textarea,input'))return;
  if(e.key==='o'||e.key==='O')toggleOv();
  if(e.key==='Escape')lb.classList.remove('on');
  if(lb.classList.contains('on')){
    if(e.key==='+'||e.key==='='){z=Math.min(14,z*1.2);apply()}
    if(e.key==='-'){z=Math.max(.4,z/1.2);apply()}
    if(e.key==='0'){z=1;x=0;y=0;apply()}
  }
});

/* ── export ──────────────────────────────────────────────────── */
const LABEL={obj:'sobre objetos foráneos',mix:'mixto',tis:'sobre tejido / anatomía',idk:'no distingo'};
function markdown(){
  const L=['# Observaciones a ojo — máscaras SAM 2 (paso 6)','',
    'Escrito desde `docs/viewers/sam2_masks_viewer.html` · '+new Date().toISOString().slice(0,10),'',
    '> La pregunta: C1 contó máscaras y nunca miró DÓNDE caen. Esto es esa mirada.',''];
  let any=false;
  for(const sec of ['C1','C2']){
    const rows=$$('.card').filter(c=>c.dataset.id.startsWith(sec));
    const hdr=sec==='C1'?'## C1 — ¿sobre qué caen las máscaras?':'## C2 — ¿el track sigue al mismo objeto?';
    const body=[];
    rows.forEach(c=>{
      const id=c.dataset.id, d=notes[id]||{};
      if(!d.v&&!(d.t||'').trim())return;
      any=true;
      const meta=$('.n',c)?.textContent.trim()||'';
      body.push(`### ${id} — ${meta}`);
      if(d.v)body.push(`**Veredicto:** ${LABEL[d.v]}`);
      if((d.t||'').trim())body.push('',d.t.trim());
      body.push('');
    });
    if(body.length){L.push(hdr,'',...body)}
  }
  if(!any)L.push('_(sin observaciones todavía)_');
  const tally={};
  Object.values(notes).forEach(d=>{if(d.v)tally[d.v]=(tally[d.v]||0)+1});
  if(Object.keys(tally).length){
    L.push('## Recuento de veredictos','');
    for(const k in LABEL)if(tally[k])L.push(`- **${tally[k]}** ${LABEL[k]}`);
    L.push('');
  }
  return L.join('\\n');
}
$('#dl').addEventListener('click',()=>{
  const b=new Blob([markdown()],{type:'text/markdown'}), u=URL.createObjectURL(b);
  const a=document.createElement('a');a.href=u;a.download='observaciones-mascaras-sam2.md';a.click();
  URL.revokeObjectURL(u);
});
$('#cp').addEventListener('click',async()=>{
  await navigator.clipboard.writeText(markdown());
  $('#cp').textContent='copiado ✓'; setTimeout(()=>$('#cp').textContent='copiar como Markdown',1400);
});
$('#clr').addEventListener('click',()=>{
  if(!confirm('¿Borrar todas las observaciones guardadas?'))return;
  notes={};save(notes);location.reload();
});
"""

page = f"""<!doctype html><html lang="es"><meta charset="utf-8">
<title>Paso 6 — revisión a ojo de las máscaras SAM 2</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>
<div class="bar">
  <b>Revisión a ojo</b>
  <label>opacidad máscara <input type="range" id="op" min="0" max="100" value="85"></label>
  <button id="ovbtn">alternar máscara <kbd>O</kbd></button>
  <span style="color:var(--mut)">clic en una imagen = zoom · rueda = acercar · arrastrar = mover</span>
  <span style="flex:1"></span>
  <button id="cp">copiar como Markdown</button>
  <button class="pri" id="dl">descargar .md</button>
  <button id="clr">borrar</button>
</div>
<div class="wrap">
<h1>Paso 6 — qué ve SAM 2 en nuestros frames</h1>
<p class="sub">C1 separación <b>6.475</b> contra 0.5 · C2 persistencia <b>0.9167</b> contra 0.90.
Esto es lo que hay detrás de esos números.</p>

<div class="task">
  <b>La pregunta que esta página existe para contestar.</b> C1 contó máscaras — <b>nunca miró dónde
  caen</b>. Un 32.0 contra 25.5 es compatible con «SAM encuentra los objetos foráneos» y también con
  «los frames con más clips son escenas más cargadas y salen más manchas, sin tocar un solo clip».
  <b>Las dos historias dan el mismo número.</b> Separarlas necesita un ojo, porque el dataset no
  tiene etiquetas de localización.
  <br><br>
  Por cada frame: ¿las siluetas caen sobre <b>instrumentos y objetos foráneos</b>, o sobre
  <b>tejido y anatomía</b>? Marca el veredicto y escribe lo que veas. Se guarda solo.
</div>

<div class="warn">
  <b>No estás buscando clips.</b> El dataset tiene <b>ocho</b> clases de objeto foráneo, y el
  <b>60.7 %</b> de las preguntas de conteo no nombra ninguna — piden contar objetos de cualquier
  tipo. Las de Clip son el 32.6 % del total. SAM es agnóstico de clase: un color es
  <i>un objeto segmentado</i>, nunca «un Clip».
  <div class="classes">
    <span>Clip 3,631</span><span>Sponge 2,414</span><span>External drain 1,435</span>
    <span>Specimen 1,425</span><span>Specimen bag 1,196</span><span>Silicone loop 1,167</span>
    <span>Needle 728</span><span>Gallstone 65</span>
  </div>
</div>

<h2>C1 — ¿sobre qué caen las máscaras?</h2>
<p class="sub">Un color por instancia sembrada. Los cuatro primeros son <code>gold ≥ 5</code>,
los cuatro últimos <code>gold == 1</code> — si la separación fuera real, deberías ver
<i>más objetos foráneos</i> arriba, no solo más manchas.</p>
<div class="grid">{c1_cards}</div>

<h2>C2 — ¿el track sigue al mismo objeto?</h2>
<p class="sub">Pasa el ratón por la imagen grande para <b>parpadear entre A y B</b> — así se ve si la
silueta se queda en su objeto o resbala. Debajo, las dos por separado.</p>
{c2_cards}
</div>
<div id="lb"><div id="lbi"></div>
  <div id="lbbar">rueda: zoom · arrastrar: mover · <kbd>O</kbd> máscara · <kbd>0</kbd> reset · <kbd>Esc</kbd> cerrar</div>
</div>
<script>{JS}</script>
</html>"""

OUT.write_text(page, encoding="utf-8")
print(f"viewer -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
