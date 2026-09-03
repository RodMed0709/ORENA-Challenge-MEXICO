import json,csv,re
from pathlib import Path
SC=Path("/tmp/claude-1000/-mnt-datos-code-ai-ORENA-proy-ORENA-Challenge-MEXICO/d1ef043a-9338-4e32-8607-42f916646287/scratchpad")
R=Path("/mnt/datos/code/ai/ORENA/proy/ORENA-Challenge-MEXICO/experiments/48-centre-probe")
meta={}
for l in (R/"runs/probe_items_v2.jsonl").read_text().splitlines():
    d=json.loads(l); meta[d["qID"]]=d
def ns(s):
    s=(s or "").strip().lower()
    if s in ("none","","no foreign objects","none.","nothing"): return frozenset()
    return frozenset(p.strip().strip(".") for p in re.split(r"[,;]| and ",s) if p.strip())
M={}
for f in sorted((SC/"v2").glob("*.csv")):
    if f.stat().st_size==0: continue
    with f.open() as fh:
        rd=csv.DictReader(fh); M[f.stem]={r["qID"]:ns(r["prediction"]) for r in rd}
veto=set()
with (SC/"veto_preds.csv").open() as fh:
    for r in csv.DictReader(fh):
        if r["yn"]=="False": veto.add((r["parent"], r["cls"].strip().lower()))
Q=sorted(set(meta)&set.intersection(*[set(d) for d in M.values()]))
G={q:ns(meta[q]["gold"]) for q in Q}
vids=sorted({meta[q]["video"] for q in Q})
def V(q,s): return frozenset(c for c in s if (q,c) not in veto)
b=lambda q: M['r42'][q]
def acc(fn,S=None):
    S=S or Q; return sum(fn(q)==G[q] for q in S)/len(S)
base=acc(b)
def jack(fn):
    w=0;ds=[]
    for v in vids:
        S=[q for q in Q if meta[q]["video"]!=v]
        d=acc(fn,S)-acc(b,S); ds.append(d); w+= d>0
    return w,min(ds),max(ds)
ENS=['r42','r42_ep2','r42_ep3','r42_ep5','a2']
arms=[
 ("r42 sola (lo enviado)", b),
 ("conjunto 5 versiones (mas corta)", lambda q: min([M[m][q] for m in ENS],key=len)),
 ("r42 + auto-veto", lambda q: V(q,M['r42'][q])),
 ("r42+veto, luego mas corta con las 5", lambda q: min([V(q,M['r42'][q])]+[M[m][q] for m in ENS[1:]],key=len)),
 ("mas corta de las 5, luego veto", lambda q: V(q,min([M[m][q] for m in ENS],key=len))),
 ("r42+veto, luego mas corta con ep2+a2", lambda q: min([V(q,M['r42'][q]),M['r42_ep2'][q],M['a2'][q]],key=len)),
]
print(f"n={len(Q)}  ({len(vids)} videos, CholecT50)\n")
print(f"{'arm':44s} {'exact':>7s} {'vs r42':>8s}  folds   min      neg     pos")
out=[]
for n,f in arms:
    a=acc(f); w,mn,mx=jack(f)
    neg=acc(f,[q for q in Q if meta[q]['kind']=='neg']); pos=acc(f,[q for q in Q if meta[q]['kind']=='pos'])
    out.append((a-base,n,a,w,mn,neg,pos))
for d,n,a,w,mn,neg,pos in sorted(out,reverse=True):
    print(f"{n:44s} {a:.4f} {d:+.4f}  {w:2d}/15 {mn:+.4f}  {neg:.4f}  {pos:.4f}")
