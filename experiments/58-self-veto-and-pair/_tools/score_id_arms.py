import csv,re,itertools
from pathlib import Path
SC=Path("/tmp/claude-1000/-mnt-datos-code-ai-ORENA-proy-ORENA-Challenge-MEXICO/d1ef043a-9338-4e32-8607-42f916646287/scratchpad/id")
def ns(s):
    s=(s or "").strip().lower()
    if s in ("none","","no foreign objects","none.","nothing"): return frozenset()
    return frozenset(p.strip().strip(".") for p in re.split(r"[,;]| and ",s) if p.strip())
M={}; gold=None; vids=None
for f in sorted(SC.glob("*.csv")):
    rows=list(csv.DictReader(f.open()))
    M[f.stem]=[ns(r["pred"]) for r in rows]
    if gold is None:
        gold=[ns(r["gold"]) for r in rows]; vids=[r["video"] for r in rows]
N=len(gold); uv=sorted(set(vids))
def acc(fn,idx=None):
    idx=idx if idx is not None else range(N); idx=list(idx)
    return sum(fn(i)==gold[i] for i in idx)/len(idx)
base=acc(lambda i: M['r42'][i])
ENS=['r42','r42_ep2','r42_ep3','r42_ep5','a2']
arms=[(m,(lambda i,m=m: M[m][i])) for m in ENS]
arms += [
 ("conj 5 (mas corta)", lambda i: min([M[m][i] for m in ENS],key=len)),
 ("r42+ep2 (mas corta)", lambda i: min([M['r42'][i],M['r42_ep2'][i]],key=len)),
 ("r42+a2 (mas corta)", lambda i: min([M['r42'][i],M['a2'][i]],key=len)),
 ("r42+ep2+a2 (mas corta)", lambda i: min([M[m][i] for m in ['r42','r42_ep2','a2']],key=len)),
 ("r42+ep5 (mas corta)", lambda i: min([M['r42'][i],M['r42_ep5'][i]],key=len)),
]
print(f"ID · {N} preguntas fo_class · {len(uv)} videos held-out de rung 42\n")
print(f"{'arm':26s} {'ID':>7s} {'vs r42':>8s}  folds")
out=[]
for n,f in arms:
    a=acc(f); folds=[acc(f,[i for i in range(N) if vids[i]!=v])-acc(lambda i: M['r42'][i],[i for i in range(N) if vids[i]!=v]) for v in uv]
    out.append((a-base,n,a,sum(d>0 for d in folds),min(folds)))
for d,n,a,w,mn in sorted(out,reverse=True):
    print(f"{n:26s} {a:.4f} {d:+.4f}  {w}/8  min {mn:+.4f}")
