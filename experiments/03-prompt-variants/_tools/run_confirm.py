"""Folder-private driver: CONFIRMATION stage of rung 03.

Reads select_results.csv (val_id ranking), picks the winning challenger by a
BROAD-lift rule (highest headline acc among challengers; flags whether fo_class
AND number also moved up vs a1_v0 — a chole-only quirk is not a real win), then
re-evaluates [winner, a1_v0] on the HELD-OUT val_ood (4000 Q) and applies the
pre-registered ship rule: ship only if the winner's val_ood CI does not overlap
a1_v0's on the headline. Also reports a0_noFO vs a1_v0 = the FO-grounding delta.
"""
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%H:%M:%S')

import prompt_variants as pv

OUT = Path('/workspace/repo/experiments/03-prompt-variants/runs')


def _load(path):
    rows = list(csv.DictReader(open(path, newline='')))
    for r in rows:
        for k in ('n',):
            r[k] = int(r[k])
        for k in ('acc', 'ci_low', 'ci_high', 'fo_class', 'fo_lo', 'fo_hi', 'number', 'num_lo', 'num_hi'):
            r[k] = float(r[k]) if r[k] not in ('', 'None', None) else None
    return {r['arm']: r for r in rows}


def _overlap(lo1, hi1, lo2, hi2):
    if None in (lo1, hi1, lo2, hi2):
        return True  # unknown CI -> treat as overlapping (conservative: do not ship)
    return lo1 <= hi2 and lo2 <= hi1


sel = _load(OUT / 'select_results.csv')
v0 = sel['a1_v0']
noFO = sel.get('a0_noFO')
challengers = [a for a in sel if a not in ('a1_v0', 'a0_noFO')]
winner = max(challengers, key=lambda a: sel[a]['acc'])
w = sel[winner]

print('\n=== val_id selection summary ===')
print('a1_v0 (baseline): acc=%.4f  fo_class=%s  number=%s' % (v0['acc'], v0['fo_class'], v0['number']))
if noFO:
    print('a0_noFO (strip FO defs): acc=%.4f  -> FO-grounding delta = %+.4f'
          % (noFO['acc'], v0['acc'] - noFO['acc']))
print('WINNER challenger on val_id: %s  acc=%.4f (v0 %.4f, delta %+.4f)'
      % (winner, w['acc'], v0['acc'], w['acc'] - v0['acc']))
broad = (w['fo_class'] is not None and v0['fo_class'] is not None and w['fo_class'] >= v0['fo_class']
         and w['number'] is not None and v0['number'] is not None and w['number'] >= v0['number'])
print('broad lift (fo_class AND number not worse than v0)?', broad)

# Confirm winner + baseline on held-out val_ood
print('\n=== confirming [%s, a1_v0] on held-out val_ood ===' % winner)
rows = pv.run_arms([winner, 'a1_v0'], 'val_ood', OUT)
pv.write_rows(rows, OUT / 'confirm_results.csv')
cw = next(r for r in rows if r['arm'] == winner)
cv = next(r for r in rows if r['arm'] == 'a1_v0')

overlap = _overlap(cw['ci_low'], cw['ci_high'], cv['ci_low'], cv['ci_high'])
ship = (not overlap) and (cw['acc'] > cv['acc']) and broad

print('\n=== VAL_OOD (held-out) RESULT ===')
print('a1_v0 : acc=%.4f CI[%.3f,%.3f]  fo_class=%s number=%s' % (cv['acc'], cv['ci_low'], cv['ci_high'], cv['fo_class'], cv['number']))
print('%-7s: acc=%.4f CI[%.3f,%.3f]  fo_class=%s number=%s' % (winner, cw['acc'], cw['ci_low'], cw['ci_high'], cw['fo_class'], cw['number']))
print('headline delta = %+.4f | CIs overlap = %s | broad lift = %s' % (cw['acc'] - cv['acc'], overlap, broad))
print('VERDICT:', 'SHIP %s' % winner if ship else 'FAITHFUL NEGATIVE (prompt does not beat baseline beyond noise)')
print('CONFIRM_DONE', flush=True)
