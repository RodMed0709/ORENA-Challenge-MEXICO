"""Folder-private driver: SELECTION stage of rung 03.

Runs all 6 arms on val_id (2252 Q), writes select_results.csv with bootstrap CIs,
prints the ranking. Winner selection (broad lift + CI) and the held-out val_ood
confirmation are done in run_confirm.py once this finishes. Headless entry so the
sweep can run detached on the pod; the notebook mirrors this logic for provenance.
"""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%H:%M:%S')

import prompt_variants as pv

ARMS = ['a0_noFO', 'a1_v0', 'a2_decisive', 'a3_instrument', 'a4_negexem', 'a5_baredigit']
OUT = Path('/workspace/repo/experiments/03-prompt-variants/runs')
OUT.mkdir(parents=True, exist_ok=True)

rows = pv.run_arms(ARMS, 'val_id', OUT)
pv.write_rows(rows, OUT / 'select_results.csv')

rows.sort(key=lambda r: -(r['acc'] or 0))
print('\n=== SELECTION (val_id 2252, ranked by headline acc, with 95% CI) ===')
for r in rows:
    print(
        '  %-14s acc=%.4f CI[%.3f,%.3f]  fo_class=%s  number=%s'
        % (r['arm'], r['acc'], r['ci_low'], r['ci_high'], r['fo_class'], r['number'])
    )
print('SELECT_DONE', flush=True)
