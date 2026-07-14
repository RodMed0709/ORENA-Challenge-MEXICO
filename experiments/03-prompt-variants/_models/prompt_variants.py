"""Rung 03 — prompt-variant experiment engine (FRAME).

Single-variable sweep over SYSTEM_PROMPT additions vs the shipped baseline (a1_v0).
Selection on val_id (2252 Q, cholecystectomy), held-out confirmation on val_ood
(4000 Q, Sigmoid resection). Reuses ``frame.run.run_baseline`` by monkeypatching
``frame.engine.SYSTEM_PROMPT`` per arm, and reads the SDK's video-then-question
hierarchical bootstrap CIs straight from each run's ``summary.csv`` (the harness
computes them; we keep them for the pre-registered decision rule).

Arms (each changes ONE thing vs a1_v0):
- a0_noFO       : baseline instruction WITHOUT the FO definitions. Recovers the
                  FO-grounding delta that shipped into v0 un-A/B'd (adversarial
                  review, finding 2).
- a1_v0         : the shipped baseline == frame.engine.SYSTEM_PROMPT (BASE + FO).
- a2_decisive   : v0 + commit-to-one-answer / never-hedge.
- a3_instrument : v0 + explicit instrument-vs-FO tie-break (when unsure -> instrument).
- a4_negexem    : v0 + worked negative disambiguation examples.
- a5_baredigit  : v0 + silent-enumerate-then-bare-integer counting protocol.

Decision protocol (pre-registered, adversarial review finding 5):
  1. Rank all arms on val_id. Winner must show a BROAD lift (headline AND
     fo_class/number moving together), not one chole-specific leaf.
  2. Confirm winner + a1_v0 on the held-out val_ood.
  3. Ship the winner ONLY if its val_ood CI (headline and fo_class/number) does
     not overlap a1_v0's. Otherwise log a faithful negative.
"""
from __future__ import annotations

import csv
from pathlib import Path

import frame.engine as _eng
from frame.config import BaselineConfig
from frame.run import run_baseline
from frame import split as sp
from focus.foreign_objects import FO_DEFINITIONS_FILE

# Derive BASE from the shipped prompt so a1_v0 is byte-identical to the baseline.
_FO = FO_DEFINITIONS_FILE.read_text()
_FULL = _eng.SYSTEM_PROMPT
BASE = _FULL[: _FULL.rindex(_FO)]  # instruction block, before the FO definitions

DECISIVE = (
    'ANSWERING: Always commit to a single best answer. Never refuse, never say '
    'cannot be determined or unclear. Pick the most likely answer from the evidence.\n'
)
INSTRUMENT = (
    'CRITICAL DISAMBIGUATION: Reusable instruments that stay connected to the outside '
    '(graspers, scissors, dissectors, hooks, trocars, staplers, cameras, suction or '
    'irrigation tips, energy devices such as LigaSure or Harmonic Scalpel) are NEVER '
    'foreign objects, no matter how metallic or object-like they look. If unsure whether '
    'something is an instrument or a foreign object, treat it as an instrument and do not '
    'count it. Only count an item as a foreign object if it matches one of the classes '
    'above AND is fully detached from any hand-held tool.\n'
)
NEGEXEM = (
    'WORKED EXAMPLES (internal rules only, never repeat these in your answer):\n'
    '- Energy device (LigaSure/Harmonic Scalpel) touching tissue -> instrument, not an FO.\n'
    '- Metal clip still loaded in the clip applier jaws, not yet released -> not yet an FO.\n'
    '- Metal clip released and lying free in the abdomen -> FO, class Clip.\n'
    '- Only the suture thread visible, needle itself off-screen -> Needle is NOT visible.\n'
    '- Specimen bag pulled through a trocar with its string visible -> the FO is the bag only, ignore the string.\n'
)
BAREDIGIT = (
    'COUNTING: silently enumerate each distinct foreign object visible in THIS frame '
    '(merged instances of the same physical object count once). Then output ONLY the final '
    'integer, nothing else: no words, no list, no punctuation. If nothing in the frame '
    'matches a foreign-object class, output 0.\n'
)


def build_prompts() -> dict:
    """The 6 arms as full SYSTEM_PROMPT strings. a1_v0 == the shipped baseline."""
    return {
        'a0_noFO': BASE,
        'a1_v0': BASE + _FO,
        'a2_decisive': BASE + _FO + '\n\n' + DECISIVE,
        'a3_instrument': BASE + _FO + '\n\n' + INSTRUMENT,
        'a4_negexem': BASE + _FO + '\n\n' + NEGEXEM,
        'a5_baredigit': BASE + _FO + '\n\n' + BAREDIGIT,
    }


def _read_summary_ci(run_dir: Path) -> dict:
    """Pull (accuracy, ci_low, ci_high, count) for the overall + per-answer_format
    rows from a completed run's summary.csv (SDK hierarchical bootstrap CIs)."""
    out: dict = {}
    p = Path(run_dir) / 'summary.csv'
    with open(p, newline='') as fh:
        for row in csv.DictReader(fh):
            lvl, name = row['level'], row['name']

            def f(k):
                v = row.get(k)
                return float(v) if v not in (None, '') else float('nan')

            rec = (f('accuracy'), f('ci_low'), f('ci_high'), int(row['count']))
            if lvl == 'overall':
                out['overall'] = rec
            elif lvl == 'answer_format':
                out[name] = rec
    return out


def run_arms(
    arms: list,
    split: str,
    out_dir,
    data_root='/workspace/orena-data',
    model_path='/workspace/models/qwen3-vl-8b',
    manifest='/workspace/repo/experiments/splits/frame_ood_v1.csv',
) -> list:
    """Run each named arm on the given split (val_id | val_ood), returning a list of
    dicts with headline + fo_class + number accuracy and their bootstrap CIs."""
    vs = sp.load_manifest(Path(manifest))
    vids = {k for k, s in vs.items() if s == split}
    prompts = build_prompts()
    assert prompts['a1_v0'] == _eng.SYSTEM_PROMPT, 'a1_v0 must equal the shipped baseline prompt'

    rows = []
    for name in arms:
        _eng.SYSTEM_PROMPT = prompts[name]
        cfg = BaselineConfig(
            data_root=Path(data_root), model_path=Path(model_path),
            out_dir=Path(out_dir), run_name=f'{split}_{name}',
        )
        rep = run_baseline(cfg, video_filter=vids)
        ci = _read_summary_ci(Path(out_dir) / f'{split}_{name}')
        ov = ci.get('overall', (rep['raw_accuracy'], float('nan'), float('nan'), rep['n_questions']))
        fo = ci.get('fo_class', (None, None, None, None))
        num = ci.get('number', (None, None, None, None))
        rows.append({
            'arm': name, 'split': split, 'n': ov[3],
            'acc': ov[0], 'ci_low': ov[1], 'ci_high': ov[2],
            'fo_class': fo[0], 'fo_lo': fo[1], 'fo_hi': fo[2],
            'number': num[0], 'num_lo': num[1], 'num_hi': num[2],
        })
        print(
            'ARMDONE %s [%s]: acc=%.4f CI[%.3f,%.3f] fo_class=%s number=%s'
            % (name, split, ov[0], ov[1], ov[2], fo[0], num[0]),
            flush=True,
        )
    return rows


def write_rows(rows: list, path) -> None:
    cols = ['arm', 'split', 'n', 'acc', 'ci_low', 'ci_high',
            'fo_class', 'fo_lo', 'fo_hi', 'number', 'num_lo', 'num_hi']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
