# Blind human adjudication — the instrument 16b could not be

## Why this exists

Probe 16b asked whether the `Clips` gold is a count of frame-visible instances, using a frozen
OWLv2 detector. It returned **`r = 0.0459`, MAE 2.95, and no detection at all on 58.6% of
frames** — but the detector finds only **0.67 clips** where the gold says **3.52**, and OWLv2 has
never seen laparoscopic tissue. The positive control (`"a surgical instrument"`) stayed alive
(1.27 detections/frame, 35% zero), so the instrument is not dead — it is simply far too weak to
tell "the label is not visible" apart from "this detector cannot see surgical clips".

That is the confound rung 12 fell into three times. So 16b's verdict is **NOT MEASURABLE**, not
NO-GO, and the question stays open:

> Is the counting gold recoverable from the frame at all?

It is the most expensive open question in the project. If the answer is no, every `number` lever
— minted zeros, pointing, counting recipes — is aimed at a ceiling, and the strategy re-aims to
`object_recognition`.

## The reference points already in the repo

| who | Spearman r vs gold |
|---|---|
| blind human (recorded in `context/ERROR_ANATOMY.md`) | **−0.17** |
| our model (rung 06) | **+0.43** |
| OWLv2, this probe | **+0.046** |

Also on record: the gold moves **±0.86 between frames ≤1 s apart**, and "total instances"
contradicts the sum of per-class counts on **≥11%** of frames.

## How to run it

```
cd local/adjudicate
python -m http.server 8000      # some browsers block fetch() over file://
# open http://localhost:8000/contar.html
```

`local/adjudicate/` is **gitignored**: its 109 JPEGs are copies of `/workspace/frames_cache`
pixels, and the single-source rule says a frame exists once and is called from there. Regenerate
the staging dir from the pod whenever it is needed; only this tool is versioned.

## Design decisions that make the result readable

- **Blind.** The dataset label is hidden until export. Showing it first would anchor the count
  and the measurement would be worthless.
- **«No se puede saber» is a first-class answer.** If a trained human cannot count them, that IS
  the finding. A forced-choice UI would manufacture agreement and hide the ceiling.
- **Stratified by gold, then shuffled.** 109 frames spanning gold 1–12 (12 each up to 4, then 10,
  tapering to 2 at gold 12), over 30 videos, 44 ID / 65 OOD. A sample tracking the natural
  distribution would be dominated by gold 1–3 and would say nothing about where the model
  actually collapses — accuracy is **0% from gold ≥5**. Order is shuffled so position leaks nothing.
- **The detector's own count travels in the export**, so the human, the label and OWLv2 can be
  compared three ways on the same frames.

## Reading the result

| outcome | what it means |
|---|---|
| high «no se puede saber» rate | the label is not recoverable from a single frame → `number` has an annotation ceiling → re-aim |
| low NA, high exact agreement | the label is sound → the deficit is ours → counting levers are live |
| low NA, large \|Δ\| | the label is *visible* but *wrong* → annotation noise, bounded and quantified |

Export writes `adjudicacion_humana.csv` with `human`, `gold`, `delta` and the detector count per
frame. Drop it next to this README and record the verdict in
`context/decisions/` — including, if it lands there, a faithful "not measurable".
