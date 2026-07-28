"""Gate — ``frame.metrics.class_f1_report`` must reproduce a macro-F1 we already have.

A new metric is only trustworthy if it lands on a number produced independently of it.
``experiments/18-count-aug/RESULTS_objrec_probe0.txt`` holds an ad-hoc macro-F1 computed
during rung 18's build, from the same committed answers, by code that no longer exists:

    06 ep3  fo_class ID  n=920  exact-set acc=0.6391  MACRO-F1=0.5116
      Clip 406 r=0.781 p=0.836 f1=0.808 | Specimen 367 r=0.831 p=0.772 f1=0.801
      Specimen Bag 278 r=0.835 p=0.862 f1=0.848 | Sponge 218 r=0.734 p=0.812 f1=0.771
      Gallstone 28 -> 0.000 | External Drain 24 r=0.250 p=0.600 f1=0.353 | Needle 1 -> 0.000

This asserts the canonical module returns exactly that off rung 06 ep3's ``predictions.json``.
Zero GPU, no judge, no network — pure pandas over the committed answers and the QA parquets.

Run it: ``python experiments/21-loss-mass/_tools/test_class_f1.py`` (RAISES on mismatch,
prints one OK line otherwise). It is a gate, not a suite: RULES §7, gates raise.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src", ROOT / "vendor" / "orena-focus" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from frame import metrics  # noqa: E402
from frame.ledger import gold_from_frame_parquets  # noqa: E402

RUN_DIR = ROOT / "experiments" / "06-vit-lora" / "runs" / "06_vit_lora_v1" / "ep3_full"
DATA_ROOT = ROOT / "external_data" / "orena-data"

# probe0's numbers. `macro_f1` is asserted to 3 dp because probe0 printed 4 and its own
# per-class F1s were rounded before the mean; the per-class table is exact.
EXPECT_ID = {"n": 920, "exact_set_acc": 0.6391, "macro_f1": 0.5116}
EXPECT_CLASSES = {
    "Clip": (406, 0.781, 0.836, 0.808),
    "Specimen": (367, 0.831, 0.772, 0.801),
    "Specimen Bag": (278, 0.835, 0.862, 0.848),
    "Sponge": (218, 0.734, 0.812, 0.771),
    "Gallstone": (28, 0.000, 0.000, 0.000),
    "External Drain": (24, 0.250, 0.600, 0.353),
    "Needle": (1, 0.000, 0.000, 0.000),
}


def main() -> None:
    if not (RUN_DIR / "predictions.json").exists():
        raise FileNotFoundError(
            f"{RUN_DIR}/predictions.json is missing — runs/ is gitignored; pull rung 06 "
            "ep3's answers from the pod before running this gate."
        )
    preds = metrics.predictions_frame(RUN_DIR)
    gold = gold_from_frame_parquets(DATA_ROOT, "test")
    rep = metrics.class_f1_report(preds, gold, n_boot=0)
    cell = rep["ID"]

    if cell["n"] != EXPECT_ID["n"]:
        raise AssertionError(f"fo_class x ID n = {cell['n']}, expected {EXPECT_ID['n']}")
    if cell["n_illegal"]:
        raise AssertionError(
            f"{cell['n_illegal']} SDK-illegal predictions on rung 06 ep3 — probe0 measured 0"
        )
    for key, tol in (("exact_set_acc", 1e-4), ("macro_f1", 1e-3)):
        got, want = cell[key], EXPECT_ID[key]
        if abs(got - want) > tol:
            raise AssertionError(f"{key} = {got:.6f}, expected {want} (tol {tol})")

    per = cell["per_class"]
    for name, (n_gold, rec, prec, f1) in EXPECT_CLASSES.items():
        if name not in per:
            raise AssertionError(f"class {name!r} missing from the per-class table")
        v = per[name]
        if v["n_gold"] != n_gold:
            raise AssertionError(f"{name}: n_gold {v['n_gold']}, expected {n_gold}")
        for key, want in (("recall", rec), ("precision", prec), ("f1", f1)):
            if abs(v[key] - want) > 5e-4:
                raise AssertionError(f"{name}.{key} = {v[key]:.4f}, expected {want}")

    # Silicone Loop is the phantom class (435 train examples, 0 gold in val): it must appear
    # in the table carrying its false positives, and must NOT enter the class-balanced mean.
    if per.get("Silicone Loop", {}).get("n_gold", 0) != 0:
        raise AssertionError("Silicone Loop is supposed to have zero gold in val")
    if cell["n_classes_supported"] != len(EXPECT_CLASSES):
        raise AssertionError(
            f"{cell['n_classes_supported']} supported classes, expected {len(EXPECT_CLASSES)} "
            "— an unsupported class leaking into the macro mean makes the headline a "
            "property of the class registry, not of the model"
        )

    print(
        f"OK  class_f1_report reproduces probe0: fo_class x ID n={cell['n']} "
        f"exact={cell['exact_set_acc']:.4f} macro_f1={cell['macro_f1']:.4f} "
        f"over {cell['n_classes_supported']} supported classes"
    )


if __name__ == "__main__":
    main()
