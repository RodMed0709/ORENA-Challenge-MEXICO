"""Rung 45 — exercise the four things the engine copy changed, with no GPU and no model.

The engine is a copy of rung 40's with four edits: a 4-bit flag, per-corpus sha pins, three
new arm declarations, and a DDP-aware effective batch. Each of those can fail silently in a
way that only shows up hours into a 27B run, so each gets checked here for free.

The DDP case is the one that matters most and is the easiest to get wrong: reusing rung 40's
literal `grad_accum 16` under `torchrun --nproc 2` trains at effective batch 32 with no error
of any kind. `world_size()` reads `WORLD_SIZE` from the environment, so this file can simulate
two GPUs on a laptop by setting it.

Run: `python test_arms.py [corpus_dir]`
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_models"))
import gen36_data_reg_arm as E  # noqa: E402

CORPUS_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/uaq_user/storage/rung45/corpus"
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def expect_raise(name: str, fn) -> None:
    try:
        fn()
    except E.ArmFailure as e:
        check(name, True, str(e).split("\n")[0][:90])
    except Exception as e:  # noqa: BLE001
        check(name, False, f"raised {type(e).__name__}, wanted ArmFailure: {e}")
    else:
        check(name, False, "did NOT raise — the guard is asleep")


def main() -> int:
    print("\n== 1. the three arms declare exactly what they move")
    for arm in ("R00", "R0", "R1"):
        cfg = _cfg(arm)
        diff = E.assert_single_variable(cfg)
        check(f"{arm} diff == declared {sorted(E.ARMS[arm])}",
              set(diff) == E.ARMS[arm], f"moved {sorted(diff)}")
    check("R1's control is R0, not the baseline", E.CONTROL_OF["R1"] == "R0")
    check("R00 moves nothing", E.ARMS["R00"] == set())

    print("\n== 2. an undeclared change is refused")
    bad = _cfg("R00")
    bad.lora_alpha = 16
    expect_raise("R00 + alpha 16 raises", lambda: E.assert_single_variable(bad))
    expect_raise("unknown arm raises", lambda: E.assert_single_variable(_cfg("R2")))

    print("\n== 3. effective batch survives DDP")
    for ws, want in ((1, 16), (2, 8), (4, 4)):
        os.environ["WORLD_SIZE"] = str(ws)
        got = _cfg("R00").gradient_accumulation_steps
        check(f"world_size {ws} -> grad_accum {want}", got == want, f"got {got}")
    os.environ["WORLD_SIZE"] = "2"
    c = _cfg("R00")
    check("2 GPUs still train at effective batch 16",
          c.per_device_train_batch_size * c.gradient_accumulation_steps * E.world_size() == 16)
    os.environ["WORLD_SIZE"] = "5"
    expect_raise("a world size that does not divide 16 raises",
                 lambda: _cfg("R00").gradient_accumulation_steps)
    os.environ["WORLD_SIZE"] = "1"

    print("\n== 4. each corpus label resolves to its OWN pinned bytes")
    check("R00 and R0 point at different files",
          _cfg("R00").train_jsonl != _cfg("R0").train_jsonl)
    for arm in ("R00", "R0"):
        cfg = _cfg(arm)
        if not Path(cfg.train_jsonl).exists():
            print(f"  SKIP  {arm} sha — {cfg.train_jsonl} not on this host")
            continue
        info = E.assert_dataset(cfg)
        check(f"{arm} sha + row count match the pin", True,
              f"{info['n_rows']} rows, {info['sha256_rehosted'][:12]}")
    swapped = _cfg("R0")
    swapped.corpus = "r18_14415"          # label says one file, arm declares the other
    check("a label swap changes the resolved path",
          swapped.train_jsonl == _cfg("R00").train_jsonl)

    print("\n== 5. NF4 is on by default and is a config field, not a constant")
    check("R00 loads in 4-bit", _cfg("R00").load_in_4bit is True)
    check("BASELINE declares it too", E.BASELINE["load_in_4bit"] is True)

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    return 1 if FAILS else 0


def _cfg(arm: str) -> "E.ArmConfig":
    kw: dict = {"arm": arm, "corpus_dir": CORPUS_DIR}
    if arm in ("R0", "R1"):
        kw["corpus"] = "r42_19384"
    if arm == "R1":
        kw["lora_dropout"] = 0.1
    return E.ArmConfig(**kw)


if __name__ == "__main__":
    raise SystemExit(main())
