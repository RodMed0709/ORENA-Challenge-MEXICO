"""Rung 14 — the off-pod verification. Folder-private (EXPERIMENT_REPO_STRUCTURE_SPEC).

Everything in rung 14 that does NOT need swift, CUDA, decord, the model weights or
the frame cache is checked here, on synthetic pixels, so the pod SMOKE only has to
cover the things that genuinely need the pod.

Run: ``python experiments/14-appearance-aug/_tools/test_appearance.py``
(also importable; ``main()`` returns the number of failures).

Covered:
  T1  flag OFF returns the SAME object (the gate, not an assertion)
  T2  determinism: same (seed, qID) → identical pixels, across two runs and two
      fresh interpreter states; different qIDs → different draws
  T3  the drawn dose matches the pre-registered marginals (Afifi 1/5 identity,
      severity-0 1/3)
  T4  the operators do what their citations say (WB warms/cools, brightness
      raises V, contrast compresses about the mean) and 5500 K is the identity
  T5  JSONL rewriting: OFF path untouched, ON path differs in ``images`` only
  T6  the within-video statistic returns NULL on data whose association is purely
      BETWEEN video, while the pooled correlation happily reports a big number —
      the rung-12d failure mode, reproduced and caught
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_EXP = Path(__file__).resolve().parents[1]
for _p in (_EXP / "_models", _EXP.parents[1] / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import appearance as A  # noqa: E402
import quality_diag as Q  # noqa: E402


def _synth(seed: int = 0, h: int = 96, w: int = 128) -> Image.Image:
    """A deterministic synthetic 'surgical-ish' frame: warm tissue field, a few
    bright desaturated specular blobs, a dark instrument bar."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 3), dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    arr[..., 0] = 150 + 40 * np.sin(xx / 11.0)
    arr[..., 1] = 70 + 25 * np.cos(yy / 9.0)
    arr[..., 2] = 65 + 20 * np.sin((xx + yy) / 13.0)
    for _ in range(6):
        cy, cx = rng.integers(6, h - 6), rng.integers(6, w - 6)
        arr[cy - 3:cy + 3, cx - 3:cx + 3] = 250
    arr[h // 3:h // 3 + 5, :] = 30
    arr += rng.normal(0, 3, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _mean_hue_ratio(img: Image.Image) -> float:
    """R/B mean ratio — rises when the image is warmed, falls when cooled."""
    a = np.asarray(img, dtype=np.float64)
    return float(a[..., 0].mean() / max(1e-6, a[..., 2].mean()))


# ── tests ────────────────────────────────────────────────────────────────────

def t1_flag_off(log) -> bool:
    img = _synth()
    g = A.gate_flag_off_identity(img)
    log(f"T1 flag-OFF identity            : {g}")
    return bool(g["PASS"])


def t2_determinism(log) -> bool:
    img = _synth()
    pol = A.AugPolicy()
    keys = [f"heico__{i:04d}" for i in range(24)] + [f"lapchole__{i:04d}" for i in range(24)]
    g = A.gate_determinism(img, pol, keys)
    log(f"T2 determinism (in-process)     : {g}")

    # Cross-process: a fresh interpreter must produce the same draws. This is the
    # check that catches a policy keyed on Python's per-process-salted hash().
    code = (
        "import sys,json;sys.path.insert(0,r'%s');import appearance as A;"
        "p=A.AugPolicy();print(json.dumps([A.draw(p,k) for k in %r]))"
        % (str(_EXP / "_models"), keys)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    fresh = json.loads(out.stdout)
    here = [A.draw(pol, k) for k in keys]
    cross = fresh == here
    log(f"T2 determinism (fresh process)  : identical_draws={cross}")

    # Two independent AugPolicy() instances must agree too (no hidden state).
    same_pol = [A.draw(A.AugPolicy(), k) for k in keys] == here
    log(f"T2 determinism (new policy obj) : identical_draws={same_pol}")
    return bool(g["PASS"]) and cross and same_pol


def t3_dose(log) -> bool:
    pol = A.AugPolicy()
    keys = [f"lapchole__{i}" for i in range(20000)]
    d = A.dose_table(pol, keys)
    log(f"T3 dose wb_K                    : {d['wb_K']}")
    log(f"T3 dose illum_op                : {d['illum_op']}")
    log(f"T3 dose illum_severity          : {d['illum_severity']}")
    log(f"T3 frac fully untouched         : {d['frac_untouched']:.4f}  (expect ~0.0667 = 1/15)")
    wb_ok = all(abs(v / d["n"] - 0.2) < 0.02 for v in d["wb_K"].values())
    sev0 = d["illum_severity"][0] / d["n"]
    sev_ok = abs(sev0 - 1 / 3) < 0.02
    unt_ok = abs(d["frac_untouched"] - 1 / 15) < 0.01
    ops_ok = A.gate_no_forbidden_ops(pol)["PASS"]
    log(f"T3 marginals ok                 : wb={wb_ok} sev0={sev_ok} untouched={unt_ok} "
        f"no_forbidden_ops={ops_ok}")
    return wb_ok and sev_ok and unt_ok and ops_ok


def t4_operators(log) -> bool:
    img = _synth()
    base_ratio = _mean_hue_ratio(img)
    warm = _mean_hue_ratio(A.wb_error(img, 2850))
    cool = _mean_hue_ratio(A.wb_error(img, 7500))
    ident = A.wb_error(img, 5500)
    log(f"T4 R/B ratio 2850K/base/7500K   : {warm:.4f} / {base_ratio:.4f} / {cool:.4f}")
    wb_ok = warm > base_ratio > cool and ident is img

    v = lambda im: np.asarray(im.convert("RGB"), dtype=np.float64).max(axis=2).mean() / 255.0
    up1, up2 = v(A.brightness(img, 1, +1)), v(A.brightness(img, 2, +1))
    dn1 = v(A.brightness(img, 1, -1))
    log(f"T4 mean V base/+s1/+s2/-s1      : {v(img):.4f} / {up1:.4f} / {up2:.4f} / {dn1:.4f}")
    br_ok = dn1 < v(img) < up1 < up2

    sd = lambda im: float(np.asarray(im, dtype=np.float64).std())
    c1, c2 = sd(A.contrast(img, 1)), sd(A.contrast(img, 2))
    log(f"T4 std base/contr s1/contr s2   : {sd(img):.2f} / {c1:.2f} / {c2:.2f}  "
        f"(ImageNet-C c=0.40 / 0.30)")
    ct_ok = c2 < c1 < sd(img)

    # the identity draw must be a true no-op on BOTH axes
    noop = A.apply_params(img, {"wb_K": 5500, "illum_op": "none", "illum_severity": 0})
    id_ok = noop is img
    log(f"T4 identity draw is a no-op     : {id_ok}")

    # a typo must raise, never silently no-op (rung 12 enhance.apply doctrine)
    try:
        A.apply_params(img, {"wb_K": 5500, "illum_op": "unsharp", "illum_severity": 1})
        typo_ok = False
    except ValueError:
        typo_ok = True
    log(f"T4 unknown op raises            : {typo_ok}")
    return wb_ok and br_ok and ct_ok and id_ok and typo_ok


def t5_jsonl(log) -> bool:
    import aug_export as AE

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cache, augd = td / "frames_cache", td / "aug_frames"
        cache.mkdir()
        rows, index = [], {}
        for i in range(12):
            qid = f"{'heico' if i % 2 else 'lapchole'}__{i:04d}"
            fname = f"{'heico' if i % 2 else 'lapchole'}__vid{i // 3}__{i}.jpg"
            _synth(i).save(cache / fname, quality=95)
            q, ans = f"How many objects are visible? #{i}", str(i % 4)
            rows.append({"messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": f"<image>{q}"},
                {"role": "assistant", "content": ans}],
                "images": [str(cache / fname)]})
            index.setdefault((fname, q, ans), __import__("collections").deque()).append(qid)

        base = td / "train.jsonl"
        base.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")
        base_sha = AE.sha256(base)

        info = AE.rewrite_jsonl_with_aug(
            base, td / "train_aug.jsonl", augd, index, A.AugPolicy(),
            manifest_csv=td / "aug_manifest.csv")
        log(f"T5 rewrite                      : {info['n_rows']} rows, "
            f"{info['unconsumed_qids']} unconsumed qIDs")

        untouched = AE.sha256(base) == base_sha
        shape = AE.gate_aug_jsonl_shape(base, td / "train_aug.jsonl", augd)
        log(f"T5 base jsonl untouched         : {untouched}")
        log(f"T5 G-SHAPE                      : {shape}")

        man = pd.read_csv(td / "aug_manifest.csv")
        log(f"T5 manifest cols                : {list(man.columns)}  n={len(man)}")

        # a row the index cannot resolve must RAISE, not fall back to a frame key
        bad = td / "bad.jsonl"
        bad.write_text(json.dumps({"messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "<image>unknown question"},
            {"role": "assistant", "content": "x"}],
            "images": [str(cache / "lapchole__vid0__0.jpg")]}) + "\n", encoding="utf-8")
        try:
            AE.rewrite_jsonl_with_aug(bad, td / "bad_aug.jsonl", augd, {}, A.AugPolicy())
            raises = False
        except KeyError:
            raises = True
        log(f"T5 unmatched row raises         : {raises}")

        # re-running must reproduce the same augmented bytes (determinism end-to-end)
        again = td / "train_aug2.jsonl"
        idx2 = {}
        for i in range(12):
            qid = f"{'heico' if i % 2 else 'lapchole'}__{i:04d}"
            fname = f"{'heico' if i % 2 else 'lapchole'}__vid{i // 3}__{i}.jpg"
            idx2.setdefault((fname, f"How many objects are visible? #{i}", str(i % 4)),
                            __import__("collections").deque()).append(qid)
        AE.rewrite_jsonl_with_aug(base, again, td / "aug_frames2", idx2, A.AugPolicy())
        same_pixels = all(
            AE.sha256(augd / p.name) == AE.sha256(p)
            for p in sorted((td / "aug_frames2").glob("*.jpg")))
        log(f"T5 rerun byte-identical frames  : {same_pixels}")

        return (untouched and shape["PASS"] and raises and same_pixels
                and info["n_rows"] == 12 and len(man) == 12)


def t6_within_video(log) -> bool:
    """The rung-12d trap, built on purpose.

    Construction: every video has its own quality level AND its own accuracy, and
    they are correlated ACROSS videos — but INSIDE each video, quality is pure
    noise with zero relation to correctness. The pooled correlation must look
    strong; the within-video delta must be a null with a CI covering 0.
    """
    rng = np.random.default_rng(7)
    rows = []
    for v in range(38):                       # effective n ≈ 38 videos
        q_level = rng.normal(100, 30)
        p_correct = float(np.clip(0.15 + 0.007 * (q_level - 100) + 0.5, 0.05, 0.95))
        for _ in range(160):
            rows.append({"video": f"v{v}", "correct": float(rng.random() < p_correct),
                         "blur_lapvar": q_level + rng.normal(0, 4)})
    df = pd.DataFrame(rows)

    w = Q.within_video_delta(df, "blur_lapvar", n_boot=1000, seed=1)
    p = Q.pooled_correlation(df, "blur_lapvar")
    log(f"T6 pooled r (CONFOUNDED)        : {p['pooled_point_biserial_r']:+.4f}  n={p['n']}")
    log(f"T6 within-video delta           : {w['delta']:+.4f}  CI[{w['ci_low']:+.4f},"
        f" {w['ci_high']:+.4f}]  videos={w['n_videos']}  excludes0={w['excludes_zero']}")
    trap_caught = abs(p["pooled_point_biserial_r"]) > 0.10 and not w["excludes_zero"]

    # positive control: a REAL within-video effect must be detected
    df2 = df.copy()
    df2["blur_lapvar"] = df2["blur_lapvar"] + 12.0 * df2["correct"]
    w2 = Q.within_video_delta(df2, "blur_lapvar", n_boot=1000, seed=1)
    log(f"T6 positive control             : {w2['delta']:+.4f}  CI[{w2['ci_low']:+.4f},"
        f" {w2['ci_high']:+.4f}]  excludes0={w2['excludes_zero']}")

    # the scorer itself, on synthetic frames
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(4):
            _synth(i).save(td / f"f{i}.jpg", quality=95)
        sc = Q.scan_frames(sorted(td.glob("*.jpg")))
        log(f"T6 scan_frames                  :\n{sc.to_string(index=False)}")
        scan_ok = len(sc) == 4 and sc[list(Q.SCORES)].notna().all().all()

    return trap_caught and w2["excludes_zero"] and scan_ok


def main() -> int:
    lines: list[str] = []

    def log(msg):
        print(msg)
        lines.append(str(msg))

    tests = [("T1 flag-off", t1_flag_off), ("T2 determinism", t2_determinism),
             ("T3 dose", t3_dose), ("T4 operators", t4_operators),
             ("T5 jsonl", t5_jsonl), ("T6 within-video", t6_within_video)]
    fails = 0
    for name, fn in tests:
        log(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        ok = fn(log)
        log(f"--> {name}: {'PASS' if ok else 'FAIL'}")
        fails += 0 if ok else 1
    log(f"\n{'=' * 72}\n{len(tests) - fails}/{len(tests)} PASS, {fails} FAIL")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
