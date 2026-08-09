"""Rung 35 — the ms-swift adapter for NTL-WAS. The seam `src/frame/loss.py` could not verify.

Loaded by ms-swift via ``--external_plugins <this file>`` and selected with
``--loss_type ntl_was``. Verified against the installed build, ``ms_swift==4.4.1``:
``trainers/mixin.py:1053`` does ``res['compute_loss_func'] = loss_map[args.loss_type](args, self)``,
and ``swift/loss/base.py`` fixes the constructor as ``__init__(args, trainer)`` and the call as
``__call__(outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs)``.

🔴 **The CE half is `CustomCrossEntropyLoss` byte-for-byte** (``swift/loss/causal_lm.py``), on
purpose. It calls the same ``per_token_loss_func`` with the same ``num_items_in_batch`` fallback,
so at ``lam = 0`` this class is arithmetically the framework's own default and the arm's ONLY
delta is the ordinal term. Re-deriving the CE here would have made "we changed the loss" and "we
changed the objective" the same edit — two variables wearing one flag.

## The rung-22 lesson, wired in

Rung 22 shipped a hook that was an arithmetic identity and whose trigger never fired, and it cost
the whole arm. So this class **counts its own firings** and exposes them: if ``n_fired`` is zero
after a smoke, the term never touched a gradient and no full run may be scheduled. A silent no-op
that trains to a perfect null is the failure mode this rung is most exposed to.

## The digit ids are resolved from the trainer's own tokenizer, not hardcoded

`src/frame/loss.py` never imports a tokenizer — that is what keeps it unit-testable offline. The
binding happens here, once, from ``trainer.processing_class``, and it **raises** if any digit is
not a single token: a silently multi-token digit would make every distance in the Wasserstein term
wrong while the run looked healthy.
"""

from __future__ import annotations

import os
import sys

for _p in ("/workspace/repo_rodri/src",):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from frame.loss import ntl_was  # noqa: E402
from swift.loss import BaseLoss, loss_map  # noqa: E402


class NTLWasLoss(BaseLoss):
    """``CE + lam * NTL-WAS``, with the CE half identical to the framework default."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        tok = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        if tok is None:
            raise RuntimeError("no tokenizer on the trainer; cannot resolve the digit token ids")
        tok = getattr(tok, "tokenizer", tok)

        ids, vals = [], []
        for d in range(10):
            enc = tok.encode(str(d), add_special_tokens=False)
            if len(enc) != 1:
                raise AssertionError(
                    f"digit {d!r} encodes to {len(enc)} tokens ({enc}); every distance in the "
                    "Wasserstein term would be wrong while the run looked healthy."
                )
            ids.append(enc[0])
            vals.append(float(d))
        self.digit_token_ids, self.digit_values = ids, vals

        # lam lives in the env so the arm is a launcher-visible variable, not a code edit
        self.lam = float(os.environ.get("NTL_LAMBDA", "0.3"))
        # 🔴 rung 22: a hook that never fires is worse than no hook. Counted, and asserted
        # by the smoke before any full run is scheduled.
        self.n_calls = 0
        self.n_fired = 0
        self.ntl_sum = 0.0
        print(f"[ntl_was] digit ids {ids} | lam {self.lam}", flush=True)

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        from swift.trainers import per_token_loss_func

        # ---- CE: swift/loss/causal_lm.py::CustomCrossEntropyLoss, unchanged ----------
        token_loss = per_token_loss_func(outputs, labels)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        loss = token_loss.sum() / num_items_in_batch

        self.n_calls += 1
        if self.lam == 0.0:      # the null arm: arithmetically the framework default
            return loss

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        ntl = ntl_was(logits, labels, self.digit_token_ids, self.digit_values)
        v = float(ntl.detach())
        if v > 0.0:
            self.n_fired += 1
            self.ntl_sum += v
        return loss + self.lam * ntl

    @property
    def stats(self) -> dict:
        return {"n_calls": self.n_calls, "n_fired": self.n_fired,
                "fire_rate": self.n_fired / max(self.n_calls, 1),
                "ntl_mean_when_fired": self.ntl_sum / max(self.n_fired, 1),
                "lam": self.lam}


loss_map["ntl_was"] = NTLWasLoss
