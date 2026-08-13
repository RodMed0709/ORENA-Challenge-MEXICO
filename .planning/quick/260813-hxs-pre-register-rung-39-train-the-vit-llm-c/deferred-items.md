# Deferred items — quick task 260813-hxs

Out-of-scope discoveries. Found, **not** fixed (SCOPE BOUNDARY: only issues directly caused by this
task's changes are auto-fixed).

## 1. Three committed `.sh` files pre-date this task

The plan's verification item 4 asks for `git ls-files '*.sh'` to be empty. It returns three:

```
submissions/02-rung21-a2/do_build.sh
submissions/02-rung21-a2/do_save.sh
submissions/02-rung21-a2/do_test_run.sh
```

They belong to the submission-02 Docker packaging, not to any experiment, and they were committed
long before this task. **Nothing this task produced is a `.sh`** — the rung-39 chain is rendered into
`/workspace/tmp` by `experiments/39-connector-lora/_tools/chain.py` and never committed.

Deferred, not fixed: deleting or converting another rung's submission tooling is a separate decision
with its own blast radius (the submission is scored and shipped). Worth one quick task to decide
whether `submissions/*/do_*.sh` is an intended exception to the no-`.sh` rule — and if it is, to say
so in `CLAUDE.md`/`AGENTS.md` so the rule stops reading as violated.
