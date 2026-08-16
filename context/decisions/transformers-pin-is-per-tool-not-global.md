---
question: Does the binding `transformers==4.57.*` pin still describe what we run?
verdict: NO — it describes ONE of our two lines and forbids the other. All four environments that actually train, quantise and serve gen-3.6 run transformers 5.x (5.5.0 to 5.15.0), and they work: ms_swift 4.4.1 sits at 5.12.1, inside its own `<5.13` cap, and the vendored SDK imports clean under 5.15.0. The pin conflated "5.x" with ">=5.13". It stays for the Qwen3-VL-8B line and is superseded for gen-3.6
status: SETTLED
date: 2026-08-16
basis: zero GPU — `pip freeze` of the four UNAM environments (`requirements/unam-*.lock`) plus a live import of the SDK under transformers 5.15.0
---

# Decision: the `transformers` pin is per-tool, not global

- **Status:** SETTLED · 2026-08-16 · **zero GPU**
- **Applies when:** anyone builds an environment from `CLAUDE.md`'s stack table, or cites the
  4.57 pin to rule out a gen-3.6 experiment.

## Question

`CLAUDE.md` fixes `transformers==4.57.*` and says, in the *What NOT to use* table, that 5.x
"breaks ms-swift cap (`<5.13`) and `qwen-vl-utils`". Meanwhile every machine that has actually
produced a gen-3.6 result runs 5.x. Which is right?

## What it gave us

The four environments on UNAM, frozen in `requirements/unam-*.lock`:

| environment | job | transformers | the tool it has to satisfy |
|---|---|---|---|
| `orena-gen36` | training, ms-swift | **5.12.1** | `ms_swift 4.4.1` — cap is `<5.13`, so **5.12.1 is inside it** |
| `orena-unsloth` | 27B training | **5.5.0** | `unsloth 2026.8.15`, `peft 0.20.0`, `trl 0.24.0` |
| `orena-quant` | FP8 | **5.14.1** | `llmcompressor 0.13.0` |
| `orena-vllm` | serving + eval | **5.15.0** | `vllm 0.27.1` |

Three of the pin's four premises do not hold:

1. **"5.x breaks the ms-swift cap."** The cap is `<5.13`. `5.12.1` satisfies it. The pin
   conflated *"5.x"* with *">= 5.13"*; only the latter is excluded.
2. **"5.x breaks `qwen-vl-utils`."** `qwen-vl-utils` is **not installed in any environment we
   own** — nothing we run imports it.
3. **"5.x breaks the SDK."** Verified false by import, today, under transformers 5.15.0:
   `focus.data.data_models`, `focus.evaluation.evaluator` and `focus.taxonomy` all load and
   `Capability.from_any` is callable. That is the path every rung-43 job took.
4. ✅ **What IS true:** gen-3.6 is a different model class (`Qwen3_5ForConditionalGeneration`,
   `model_type: qwen3_5`) and 4.57 cannot load it at all. The 5.x line is not a preference there,
   it is the floor.

## The one real migration cost, and it is bounded

Rung 38 hit it: **5.x's `apply_chat_template` indexes every message's `content` for
`content["type"]`**, so a message carrying a bare string raises `TypeError: string indices must be
integers`. Fixed in place with typed content parts, which both template generations accept. That
is the shape of a 5.x port — real, small, and already paid once.

## Verdict

**Two lines, two pins.**

- **Qwen3-VL-8B line** (A2, rung 21, rung 42 — what we ship today): `transformers==4.57.*` stands.
  Nothing here is a reason to move a shipping model.
- **gen-3.6 line** (rungs 38 / 40 / 43 / 44): `transformers>=5.5,<5.13` when ms-swift is in the
  loop, and whatever the serving stack requires when it is not. `requirements/unam-*.lock` is the
  record of what actually ran.

⚠️ **This is a documentation correction, not a licence to upgrade the 8B.** The shipped submission
is built on the 4.57 line and nothing here proposes moving it. What is retired is the blanket
prohibition, which was making the documented stack unbuildable for half our own experiments.

## Sources

- `requirements/unam-gen36.lock`, `unam-unsloth.lock`, `unam-quant.lock`, `unam-vllm.lock`.
- `experiments/38-gen36-ft-screen/README.md` — the `apply_chat_template` breakage and its fix.
- Related: [[backbone-generation-is-not-the-lever]] ·
  [[gen36-fails-the-8b-recipe-not-the-backbone-test]].
