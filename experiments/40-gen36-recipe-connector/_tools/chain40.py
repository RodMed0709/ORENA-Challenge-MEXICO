"""Rung 40 — render the unattended chain. NEVER a committed `.sh`.

Importable renderer, following rung 39's `_tools/chain.py`. What is committed is
this module — reproducible, reviewable, diffable. What it produces is pod scratch
and is deleted when done. A committed `.sh` is forbidden (CLAUDE.md,
CONSTITUTION §VIII.1, §IX.3) and a `.py` launcher is forbidden too, so the chain is
*rendered* and handed to `nohup bash`.

The chain runs ~16 h with nobody awake. Four things it must survive, each of which
has already cost this project something:

* **the pod outliving the work.** A `trap ... EXIT` fires on ANY exit — normal end,
  error, `kill`, crash. "Stops when it finishes" is not the same as "stops no
  matter what", and only the second is safe when legokna is asleep.
* **the disk quota.** `df` LIES on this volume (it reports the MooseFS cluster,
  not the ~670 GB quota). A run already died `Disk quota exceeded` MID-MERGE, after
  training. So: rung 38's merge is deleted BEFORE anything starts, and arm A's is
  deleted before arm B begins.
* **losing 14 h of work to a late failure.** The volume IS the record: it outlives the
  pod and every artifact is readable from S3 with nothing running. 📌 There is no
  commit step — there used to be one, and the 2026-08-14 rehearsal showed it had never
  saved anything, because `.gitignore:34` ignores `experiments/*/runs/**`. It printed
  "committing the evidence" and then `nothing to commit`, twice. Removed: a step that
  prints reassurance and preserves nothing is worse than no step at all.
* **an arm that trains and cannot be scored.** Training and eval live in DIFFERENT
  envs (`unsloth` vs the `focus` SDK in `envs/infer`), and the eval's rc used to be
  ignored outright. Now the eval env is proved before a GPU is touched, and
  `check_eval` stops the chain on a failed score instead of walking past it.
* **a HUNG run.** The `trap` fires on exit; a training that stalls never exits, so
  nothing fires and the pod bills all night. This is the most likely expensive
  failure — likelier than SIGKILL. The watchdog watches the log's **mtime**, which is
  only meaningful because `_Pulse` deliberately writes every 60 s. 🔴 It used to rest
  on "the 27B logs a line every ~24 s", which was FALSE under `papermill --log-output`
  and cost a healthy arm on 2026-08-14. Set `stale_seconds=0` when a human is watching.

The watchdog is rendered as a SEPARATE script and launched with `setsid nohup`, so
it is not a child of the chain. A child would die with a process-group kill, which
is exactly the case it exists to cover.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# The branch this rung lives on. NOT main: rung 40 is unfinished work and the
# branch is deliberately unmerged.
BRANCH = "task/gen36-recipe-and-connector"


@dataclass
class ChainConfig:
    pod_id: str = ""                      # filled in when legokna launches the pod
    repo_root: str = "/workspace/repo_leo"
    exp_dir: str = "experiments/40-gen36-recipe-connector"
    env_python: str = "/workspace/envs/unsloth/bin/python"
    # The eval runs in the SAME env as training, and the seam is kept only because the
    # eval genuinely differs in its HF_HOME (see eval_hf_home).
    #
    # 🔴 The 2026-08-14 detour, recorded so it is not repeated: the eval first died on
    # `ModuleNotFoundError: No module named 'focus'` here, and the obvious fix — point
    # it at envs/infer, which HAS focus installed — was wrong. envs/infer pins
    # transformers 4.57.6, which cannot load `Qwen3_5ForConditionalGeneration` at all
    # (`AutoConfig: KeyError 'qwen3_5'`), so it traded a missing import for a model that
    # will not load, and cost a second rehearsal to find out. envs/judge35 has
    # transformers 5.14.1 but no papermill and no decord. The training env has
    # transformers 5.5.0, papermill, decord and pandas — and `focus` is VENDORED in the
    # repo, which is how rung 38's eval gets it. `eval_arm.ensure_paths()` is the whole
    # fix. Gate 1b now checks the CLASS LOADS, not merely that imports resolve.
    eval_python: str = "/workspace/envs/unsloth/bin/python"
    key_file: str = "/workspace/tmp/leo_runpod_key"
    script_path: str = "/workspace/tmp/leo_chain40.sh"
    log_path: str = "/workspace/tmp/leo_chain40.log"

    # 51.8 GB of rung 38's merged 27B — closed, scored, and regenerable from its
    # adapter. Deleting it takes the volume from ~136 GB free to ~188 GB.
    stale_merge: str = (
        "/workspace/repo_leo/experiments/38-gen36-ft-screen/runs/38_qwen36_27b_v1/merged"
    )

    # 🔴 HF_HOME is EMPTY on the pod and there are TWO caches with different
    # contents: the 27B is in /workspace/hf_cache (61 G) and NOT in
    # /workspace/.cache/huggingface (7.6 G, the judge). Unset, `from_pretrained`
    # re-downloads 52 GB — hours lost and the disk plan broken. Resolved by
    # LOOKING, per [[pod-hf-home-is-not-set]], not by hardcoding a guess.
    hf_home: str = "/workspace/hf_cache"
    model_cache_dir: str = "models--Qwen--Qwen3.6-27B"

    # 🔴 The eval needs the OTHER cache. Its judge is `Qwen/Qwen3-4B`
    # (src/frame/config.py:102) and that model is in /workspace/.cache/huggingface,
    # NOT in /workspace/hf_cache where the 27B lives. Exporting one HF_HOME for the
    # whole chain is right for training and wrong for the judge — which is the exact
    # trap [[pod-hf-home-is-not-set]] records: two caches, different contents, and
    # hardcoding either one breaks a job. So the eval gets its own, and the judge is
    # gated up front rather than discovered after an arm has trained.
    eval_hf_home: str = "/workspace/.cache/huggingface"
    judge_cache_dir: str = "models--Qwen--Qwen3-4B"

    watchdog_path: str = "/workspace/tmp/leo_watchdog40.sh"
    watchdog_log: str = "/workspace/tmp/leo_watchdog40.log"

    run_arm_b: bool = True                # False = stop after arm A and its eval

    # Turn the pod off when the chain exits (trap) and run the watchdog at all.
    # False is for REHEARSALS ONLY, and legokna asked for it for a concrete reason:
    # while debugging, every failed attempt stopped the pod and had to be restarted by
    # hand, which is friction with no information — the trap is already proved, twice,
    # including down the failure branch. 🔴 With this off NOTHING stops the pod: not the
    # trap, not the watchdog. It bills until a human says otherwise. `render` refuses to
    # emit a non-smoke chain with it off, so the debugging setting cannot ride along
    # into the 16 h run — which is exactly the shape of mistake it would be worst in.
    stop_pod_on_exit: bool = True
    expect_unsloth: str = "2026.8.15"
    expect_unsloth_zoo: str = "2026.8.10"

    # Rehearsal mode. The arms run on Qwen3.5-4B against synthetic data
    # (`ArmConfig.smoke`), so the WHOLE chain — papermill, the pulse, the watchdog,
    # the eval and the self-stop — is exercisable end to end on a cheap
    # small GPU before the 27B gets a morning. The rung-40 arm A loss on 2026-08-14
    # was a plumbing failure, not a science failure, and plumbing is testable at
    # 1/8th the price. Must be in EU-RO-1: the volume does not leave its region.
    smoke: bool = False
    smoke_model_cache_dir: str = "models--Qwen--Qwen3.5-4B"
    # Long enough that training outlasts several watchdog checks — a rehearsal that
    # ends before the watchdog looks twice proves nothing about the watchdog.
    smoke_steps: int = 40

    # 🔴 The old premise here — "a stalled 27B is silent; a live one writes a line
    # every ~24 s" — was FALSE, and it cost the 2026-08-14 run. Under
    # `papermill --log-output` a live trainer writes NOTHING: its progress bar is a
    # `display_data` object and papermill only forwards streams. Liveness is now
    # produced deliberately by `_Pulse` in the arm engine (every 60 s, all phases,
    # merge included) instead of being assumed from a side effect.
    stale_seconds: int = 2700             # 45 min without log output => hung
    max_seconds: int = 16 * 3600          # hard wall clock, last resort
    stop_retries: int = 5                 # the stop call itself can fail on a flaky link

    # Run names, kept in ONE place because the arm engine appends `_smoke` itself
    # (`ArmConfig.run_dir`) and the eval must be pointed at the directory the arm
    # actually wrote. Hardcoding them twice is how a rehearsal evaluates an empty dir.
    @property
    def suffix(self) -> str:
        return "_smoke" if self.smoke else ""

    @property
    def run_a(self) -> str:
        return f"40_A_alpha_v1{self.suffix}"

    @property
    def run_b(self) -> str:
        return f"40_B_connector_v1{self.suffix}"


def _arm_params(cfg: ChainConfig, exp: str) -> str:
    """The papermill params both arms take. The notebooks' own defaults are UNAM's;
    the chain always names the machine explicitly so a rehearsal on the pod cannot
    inherit a path from the wrong box."""
    return (
        f'-p WORK_DIR "{exp}/runs" -p HF_HOME "{cfg.hf_home}" '
        f'-p SMOKE_STEPS {cfg.smoke_steps}'
    )


def _shutdown_block(cfg: ChainConfig) -> str:
    """The trap + watchdog, or an explicit note that neither is armed."""
    if not cfg.stop_pod_on_exit:
        # Loud on purpose. The one thing worse than a pod that stops when you did not
        # want it to is a pod that does not stop when you assumed it would.
        return (
            '# --- 0. NOTHING STOPS THIS POD ---------------------------------------------\n'
            'echo "🔴 stop_pod_on_exit=False — no trap, no watchdog. This pod bills until"\n'
            'echo "   a human stops it. Rehearsal setting; render() forbids it when smoke=False."'
        )
    return f"""# --- 0. the pod stops on ANY exit ------------------------------------------------
# Registered FIRST, before anything can fail. Reads the key at fire time and deletes
# the file, so the credential never outlives the run.
K=$(cat {cfg.key_file})
stop_pod() {{
  echo "===== stopping pod {cfg.pod_id} — $(date -u) ====="
  # Retried: the stop call itself can fail on a flaky link, and a single 000 would
  # leave the pod billing all night. Verified by reading the status back, because a
  # 200 alone proves nothing (measured 2026-08-13 on the test pod).
  for i in $(seq 1 {cfg.stop_retries}); do
    C=$(curl -s -o /dev/null -w "%{{http_code}}" -X POST \\
        -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id}/stop)
    echo "stop attempt $i -> HTTP $C"
    sleep 5
    S=$(curl -s -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id} | grep -o '"desiredStatus":"[A-Z]*"')
    echo "  status now: $S"
    case "$S" in *EXITED*) echo "pod confirmed stopped"; break;; esac
    sleep 20
  done
  rm -f {cfg.key_file}
}}
trap stop_pod EXIT

# --- 0b. the watchdog: covers what the trap CANNOT ---------------------------------
# `trap` never fires on SIGKILL, and it never fires on a HUNG run because nothing
# exits. The watchdog is detached with setsid so a process-group kill cannot take it
# with the chain.
setsid nohup bash {cfg.watchdog_path} $$ > {cfg.watchdog_log} 2>&1 &
echo "watchdog detached (pid guard on $$) -> {cfg.watchdog_log}\""""


def render(cfg: ChainConfig) -> str:
    if cfg.stop_pod_on_exit and not cfg.pod_id:
        raise AssertionError(
            "pod_id is empty — the chain would run and then fail to stop the pod, "
            "which is the one failure this whole design exists to prevent."
        )
    if not cfg.stop_pod_on_exit and not cfg.smoke:
        raise AssertionError(
            "stop_pod_on_exit=False is a REHEARSAL setting and this is not a rehearsal "
            "(smoke=False). The real chain runs ~16 h and ends while everyone is asleep; "
            "without the trap the pod bills until someone notices in the morning. If you "
            "genuinely mean it, set smoke=True or change this assertion deliberately."
        )
    exp = f"{cfg.repo_root}/{cfg.exp_dir}"
    arm_b_block = _arm_b(cfg, exp) if cfg.run_arm_b else (
        '\necho "=== arm B skipped by config (run_arm_b=False) ==="\n'
    )
    # The cache gate must check the model that will ACTUALLY be loaded — in rehearsal
    # that is the smoke model, not the 27B. Fatal in BOTH modes: the rehearsal model
    # was chosen *because* it is already cached (legokna's call, 2026-08-14), so a
    # missing one means the model choice changed, and the right answer to that is to
    # stop and say so, not to quietly spend a download on it.
    shutdown_block = _shutdown_block(cfg)
    arm_params = _arm_params(cfg, exp)
    wanted_model = cfg.smoke_model_cache_dir if cfg.smoke else cfg.model_cache_dir
    why = ("a rehearsal runs on an ALREADY CACHED model — do not download one"
           if cfg.smoke else
           "training would re-download 52 GB and blow the disk plan")
    model_missing = (
        f'  echo "FATAL: {wanted_model} is not in {cfg.hf_home}/hub."\n'
        f'  echo "{why}. Fix HF_HOME or the model choice before relaunching."\n'
        f'  exit 1'
    )
    return f"""#!/usr/bin/env bash
# RENDERED by _tools/chain40.py — do not edit here, and do not commit this file.
set -uo pipefail          # NOT -e: a failing arm must still reach its handler and the trap
echo "===== chain40 start $(date -u) ====="

{shutdown_block}

cd {exp} || exit 1

# --- 0c. HF_HOME, and PROOF the model is really in it ------------------------------
# Exported before anything imports transformers. The assert exists because the
# failure mode is silent and expensive: an unset HF_HOME does not error, it
# re-downloads 52 GB.
export HF_HOME={cfg.hf_home}
if [ ! -d "{cfg.hf_home}/hub/{wanted_model}" ]; then
{model_missing}
fi
echo "HF_HOME={cfg.hf_home} — {wanted_model} present"

# --- 1. the versions the gates were measured against -----------------------------
# A mismatch means the merge gate and G4 do not transfer, so nothing below is
# interpretable. Fail here, before a 6 h arm.
{cfg.env_python} - <<'PY' || exit 1
import importlib.metadata as md, sys
want = {{"unsloth": "{cfg.expect_unsloth}", "unsloth_zoo": "{cfg.expect_unsloth_zoo}"}}
bad = {{p: md.version(p) for p in want if md.version(p) != want[p]}}
if bad:
    print("VERSION MISMATCH:", bad, "expected", want)
    sys.exit(1)
print("versions OK:", want)
PY

# --- 1b. the EVAL environment, proved before a GPU is touched ---------------------
# 🔴 The 2026-08-14 rehearsal trained an arm to completion and only THEN discovered
# the eval could not import `focus`. Cost at rehearsal scale: 15 minutes. Cost at 27B
# scale: 16 hours of training and zero scores. The eval's imports are checkable in two
# seconds, so they are checked first — a gate that runs after the expensive part is
# not a gate, it is a postmortem.
if [ ! -x "{cfg.eval_python}" ]; then
  echo "FATAL: eval python {cfg.eval_python} is missing or not executable."
  exit 1
fi
{cfg.eval_python} - <<'PY' || exit 1
import glob, json, sys
sys.path.insert(0, "{cfg.repo_root}/{cfg.exp_dir}/_tools")
try:
    # The SAME sys.path setup score() uses — not a re-creation of it.
    from eval_arm import ensure_paths
    ensure_paths("{cfg.repo_root}")
    from focus.enums import Track            # the import that failed on 2026-08-14
    from frame.run import run_baseline       # what eval_arm.score() actually calls
    from screen_engine import GenericVLMEngine   # noqa: F401
except Exception as e:
    print(f"EVAL ENV BROKEN: {{type(e).__name__}}: {{e}}")
    print("The arms would train for hours and produce no score. Fix the env first.")
    sys.exit(1)

# 🔴 Imports resolving is NOT enough, and believing it was cost a whole rehearsal.
# Under envs/infer every import above succeeded and the eval still died inside
# `GenericVLMEngine.load()` with `AutoConfig: KeyError 'qwen3_5'` — this transformers
# simply has no entry for the architecture. So ask the question that actually matters:
# can THIS interpreter map the model type the merged checkpoint will carry? The merge
# does not exist yet, but it inherits `model_type` from the base, which is right here.
cfgs = sorted(glob.glob("{cfg.hf_home}/hub/{wanted_model}/snapshots/*/config.json"))
if not cfgs:
    print("EVAL ENV GATE: cannot read a config.json for {wanted_model} — cannot verify")
    sys.exit(1)
mt = json.load(open(cfgs[0])).get("model_type")
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
import transformers
if mt not in CONFIG_MAPPING:
    print(f"EVAL ENV BROKEN: transformers {{transformers.__version__}} does not know "
          f"model_type {{mt!r}} — AutoConfig will raise KeyError inside the engine,")
    print("after the arm has trained. Use an interpreter whose transformers has it.")
    sys.exit(1)
print(f"eval env OK — imports resolve AND transformers {{transformers.__version__}} "
      f"maps model_type {{mt!r}}, under {cfg.eval_python}")
PY

# --- 1c. the JUDGE, in the OTHER cache -------------------------------------------
# `eval_arm.score()` scores through frame.run -> TransformersJudge, whose model is
# `Qwen/Qwen3-4B` (src/frame/config.py:102). It is in {cfg.eval_hf_home}, NOT in
# {cfg.hf_home} with the 27B. Without this the eval silently re-downloads ~8 GB, or
# fails outright if the network is not there — after the arm has already trained.
if [ ! -d "{cfg.eval_hf_home}/hub/{cfg.judge_cache_dir}" ]; then
  echo "FATAL: the judge {cfg.judge_cache_dir} is not in {cfg.eval_hf_home}/hub."
  echo "The arms would train and then fail to be scored. Fix the judge cache first."
  exit 1
fi
echo "judge OK — {cfg.judge_cache_dir} present in {cfg.eval_hf_home}"

# --- 1d. the eval's DATA and its CONTROL -----------------------------------------
# Both read from `EvalConfig`, so this gate asks the config itself rather than
# repeating its paths here — a gate that hardcodes what it checks stops checking the
# moment the config moves. The control only matters outside smoke (`paired_vs_control`
# is skipped at n=40), so without this gate a broken control path would surface for
# the first time on the real 16 h run, at the very last step.
{cfg.eval_python} - <<'PY' || exit 1
import sys
from pathlib import Path
sys.path.insert(0, "{cfg.repo_root}/src")
sys.path.insert(0, "{cfg.repo_root}/{cfg.exp_dir}/_tools")
from eval_arm import EvalConfig, build_baseline_config
c = EvalConfig()
if not Path(c.control_results_csv).exists():
    print(f"EVAL DATA GATE FAILED: control_results_csv {{c.control_results_csv}} missing")
    sys.exit(1)
# Build the REAL config via the same helper score() uses, then actually LOAD the
# items. This is the two-second version of the eval's first minute: it catches a wrong
# data_root, a str where BaselineConfig wants a Path, and a missing parquet — all three
# of which were found the expensive way on 2026-08-14, after an arm had trained.
try:
    from frame.data import load_frame_items
    items = load_frame_items(build_baseline_config(c, Path("/nonexistent-merge-ok-here")))
except Exception as e:
    print(f"EVAL DATA GATE FAILED: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
if not items:
    print(f"EVAL DATA GATE FAILED: data_root {{c.data_root}} yielded ZERO items")
    sys.exit(1)
n = sum(1 for _ in open(c.control_results_csv)) - 1
print(f"eval data OK — {{len(items)}} items from {{c.data_root}}, control has {{n}} rows")
PY

# --- 2. reclaim the quota BEFORE training ----------------------------------------
# `df` reports the MooseFS cluster and not the quota, so measure with du.
echo "--- volume before ---"; du -sx /workspace 2>/dev/null | tail -1
if [ -d "{cfg.stale_merge}" ] && [ "{cfg.smoke}" != "True" ]; then
  echo "removing rung 38's merged 27B (regenerable from its adapter)"
  rm -rf "{cfg.stale_merge}"
elif [ "{cfg.smoke}" = "True" ]; then
  echo "rehearsal: KEEPING rung 38's merged 27B — a 4B smoke merge is ~9 GB, not 52"
fi
echo "--- volume after ---";  du -sx /workspace 2>/dev/null | tail -1

# --- helpers ---------------------------------------------------------------------
# There is NO commit step. It was removed on 2026-08-14 after the rehearsal showed it
# had never done anything: `.gitignore:34` is `experiments/*/runs/**`, so every
# "committing the evidence" line was followed by `nothing to commit`. The volume is
# the record — it outlives the pod and the artifacts are readable from S3 with the
# pod off. A step that prints reassurance and preserves nothing is worse than no step,
# because it is read as a guarantee at 3 a.m.

# run_nb <python> <notebook> <output-tag> [extra papermill args...]
# 🔴 The interpreter is an ARGUMENT because training and eval need DIFFERENT envs:
# unsloth is in {cfg.env_python}, and the `focus` SDK the eval imports is NOT — it
# lives in {cfg.eval_python}. The 2026-08-14 rehearsal ran the eval under the training
# python and died on `ModuleNotFoundError: No module named 'focus'`. At 27B scale that
# is 16 h of training and zero scores.
run_nb() {{
  local py="$1"; local nb="$2"; local tag="$3"; shift 3
  echo "===== $nb start $(date -u) ====="
  "$py" -m papermill "$nb" "/workspace/tmp/leo_out_${{tag}}.ipynb" \\
    -p SMOKE {cfg.smoke} "$@" --log-output
  local rc=$?
  echo "===== $nb end rc=$rc $(date -u) ====="
  return $rc
}}

# An eval that fails is not a footnote. The old chain ignored the eval's rc entirely,
# sailed on to the next arm, and that is precisely how a broken eval would have gone
# unnoticed for a whole night. Loud, and it stops the chain.
run_eval() {{  # run_eval <output-tag> <merged-dir> <run-name>
  # The eval needs BOTH its own interpreter and its own HF_HOME — the judge lives in
  # a different cache from the 27B. In a subshell so the export cannot leak back and
  # point the next arm's training at the wrong cache.
  ( export HF_HOME="{cfg.eval_hf_home}"
    run_nb {cfg.eval_python} 04_eval.ipynb "$1" -p MERGED_DIR "$2" -p RUN_NAME "$3" )
}}

check_eval() {{  # check_eval <rc> <which>
  if [ "$1" -ne 0 ]; then
    echo "🔴 EVAL $2 FAILED (rc=$1) — the arm trained but produced NO score."
    echo "   Nothing below this point is interpretable. Stopping."
    return 1
  fi
  echo "eval $2 OK"
}}

# --- 3. ARM A: lora_alpha 32 -> 16 -----------------------------------------------
run_nb {cfg.env_python} 02_alpha_arm.ipynb arm_a {arm_params}
RC_A=$?
if [ $RC_A -ne 0 ]; then
  echo "ARM A FAILED (rc=$RC_A) — its RESULTS_arm.json holds the evidence. Stopping."
  exit $RC_A
fi

run_eval eval_a "{cfg.repo_root}/{cfg.exp_dir}/runs/{cfg.run_a}/merged" {cfg.run_a}
check_eval $? A || exit 1
{arm_b_block}
echo "===== chain40 end $(date -u) ====="
"""


def _arm_b(cfg: ChainConfig, exp: str) -> str:
    return f"""
# --- 4. reclaim arm A's merge before arm B ---------------------------------------
# Two 27B merges are ~104 GB and the volume has ~136 GB free. Sequential with a
# delete in between keeps the peak at ONE merge. Arm A is SCORED by this point —
# `check_eval` above guarantees it, or we never got here — so its merge is
# regenerable from its adapter and expendable.
A_MERGED="{cfg.repo_root}/{cfg.exp_dir}/runs/{cfg.run_a}/merged"
if [ -d "$A_MERGED" ]; then
  echo "removing arm A's merge (already scored — check_eval passed)"
  rm -rf "$A_MERGED"
fi
du -sx /workspace 2>/dev/null | tail -1

# --- 5. ARM B: the connector trains, at 4e-5 -------------------------------------
run_nb {cfg.env_python} 03_connector_arm.ipynb arm_b {_arm_params(cfg, exp)}
RC_B=$?
if [ $RC_B -ne 0 ]; then
  echo "ARM B FAILED (rc=$RC_B) — its RESULTS_arm.json holds the evidence. Stopping."
  exit $RC_B
fi

run_eval eval_b "{cfg.repo_root}/{cfg.exp_dir}/runs/{cfg.run_b}/merged" {cfg.run_b}
check_eval $? B || exit 1
"""


def render_watchdog(cfg: ChainConfig) -> str:
    """The watchdog. Detached from the chain, and it covers what the trap cannot.

    Three conditions, any of which stops the pod:

    * **the chain is gone** — `kill -0` on its PID fails. `trap ... EXIT` does NOT
      fire on SIGKILL, and the OOM killer uses SIGKILL. With a 27B that is not
      hypothetical.
    * **the log went stale** — a live run writes a line every ~24 s, so 45 minutes of
      silence means hung. The trap cannot help here: a hung process never exits, so
      nothing fires and the pod bills until morning. This is the likeliest expensive
      failure, and it is the reason this file exists.
    * **the wall clock** — last resort, if even the log hangs in some way we did not
      foresee.

    It stops the pod itself rather than signalling the chain: if the chain is hung or
    already dead, there is nobody left to ask.
    """
    if not cfg.pod_id:
        raise AssertionError("pod_id is empty — a watchdog that cannot stop the pod is decoration")
    # The stale limb is the one that misfired on 2026-08-14 and killed a healthy 27B.
    # It is kept, not deleted: for a genuinely unattended overnight run it is the only
    # thing that catches a hang, and with `_Pulse` feeding the log it is now trustworthy.
    # But when a human is watching the log themselves, they ARE the stale detector, and
    # two of them is one too many — so `stale_seconds=0` removes the limb outright
    # rather than setting a threshold so high it silently never fires.
    stale_limb = (
        f'  if [ -f "{cfg.log_path}" ]; then\n'
        f'    AGE=$(( $(date +%s) - $(stat -c %Y "{cfg.log_path}") ))\n'
        f'    if [ "$AGE" -gt {cfg.stale_seconds} ]; then\n'
        f'      stop_now "log silent for ${{AGE}}s (the pulse writes every 60s)"\n'
        f'    fi\n'
        f'  fi'
        if cfg.stale_seconds > 0 else
        '  : # stale detection DISABLED by config (stale_seconds=0) — a human is watching'
    )
    stale_banner = (f"stale>{cfg.stale_seconds}s" if cfg.stale_seconds > 0
                    else "stale detection OFF (a human is watching)")
    return f"""#!/usr/bin/env bash
# RENDERED by _tools/chain40.py — do not edit here, and do not commit this file.
CHAIN_PID="${{1:-0}}"
START=$(date +%s)
K=$(cat {cfg.key_file})
echo "watchdog up — chain pid $CHAIN_PID, {stale_banner}, wall>{cfg.max_seconds}s"

stop_now() {{
  echo "WATCHDOG STOPPING POD: $1 — $(date -u)"
  for i in $(seq 1 {cfg.stop_retries}); do
    C=$(curl -s -o /dev/null -w "%{{http_code}}" -X POST \\
        -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id}/stop)
    echo "  stop attempt $i -> HTTP $C"
    sleep 5
    S=$(curl -s -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id} | grep -o '"desiredStatus":"[A-Z]*"')
    case "$S" in *EXITED*) echo "  confirmed stopped"; exit 0;; esac
    sleep 20
  done
  exit 0
}}

while true; do
  sleep 300

  # 1) the chain vanished without stopping the pod => SIGKILL / OOM
  if [ "$CHAIN_PID" != "0" ] && ! kill -0 "$CHAIN_PID" 2>/dev/null; then
    sleep 60   # grace: let a normal exit's own trap win the race
    S=$(curl -s -H "Authorization: Bearer $K" \\
        https://rest.runpod.io/v1/pods/{cfg.pod_id} | grep -o '"desiredStatus":"[A-Z]*"')
    case "$S" in *EXITED*) echo "chain ended and the pod is already stopping — done"; exit 0;; esac
    stop_now "chain pid $CHAIN_PID is gone and the pod is still up"
  fi

  # 2) the log went stale => hung.  Skipped entirely when stale_seconds == 0.
{stale_limb}

  # 3) wall clock
  if [ $(( $(date +%s) - START )) -gt {cfg.max_seconds} ]; then
    stop_now "wall clock exceeded"
  fi
done
"""


def write(cfg: ChainConfig, path: str | Path | None = None) -> str:
    """Render into pod scratch and return the path. RAISES if aimed at a checkout.

    A committed `.sh` is exactly what this module exists to avoid, and the cheapest
    way to violate that is a convenient default.
    """
    target = PurePosixPath(str(path) if path is not None else cfg.script_path)
    posix = target.as_posix()
    if cfg.repo_root.rstrip("/") in posix or "/repo" in posix:
        raise AssertionError(
            f"refusing to write the chain to {posix} — it is inside a repo checkout."
        )
    Path(posix).parent.mkdir(parents=True, exist_ok=True)
    Path(posix).write_text(render(cfg))
    Path(posix).chmod(0o755)
    # The watchdog goes beside it: a chain rendered without its watchdog would run
    # perfectly and leave the pod up on the two failures the trap cannot catch.
    wd = Path(cfg.watchdog_path)
    wd.parent.mkdir(parents=True, exist_ok=True)
    wd.write_text(render_watchdog(cfg))
    wd.chmod(0o755)
    return posix


def launch_line(cfg: ChainConfig) -> str:
    """The one line a human types. Printed, never executed by this module."""
    return (
        f"# 1) put the RunPod API key in {cfg.key_file} (the chain deletes it)\n"
        f"# 2) verify SSH works FIRST — it did not come up on a test pod on 2026-08-13\n"
        f"#    the chain launches {cfg.watchdog_path} itself; do not start it by hand\n"
        f"nohup bash {cfg.script_path} > {cfg.log_path} 2>&1 &\n"
        f"tail -f {cfg.log_path}"
    )
