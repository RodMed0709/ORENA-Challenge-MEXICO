---
task: canonical-eval-metrics-module
slug: 260717-t2f-canonical-eval-metrics-module-src-frame-
branch: task/eval-canonical            # off main — do NOT touch main
type: quick
date: 2026-07-17
autonomous: false                       # task 4 real-df smoke needs pod artifacts
requirements: [EVAL-CANON]              # kill the recurring leaf-vs-group drop-bug class
files_modified:
  - src/frame/metrics.py                # NEW — single source of truth for scoring
  - experiments/02-lora-sft/_tools/test_metrics_canonical.py   # NEW — reproduces known numbers
  - src/frame/run.py                    # MODIFY — headline → bucket_mean
  - src/frame/ledger.py                 # NEW — RESULTS.md builder
  - RESULTS.md                          # GENERATED output (root ledger)
  - context/decisions/eval-canonical.md # NEW — brain decision note
  - context/INDEX.md                    # MODIFY — link decision + knowledge pointer

must_haves:
  truths:
    - "Every scored row maps its leaf primary to a capability GROUP — no group-name filter ever drops a leaf-valued row again."
    - "ID/OOD is derived from the qID prefix (heico=OOD, lapchole=ID), never from results_df['ood'] (all-False on public data)."
    - "bucket_mean over the 4 real buckets reproduces rung-02 (~0.550) and rung-05 a0_real (0.5503)."
    - "Gate assertions RAISE on malformed input and cannot be silently disabled."
    - "run.py headline is bucket_mean; pre_evaluation_score survives only as a labeled reference field."
    - "RESULTS.md is regenerated from experiments/*/RESULTS.csv, sorted by bucket_mean."
  artifacts:
    - path: "src/frame/metrics.py"
      provides: "stratified_report + gate assertions (single source of truth for scoring)"
      contains: "def stratified_report"
    - path: "experiments/02-lora-sft/_tools/test_metrics_canonical.py"
      provides: "offline synthetic-df checks + pod-gated rung-02 reproduction"
    - path: "src/frame/ledger.py"
      provides: "build_results_md() aggregating experiments/*/RESULTS.csv"
    - path: "context/decisions/eval-canonical.md"
      provides: "settled verdict registered in the brain"
  key_links:
    - from: "src/frame/metrics.py"
      to: "focus.taxonomy.Capability.group"
      via: "Capability.from_any(leaf).group.value — SAME mapping as split._group"
    - from: "src/frame/run.py::_kpi_report"
      to: "src/frame/metrics.py::stratified_report"
      via: "headline bucket_mean replaces evaluator.pre_evaluation_score"
---

<objective>
Build ONE canonical evaluation-metrics module (`src/frame/metrics.py`) so every experiment
imports the same scoring code instead of re-deriving metrics in prose. This kills a recurring
defect class: `results_df["primary"]` holds a taxonomy LEAF (e.g. `object_identification`,
n=2457) but the scored buckets are GROUPS (`object_recognition`, n=3421). Filtering group
names against the leaf column silently drops questions (964 dropped in rung 07, commit 8839ee5,
caught by gate G3, reverted 881d057). The correct leaf→group mapping already exists —
`split._group()` (src/frame/split.py:76) → `focus.taxonomy.Capability.group` (taxonomy.py:85) —
but is only used for split coverage, never for scoring.

Purpose: single source of truth for scoring + importable gates that RAISE, so the drop-bug
cannot recur silently.
Output: `metrics.py` (report + gates), a reproduction test, `run.py` migrated to `bucket_mean`,
an auto-built root `RESULTS.md` ledger, and a brain decision note.

Additive / single-variable: existing behavior is preserved until Task 4 migrates `run.py`.
`pre_evaluation_score` is never deleted — it is demoted to a clearly-labeled reference field.
</objective>

<context>
@src/frame/split.py        # _group (76), per_bucket_report (289), split_summary (321), load_manifest (230), apply_split (263)
@src/frame/run.py          # _kpi_report (59) — currently headlines the BROKEN evaluator.pre_evaluation_score (64)
@src/frame/data.py         # FrameItem (43), qID prefix f"{ds}__" (110); ood=all-False (87)
@src/frame/delta.py        # correctness_frame (29), question_metadata (40) — schema-light patterns to reconcile with
@experiments/02-lora-sft/RESULTS.csv        # acc_OOD 0.5918, pre_evaluation_score 0.7079 (NO bucket_mean col)
@experiments/05-bottleneck-audit/RESULTS.csv # a0_real bucket_mean 0.5503; the 4-bucket layout + trivial-floor columns

<interfaces>
<!-- Confirmed from vendor source. Executor uses these directly — no re-exploration. -->

focus.taxonomy.Capability (vendor/orena-focus/src/focus/taxonomy.py):
  Capability.from_any(value) -> Capability | None   # robust str/name/value/code → enum (line 120); returns None on junk
  cap.group -> Capability                            # leaf → group; group → self (line 85, _PARENT_MAP line 179)
  cap.value -> str                                   # e.g. "object_recognition"
  Capability.groups() -> tuple[Capability, ...]      # the 5 groups
  # Canonical leaf-string → group-string:  Capability.from_any(leaf).group.value
  # This is the SAME .group that split._group(item) uses on item.reference.primary (split.py:76-80).

results_df schema (focus.evaluation.evaluator.Evaluator._make_row, evaluator.py:373-392):
  columns = [qID, video, ood, clinical, primary, answer_format, latency, timed_out, correctness]
  # primary  = ref.primary.value  → a LEAF value string (NOT a group)         [the bug source, line 388]
  # ood      = ref.ood            → all-False on public data (data.py:87)      [do NOT use for ID/OOD]
  # qID      = f"{dataset}__{row_id}"  → prefix is the REAL ID/OOD signal      [heico=OOD, lapchole=ID; data.py:110]
  # video    = req.videoID (raw id; COLLIDES across datasets → namespace by dataset for video-level bootstrap)

focus.evaluation.evaluator.Evaluator (evaluator.py):
  .pre_evaluation_score(results_df) -> (score, buckets_df)   # splits ID/OOD by results_df["ood"] → BROKEN for us (line 402/440)
  ._hierarchical_summary(df) -> summary_df                   # two-level video→question bootstrap (line 464)
  # summary_df level="answer_format" carries per-format accuracy + ci_low/ci_high (n_boot=1000, seed=42)
</interfaces>
</context>

<tasks>

<task type="auto" id="0-spike">
  <name>Task 0 (SPIKE, read-only): confirm the SDK number/hierarchical estimate API before coding</name>
  <files>(none — investigation only; findings recorded in the metrics.py module docstring in Task 1)</files>
  <action>
    Read from vendor/orena-focus/ ONLY (no edits) to nail down two API facts the report depends on:

    1. The per-format hierarchical estimate for `number`. The stated deliverable asks stratified_report
       to surface "the SDK hierarchical estimate for number." Confirm the real API:
       - grep vendor/orena-focus/src/focus/data/formats.py for the `number` format class and any
         hierarchical / interval / tolerance estimator it exposes (see JUDGE_FORMATS, fmt.compare,
         fmt.read referenced in evaluator.py:326-351).
       - Confirm whether the "hierarchical estimate for number" is (a) just the answer_format="number"
         row of Evaluator._hierarchical_summary (video→question bootstrap, evaluator.py:464), or
         (b) a dedicated estimator inside the number format. Record which, with file:line.
    2. Confirm Capability.from_any(leaf_value).group.value is the canonical leaf→group and that it is
       the SAME mapping split._group uses (taxonomy.py:85/179 _PARENT_MAP; split.py:76). Confirm from_any
       returns None (not raise) on junk so a gate can catch un-mappable leaves.

    Write the two confirmed facts (with file:line) as a short "SDK API — confirmed" block that Task 1
    pastes into the metrics.py module docstring. If fact (1) is ambiguous, default to interpretation (a)
    — the answer_format="number" row of _hierarchical_summary — and note the assumption explicitly.
  </action>
  <verify>
    <automated>MISSING — read-only spike; verified by the two file:line facts being cited in Task 1's metrics.py docstring.</automated>
  </verify>
  <done>Two API facts (number estimate source; from_any().group.value) confirmed with file:line, ready to paste into metrics.py. No files changed, no commit.</done>
</task>

<task type="auto" id="1-metrics-core" tdd="true">
  <name>Task 1: NEW src/frame/metrics.py — stratified_report core (pure-python, offline-verifiable)</name>
  <files>src/frame/metrics.py</files>
  <behavior>
    On a synthetic results_df (hand-built, known correctness) with the confirmed schema:
    - leaf→group: a row with primary="object_identification" is bucketed under "object_recognition".
    - ID/OOD from qID: qID "heico__12" → OOD; "lapchole__7" → ID; never reads results_df["ood"].
    - bucket_mean = unweighted mean over the populated {group}×{ID,OOD} buckets AFTER dropping any
      bucket with n < min_bucket_n (default 2) → drops temporal_grounding (n=1). For a synthetic df
      with 4 buckets at accuracies [0.60, 0.61, 0.42, 0.57] → bucket_mean == 0.55 (assert to 1e-9).
    - per-format accuracy table includes trivial floors (majority-class + train-prior) computed vs the
      EVAL set split by ID/OOD, matching the column shape in 05-bottleneck-audit/RESULTS.csv.
    - bootstrap CIs are reproducible for a fixed seed.
  </behavior>
  <action>
    Create src/frame/metrics.py as the single source of truth for scoring predictions. Paste the
    Task-0 "SDK API — confirmed" block into the module docstring.

    Helpers (private, single place so a schema change is one edit — mirror split.py's accessor pattern):
      _leaf_to_group(leaf: str) -> str:
        return Capability.from_any(leaf).group.value      # SAME mapping as split._group; NEVER prose.
        Raise/flag via the gate if from_any returns None.
      _dist_from_qid(qid: str) -> str:
        prefix = qid.split("__", 1)[0]; return "OOD" if prefix == "heico" else "ID"
        (heico=OOD per data.py:110 / split.py:19-22; lapchole=ID). NEVER results_df["ood"].
      _video_key(row) -> tuple: (prefix_from_qid, row["video"])   # namespace video by dataset —
        req.videoID collides across datasets (data.py note), would corrupt the video-level bootstrap.

    Public:
      def stratified_report(results_df, video_split=None, *, min_bucket_n=2, n_boot=1000, seed=42,
                            train_results_df=None) -> dict[str, pd.DataFrame | float]:
        Returns a dict with at least:
          - "bucket_mean": float over the populated 4 real buckets (group×{ID,OOD}, drop n<min_bucket_n).
          - "by_bucket": DataFrame [capability_group, distribution, accuracy, n] (leaf→group; ID/OOD from qID).
          - "by_format": DataFrame [answer_format, distribution, accuracy, n, ci_low, ci_high,
                          floor_majority, floor_train_prior]  — trivial floors computed vs the EVAL set
                          split by ID/OOD (majority-class of the eval answers; train-prior if
                          train_results_df/priors supplied, else NaN with a logged note).
          - "acc_ID", "acc_OOD": float macro/flat means over the ID and OOD halves (acc_OOD is the
            number the leaderboard weights ~half).
          - "number_estimate": the SDK per-format hierarchical estimate for `number`, sourced per Task-0.
        Reuse, do NOT duplicate:
          - leaf→group via Capability (taxonomy.py:85) — the exact call split._group makes.
          - the two-level video→question bootstrap: either call Evaluator._hierarchical_summary and read
            its answer_format rows, or replicate its algorithm (evaluator.py:464-513) keyed on _video_key.
            Prefer calling the SDK to stay single-source; note the choice in the docstring.
        Accept video_split (from split.load_manifest) as an OPTIONAL override of the qID-prefix ID/OOD
        (some experiments use a custom OOD procedure); default None = qID-prefix rule. Document that the
        qID rule is the default and the manifest is the override, never the reverse.

    Keep it schema-light like delta.py: reuse delta.correctness_frame's tolerant column detection if
    normalizing correctness. Pure pandas/numpy + focus.taxonomy import; no torch, no GPU.
  </action>
  <verify>
    <automated>python -c "import pandas as pd; from frame.metrics import stratified_report; df=pd.DataFrame([{'qID':f'{d}__{i}','video':f'{d}_v{i%3}','primary':p,'answer_format':'number','correctness':c,'latency':1.0,'timed_out':False,'ood':False} for d in ('lapchole','heico') for i,(p,c) in enumerate([('object_identification',1),('object_identification',1),('object_aggregation',0),('object_aggregation',1)])]); r=stratified_report(df); print('bucket_mean',round(r['bucket_mean'],4)); assert 'by_bucket' in r and 'acc_OOD' in r"</automated>
  </verify>
  <done>metrics.py imports cleanly; stratified_report returns bucket_mean + by_bucket + by_format + acc_ID/acc_OOD on a synthetic df; leaf→group and ID/OOD-from-qID verified offline (no GPU, no pod). Commit: "feat(metrics): canonical stratified_report — leaf->group + qID-derived ID/OOD (src/frame/metrics.py)"</done>
</task>

<task type="auto" id="2-metrics-gates">
  <name>Task 2: gate assertions in metrics.py that RAISE (importable, cannot be silently disabled)</name>
  <files>src/frame/metrics.py</files>
  <action>
    Add five gate functions to metrics.py. Each RAISES AssertionError/ValueError with a diagnostic
    message (qIDs / counts), so an experiment that imports them cannot silently pass a broken df.
    These encode the exact failure modes from the git archaeology (964 dropped, rung 07):

      assert_all_rows_grouped(results_df):
        every results_df["primary"] leaf maps to a real Capability group via _leaf_to_group; raise
        listing any un-mappable / dropped leaves (this is the 964-drop guard, ex-gate G3).
      assert_bucket_counts(results_df, expected: dict[tuple[str,str], int]):
        recompute {group}×{ID,OOD} counts and assert they equal `expected` (e.g. object_recognition
        totals 3421, NOT the object_identification leaf n=2457) — catches a group-name-vs-leaf filter.
      assert_floors_vs_eval_set(report):
        assert every by_format accuracy is >= its trivial floor (or explicitly flag a below-floor row);
        floors computed vs the EVAL set split by ID/OOD, never a global prior.
      assert_no_dup_qid(results_df):
        assert results_df["qID"].is_unique (mirrors run.py:130 + evaluator.py:191 duplicate-qID abort).
      assert_ood_from_qid(results_df):
        assert every qID prefix ∈ {"heico","lapchole"} and that ID/OOD was derived from it — raise if
        anyone passed a df whose distribution silently came from the all-False ood column.

    Export them at module level (no flag, no env toggle — importable and unconditional).
  </action>
  <verify>
    <automated>python -c "import pandas as pd; from frame import metrics as m; import sys; bad=pd.DataFrame([{'qID':'lapchole__1','video':'v','primary':'not_a_capability','answer_format':'number','correctness':1,'ood':False}]);
try: m.assert_all_rows_grouped(bad); sys.exit('gate did NOT raise on bad leaf')
except (AssertionError, ValueError): pass
dup=pd.DataFrame([{'qID':'lapchole__1'},{'qID':'lapchole__1'}]);
try: m.assert_no_dup_qid(dup); sys.exit('dup gate did NOT raise')
except (AssertionError, ValueError): print('gates raise OK')"</automated>
  </verify>
  <done>All five gates importable from frame.metrics; each raises on its crafted-bad input and passes on a clean df. Verified offline. Commit: "feat(metrics): add raising gate assertions (all_rows_grouped, bucket_counts, floors, no_dup_qid, ood_from_qid)"</done>
</task>

<task type="auto" id="3-repro-test">
  <name>Task 3: reproduction test in experiments/02-lora-sft/_tools/ (offline synthetic + pod-gated real numbers)</name>
  <files>experiments/02-lora-sft/_tools/test_metrics_canonical.py</files>
  <action>
    Create the test INSIDE the owning experiment's _tools/ (BINDING: never a top-level tests/ folder).
    Two clearly-separated parts:

    PART A — OFFLINE (always runs, pure-python, no pod):
      - Build the synthetic results_df from Task 1's behavior block and assert bucket_mean == 0.55
        (1e-9), that temporal_grounding n=1 is dropped, and that swapping a group name into a leaf
        filter would drop rows (assert the gate catches it).
      - Run all five gates: assert they RAISE on crafted-bad dfs and pass on the clean df.
      - Cross-check against a KNOWN CSV that already carries bucket_mean: load
        experiments/05-bottleneck-audit/RESULTS.csv, take a0_real, and assert the recorded
        bucket_mean == 0.5503 equals the mean of its 4 recorded per-bucket columns
        (acc_bucket_object_recognition_{ID,OOD} + acc_bucket_aggregation_{ID,OOD}) — a self-consistency
        check on the 4-bucket definition that needs NO saved predictions.

    PART B — POD-GATED (skips if artifacts absent):
      - Locate rung-02 saved results.csv (evaluator output; e.g. experiments/02-lora-sft/runs/*/results.csv
        — gitignored, present only on the pod). If missing, SKIP with a clear message.
      - If present: load it, run gates, call stratified_report, and assert bucket_mean ≈ 0.550 (atol 0.005)
        and acc_OOD ≈ 0.592 (atol 0.005) — the known-good rung-02 numbers.
      Use plain asserts + a __main__ runner (no pytest dependency required); print PASS/SKIP per part.
  </action>
  <verify>
    <automated>python experiments/02-lora-sft/_tools/test_metrics_canonical.py</automated>
  </verify>
  <done>Part A passes offline anywhere (synthetic bucket_mean==0.55, gates raise, 05 CSV self-consistent to 0.5503); Part B either reproduces rung-02 (0.550/0.592) on the pod or SKIPs cleanly when results.csv is absent. Commit: "test(metrics): reproduce rung-02 numbers + gate coverage (experiments/02-lora-sft/_tools)"</done>
</task>

<task type="auto" id="4-migrate-run">
  <name>Task 4: migrate src/frame/run.py headline pre_evaluation_score → bucket_mean</name>
  <files>src/frame/run.py</files>
  <action>
    Depends on Tasks 1-3 landing (metrics.py + test must exist and pass first).
    Edit _kpi_report (run.py:59) ONLY — do not touch inference (_infer_all) or the run_baseline
    memory-hygiene / latency plumbing:
      - Import frame.metrics; call stratified_report(results_df) to produce the headline.
      - Set report["bucket_mean"] as the HEADLINE metric (the number we track).
      - Keep report["pre_evaluation_score"] (from evaluator.pre_evaluation_score, run.py:64) but RENAME
        its role to a clearly-labeled reference field, e.g. report["pre_evaluation_score_reference"] with
        a comment: "SDK pre_eval — splits ID/OOD by the all-False ood column; kept for reference, NOT the headline."
      - Populate acc_ID / acc_OOD / by_bucket from stratified_report so by_group_distribution comes from
        the qID-derived ID/OOD, not results_df["ood"].
      - PRESERVE exactly: the `del engine; gc.collect(); empty_cache()` before the judge (run.py:142-149),
        the `del judge, evaluator` before return (run.py:174-177), and the df_e0 / responses latency read
        (run.py:61, lat series). Update the final log line (run.py:180) to print bucket_mean.
    Additive: keep raw_accuracy, latency block, by_answer_format, timeout count unchanged.
  </action>
  <verify>
    <automated>python -c "import ast; s=open('src/frame/run.py').read(); assert 'bucket_mean' in s and 'stratified_report' in s and 'pre_evaluation_score_reference' in s, 'headline not migrated'; ast.parse(s); print('run.py migrated + parses OK')"</automated>
  </verify>
  <done>_kpi_report headlines bucket_mean via metrics.stratified_report; pre_eval survives only as a labeled reference field; memory hygiene + latency read byte-preserved; run.py parses. NOTE: a full real-run smoke needs the pod (model + judge) — offline verify is import/parse + the Task-3 synthetic path. Commit: "refactor(run): headline bucket_mean (metrics.stratified_report); demote pre_eval to reference field"</done>
</task>

<task type="auto" id="5-ledger">
  <name>Task 5: NEW src/frame/ledger.py — build root RESULTS.md from experiments/*/RESULTS.csv (INDEPENDENT)</name>
  <files>src/frame/ledger.py, RESULTS.md</files>
  <action>
    Independent of Tasks 1-4 (can land in parallel). Create src/frame/ledger.py with:
      def build_results_md(repo_root=".", out="RESULTS.md") -> Path:
        - glob experiments/*/RESULTS.csv, read each (they are HETEROGENEOUS: 02-lora-sft has
          pre_evaluation_score + acc_OOD but NO bucket_mean column; 05-bottleneck-audit HAS bucket_mean).
          Coalesce a common view: at minimum [experiment, run/arm, bucket_mean, acc_ID, acc_OOD, verdict].
          When bucket_mean is absent, leave it blank (do NOT fabricate) and note it in the row.
        - concat, sort DESC by bucket_mean (NaN/blank sink to the bottom), and write a Markdown table to
          the repo-root RESULTS.md with a one-line header explaining bucket_mean is the canonical headline.
        RESULTS.md is GENERATED output — notebooks call build_results_md(); the file itself is a build artifact.
    Put the builder in src/frame/ so notebooks import it (per deliverable #4). No hand-editing of RESULTS.md.
  </action>
  <verify>
    <automated>python -c "from frame.ledger import build_results_md; build_results_md('.'); import pathlib; t=pathlib.Path('RESULTS.md').read_text(); assert 'bucket_mean' in t and '02' in t and '05' in t; print('RESULTS.md built + sorted')"</automated>
  </verify>
  <done>ledger.build_results_md() aggregates every experiments/*/RESULTS.csv (heterogeneous columns handled), sorts by bucket_mean, and writes root RESULTS.md; runs fully offline. Commit: "feat(ledger): auto-build root RESULTS.md from experiments/*/RESULTS.csv (src/frame/ledger.py)"</done>
</task>

<task type="auto" id="6-brain">
  <name>Task 6: brain registration — decision note + INDEX links (docs, land last)</name>
  <files>context/decisions/eval-canonical.md, context/INDEX.md</files>
  <action>
    1. Create context/decisions/eval-canonical.md in the mandated format (INDEX.md:40):
       Question / What we sought / What it gave us / Verdict / Sources.
       - Question: how do we stop eval bugs from recurring (leaf-vs-group drop; ID/OOD from all-False ood col)?
       - What we sought: one importable scoring module + gates that raise.
       - What it gave us: src/frame/metrics.py (stratified_report + 5 gates), run.py headlines bucket_mean.
       - Verdict: SETTLED — all scoring imports frame.metrics; leaf→group ALWAYS via Capability.group;
         ID/OOD ALWAYS from qID prefix; pre_evaluation_score is reference-only.
       - Sources: rung-07 drop bug (commit 8839ee5, reverted 881d057, gate G3); split.py:76; evaluator.py:388,402;
         data.py:110; 02/05 RESULTS.csv.
    2. Edit context/INDEX.md:
       - Under "## DECISIONS" add: [[eval-canonical]] — one-line summary.
       - Under "## KNOWLEDGE" add: "Evaluations → src/frame/metrics.py + gates (canonical scoring)".
    Keep INDEX.md to one screen (its own rule, line 41) — point, do not contain.
  </action>
  <verify>
    <automated>python -c "import pathlib; d=pathlib.Path('context/decisions/eval-canonical.md').read_text(); i=pathlib.Path('context/INDEX.md').read_text(); assert all(k in d for k in ('Question','Verdict','Sources')); assert 'eval-canonical' in i and 'metrics.py' in i; print('brain registered')"</automated>
  </verify>
  <done>context/decisions/eval-canonical.md exists in the mandated format; context/INDEX.md links it under DECISIONS and adds the Evaluations->metrics.py pointer under KNOWLEDGE; INDEX stays one screen. Commit: "docs(brain): register eval-canonical decision + INDEX pointers"</done>
</task>

</tasks>

<verification>
Sequencing (nothing breaks mid-way):
- Wave 1 (parallel, offline): Task 0 spike (read-only) · Task 1 metrics core · Task 5 ledger (independent).
- Wave 2 (offline): Task 2 gates (same file as Task 1 → after it) · Task 3 test (needs metrics.py).
- Wave 3: Task 4 run.py migration (ONLY after metrics.py + test are green) · Task 6 brain docs (last).
metrics.py + its test land BEFORE run.py is migrated; ledger + brain are independent.

File ownership (no mid-task conflicts):
- Tasks 1 & 2 both own src/frame/metrics.py → strictly sequential, two commits.
- All other tasks own disjoint files.

Offline vs pod (the runtime unknown):
- OFFLINE, verifiable anywhere (pure-python on synthetic / already-committed CSVs):
  Task 0, Task 1, Task 2, Task 3 Part A, Task 4 (import/parse only), Task 5, Task 6.
- POD-ONLY (needs rung-02 saved predictions/results.csv — gitignored under experiments/02-lora-sft/runs/):
  Task 3 Part B (reproduce bucket_mean≈0.550 / acc_OOD≈0.592) and any full real-run smoke of run.py.
  These SKIP cleanly off-pod; run them on the RunPod pod that holds the artifacts.
</verification>

<success_criteria>
- src/frame/metrics.py is the single scoring source: stratified_report returns bucket_mean over the
  4 real buckets, per-bucket×{ID,OOD}, per-format accuracy + trivial floors + bootstrap CIs + the SDK
  number estimate; leaf→group ALWAYS via Capability.group; ID/OOD ALWAYS from the qID prefix.
- Five gates in metrics.py RAISE and are importable (cannot be silently disabled).
- Offline test reproduces the 4-bucket definition (synthetic bucket_mean==0.55; 05 CSV self-consistent
  to 0.5503); pod test reproduces rung-02 (0.550 / 0.592) or skips cleanly.
- run.py headlines bucket_mean; pre_evaluation_score demoted to a labeled reference field; memory
  hygiene + latency read preserved.
- Root RESULTS.md is regenerated by frame.ledger, sorted by bucket_mean.
- context/decisions/eval-canonical.md registered and linked in context/INDEX.md.
- Every commit lands on task/eval-canonical; main is never touched; NO Claude co-author trailer; English only.
</success_criteria>

<notes>
- API uncertainty flagged as Task 0 (spike): the exact SDK source for the "hierarchical estimate for
  number" — resolve from vendor/orena-focus/ formats.py vs Evaluator._hierarchical_summary BEFORE coding
  Task 1. Default assumption if ambiguous: the answer_format="number" row of _hierarchical_summary.
- The leaf-vs-group bug is the reason this module exists: results_df["primary"] = ref.primary.value is a
  LEAF (evaluator.py:388); filtering a GROUP name against it silently drops rows (964 in rung 07). Gates
  assert_all_rows_grouped + assert_bucket_counts encode the ex-gate-G3 guard that caught it.
- ID/OOD: results_df["ood"] is all-False on public data (data.py:87) — the SDK pre_evaluation_score is
  broken for us because of this. qID prefix (heico=OOD, lapchole=ID) is the only correct signal.
- video-level bootstrap must namespace video by dataset (req.videoID collides across datasets) or the CIs
  silently merge two surgeries' frames.
</notes>
