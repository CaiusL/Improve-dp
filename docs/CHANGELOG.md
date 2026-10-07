# Changelog

## 1.3.7

Documentation only; no code, data, table or figure changed.

- Removed the `Formerly …` provenance lines from the round reports, and reworded the two headers
  that repeated an earlier filename, so no report's own text names its pre-1.3.6 file. The mapping
  is unchanged where a reader looks for it: the README's naming conventions, this changelog,
  `docs/RELEASE_CHECK.md` section 6, and `RENAME_MAP.md` in the working archive.
- `REPORT_ROUND_7.md`: the version note now says only that version 1 is kept unedited in the
  working archive, without naming it or the review record. `REPORT_ROUND_6.md`: the closing section
  says those corrections "were shipped as a separate file in earlier releases" instead of naming
  it.

## 1.3.6

Documentation only; no code, data, table or figure changed. The round reports are now named by the
round they document, and two files a reader of this archive could not check were removed
(128 files, down from 130).

### Changed

- **Round reports renamed** (the `Formerly …` lines this entry mentions were removed again
  in 1.3.7): `REPORT_NOTES.md` →
  `REPORT_ROUND_1.md`, `REPORT_ADDENDUM.md` → `REPORT_ROUND_2.md`, `REPORT_ADDENDUM_2.md` →
  `REPORT_ROUND_3.md`, `REPORT_ADDENDUM_3.md` → `REPORT_ROUND_4.md`, `REPORT_ADDENDUM_4.md` →
  `REPORT_ROUND_5.md`, `REPORT_ADDENDUM_5.md` → `REPORT_ROUND_6.md` and `REPORT_ADDENDUM_6.md` →
  `REPORT_ROUND_7.md`. Every cross-reference in the README, this file and the reports was
  rewritten. The working archive keeps the original names; its `RENAME_MAP.md` and README carry
  the mapping.
- `REPORT_ADDENDUM_5_FIXES.md` was **merged into `REPORT_ROUND_6.md`** instead of shipping
  alongside it: its corrections are applied in that report's body and the list survives as the
  report's closing "Corrections applied when this file was packaged" section.
- `REPORT_ADDENDUM_6.md` was replaced by its version 2 (`REPORT_ADDENDUM_6_v2.md`), which the
  working archive and `FINAL_REPORT.md` both record as superseding it. It adds the third review's
  six text corrections and changes no number.
- `STAGE_REPORT_v2_CHECK.md` was **dropped**: it verifies `STAGE_REPORT_v2.md`, which the archive
  does not contain, and the script that produced it is not shipped either.
- `README.md`: the round table's report column, the "Full detail" paragraph and the naming
  conventions use the new names; the round-6 row's `PullCubeTool-v1` difference was corrected from
  +15.82 pp to +15.88 pp, and a fourth convention bullet records the merge and the dropped check.

### Fixed

- `REPORT_ROUND_6.md` §C: the `PullCubeTool-v1` result table was rebuilt from
  `results/r7_runlevel_arms.csv` (mean 27.50%, sd 6.87, `success_at_end` 27.38%) when the
  duplicated block holding the stale copy was deleted; the stale row read 27.44%, sd 6.82,
  +15.82 pp.
- Three references in that report that called the +11.0 pp (p = 0.00068)
  adaptive-vs-`fixed:8` figure a round-4 result now say round 5 (`REPORT_ROUND_5.md` §2.2 is where
  it was measured).

### Verified

- `tests/` **21/21**; `verify_results.py` **3 526 / 0**; `r5_integrity.py` **4 196 / 0**;
  `FINAL_RESULTS.csv` **273 rows**; `MANIFEST.md` **5 392 files / 53 942 episodes**; safety scan
  clean.

## 1.3.5

Documentation only; no code, data, table or figure changed.

- `README.md` gained **"What a 'round' is, and what each round did"**: a per-round table — what each
  round tried or tested, its outcome in one line, the report that carries it and its
  pre-registrations in `configs/` — plus the three numbering conventions that otherwise make the
  archive look as though it covers only rounds 5-7:
  - a **round is one improvement attempt or one test**, not a release and not a batch of runs; each
    round pre-registers its settings before its first rollout or training run and reports the
    outcome whether or not it supports the change (round 3's `PickCube-v1` study is a negative
    result);
  - **the reports were named by addendum number, offset from the round number** —
    `REPORT_NOTES.md` was round 1, then `REPORT_ADDENDUM.md` was round 2 through
    `REPORT_ADDENDUM_6.md` for round 7, so addendum *N* reported round *N + 1* (1.3.6 renamed
    them to `REPORT_ROUND_1.md` … `REPORT_ROUND_7.md`, so this archive names rounds directly);
  - **`rN_*` marks the code a round introduced, not the data it used** — rounds 1-4 were answered
    with the generic toolkit, so there are no `r1_*`-`r4_*` scripts, and rounds 1-4 ship as 59 rows
    inside `FINAL_RESULTS.csv` rather than as separate tables. Rounds also group into releases
    (1-4 → 1.0.0, 5 → 1.1.0, 6 → 1.2.0, 7 → 1.3.0; the 1.3.x patches are corrections on round-7
    content).
- The README also carried a stale "this archive is release 1.3.2" sentence; it is corrected.
- `docs/RELEASE_CHECK.md` sections 2 and 4 claimed the archive ships the rounds 1-4 per-block
  tables. It ships their **rows** and, from that era's tables, only `r4_fresh_final_*.csv`; the
  others — `pushcube_test100_*`, `pickcube_val_*`, `confirm_summary_*`, `partial_test_*`,
  `final_*` and the `r3_*` cross-task analyses — are not in this release, and no shipped script
  regenerates the `r3_*` ones. Both sentences now say so, and the section 6 heading covers
  1.3.1-1.3.5.

## 1.3.4

Documentation only; no code, data, table or figure changed.

- `docs/RELEASE_CHECK.md` rewritten so it describes the package as shipped rather than the
  round-4 state it had described since 1.0.0: a current-state table (version and commit chain,
  archive size and per-area file counts, 21/21 tests, 4 196/0 integrity, 3 526/0 verification,
  273 rows, 5 392 files / 53 942 episodes), an updated contents and exclusions inventory (raw
  logs 245 MB / 5 432 chunk files with the 40 `AMBIGUOUS_provenance` files noted, `.git` not in
  the zip, licence and remote status), and a "how to verify this release" section whose commands
  were executed as written.
- It also separates what the documented pipeline **does** regenerate from the raw logs —
  11 shipped tables and logs byte-for-byte plus all 38 figure files (19 PNG byte-identical,
  19 PDF identical ignoring the embedded creation date) — from what it **does not**
  (`r5_frontier_*`, `r5_timing_clean.csv`, `r4_fresh_final_*`, `r6_rawfeat_check.json`,
  `r7_failure.csv`, `r7_p0_*.json` and the rounds 1-4 per-task tables, which need a specific
  invocation or an argument file). That last clause overstated the inventory and is corrected in
  1.3.5. Previously the file implied the whole package was
  reproducible from the folder alone.
- The round-by-round verification numbers are kept as history under their own version labels, and
  the 1.3.1-1.3.3 corrections are consolidated into one section.

## 1.3.3

Third review pass: the README's "Reproducing the results" section was executed command by command.
It did not work as written, and the check exposed four defects in the shipped scripts.

### Fixed

- `scripts/r5_sampling_analysis.py`, `scripts/r5_training_analysis.py`: both called `HERE`
  before defining it (`NameError` on import), so the two documented round-5 analyses could not run
  at all. The path setup is ordered correctly now; both run and reproduce their shipped tables.
- `scripts/run_matrix.py`: used `args.sampler` without ever defining `--sampler`, so every real
  (non-dry-run) matrix crashed with `AttributeError`. The flag is added and forwarded to
  `run_rollout.py`.
- `scripts/r7_parity_figure.py`: the shipped copy was the pre-completion version, which still
  marked six rows "pending / to be filled" — running it produced a parity figure contradicting
  both the shipped PNG and the addendum. The final version (the one that drew the shipped figure)
  is now in the release, and re-running it reproduces that PNG byte-for-byte.
- `scripts/run_rollout.py`, `scripts/run_matrix.py`, `src/adaptive_replan/rollout.py`: the
  `--feature-check` passthrough promised by the 1.3.0 changelog and the README did not exist in
  the release, so `r6_verify_rawfeat.py` had nothing to read. `--decision-log --feature-check` now
  records the raw observation row plus the features recomputed from the normalized vector; it was
  tested end to end (0 feature and rule mismatches, non-vacuous on a normalized checkpoint).

### Changed

- `README.md` "Reproducing the results": the shell block had an unmatched code fence (five
  commands rendered as prose), `--tasks` used semicolons where the parser wants commas, the
  round-7 figure script was missing, the round-5 sweep analysed 3 of the 9 samplers, and six
  commands omitted `--raw-root` although this archive ships `results/raw/` empty. The section now
  defines `RAW` once, is a single fenced block, and lists the full nine-sampler sweep.

### Verified

- Every runnable command in the section was executed against the raw logs: **18/18 pass**
  (4 skipped: three need argument placeholders, one needs checkpoints).
- The flow was then run end to end from a clean extraction of this release: **30 shipped tables
  and logs reproduce byte-for-byte** (`FINAL_RESULTS.csv`, `MANIFEST.md`, all `r5_*`,
  `r6_runlevel_*`, `r7_*`, `r4_fresh_final_*`), **all 19 shipped PNG figures are byte-identical**,
  and **all 19 PDFs are identical once the embedded creation date is ignored**. Four further CSVs
  the flow writes (`r6_runlevel_arms`, `r6_runlevel_mcnemar`, `pushcube_fresh_summary`,
  `pushcube_fresh_paired`) are new outputs, not replacements of shipped files.
- `verify_results.py` 3 526 values / 0 mismatches; `r5_integrity.py` 4 196 chunk files /
  0 problems; `tests/` 21/21.

## 1.3.2

Two analysis/packaging defects found by a second external review of the round-7 numbers, plus the
resulting number corrections. Success rates are unchanged everywhere; only derived costs, one
merge and rounding were wrong.

### Fixed

- `scripts/r7_runlevel.py`: the clean per-call time was looked up under the arm name `co` while
  the co-trained frontier cells carry `co_tm`, so every DDPM-20 / DDIM-20 frontier cell fell back
  to the DDPM-100 time (145.0 ms); the `("ddim", 20)` entry (28.5 ms) was missing as well. The
  arm name and the missing entry are added, and `results/r7_runlevel_summary.csv` plus the two
  frontier figures are regenerated. Example: `PickCube-v1` `fixed:8` at DDPM-20 / DDIM-20 is
  0.72 s / 0.71 s per episode, not 3.62 s; adaptive is 1.82 s / 1.74 s, not 9.19 s / 8.85 s.
- `scripts/pack_final_results.py`: the round-5 frontier reused the first 100 sweep episodes
  (seeds 400000-400099) for its three `fixed:8` cells, and the round-5 analysis merged them, but
  the packer did not — so `FINAL_RESULTS.csv` shipped those three `r5_front_*` rows at n = 100.
  The reused half is now folded in by seed and the rows match the analysis: n = 200,
  76.0 / 81.5 / 80.5% (DDPM-100 / DDPM-20 / DDIM-5). `verify_results.py` shares this collection
  code, so it now checks the merged rows too.
- `scripts/r7_figures.py`: the cotrain panel was still being generated here although amendment 7a
  cut the co-training comparison and the shipped figure set has no cotrain figure; the call is
  replaced by the skip notice, so re-running the script reproduces the shipped set exactly.

### Changed

- `docs/REPORT_ROUND_7.md`: the two P1 frontier tables carry the corrected `derived s/ep`
  values; the P3 cell means were re-derived from the packaged table and now follow a stated
  half-up rounding convention (PickCube DDIM-5 default cell 63.13%, not 63.10%; PullCubeTool V1
  `fixed:8` range [20.0-34.5]; Task F V1 difference -5.88 pp); revision note (2) records all of it.
- `docs/REPORT_ROUND_6.md` corrections list: two statements this pass had missed — §B's last two per-run
  differences per row were stale (they contradicted the row's own mean), and §F's PickCube V1
  difference is -5.88 pp, not -5.87.
- Re-verified after the fixes: `r5_integrity.py` 4 196 chunk files / 0 problems,
  `verify_results.py` 3 526 values / 0 mismatches, release tests 21/21, and all 24 frontier rows
  plus all 16 P3 cells re-checked against `r7_runlevel_summary.csv`.

## 1.3.1

Documentation correction only; no code, data, table or figure changed.

- `docs/REPORT_ROUND_7.md`: an external review found stale interim numbers and unfinished
  wording left in the round-7 report. Corrected: the `PullCubeTool-v1` V3 run values
  (30.5 / 22.5 / 26.0, mean 26.3% — the flag and the +14.7 pp difference are unchanged and match
  the shipped CSVs); the `PullCubeTool-v1` sampler paragraph (final eight paired comparisons and
  the 0.3-8.2 pp compatible-loss range, replacing an interim "4.6-7.6 pp on the cells computed so
  far"); the P3 post-hoc best cell on that task (V1 `fixed:8` at DDPM-100, 27.50%, not the DDIM-5
  cell); the adaptive penalty on `PickCube-v1` (-14.75 pp on B0 / -5.87 pp on V1); the parity
  table's three variant rows, which still said "pending"; the V2b `PickCube-v1` remark ("every run
  at or above the baseline mean" — one of the three is 0.9 pp below it); the round-7 raw-tree
  episode counts (12 analysed frontier cells / 2 400 episodes, with the quarantined superseded
  run of 30 episodes listed separately). A revision note was added at the top of the addendum and
  the full correction list is in `NOTES.md` (round 7 post-release).
- Re-verified after the edit: `r5_integrity.py` 4 196 chunk files / 0 problems,
  `verify_results.py` 3 526 values / 0 mismatches, release tests 21/21.

## 1.3.0

Round 7 (report addendum 6): parity for `PickCube-v1` and `PullCubeTool-v1` — sampler x horizon
frontiers on the co-trained checkpoint with fresh seed blocks, the prediction-horizon/encoder
variants on the other two tasks, the completed four-way combination, our own co-trained models
compared with single-task runs, failure analysis of the trained models, and a consistency pass
over round 6's report.

### Added

- `scripts/r7_runlevel.py` — round-7 run-level summary/arm tables (one row per training run and
  schedule) covering the round-5/6/7 raw roots.
- `scripts/r7_tests.py` — every round-7 pre-registered comparison into one table with, per row,
  the arm sizes, permutation count, smallest attainable p, statistic, difference, bootstrap
  interval, the p-value where the design is valid, and the descriptive flag where it is not.
- `scripts/r7_p0_dimstats.py` — the normalization diagnostic: per-dimension mean/sd of the
  training demonstrations, the V1 floor rule as implemented, and the dimensions most amplified.
- `scripts/r7_failure_classify.py` — failure-phase classification (thresholds unchanged from
  round 3) over episodes logged with `--decision-log`.
- `scripts/r7_parity_figure.py` — the parity matrix figure (supported / not established /
  contradicted per task).
- `configs/round7_*.json` — the round-7 pre-registered grids, triggers and decision rules.
- `runlevel.descriptive_flag()` — the flag used when a design cannot reach significance by
  construction ("suggestive" iff the arm mean is outside the baseline's min-max range and every
  run is on the same side of the baseline mean), with hand-computed tests.

### Changed

- `runlevel.py` gained `descriptive_flag`; `run_rollout`/`run_matrix` gained a
  `--feature-check` passthrough in the evaluation helper (raw observation logging used by the
  round-7 diagnostic).
- Fixed a round-6 defect in the DDIM comparison pairing: the DDPM-100 baseline was matched by
  sampler and step count, so DDIM cells were compared with themselves (all differences 0.0 pp).
  The baseline is now always the same run's DDPM-100 cell; the corrected rows reproduce the
  values quoted in `REPORT_ROUND_6.md` and extend them to all 12 runs.

### Notes

- No new timing benchmark is taken except for architectures not timed before (timed once,
  sequentially, on an idle machine); policy seconds per episode remain derived else.
- P6 (does the adaptive gain depend on policy quality?) is reported as *not testable without
  retraining*: all six B0 `PushCube-v1` runs saved only final checkpoints (verified by weight
  hashes).

## 1.2.0

Round 6 (report addendum 5): run-level statistics for training-time comparisons, the adaptive
rule evaluated on the trained single-task models, the normalization x adaptive x sampling
combination, and a descriptive co-training comparison.

### Added

- `src/adaptive_replan/runlevel.py` — the run-level statistics used for every round-6 claim:
  exact two-sided permutation test on two groups of run-level rates (enumerating all
  `C(n1+n2, n1)` assignments), exact sign-flip test on paired run differences, percentile
  bootstrap interval for the difference in means, and a per-arm summary helper.
- `tests/test_runlevel.py` — hand-computed cases for both permutation tests (including the
  smallest attainable two-sided p for a 4-vs-4 comparison, 2/70) and a synthetic-observation
  test that the adaptive rule's geometric features are computed from the raw observation and
  change under normalization (the raw-feature requirement of round 6).
- `scripts/r6_runlevel.py` — builds `results/r6_runlevel_summary.csv` (one row per training run
  and schedule) and `results/r6_runlevel_tests.csv` (every pre-registered run-level test with
  arm sizes, statistic, p-value, difference and bootstrap interval).
- `scripts/r6_figures.py` — the four round-6 figure families (per-run strip plots; the
  eight-cell success-versus-derived-seconds combination chart; V1 vs B0 per task; co-trained
  checkpoint versus single-task runs).
- `scripts/r6_verify_rawfeat.py` — offline verification that the rule used the raw observation:
  recomputes `d_tcp`/`d_goal` from the raw rows logged during real episodes and compares them
  with the values the rule used.
- `configs/round6_*.json` — the pre-registered round-6 grids and decision rules.

### Changed

- `scripts/pack_final_results.py` discovers the round-6 raw roots (`r6_*`) as additional
  seed blocks, so `FINAL_RESULTS.csv` and `MANIFEST.md` cover them without hand-maintained
  lists; `verify_results.py` inherits the same discovery.

### Notes

- Round 6 takes no new timing benchmark: policy time per episode is derived from the round-5
  clean per-call measurements and is labelled "derived" wherever it appears.
- No run is discarded for a bad result; crashed runs are retried once from scratch with the
  same seed and every run is reported.

## 1.1.0

Round 5 (report addendum 4): diffusion-sampling sweep, execution-horizon-by-sampler
frontier, a clean inference-timing benchmark, and training-time variants of the DP
observation encoder / prediction horizon on `PushCube-v1`.

### Added

- `scripts/r5_sampling_analysis.py` — sampler x steps analysis: per-cell Wilson summaries and
  paired comparisons against the 100-step DDPM baseline, keyed by
  (task, sampler, diffusion_steps, schedule) instead of schedule alone, with the
  "largest compatible loss" and "excludes a loss > 10 pp" columns.
- `scripts/r5_training_analysis.py` — training-variant evaluation analysis: per-run Wilson
  summaries, paired McNemar/Newcombe against each baseline seed, and the pre-registered
  decision rule applied to variant groups (trailing seed digit stripped).
- `scripts/r5_figures.py` — the three round-5 figure families: PushCube success-versus-seconds
  frontier, success-versus-sampling-steps for all three tasks (DDPM vs DDIM), and the
  training-variant per-run points with means.
- `scripts/r5_integrity.py` — pre-analysis guard for the raw logs: unparsable lines, duplicate
  seeds and wrong episode counts per chunk file.
- `configs/round5_sampling_sweep.json`, `configs/round5_frontier.json`,
  `configs/round5_timing.json`, `configs/round5_training_variants.json` and the three
  `configs/model_v*.json` variant model-configs — the pre-registered grids and settings.
- `run_rollout.py` / `run_matrix.py`: `--sampler {ddpm,ddim}`, which rebuilds the checkpoint's
  scheduler as a `diffusers.DDIMScheduler` from its trained betas (epsilon prediction,
  `eta = 0`, `clip_sample = True`) before `set_timesteps`. `--raw-root` added to
  `run_matrix.py` so round-5 phases can live under a subdirectory.

### Changed

- `pack_final_results.py` discovers nested round-5 phases
  (`sampling_sweep/<phase>`, `training_variants/eval_<model>`) and lists them in
  `FINAL_RESULTS.csv`; block discovery is shared with the manifest writer.
- `MANIFEST.md` now inventories every raw log under the raw root (previously only
  top-level phases) and its reported total matches the number of listed rows. The 40
  `AMBIGUOUS_provenance` files stay excluded everywhere.
- `docs/REPORT_ROUND_5.md` added.

## 1.0.0

First packaged release. It consolidates four rounds of work that had accumulated as a
set of flat scripts in the working archive.

### Added

- `src/adaptive_replan/` — library split out of the original scripts:
  - `envs.py` (checkpoint loading, environment construction, seed helpers)
  - `features.py` (observation field offsets, geometric features, phase labels)
  - `policies.py` (all execution-horizon schedules)
  - `rollout.py` (single-episode rollout)
  - `statistics.py` (Wilson, exact McNemar, Newcombe paired difference, power)
- `scripts/run_rollout.py` — runs one schedule for a list of seeds.
- `scripts/run_matrix.py` — parallel driver over a schedule x seed matrix, resumable.
- `scripts/analyze_results.py` — summary, paired and figure-ready tables from raw logs.
- `scripts/make_figures.py` — per-task and combined success-versus-cost figures.
- `scripts/verify_results.py` — recomputes `results/FINAL_RESULTS.csv` from the raw logs.
- `scripts/pack_final_results.py` — builds `FINAL_RESULTS.csv`, the paired table and `MANIFEST.md`.
- `tests/test_core.py` — fast unit tests for schedules, features and statistics.
- `configs/` — the pre-registered settings of every experiment.

### Replaced

- `adaptive_rollout.py` + `dp_common.py` -> `src/adaptive_replan/` plus
  `scripts/run_rollout.py`. The two originals mixed library code, CLI parsing and
  environment setup in a single file.
- `analysis.py` -> `scripts/analyze_results.py`.
- `trace_figure.py` and `pushcube_failure.py` -> `scripts/failure_analysis.py`.
- `select_ckpt.py` -> `scripts/select_checkpoint.py`.
- `task_c_analysis_r3.py` and parts of `task_e_packaging.py` -> `scripts/make_figures.py`
  and `scripts/pack_final_results.py`.

### Changed

- All absolute paths removed. The checkpoint root and the teammate's code root are now
  `--runs-root` / `--teammate-code` or `DP_RUNS_ROOT` / `DP_TEAMMATE_CODE`.
- The sampling-step override is now recorded in each chunk's metadata
  (`diffusion_steps`); an earlier run stored the same phase directory for two step counts
  and its provenance had to be recovered by hand.
- A re-run chunk is truncated rather than appended, so a killed chunk cannot duplicate seeds.
- `analyze_results.py` gained `--include-adaptive-pairs` (also pairs adaptive-family
  schedules against each other, e.g. the state rule versus the time-window rules).
- `verify_results.py` gained `--raw-root` and now checks `FINAL_RESULTS.csv` per seed
  block, sharing the block definitions of `pack_final_results.py`; the earlier version
  pooled a schedule's rows across blocks and reported false mismatches once a schedule
  appeared in more than one block.

### Fixed

- `scripts/run_rollout.py` now bootstraps `src/` on `sys.path` like the other scripts;
  in the first packaging pass it could only run with the package otherwise importable.
- Source comments removed throughout with a `tokenize`-based pass (shebangs and
  `# noqa` pragmas kept; see `docs/RELEASE_CHECK.md` for the list).

### Removed

- Dead helpers and duplicated copies of the analysis scripts from earlier rounds.
- Internal notes, agent briefs and working logs (kept in the working archive only).
