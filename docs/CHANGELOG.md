# Changelog

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
- `docs/REPORT_ADDENDUM_4.md` added.

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
