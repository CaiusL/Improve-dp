# Changelog

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
