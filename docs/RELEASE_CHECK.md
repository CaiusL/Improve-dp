# Release check

Date of packaging: round 4 (final). Everything below was executed on the packaging
machine; the commands are in the README and can be re-run from this folder.

## Final tree

```
adaptive-replanning-dp/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                     8 pre-registration JSON files (rounds 1-4)
├── docs/
│   ├── CHANGELOG.md
│   ├── RELEASE_CHECK.md         (this file)
│   ├── REPORT_NOTES.md          round-1 report (sanitised copy)
│   ├── REPORT_ADDENDUM.md       round-2 report (sanitised copy)
│   ├── REPORT_ADDENDUM_2.md     round-3 report (sanitised copy)
│   ├── REPORT_ADDENDUM_3.md     round-4 report (sanitised copy)
│   └── STAGE_REPORT_v2_CHECK.md verification of the stage report
├── results/
│   ├── FINAL_RESULTS.csv        59 rows: every (task, seed block, schedule) cell
│   ├── MANIFEST.md              1 446 raw-log files, 14 082 episodes, SHA-256 each
│   ├── r4_fresh_final_summary.csv / r4_fresh_final_paired.csv   (22 pairs)
│   └── fig_*.{png,pdf}          per-task and combined success-versus-cost figures
├── scripts/                     8 CLI scripts (run, analyse, figures, pack, verify, …)
├── src/adaptive_replan/         6 library modules
└── tests/test_core.py           15 unit tests
```

Total size ≈ 1 MB; largest files: `results/fig_combined_success_vs_cost.png` (172 KB),
`results/MANIFEST.md` (144 KB), per-task PNGs (60-64 KB).

## Excluded (and why)

| Item | Reason |
|---|---|
| Model checkpoints (`*.pt`) | teammate-owned; not ours to republish |
| Demonstration data (`*.h5`) | teammate-owned |
| Teammate's `diffusion_policy` code / ManiSkill fork | teammate-owned; replaced by `--runs-root` / `--teammate-code` and `DP_RUNS_ROOT` / `DP_TEAMMATE_CODE` |
| Per-episode raw logs (83 MB, 1 446 JSONL) | exceed the ~50 MB guideline; `results/MANIFEST.md` lists every file with episode count, seed range and SHA-256 so the external archive can be verified and pointed at via `--raw-root` |
| Working archive (notes, agent briefs, stage reports, earlier scratch results) | internal working material; the four sanitised report copies carry the findings |
| `STAGE_REPORT_v2.md` | the user's own document; will be added by the user |
| venvs, caches, `.DS_Store`, scratch files | not source |

**Not reproducible from this folder alone** (need the external artefacts above): the raw
logs behind `MANIFEST.md`, any new rollout (checkpoint + teammate code required), and the
figures/CSVs when regenerated from raw (the shipped summaries were built from the external
raw tree; regenerate with `--raw-root` pointing at it).

## Verification runs

- `python -m py_compile` on all 15 Python files: pass.
- `bash -n`: no shell scripts shipped. `ruff`/`black`: not installed on the packaging
  machine; skipped (no reformatting was needed after the comment strip).
- `tests/test_core.py`: **15/15 passed** (pure Python, no environment needed).
- `scripts/verify_results.py --raw-root <external raw tree>`: **744 values checked,
  0 mismatches** — every row of `FINAL_RESULTS.csv` (both success metrics, Wilson
  intervals, calls, mean horizons, decision fractions, wall/inference seconds) recomputes
  from the per-episode logs using only this folder's scripts.
- `scripts/pack_final_results.py` and `scripts/make_figures.py` were re-run from this
  folder against the raw tree and reproduced the shipped CSV and figures.

### Regression: release code vs original logs

5 seeds × 3 schedules × 3 tasks (45 episodes) with the release scripts; compared per
episode on `success_once`, `success_at_end`, `n_inference`, `mean_horizon`,
`episode_len`, `n_env_steps`, `return`: **315 values, 0 differences**.

| Task | fixed:2 | fixed:8 | adaptive |
|---|---|---|---|
| `PushCube-v1` (seeds 200000-200004) | identical | identical | `adaptive_push:d_tcp=0.08,d_goal=0.10` identical |
| `PickCube-v1` (seeds 300000-300004) | identical | identical | `adaptive_push:d_tcp=0.03,d_goal=0` identical |
| `PullCubeTool-v1` (seeds 200000-200004) | identical | identical | `adaptive_dist:d=0.30` identical |

Full-block check: `PushCube-v1` `fixed:2` on **all** seeds 200000-200199 re-run with the
release code: **152/200 successes — exactly the pre-registered expectation — and every
episode identical to the original log on all seven fields.**

Bugs found while running these checks and fixed in this folder (no logic or defaults
changed, see `CHANGELOG.md`): a missing `sys.path` bootstrap in `run_rollout.py`;
`verify_results.py` keying rows across seed blocks; `analyze_results.py` needed the
opt-in `--include-adaptive-pairs` to reproduce the adaptive-vs-window rows.

## Seed-range audit (from `results/MANIFEST.md`)

Every phase's seed ranges were checked against the pre-registration: all calibration,
tuning and validation phases touch only the validation block (seeds 100000-100099);
test blocks use only 200000-200199 (rounds 1-2) or 300000-300199 (rounds 3-4); the
reduced-steps runs use the 300000-300099 sub-block of the fresh PushCube block. No phase
leaks into a seed block reserved for another role. (`PickCube-v1`'s round-3 test block and
`PushCube-v1`'s round-4 fresh block share the numeric range 300000-300199 by design: the
range was fresh per task, and the two tasks were never compared on shared seeds.)

## Comments removed

A `tokenize`-based pass (not regex) stripped all comments from the shipped Python files;
strings containing `#` (matplotlib colour codes, CSV text) were untouched. Kept comments
(18 in total):

- shebangs `#!/usr/bin/env python` (8): `analyze_results.py`, `failure_analysis.py`,
  `make_figures.py`, `pack_final_results.py`, `run_matrix.py`, `run_rollout.py`,
  `select_checkpoint.py`, `verify_results.py`
- `# noqa: E402` (9): `analyze_results.py`, `pack_final_results.py`, `run_matrix.py`,
  `select_checkpoint.py`, `verify_results.py`, `make_figures.py` ×3, `test_core.py`
- `# noqa: F401` (1): `test_core.py`

## Docstrings

Every module/function/class docstring was audited via AST for absolute paths, personal
information, experiment results and internal notes: **none contained any**, so none were
shortened. Two docstrings were reworded to describe the new `--raw-root` behaviour
(`verify_results.py`). The word "teammate" appears in dependency documentation only, as
an anonymous role ("a teammate-supplied checkpoint"); no handle or real names anywhere.

## Safety scan

Patterns searched across the whole folder (case-insensitive): `/Users/`, `caius`,
`Desktop`, `HKU`, e-mail addresses, `token|secret|password|api_key|BEGIN *KEY`,
run-directory names that could identify the teammate's repository.

Result: **0 hits anywhere in code, configs, results or README**, and none in the report
copies except the self-referential list above (this section necessarily names the patterns
it searched for, plus the substring "token" inside the word "tokenize" in the changelogs
and one sentence in `docs/REPORT_ADDENDUM_3.md` describing the plan).
One deliberate survivor for the user to review:
the course code appears once, in `docs/REPORT_NOTES.md` (a historical report whose
conclusions depend on it); remove it there if public attribution is a concern.
No remote was added and nothing was pushed; there is **no `LICENSE` file** — the choice
is left to the repository owner (course policy and the teammate's rights over the
checkpoint/data should be checked; ManiSkill and Diffusion Policy are separate projects
with their own licences).
