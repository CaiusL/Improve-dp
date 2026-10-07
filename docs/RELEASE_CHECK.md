# Release check

Release **1.3.6** (round-7 content; 1.3.1-1.3.6 are documentation and correction patches on it),
packaged 2026-10-07. This file states what the shipped archive contains, what it deliberately
leaves out, and how each claim was checked. Section 1 is the current state of the package;
section 5 keeps the round-by-round verification history with the numbers **as they stood at that
round**; section 6 records the corrections made after review.

## 1. Current state

| Item | Value |
|---|---|
| Version / commits | 1.3.6 — round-7 content 1.3.0 `2612833`, docs 1.3.1 `004eac0`, number corrections 1.3.2 `a36089e`, release prep `1ef43ae`, reproduction fixes 1.3.3 `e23207c`; the 1.3.4-1.3.6 docs commits are the ones carrying this file |
| Archive | `dp-replan-release-v1.3.6.zip`, **1.9 MB, 128 files** (git history is not in the zip) |
| Unpacked size | ≈3.4 MB — `results` 2.7 MB, `docs` 268 KB, `scripts` 200 KB, `configs` 108 KB, `src` 100 KB, `tests` 16 KB |
| Unit tests | **21/21** (`tests/test_core.py` 15 + `tests/test_runlevel.py` 6) |
| Raw-log integrity | **4 196 chunk files, 0 problems** (`scripts/r5_integrity.py`) |
| Published values | **3 526 recomputed, 0 mismatches** (`scripts/verify_results.py`) |
| Packaged table | `results/FINAL_RESULTS.csv`, **273 rows** |
| Raw-log inventory | `results/MANIFEST.md`, **5 392 files / 53 942 episodes**, SHA-256 per file |
| Figures | 19 figure families, 38 files (PNG + PDF) |
| Reproduction | **11 shipped tables and logs byte-identical; all 38 figure files reproduce** (section 4) |
| Licence / remote | no `LICENSE`; no git remote; nothing pushed |

## 2. What is in the archive

```
adaptive-replanning-dp/       128 files in total
├── README.md                    overview, quick start, "Reproducing the results"
├── requirements.txt             numpy, torch, matplotlib, diffusers, gymnasium, ManiSkill 3.0.1
├── .gitignore
├── configs/           27 JSON   24 pre-registrations (rounds 1-4: 8, round 5: 4, round 6: 5,
│                                round 7: 7) + 3 model specs for the V2a/V2b/V3 variants
├── docs/               9 files  CHANGELOG, RELEASE_CHECK (this file), and one report per round:
│                                REPORT_ROUND_1 … REPORT_ROUND_7 (renamed from REPORT_NOTES /
│                                REPORT_ADDENDUM* in 1.3.6; see section 6 for the mapping)
├── results/           59 files  FINAL_RESULTS.csv (273 rows, one row per reported cell,
│                                including the 59 round-1-4 rows), per-block source tables
│                                (r5_*, r6_*, r7_*, plus r4_fresh_final_* from the earlier
│                                rounds), MANIFEST.md (5 392 raw files), r7_p0_*.json,
│                                19 figures as PNG + PDF
├── scripts/           21 files  rollouts (run_rollout, run_matrix, select_checkpoint), analyses
│                                (r5_*, r6_*, r7_*, analyze_results, failure_analysis), figures
│                                (make_figures, r5_figures, r6_figures, r7_figures,
│                                r7_parity_figure), packaging/verification (pack_final_results,
│                                verify_results, r5_integrity)
├── src/adaptive_replan/ 7 files envs, features, policies, rollout, runlevel, statistics, __init__
└── tests/              2 files  test_core.py (15), test_runlevel.py (6)
```

Largest files: `results/MANIFEST.md` (616 KB), `results/fig_r7_parity.png` (177 KB),
`results/fig_combined_success_vs_cost.png` (174 KB), `results/fig_r5_steps.png` (131 KB).

The shipped Python is comment-free apart from shebangs and `# noqa` pragmas, with one deliberate
exception: `scripts/pack_final_results.py` carries two short comments explaining the round-5
frontier merge added in 1.3.2. Every module/function/class docstring was audited for absolute
paths, personal information and internal notes; none contained any.

## 3. What is excluded (and why)

| Item | Reason |
|---|---|
| Model checkpoints (`*.pt`) | teammate-owned; not ours to republish. Supplied through `--runs-root` / `DP_RUNS_ROOT` |
| Demonstration data (`*.h5`) | teammate-owned. Supplied through the training-data manifest |
| Teammate's `diffusion_policy` code / ManiSkill fork | teammate-owned; replaced by `--teammate-code` / `DP_TEAMMATE_CODE` |
| Per-episode raw logs | **245 MB, 5 432 JSONL chunk files** on the packaging machine (the manifest lists 5 392: the 40 `AMBIGUOUS_provenance` files are excluded). Too large to ship; `results/MANIFEST.md` gives each file's episode count, seed range and SHA-256 so an external copy can be verified and pointed at with `--raw-root` |
| Git history (`.git`, 3.1 MB) | not part of the zip; the repository itself carries it |
| Working archive (notes, agent briefs, stage reports, scratch results) | internal working material; the sanitised report copies carry the findings |
| `STAGE_REPORT_v2.md` | the user's own document; added by the user |
| `STAGE_REPORT_v2_CHECK.md` and the other stage reports | the check verifies `STAGE_REPORT_v2.md`, which is not in the archive, and the script that produced it is not shipped either, so a reader could neither read the checked document nor reproduce the check. Shipped in 1.0.0-1.3.5, dropped in 1.3.6; its findings are in `REPORT_ROUND_4.md` |
| venvs, caches, `.DS_Store`, scratch files | not source |

**Falsifiable in one step:** unzip this archive and run section 4 — everything except the raw
logs is inside it. New rollouts additionally need the checkpoint and the teammate's code. Six
shipped CSVs and three JSON records are not regenerated by the documented commands and need
either a specific script invocation or an argument file (the frontier table, clean timing, the
raw-feature record, the failure spec, the diagnostics manifest and the two `r4_fresh_final_*`
tables) — see section 4.

## 4. How to verify this release

From the unpacked archive root, with the raw logs at `$RAW`:

```bash
# unit tests (pytest is not installed in the analysis environment; the files are pytest-style)
python - <<'EOF'
import importlib.util, pathlib
for f in sorted(pathlib.Path("tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for n in sorted(x for x in dir(m) if x.startswith("test_")):
        getattr(m, n)()
print("tests pass")
EOF

python scripts/r5_integrity.py --raw-root "$RAW"        # 4 196 chunk files, 0 problems
python scripts/verify_results.py --raw-root "$RAW"      # 3 526 values, 0 mismatches
python scripts/r7_runlevel.py --raw-root "$RAW"         # r7_runlevel_{summary,arms}.csv
python scripts/r7_tests.py --raw-root "$RAW"            # r7_tests.csv (every pre-registered test)
python scripts/pack_final_results.py --raw-root "$RAW" --with-manifest
```

The full pipeline is in the README's "Reproducing the results" section; running it end to end
from a clean extraction was verified as follows (**1.3.3**):

- **Every runnable command passes: 18/18** (4 skipped: three need argument placeholders, one needs
  the checkpoint).
- **11 shipped tables and logs are regenerated byte-for-byte**: `FINAL_RESULTS.csv`, `MANIFEST.md`,
  `r5_sampling_{summary,paired}.csv`, `r5_training_{summary,paired}.csv`,
  `r6_runlevel_{summary,tests}.csv`, `r7_runlevel_{summary,arms}.csv`, `r7_tests.csv`.
- **All 19 figure families (38 files) reproduce**: every PNG byte-identical; every PDF identical
  once the embedded `CreationDate` is ignored.
- The flow also writes four CSVs that are **new outputs, not replacements**
  (`r6_runlevel_arms`, `r6_runlevel_mcnemar`, `pushcube_fresh_{summary,paired}`).
- Not regenerated by the documented commands, and shipped as built during their rounds:
  `r5_frontier_{summary,paired}.csv`, `r5_timing_clean.csv`, `r4_fresh_final_*.csv`,
  `r6_rawfeat_check.json`, `r7_failure.csv`, `r7_p0_{dimstats,zscores}.json`.
- Rounds 1-4 ship as **rows, not tables**. Their 59 cells are in `FINAL_RESULTS.csv` and are
  re-derived from the raw logs by `pack_final_results.py` (the block list at the top of that script
  is the round-1-4 phase list), but of the tables those rounds produced in the working archive only
  `r4_fresh_final_*.csv` is included; `pushcube_test100_*`, `pickcube_val_*`, `confirm_summary_*`,
  `partial_test_*`, `final_*` and the `r3_*` cross-task analyses are not part of this release, and
  no shipped script regenerates the `r3_*` ones.

## 5. Verification history, as it stood at each round

### Rounds 1-4 (packaged as 1.0.0)

- `python -m py_compile` on all 15 Python files: pass. `bash -n`: no shell scripts shipped.
  `ruff`/`black`: not installed on the packaging machine; skipped.
- `tests/test_core.py`: **15/15** pass (pure Python).
- `scripts/verify_results.py --raw-root <external raw tree>`: **744 values checked, 0 mismatches**
  — every row of the then-59-row `FINAL_RESULTS.csv` recomputed from the per-episode logs using
  only this folder's scripts. `pack_final_results.py` and `make_figures.py` re-ran and reproduced
  the shipped CSV and figures.
- Regression, release code vs original logs: 5 seeds x 3 schedules x 3 tasks (45 episodes) with
  the release scripts — **315 values, 0 differences** on `success_once`, `success_at_end`,
  `n_inference`, `mean_horizon`, `episode_len`, `n_env_steps`, `return`. Full-block check:
  `PushCube-v1` `fixed:2` on all seeds 200000-200199 — **152/200 successes, every episode
  identical** on all seven fields.
- Bugs found while running these checks and fixed (no logic or defaults changed): a missing
  `sys.path` bootstrap in `run_rollout.py`; `verify_results.py` keying rows across seed blocks;
  `analyze_results.py` needing the opt-in `--include-adaptive-pairs`.
- Seed-range audit from `results/MANIFEST.md`: all calibration/tuning/validation phases touch only
  seeds 100000-100099; test blocks use 200000-200199 (rounds 1-2) or 300000-300199 (rounds 3-4);
  the reduced-steps runs use 300000-300099. No phase leaks into a block reserved for another role;
  `PickCube-v1`'s round-3 test block and `PushCube-v1`'s round-4 fresh block share 300000-300199 by
  design (fresh per task; the tasks were never compared on shared seeds).
- Safety scan and comment/docstring audit as described in section 7.

### Round 5 (1.1.0)

- `scripts/r5_integrity.py` over the whole round-5 raw tree: **810 chunk files, 0 problems**.
  Written after a double-launch race tore 7 chunk files; those files were deleted and the affected
  model re-run from scratch rather than patched.
- The package was **ported** (not copied) for the round-5 features: `envs.py` `load_run` accepts a
  direct `.pt` path, checkpoints without a `compat` block, and an `obs_norm` block; `rollout.py`
  applies that affine before `get_action`; `run_rollout.py`/`run_matrix.py` gained
  `--sampler {ddpm,ddim}`. Equivalence against the working implementation on identical seeds and
  checkpoints: DDPM-100 `fixed:8` seeds 400000-400002, DDIM-5 seeds 400000-400002 and a
  normalized-observation checkpoint seeds 500000-500002 — identical.
- `pack_final_results.py` reproduced the previously shipped round-4 rows **byte-identically**
  (0 of 59 changed) while adding round-5 rows; `verify_results.py` then checked **1 329 values,
  0 mismatches**.
- `MANIFEST.md` was verified at checksum level against the previous version: **0 of 1 406 old
  checksums missing, exactly 600 new**. The old file's reported total (1 446) had exceeded its
  listed rows (1 406) because the 40 excluded `AMBIGUOUS_provenance` files were counted; fixed.

### Round 6 (1.2.0)

- `tests/test_runlevel.py` plus `tests/test_core.py`: **19/19** pass, including hand-computed
  permutation and sign-flip cases and the raw-feature rule test.
- Raw-feature requirement (the rule must read the raw observation, not the normalized one):
  verified on two real episodes of a normalized-observation checkpoint (191 decisions) by
  recomputing `d_tcp`/`d_goal` from the logged raw observation rows with this package's own feature
  implementation — **0 feature mismatches, 0 rule mismatches**, and the normalized values differed
  in every decision (the check is not vacuous). Record: `results/r6_rawfeat_check.json`.
- `r5_integrity.py` over the round-5 and round-6 roots: **2 530 chunk files, 0 problems**.
- `pack_final_results.py` discovers the `r6_*` raw roots automatically; `verify_results.py`
  checked **2 447 values, 0 mismatches**; `FINAL_RESULTS.csv` then held **190 rows** and
  `MANIFEST.md` **3 726 files / 37 282 episodes**.
- Every training run and rollout phase is protected by an atomic lock; no phase can start twice,
  and stale locks are broken only when the owner process is gone.

### Round 7 (1.3.0)

- Tests: **21/21** (round 6: 19; round 7 adds `runlevel.descriptive_flag` cases and the
  permutation-count identities 924 / 210 / 70 / 35 quoted as smallest attainable p values).
- Round-7 evaluators are lock-guarded. Orphaned children of a superseded launch kept running under
  old phase names after their parent was stopped; their partial output was quarantined as
  `_superseded_orphan_*` and excluded from every analysis, and the live queues were re-run under
  unique per-task phase names.
- `r5_integrity.py` covering round-5 to round-7 roots: **4 196 chunk files, 0 problems**.
- `verify_results.py` over the full raw tree: **3 526 values, 0 mismatches**;
  `FINAL_RESULTS.csv` **273 rows**, `MANIFEST.md` **5 392 files / 53 942 episodes**.
- A round-6 defect was found while preparing the round-7 consistency pass: the DDIM-vs-DDPM
  pairing compared cells with themselves (see CHANGELOG 1.3.0). The regenerated
  `results/r6_runlevel_mcnemar.csv` reproduces the values quoted in `REPORT_ROUND_6.md` §D and
  extends them to all 12 runs.

## 6. Corrections after review (1.3.1 → 1.3.6)

Three review passes checked the released numbers and commands against the raw logs and the packaged
tables; 1.3.4-1.3.6 are documentation and renaming passes. Every correction was applied to the shipped
artefacts and re-verified; no success rate changed in any of them.

- **1.3.1 (documentation).** Stale interim values and unfinished wording in
  `REPORT_ROUND_7.md`: the `PullCubeTool-v1` V3 run values (30.5 / 22.5 / 26.0%, mean 26.3% —
  the flag and the +14.7 pp difference were already correct and match the CSVs), the
  `PullCubeTool-v1` frontier sampler paragraph, the P3 post-hoc best cell (V1 `fixed:8` at
  DDPM-100, 27.50%), the `PickCube-v1` adaptive penalty, the parity table's three variant rows and
  the round-7 raw-tree episode counts. No CSV, table or figure changed.
- **1.3.2 (two pipeline defects).** `r7_runlevel.py` looked the clean per-call time up under the
  arm name `co` while the co-trained frontier cells carry `co_tm` (and the `("ddim", 20)` entry was
  missing), so all 16 DDPM-20 / DDIM-20 frontier cells used 145.0 ms and the derived `s/ep`
  column, two P1 tables and `fig_r7_frontier_{pick,pull}` were ~4x too high.
  `pack_final_results.py` never folded the round-5 frontier's reused sweep half into its three
  `r5_front_*` `fixed:8` rows, so `FINAL_RESULTS.csv` shipped them at n = 100; they are now n = 200
  (76.0 / 81.5 / 80.5%). `r7_figures.py` no longer generates the cotrain panel (amendment 7a), so
  re-running it reproduces the shipped figure set. The round-6 corrections list gained items 9-11
  (now the closing section of `REPORT_ROUND_6.md`).
- **1.3.3 (reproduction defects).** Executing the README's "Reproducing the results" section found
  four broken scripts and five documentation defects. Both round-5 analysis scripts crashed on
  import (`HERE` used before definition); `run_matrix.py` referenced `args.sampler` without
  defining `--sampler`, so no real matrix could run; `r7_parity_figure.py` was the stale
  "pending / to be filled" version, contradicting the shipped PNG and the addendum; and the
  `--feature-check` passthrough promised in 1.3.0 and used by `r6_verify_rawfeat.py` did not exist
  — it is now implemented and tested (0 feature/rule mismatches, non-vacuous on a normalized
  checkpoint). In the README: an unmatched code fence, `--tasks` with semicolons where the parser
  wants commas, the missing `r7_figures.py`, a round-5 sweep listing 3 of the 9 samplers, and six
  commands without `--raw-root` although `results/raw/` ships empty.
- Re-verified at 1.3.3: `tests/` **21/21**; `r5_integrity.py` **4 196 / 0**; `verify_results.py`
  **3 526 / 0**; the README flow **18/18** runnable commands, **11 shipped tables/logs and all 38
  figure files** reproducing (section 4); all 24 round-7 frontier rows and all 16 P3 combination
  cells of `REPORT_ROUND_7.md` re-checked against `results/r7_runlevel_summary.csv`;
  `FINAL_RESULTS.csv` still **273 rows**, `MANIFEST.md` still **5 392 files / 53 942 episodes**.
- **1.3.4 (documentation).** This file was rewritten to describe the package as shipped rather than
  the round-4 state it had described since 1.0.0: a current-state table, an updated contents and
  exclusions inventory, a "how to verify this release" section whose commands were executed as
  written, and a separation of what the documented pipeline does regenerate from the raw logs from
  what it does not. No code, data, table or figure changed.
- **1.3.5 (documentation).** Two accuracy fixes plus the README's round section; no code, data,
  table or figure changed.
  - `README.md` gained "What a 'round' is, and what each round did": a per-round table (what the
    round tried or tested, its outcome, its report, its pre-registrations) and the three numbering
    conventions that make the archive look round-5-to-7-centric — the addendum offset as it stood
    then (`REPORT_NOTES.md` is round 1, then `REPORT_ADDENDUM.md` is round 2 through `_6` for
    round 7; renamed to `REPORT_ROUND_1` … `REPORT_ROUND_7` in 1.3.6),
    `rN_*` marking the code a round introduced rather than the data it used, and the
    round-to-release mapping (rounds 1-4 → 1.0.0, 5 → 1.1.0, 6 → 1.2.0, 7 → 1.3.0). The stale
    "this archive is release 1.3.2" sentence was corrected.
  - Sections 2 and 4 said the archive ships the rounds 1-4 per-block tables. It ships their rows —
    59 of the 273 in `FINAL_RESULTS.csv`, re-derived from the raw logs by `pack_final_results.py` —
    and of that era's tables only `r4_fresh_final_*.csv`; the others, including the `r3_*`
    cross-task analyses, are not in this release. Both sentences now say that, and the 1.3.4
    changelog entry that repeated the old wording is qualified there. Found while answering a
    reader's question about why the archive appears to cover only rounds 5-7.

- **1.3.6 (documentation and renaming).** The round reports are now named by the round they
  document, and two files that a reader of this archive could not check were removed. No code,
  data, table or figure changed; the archive holds 128 files rather than 130.
  - Renamed, each with a `Formerly …` line under its title: `REPORT_NOTES.md` →
    `REPORT_ROUND_1.md`, `REPORT_ADDENDUM.md` → `REPORT_ROUND_2.md`, `REPORT_ADDENDUM_2.md` →
    `REPORT_ROUND_3.md`, `REPORT_ADDENDUM_3.md` → `REPORT_ROUND_4.md`, `REPORT_ADDENDUM_4.md` →
    `REPORT_ROUND_5.md`, `REPORT_ADDENDUM_5.md` → `REPORT_ROUND_6.md` and `REPORT_ADDENDUM_6.md` →
    `REPORT_ROUND_7.md`. Every reference in the README, the changelog and the reports was
    rewritten; the working archive keeps the old names and its `RENAME_MAP.md` carries the map.
  - `REPORT_ADDENDUM_5_FIXES.md` was **merged into `REPORT_ROUND_6.md`**. Its corrections are
    applied in the body (items 1-6, 9, 10) and the record survives as that report's closing
    "Corrections applied when this file was packaged" section, which also keeps items 7, 8 and 11
    (already correct, or notes about script defects fixed elsewhere). Applying item 1 showed that
    the duplicated block it deletes held the only `PullCubeTool-v1` result table in that report, so
    §C was rebuilt with that task's table taken from `results/r7_runlevel_arms.csv` — mean 27.50%
    (not 27.44%), sd 6.87 (not 6.82), `success_at_end` 27.38%, difference +15.88 pp (not +15.82).
    New item 12: the +11.0 pp (p = 0.00068) adaptive-vs-`fixed:8` figure is a **round-5**
    measurement (`REPORT_ROUND_5.md` §2.2), not a round-4 one; three references were corrected.
  - `REPORT_ADDENDUM_6.md` was replaced by `REPORT_ADDENDUM_6_v2.md` — the version 2 that the
    working archive and `FINAL_REPORT.md` both record as superseding it. The copies shipped in
    1.0.0-1.3.5 carried revision notes (1)-(2) only; v2 adds the third review's six text
    corrections, with no number changing (for example "p >= 0.20" → "p >= 0.199", "six paired
    sampler comparisons" → "eight", and the PushCube adaptive citation moved from round 4 to
    round 5). v2's pointer to the review record `REVIEW_ADDENDUM_6.md` is marked
    working-archive-only.
  - `STAGE_REPORT_v2_CHECK.md` was **dropped**. It verifies `STAGE_REPORT_v2.md`, which this
    archive does not contain, and `verify_stage_report_v2.py`, which produced it, is not shipped
    either, so a reader could neither read the checked document nor reproduce the check. Its
    findings are reported in `REPORT_ROUND_4.md` §Task 0, which now points at the working archive.

## 7. Safety and licensing

Patterns searched across the whole folder (case-insensitive): absolute home paths, the account
name, `Desktop`, the institution, e-mail addresses, `token|secret|password|api_key`, key
headers, and run-directory names that could identify the teammate's repository.

Result: **no hits in code, configs, results or the README**, and no checkpoint, demonstration or
teammate-owned file is present. The only matches anywhere are self-referential or deliberate:

- the scan sections of this file and of `REPORT_ROUND_5.md` list the patterns they searched;
- one prose mention of "Desktop originals" in `REPORT_ROUND_5.md`;
- the substring `token` inside "tokenize"/"tokens" in two changelog/report sentences;
- the course code appears once, in `docs/REPORT_ROUND_1.md`, a historical report whose conclusions
  depend on it — remove it there if public attribution is a concern.

No git remote was added and nothing was pushed; there is **no `LICENSE` file** — the choice is left
to the repository owner (course policy and the teammate's rights over the checkpoint and data
should be checked; ManiSkill and Diffusion Policy are separate projects with their own licences).