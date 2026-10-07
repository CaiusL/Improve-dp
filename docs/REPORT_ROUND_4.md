# REPORT_ROUND_4.md

Formerly `REPORT_ADDENDUM_3.md`.

Fourth round: verification of `STAGE_REPORT_v2.md`, fresh-seed confirmation on `PushCube-v1`
with a time-window search, reduced-diffusion-steps on fresh seeds with a clean sequential timing
benchmark, and an exploratory `PickCube-v1` stillness check. First written as a battery-safe
snapshot; the fresh block has since completed to its full pre-registered n = 200 per setting and
the timing benchmark has been run with nothing else on the machine, so this document now carries
final numbers.

Primary metric `success_once` everywhere; `success_at_end` reported alongside.
Protocol unchanged (explicit `env.reset(seed=…)`, paired seeds, torch seed `1000000 + env_seed`).

---

## Task 0 — verification of `STAGE_REPORT_v2.md`

Full item-by-item table (in the working archive, not redistributed):
**`STAGE_REPORT_v2_CHECK.md`** (`verify_stage_report_v2.py`,
129/135 checks OK). **No headline claim changes.** Corrections and reconciliations:

1. **One substantive error:** §4.3's "adaptive worse than `fixed:8` … 8 vs **43** discordant
   seeds" — the discordant counts are **8 vs 29**; 43 is the *both-succeeded* column of
   `results/pickcube_test_final_paired.csv`, misread when the addendum was written. p = 0.0008
   and the verdict are unchanged. The same misread produced two wrong count pairs in
   `REPORT_ROUND_3.md` (vs random(2.89): **26 vs 27**; vs random(3.76): **28 vs 32**).
2. **`qpos0`-control calls reconciled:** the test table (95.6 calls, mean horizon 2.16) is right;
   the addendum's "≈78 calls" was `200 / 2.577` computed on the **validation** calibration seeds
   and then compared against the primary rule's **test** figure — a validation-vs-test mix-up.
   Consequence: the addendum's "the win is conservative because the control is cheaper" is wrong;
   on test the control spent *more* compute (95.6 vs 88.9) and still scored 76.5% vs 91.0%.
3. **Test-seed short-decision fractions (computed as requested):** primary adaptive **95.8%**
   short / 4.2% long; inverted rule **98.2%** short; `qpos0` control 98.4%; both time windows
   95.5%. The "about 98%" claim for the inverted rule is **confirmed** — and extended: because
   matching a mean horizon of 2.25 forces ~96–98% short decisions, the inverted and `qpos0`
   controls are near-`fixed:2` in behaviour, and the honest reading is that the primary rule's
   edge comes from *where its rare long chunks go*.
4. **`PickCube-v1` validation baselines:** `fixed:2` = **17/100 = 17.0%** (the "17.2%" in
   `REPORT_ROUND_3.md` came from 17/99 of a partial run), `fixed:4` = 30/100, `fixed:8`
   = 49/100. v2's "about 17%" is correct.
5. **`PickCube-v1` metric divergence confirmed:** test block 256/1600 = **16.00%** mid-only;
   earlier sanity sample 5/60 = 8.33%; all six per-schedule `success_at_end` rates reproduce;
   zero reverse violations anywhere in any task.
6. **Provenance defect found and fixed during Task D setup** (disclosed in NOTES.md): the first
   launch reused one phase directory for two step counts, so the 20-step run silently did nothing,
   and `diffusion_steps` was not recorded in metadata. The ambiguous directory was moved aside
   (`pushcube_steps_fresh_AMBIGUOUS_provenance`, excluded from all analysis), metadata now records
   the step count, and both step counts were re-run under distinct phases. Also: `adaptive_rollout.py`
   now truncates re-run chunks instead of appending (duplicate-seed prevention from round 3).

## Task B — `PushCube-v1` fresh-seed confirmation

### B.1 Time-window search on validation seeds (COMPLETE, n = 100 each)

Matching the primary rule's test mean horizon 2.25 requires `W = 170` and hence `s ≤ 30`:
**`s = 60` and `s = 90` were dropped as infeasible** (best achievable mean horizons 2.58 / 3.02,
outside ±0.15) before any rollout — analysis in the round-4 pre-registration.

| s | W | success | 95% CI | calls | mean h |
|---|---|---|---|---|---|
| 0 | 170 | 86/100 = 0.860 | [0.779, 0.915] | 89.0 | 2.270 |
| **10** | 170 | **86/100 = 0.860** | [0.779, 0.915] | **87.0** | 2.345 |
| 20 | 170 | 83/100 = 0.830 | [0.745, 0.891] | 88.0 | 2.341 |
| 30 | 170 | 84/100 = 0.840 | [0.756, 0.899] | 88.0 | 2.273 |

No pair differs significantly (all p ≥ 0.82). Selected by the pre-registered rule
(success → fewer calls → order): **s = 10, W = 170**.

### B.2 Fresh block, seeds 300000–300199 — **COMPLETE, n = 200 per setting**

| schedule | succ | rate | 95% CI | calls | mean h |
|---|---|---|---|---|---|
| **primary adaptive** (0.08/0.10) | 178/200 | **0.890** | [0.839, 0.926] | 88.8 | 2.26 |
| time window s=30 (pre-specified) | 174/200 | 0.870 | [0.816, 0.910] | 88.0 | 2.27 |
| TCP-only adaptive (0.05/0) | 171/200 | 0.855 | [0.799, 0.897] | **63.4** | 3.33 |
| time window s=10 (B.1-selected) | 171/200 | 0.855 | [0.799, 0.897] | 87.0 | 2.35 |
| random(mean=2.25) | 165/200 | 0.825 | [0.766, 0.871] | 89.2 | 2.25 |
| `fixed:4` | 160/200 | 0.800 | [0.739, 0.850] | 50.0 | 4.00 |
| `fixed:2` | 157/200 | 0.785 | [0.723, 0.836] | 100.0 | 2.00 |
| `fixed:8` | 149/200 | 0.745 | [0.680, 0.800] | **25.0** | 8.00 |

Pre-registered questions (exact McNemar; Newcombe 95% CI for the paired difference; full table in
`results/r4_fresh_final_paired.csv`):

- **Q1 — replication: YES.** Primary vs `fixed:4`: **+9.0 pp** [+1.9, +16.1], 24 vs 6,
  **p = 0.00143**. Primary vs `fixed:2`: **+10.5 pp** [+3.3, +17.7], 32 vs 11,
  **p = 0.00191**. Better with p < 0.05 on both → *replicated* by the pre-registered rule.
- Also pre-specified: primary vs `fixed:8`: +14.5 pp [+7.0, +21.9], 34 vs 5, **p = 0.000002**;
  primary vs `random(2.25)`: +6.5 pp [−0.4, +13.4], 24 vs 11, **p = 0.041** (significant at
  α = 0.05, though it does not survive a 9-test Bonferroni within this block alone; the same
  comparison was p = 0.0005 in the block-1 family).
- **Q2 — vs selected window (s=10):** +3.5 pp [−3.1, +10.1], 17 vs 10, p = 0.248 —
  the data **cannot distinguish** them.
- **Q3 — vs pre-specified late window (s=30):** +2.0 pp [−4.5, +8.5], 12 vs 8, p = 0.503 —
  **cannot distinguish**. (The two windows are also indistinguishable from each other, p = 0.711.)
- **Q4 (secondary) — TCP-only vs `fixed:4`:** +5.5 pp, p = 0.080 — not significant at n = 200
  (the as-of-save partial sample had shown p = 0.043; the full block supersedes it).
  TCP-only vs matched random: +3.0 pp, p = 0.44. Note the TCP-only rule still offers the best
  success-per-call point on the frontier (85.5% at 63.4 calls).

**Bonferroni within this block (9 reported tests, α = 0.0056):** the wins over `fixed:2`,
`fixed:4` and `fixed:8` survive; `adaptive` vs `random(2.25)` does not (p = 0.041).
Across both blocks the direction of every comparison agrees; only its magnitude varies.

Note (reported regardless of direction): the as-of-save partial sample had shown `fixed:2`
marginally above `fixed:4`; at full n = 200 the usual order returns (`fixed:4` 80.0% vs
`fixed:2` 78.5%) and the two are statistically indistinguishable (p = 0.71 in block 1's
comparison terms). The adaptive rule's margin over the **best** fixed baseline is unaffected.

**Escalation check: the fresh block did NOT contradict the headline — Q1 met its pre-registered
replication rule. No stop required.**

## Task D — reduced diffusion steps, fresh seeds 300000–300099 (COMPLETE at n = 100 per cell)

Paired on identical seeds vs the 100-step runs (the fresh block, first 100 seeds).
"loss" = rate(100) − rate(reduced); "max loss" = upper 95% bound of that difference — the largest
reduction the data is compatible with.

| schedule | steps | n | r100 | r_rec | loss [95% CI] | **max compatible loss** | p |
|---|---|---|---|---|---|---|---|
| `fixed:2` | 50 | 100 | 78.0 | 79.0 | −1.0 [−12.4, +10.4] | **10.4 pp** | 1.000 |
| `fixed:2` | 20 | 100 | 78.0 | 81.0 | −3.0 [−14.2, +8.2] | **8.2 pp** | 0.508 |
| `fixed:4` | 50 | 100 | 79.0 | 82.0 | −3.0 [−14.0, +8.0] | **8.0 pp** | 0.629 |
| `fixed:4` | 20 | 100 | 79.0 | 80.0 | −1.0 [−12.2, +10.2] | **10.2 pp** | 1.000 |
| `fixed:8` | 50 | 100 | 71.0 | 74.0 | −3.0 [−15.2, +9.3] | **9.3 pp** | 0.711 |
| `fixed:8` | 20 | 100 | 71.0 | 75.0 | −4.0 [−16.1, +8.3] | **8.3 pp** | 0.585 |
| adaptive | 50 | 100 | 90.0 | 90.0 | 0.0 [−8.7, +8.7] | **8.7 pp** | 1.000 |
| adaptive | 20 | 100 | 90.0 | 87.0 | +3.0 [−6.1, +12.2] | **12.2 pp** | 0.508 |

**Statement (avoiding "no loss"):** at n = 100 per cell, reducing sampling from 100 to 50 or 20
steps produced **no statistically detectable change in success in any configuration** (all p ≥ 0.51),
and observed point estimates were mostly *upward*; however the data remain compatible with
**losses of up to ~8–12 pp** (worst case ~12 pp for the adaptive rule at 20 steps). Equivalence is
not established; the largest loss compatible with the data is in the table.

**Clean timing benchmark (final requirement; run sequentially with zero other jobs on the
machine, one worker, `OMP_NUM_THREADS=1`, 10 episodes of `fixed:2` = 100 policy calls per episode
per step count):**

| sampling steps | ms per policy call (mean ± sd) | inference s per episode (mean ± sd) | speed-up |
|---|---|---|---|
| 100 | **142.6 ± 2.9** | **14.26 ± 0.29** | 1.0× |
| 50 | **72.2 ± 0.4** | **7.22 ± 0.04** | **1.98×** |
| 20 | **28.9 ± 0.1** | **2.89 ± 0.01** | **4.94×** |

These supersede all earlier load-confounded numbers (e.g. 421 ms/call under 10-worker contention)
and the interim ratios are confirmed: ~2.0× at 50 steps, ~4.9× at 20 steps. The 100-step value
(142.6 ms) also reproduces round one's single-thread reference (149 ms).

## Task C — `PickCube-v1` stillness hypothesis (EXPLORATORY, validation only)

Rule inverted relative to round 3: **long (8) when `d_obj_goal < B`, short (L) otherwise**;
2–4 configurations on validation seeds, references `fixed:4` / `fixed:8` (n = 100 each, phase
`pickcube_stillness`).

| configuration | `success_once` | `success_at_end` | calls | long-decision fraction |
|---|---|---|---|---|
| invert, B=0.05, L=2 | 18.0% | 8.0% | 86.8 | 5.2% |
| invert, B=0.05, L=4 | 32.0% | 12.0% | 43.7 | 14.8% |
| invert, B=0.10, L=2 | 19.0% | 9.0% | 74.2 | 11.8% |
| invert, B=0.10, L=4 | **33.0%** | **14.0%** | 40.6 | 23.6% |
| `fixed:4` | 30.0% | 6.0% | 50.0 | 0% |
| **`fixed:8`** | **49.0%** | 11.0% | **25.0** | 100% |

**No configuration beats `fixed:8` under the primary metric** (best inverted 33% vs 49%,
p ≤ 0.009 against it); two `L=4` variants nominally edge it under `success_at_end`
(12–14% vs 11%) but not significantly (p ≥ 0.51). Per the pre-registered decision rule,
**a fresh-seed test block for this hypothesis is NOT justified.** What the experiment hints at:
holding long chunks near the goal *does* improve still-at-end success (14% vs the forward rule's
2.5% and `fixed:4`'s 6%), consistent with the stillness mechanism, but at 40 calls it is still far
from `fixed:8`, which achieves it by never replanning at all.

## Task E — packaging status (final)

- `results/FINAL_RESULTS.csv` — 59 rows, all (task, schedule, seed-block) cells incl. both metrics,
  Wilson intervals, calls, mean horizons, short/long decision fractions, wall and inference
  seconds. Regenerated from the **complete** raw logs (fresh block at full n = 200).
- `results/MANIFEST.md` — 1 446 raw log files, 14 082 episodes, SHA-256 per file.
- Figures: `fig_combined_success_vs_cost.png/.pdf` (3 panels) + per-task PNG/PDF, regenerated
  after the fresh block completed.
- Paired tables: `results/r4_fresh_final_summary.csv` and `results/r4_fresh_final_paired.csv`
  — 22 pairs at full n = 200 (fixed/random baselines × adaptives, plus the adaptive-vs-adaptive
  and window-vs-window rows needed for Q2/Q3; `analyze_results.py --include-adaptive-pairs`).
  The `r4_fresh_asof_save_*` files are retained for provenance; superseded.
- Clean sequential timing benchmark raw logs: `results/timing_clean/r4_timing_{100,50,20}.jsonl`.
- README updated with a round-4 section.

## Task F — release status (final)

`~/dp_replan_release/adaptive-replanning-dp/` (≈1 MB, 45 files): `src/adaptive_replan/` (envs,
features, policies, rollout, statistics), `scripts/` (run_rollout, run_matrix, analyze_results,
make_figures, pack_final_results, verify_results, failure_analysis, select_checkpoint),
`configs/` (8 pre-registration JSONs), `tests/test_core.py` (**15/15 passing**), README,
requirements.txt, .gitignore, `docs/` (the four report files + `STAGE_REPORT_v2_CHECK.md` +
CHANGELOG + RELEASE_CHECK), `results/` (FINAL_RESULTS.csv, MANIFEST.md, r4 fresh tables,
figures).

- **Comments:** tokenize-based strip; only shebangs (8) and `# noqa` pragmas (10) remain — list
  in `docs/RELEASE_CHECK.md`. All files `py_compile`; tests still 15/15.
- **Docstrings:** audited via AST; none contained absolute paths, results or personal notes, so
  none were shortened; two were reworded to describe the new `--raw-root` behaviour.
- **Raw logs (83 MB > 50 MB):** excluded; `results/MANIFEST.md` ships per-file episode counts,
  seed ranges and SHA-256 checksums so an external copy can be verified. Stated in README
  ("Reproducing the results") and `docs/RELEASE_CHECK.md`.
- **Bug fixes found by release checks (no logic/defaults changed):** `run_rollout.py` lacked the
  `sys.path` bootstrap (crashed on `--help`); `verify_results.py` pooled a schedule across seed
  blocks and reported false mismatches once a schedule existed in two blocks — it now shares the
  block map of `pack_final_results.py` and checks **744 values, 0 mismatches** against the full
  raw tree. `analyze_results.py` gained `--include-adaptive-pairs` (default off) so the
  adaptive-vs-window rows (Q2/Q3) ship reproducible. All three noted in `docs/CHANGELOG.md`.
- **Regression (release code vs stored rows):** 3 tasks × {fixed:2, fixed:8, one adaptive rule} ×
  5 seeds = 45 episodes; every field (successes, call counts, mean horizon, episode length,
  return) **identical** to the original logs — 315 values, 0 differences.
- **Full-block replication:** `PushCube-v1` `fixed:2` on seeds 200000-200199 re-run with the
  release code; see `docs/RELEASE_CHECK.md` for the result (expected 152/200).
- **Safety scan:** no hits for the home path, username, `the external directory`, e-mails or tokens; the
  teammate is referenced only as "a teammate-supplied checkpoint" (no handle or names).
  One flagged item for the user: the course code appears in the historical report copies under
  `docs/`.
- **Licence:** no `LICENSE` file; choice left to the user (course + teammate permissions;
  ManiSkill is separately licensed). Nothing pushed; a single local `git init` + commit was
  made for review convenience; **no remote added**.

## Not finished

1. `PegInsertionSide-v1` was dropped from scope (no checkpoint) — recorded here for completeness.
2. The 20/50-step results are n = 100 per cell; the intervals on the largest compatible loss
   (8.0–12.2 pp) are as reported. A better-powered equivalence test was not run.
