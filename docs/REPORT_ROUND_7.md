# REPORT_ROUND_7.md — Round 7: parity for PickCube-v1 and PullCubeTool-v1

**Version 2 (2026-10-07).** This is version 2 of the round-7 report: version 1 is kept unedited in
the working archive, per the project convention. All numbers were independently recomputed from
the raw per-episode logs before this version was produced (0 numeric mismatches; the independent
review record is in the working archive); v2 changes wording and labels only, as listed in
revision note (3). No result, verdict or test value changes.

Brief: *parity for `PickCube-v1` and `PullCubeTool-v1`, plus the questions brief 6 opened.*
Scope: Panda only. Nothing in an earlier report is edited by this file; the user merges. Every
pre-registered rule was fixed in `NOTES.md` at 2026-10-06 13:10, before the first round-7 run.

Revision notes.
(1) 2026-10-06, first external review — stale interim numbers and "pending" / "will be appended"
sentences: the `PullCubeTool-v1` V3 run values are 30.5 / 22.5 / 26.0 (mean 26.3%; flag
unchanged), the `PullCubeTool-v1` frontier sampler paragraph carries the final eight paired
comparisons, the post-hoc best cell on that task is V1 `fixed:8` at DDPM-100 (27.50%), and the
parity table's three variant rows are filled in.
(2) 2026-10-06, second external review — two pipeline defects behind the numbers themselves, both
fixed and both regenerated from the raw logs:
(a) `r7_runlevel.py` looked the clean per-call time up under the arm name `co`, so every
co-trained frontier cell fell back to 145.0 ms and the DDPM-20 / DDIM-20 `derived s/ep` entries
were ~4x too large; the table now carries the `co_tm` arm and the missing DDIM-20 entry
(28.5 ms), and the frontier tables, `r7_runlevel_summary.csv` and the two frontier figures are
regenerated (no success rate changed).
(b) `pack_final_results.py` never folded the round-5 frontier's reused sweep half into its three
`r5_front_*` `fixed:8` rows, which therefore shipped at n = 100; the packer now merges the reused
seeds and `FINAL_RESULTS.csv` matches the round-5 analysis (n = 200, 76.0 / 81.5 / 80.5%).
Also in this revision: the P3 cell means were re-derived from the packaged table and now follow
the half-up convention stated in the method (sub-0.05 pp drift; e.g. the PickCube DDIM-5 default
cell is 63.13%, not 63.10%), the PullCubeTool V1 `fixed:8` range is [20.0-34.5], and the Task F
V1 difference is -5.88 pp. Full list in `NOTES.md` (`CORRECTION`, round 7 post-release).
(3) 2026-10-07, third external review (independent recomputation of every round-7 statistic from
the raw logs — 0 numeric mismatches) — six text corrections, no numbers
change: (i) P1(b) on `PickCube-v1` has **eight** paired sampler comparisons, not six (the sentence
already listed eight losses); (ii) the same paragraph's "all p >= 0.20" becomes "all p >= 0.199"
(the `fixed:2` DDPM-20 comparison has p = 0.1996); (iii) the parity table's PushCube adaptive
citation "(round 4: +11.0 pp vs `fixed:8`, p = 0.00068)" belongs to **round 5** (round 4's
fresh-block comparison was +14.5 pp, p = 0.000002 — both are now cited); (iv) P1(a) on
`PickCube-v1` references **round 6's** Task F, not round 7's; (v) the V2b `PullCubeTool-v1`
permutation difference is **+8.0 pp** under the stated rounding convention (exact +8.04; "+8.1"
came from differencing rounded means), matching the variant-summary table; (vi) the method's
"rounded half-up" claim is relaxed — values are rounded to the precision shown, and exact
midpoints in binary floating point may round either way (e.g. the derived 3.625 s/ep prints as
3.62). A note that P2 run lists are sorted ascending (not seed order) is added to P2.
Found while preparing the final report (added to this v2): (vii) P1(b) on `PickCube-v1`
stated "`fixed:8` DDIM-20 is 5.5 pp below DDPM-100 at end (11.5% vs 19.0%)" — the two
verified rates differ by **7.5 pp**; the parenthetical was right and the derived difference
was wrong. Related observation about an earlier round (not edited here): `REPORT_ROUND_5.md`
§2.2's "None of these survive a Bonferroni across the 9 combinations" is inconsistent with
its own p-values — the DDPM-100 adaptive-vs-`fixed:8` comparison (p = 0.00068) survives
α = 0.05/9 = 0.0056; `FINAL_REPORT.md` states the corrected reading.

Method (unchanged from brief 6, restated because several round-7 designs are smaller): the
training run is the unit of analysis; exact two-sided permutation tests enumerate all
`C(n1+n2, n1)` assignments and the smallest attainable p is stated for every design
(6v6 = 924 -> 0.0022; 4v6 = 210 -> 0.0095; 4v4 = 70 -> 0.0286; 3v4 = 35 -> 0.057; 3v3 = 20 ->
0.10). Within-model comparisons on identical seeds use exact McNemar on episodes. Policy seconds
per episode are **derived** as (policy calls) x (clean ms per call) from the round-5 measurements:
DDPM-100 145.0 ms (142.8 ms for normalized observations), DDPM-20 28.7 ms, DDIM-20 28.5 ms,
DDIM-5 7.2 ms. Percentages and differences are rounded to the precision shown (nominal
convention half-up; exact binary midpoints such as 3.625 may round either way); every
test uses unrounded values.

---

## P0 — why does normalization hurt `PickCube-v1`? (diagnostic, no training)

Per-dimension statistics of the 200 training demonstrations, the V1 floor rule, and the largest
observed z-scores per decision (10 episodes of one V1 checkpoint per task).

| task | median sd | dims with sd < 1% of median | largest scale factor | max abs z | decisions with any abs z > 10 |
|---|---|---|---|---|---|
| `PushCube-v1` | 0.0471 | 4 | x15 700 | 6558 | 16.8% |
| `PickCube-v1` | 0.0608 | **0** | **x961** | **57.7** | 13.6% |
| `PullCubeTool-v1` | 0.0779 | 1 | x2810 | 301 | 44.0% |

**Pre-registered trigger: NOT MET** (part 1: no `PickCube-v1` dimension below 1% of the median sd;
part 2: the share of decisions with any abs z > 10 is 13.6% on `PickCube-v1`, *lower* than both
other tasks). **V1b is not trained, per the rule.** The pre-registered trigger was therefore not
met, and this diagnostic (ten episodes of one checkpoint per task) **gives no support** to the
"tiny standard deviation amplified at test time" explanation: `PickCube-v1` shows the mildest
normalization amplification of the three tasks and is the only task where normalization hurts.
The cause remains unexplained by this diagnostic; the task's semantics (the binary `is_grasped`
flag, the arm-must-be-still criterion, or the learning dynamics) are the remaining candidates.
This is a statement about the absence of support, not a refutation of other mechanisms.

---

## P1 — sampler x horizon frontier on the co-trained checkpoint (fresh seeds)

Pre-registered DDIM choice from the round-5 paired tables: no DDIM setting had a paired 95% upper
loss bound <= 10 pp on either task (best `PickCube-v1` 22.0 pp at 50 steps, best
`PullCubeTool-v1` 13.0 pp at 50 steps), so **DDIM-20** for both tasks. Fresh blocks:
`PickCube-v1` 800000-800199, `PullCubeTool-v1` 900000-900199, n = 200 per cell.

### `PickCube-v1` — COMPLETE (12/12 cells)

| schedule | sampler | `success_once` | `success_at_end` | calls/ep | mean horizon | derived s/ep |
|---|---|---|---|---|---|---|
| `fixed:2` | DDPM-100 | 17.5% | 3.5% | 100.0 | 2.00 | 14.50 |
| `fixed:2` | DDPM-20 | 13.0% | 4.5% | 100.0 | 2.00 | 2.87 |
| `fixed:2` | DDIM-20 | 21.0% | 5.0% | 100.0 | 2.00 | 2.85 |
| `fixed:4` | DDPM-100 | 29.0% | 7.0% | 50.0 | 4.00 | 7.25 |
| `fixed:4` | DDPM-20 | 31.0% | 9.0% | 50.0 | 4.00 | 1.44 |
| `fixed:4` | DDIM-20 | 27.0% | 5.5% | 50.0 | 4.00 | 1.43 |
| **`fixed:8`** | **DDPM-100** | **46.5%** | **19.0%** | **25.0** | 8.00 | **3.62** |
| `fixed:8` | DDPM-20 | 50.0% | 13.5% | 25.0 | 8.00 | 0.72 |
| `fixed:8` | DDIM-20 | 42.5% | 11.5% | 25.0 | 8.00 | 0.71 |
| adaptive | DDPM-100 | 34.5% | 8.5% | 60.3 | 3.96 | 8.75 |
| adaptive | DDPM-20 | 38.5% | 8.0% | 63.4 | 3.70 | 1.82 |
| adaptive | DDIM-20 | 37.0% | 6.5% | 61.0 | 3.79 | 1.74 |

**(a) Is the adaptive rule better than the best fixed schedule at DDPM-100?** No — it is
**significantly worse**. Exact McNemar on identical seeds (n = 200):

| comparison | difference | 95% interval | p |
|---|---|---|---|
| adaptive vs `fixed:2` | **+17.0 pp** | [+8.4, +25.3] | 1.7e-05 |
| adaptive vs `fixed:4` | +5.5 pp | [-3.6, +14.5] | 0.200 |
| adaptive vs `fixed:8` (the best fixed on this block) | **-12.0 pp** | [-21.3, -2.4] | **0.000272** |

This **replicates the earlier direction on a brand-new seed block**: on `PickCube-v1` the
adaptive rule beats `fixed:2` clearly, is indistinguishable from `fixed:4`, and is significantly
*worse* than `fixed:8` — the same ordering as round 3 (adaptive 25.5% vs `fixed:8` 36.0%,
p = 0.0008) and round 6's Task F. Note the frontier is *degenerate* here: at DDPM-100 `fixed:8` has
both the highest success **and** the lowest derived policy time (3.62 s vs the adaptive rule's
8.75 s), so on this task there is no success-versus-cost trade-off to exploit — the rule spends
2.4x the policy time to lose 12 points. (The DDPM-20/DDIM-20 cells are cheaper still — 0.72 s and
0.71 s for `fixed:8`, 1.82 s and 1.74 s for adaptive — but the same ordering holds at every
sampler, and the pre-registered comparison is at DDPM-100.)

**(b) Do the cheaper samplers lose success?** No significant change in any of the eight paired
comparisons (all p >= 0.199), with point estimates between -4.5 and +4.0 pp. The largest losses
compatible with the data: `fixed:2` DDPM-20 11.6 pp, `fixed:2` DDIM-20 4.3 pp, `fixed:4` DDPM-20
6.9 pp, `fixed:4` DDIM-20 10.7 pp, `fixed:8` DDPM-20 6.2 pp, `fixed:8` DDIM-20 **13.6 pp**;
adaptive DDPM-20 5.4 pp, adaptive DDIM-20 6.8 pp. Equivalence is not established for any cell.
On the secondary metric, `fixed:8` DDIM-20 is 7.5 pp below DDPM-100 at end (11.5% vs 19.0%).

**(c) Success versus cost:** figure `results/fig_r7_frontier_pick.png/.pdf`.

### `PullCubeTool-v1` — COMPLETE (12/12 cells)

| schedule | sampler | `success_once` | `success_at_end` | calls/ep | mean horizon | derived s/ep |
|---|---|---|---|---|---|---|
| `fixed:2` | DDPM-100 | 8.0% | 8.0% | 300.0 | 2.00 | 43.50 |
| `fixed:2` | DDPM-20 | 9.0% | 9.0% | 300.0 | 2.00 | 8.61 |
| `fixed:2` | DDIM-20 | 5.0% | 5.0% | 300.0 | 2.00 | 8.55 |
| `fixed:4` | DDPM-100 | 9.5% | 9.5% | 150.0 | 4.00 | 21.75 |
| `fixed:4` | DDPM-20 | 9.0% | 9.0% | 150.0 | 4.00 | 4.30 |
| `fixed:4` | DDIM-20 | 11.5% | 11.5% | 150.0 | 4.00 | 4.28 |
| `fixed:8` | DDPM-100 | 10.0% | 10.0% | 75.0 | 8.00 | 10.88 |
| `fixed:8` | DDPM-20 | 15.0% | 15.0% | 75.0 | 8.00 | 2.15 |
| `fixed:8` | DDIM-20 | 7.5% | 7.5% | 75.0 | 8.00 | 2.14 |
| adaptive | DDPM-100 | 8.5% | 8.5% | 148.4 | 5.05 | 21.52 |
| adaptive | DDPM-20 | 14.5% | 14.5% | 150.0 | 5.07 | 4.30 |
| adaptive | DDIM-20 | 9.5% | 9.0% | 150.8 | 5.08 | 4.30 |

**(a) Is the adaptive rule better than the best fixed schedule at DDPM-100?** No — and this
replicates the earlier null. Exact McNemar (n = 200): adaptive vs `fixed:2` **+0.5 pp** (p = 1.0),
vs `fixed:4` **-1.0 pp** (p = 0.851), vs `fixed:8` (the best fixed, 10.0%) **-1.5 pp**
(p = 0.581). With the task's base rate near 10%, n = 200 resolves only about +-4 pp, so the honest
statement is "no advantage, with the interval compatible with losses up to ~7 pp" — the adaptive
rule spends 2x the policy time (21.5 s vs 10.9 s) for no measurable gain.

**(b) Do the cheaper samplers lose success?** No significant change in any of the eight paired
comparisons (n = 200 per cell, all p >= 0.065): `fixed:2` DDPM-20 +1.0 pp (p = 0.839), `fixed:2`
DDIM-20 -3.0 pp (p = 0.263), `fixed:4` DDPM-20 -0.5 pp (p = 1.0), `fixed:4` DDIM-20 +2.0 pp
(p = 0.618), `fixed:8` DDPM-20 +5.0 pp (p = 0.143), `fixed:8` DDIM-20 -2.5 pp (p = 0.473),
adaptive DDPM-20 +6.0 pp (p = 0.065), adaptive DDIM-20 +1.0 pp (p = 0.864); every row is in
`results/r7_tests.csv`. The largest losses compatible with the data are 0.3-8.2 pp;
`fixed:8` DDPM-20 is nominally the best cell (15.0%). Equivalence is not established for any cell.

**(c) Success versus cost:** figure `results/fig_r7_frontier_pull.png/.pdf`. Note `success_once`
and `success_at_end` nearly coincide for this task (gaps of at most 0.5 pp; the cube is either
pulled or not), unlike `PickCube-v1` where the two metrics diverge sharply.

## P2 — variants V2a/V2b/V3 on the other two tasks (3 seeds each)

B0 (4 runs) and V1 (4 runs) per task are reused from brief 6. Variants are one change each on the
same single-task code, data and 30 000 updates; evaluation is `fixed:8`, DDPM-100, n = 200 per
run, on the brief-6 seed blocks so the baselines are comparable. The pre-registered **descriptive
flag** is "suggestive" iff the variant's mean run-level rate lies outside the baseline's min-max
range *and* all its runs are on the same side of the baseline mean; otherwise "within training
noise". No hypothesis test is claimed: 3 vs 4 runs gives C(7,3) = 35 permutations, so the smallest
attainable two-sided p is 2/35 = **0.057** and no variant can reach alpha = 0.05 by construction.
Run values in the tables below are listed **ascending** (not in seed order); e.g. the
`PullCubeTool-v1` B0 runs are seed 0 = 13.5, seed 1 = 15.0, seed 2 = 9.0, seed 3 = 9.0, and the
V1 runs are seed 0 = 20.0, seed 1 = 34.5, seed 2 = 23.5, seed 3 = 32.0. The P5 median-run seed ids
refer to these true seed orders.

### V2b — prediction horizon 32 — COMPLETE (3 runs per task)

| task | arm | runs | mean | flag |
|---|---|---|---|---|
| `PickCube-v1` | B0 (baseline) | 46.0, 57.0, 63.0, 75.5 | 60.4% | — |
| `PickCube-v1` | V1 (reference) | 29.5, 39.5, 42.0, 49.5 | 40.1% | suggestive (negative) |
| `PickCube-v1` | **V2b** | 59.5, 79.0, 83.5 | **74.0%** | **within training noise** |
| `PullCubeTool-v1` | B0 (baseline) | 9.0, 9.0, 13.5, 15.0 | 11.6% | — |
| `PullCubeTool-v1` | V1 (reference) | 20.0, 23.5, 32.0, 34.5 | 27.5% | suggestive (positive) |
| `PullCubeTool-v1` | **V2b** | 17.0, 20.0, 22.0 | **19.7%** | **suggestive** |

- **`PullCubeTool-v1`: suggestive.** V2b's mean (19.7%) is above the baseline's maximum (15.0%)
  and all three runs are above the baseline mean (11.6%) — the flag is met. The permutation
  difference is +8.0 pp (p cannot fall below 0.057 at 3 vs 4 runs, and it does not), so this is a
  flag, not an established improvement. `success_at_end` matches `success_once` for this task.
- **`PickCube-v1`: within training noise.** V2b's mean (74.0%) is 13.6 pp above the baseline's
  (60.4%) and *below* the baseline's best single run (75.5%), so the mean does not lie outside the
  baseline's range; the flag is not met. Worth noting as an observation (not a claim): V2b is the
  only variant on `PickCube-v1` whose every run clears the baseline's worst run (46.0), and two of
  its three runs (79.0, 83.5) are above the baseline mean, with the third (59.5) just below it,
  whereas V1 is clearly harmful there.

### V2a — prediction horizon 8 (execution clamped to 7) — COMPLETE (3 runs per task)

| task | arm | runs | mean | flag |
|---|---|---|---|---|
| `PickCube-v1` | B0 (baseline) | 46.0, 57.0, 63.0, 75.5 | 60.4% | — |
| `PickCube-v1` | V1 (reference) | 29.5, 39.5, 42.0, 49.5 | 40.1% | suggestive (negative) |
| `PickCube-v1` | **V2a** | 17.5, 21.0, 23.0 | **20.5%** | **suggestive (negative)** |
| `PullCubeTool-v1` | B0 (baseline) | 9.0, 9.0, 13.5, 15.0 | 11.6% | — |
| `PullCubeTool-v1` | V1 (reference) | 20.0, 23.5, 32.0, 34.5 | 27.5% | suggestive (positive) |
| `PullCubeTool-v1` | **V2a** | 3.5, 5.5, 7.0 | **5.3%** | **suggestive (negative)** |

A shorter prediction horizon is **harmful on both tasks** — and by the largest margin of any
variant measured here: -39.9 pp on `PickCube-v1` (every run below the baseline's worst) and
-6.3 pp on `PullCubeTool-v1` (every run below the baseline's worst). Because the execution
horizon is clamped to the legal maximum (`pred_horizon - obs_horizon + 1` = 7), this variant also
replans more often: 29 calls per episode at `fixed:8` instead of 25. Both tasks are consistent
with the pre-registered flag on the negative side; the smallest attainable p at 3 vs 4 runs is
0.057, so these remain flags rather than tests.

Taken together with V2b (horizon 32: 74.0% on `PickCube-v1`, "within training noise";
19.7% on `PullCubeTool-v1`, "suggestive"), the prediction-horizon axis on these tasks points one
way: **the 16-step horizon of the baseline is not the limiting factor, but shortening it is
clearly harmful**, while lengthening it is neutral-to-mildly-helpful.

### V3 — encoder width 256 — COMPLETE (3 runs per task)

| task | arm | runs | mean | flag |
|---|---|---|---|---|
| `PickCube-v1` | B0 (baseline) | 46.0, 57.0, 63.0, 75.5 | 60.4% | — |
| `PickCube-v1` | V1 (reference) | 29.5, 39.5, 42.0, 49.5 | 40.1% | suggestive (negative) |
| `PickCube-v1` | **V3** | 44.5, 59.0, 89.5 | **64.3%** | **within training noise** |

V3's mean is 3.9 pp above the baseline mean and *inside* the baseline's range, so the flag is not
met — but the striking feature is its spread: **44.5 to 89.5 pp, a 45-point range across three
training seeds**, the widest seen in this project on any task (round 5's `PushCube-v1` V3 had a
12.5-point spread). A single V3 run reached 89.5% — the best PickCube number recorded anywhere —
while another was below every baseline run. One seed is not evidence, and this is exactly why the
pre-registered unit is the run and why the flag is descriptive.

**`PullCubeTool-v1`: suggestive.** V3's runs are 30.5, 22.5, 26.0 (mean **26.3%**), above the
baseline's maximum (15.0%) and all above the baseline mean (11.6%) — the flag is met, +14.7 pp.
As with V2b on this task, the flag is descriptive: 3 vs 4 runs cannot reach α = 0.05 by
construction (smallest attainable p = 0.057).

### Variant summary (all four variants, both tasks)

| variant | `PickCube-v1` (baseline 60.4%) | flag | `PullCubeTool-v1` (baseline 11.6%) | flag |
|---|---|---|---|---|
| V1 normalization (reference) | 40.1% | suggestive (negative) | 27.5% | suggestive (positive) |
| V2a horizon 8 | 20.5% | **suggestive (negative)** | 5.3% | **suggestive (negative)** |
| V2b horizon 32 | 74.0% | within training noise | 19.7% | **suggestive (positive)** |
| V3 encoder 256 | 64.3% (spread 44.5-89.5) | within training noise | 26.3% | **suggestive (positive)** |

The two tasks answer differently, and the split tracks the baseline's competence: on
`PullCubeTool-v1` — where the baseline is weak (11.6%) — **three of the four variants are
suggestive-positive** (normalization +15.9 pp, the wider encoder +14.7 pp, the longer horizon
+8.0 pp) while the shorter horizon is clearly negative; on `PickCube-v1` — where the baseline is
strong (60.4%) — the longer horizon and the wider encoder both avoid harm (within training noise)
while normalization and the shorter horizon are negative, the latter catastrophically. All four
rows are flags, not tests: with three runs against four the smallest attainable two-sided p is
0.057, and six comparisons per task make a corrected threshold unattainable by construction.

All variant training and evaluation is complete: V2a, V2b and V3 all have three runs per task,
each evaluated at `fixed:8`, DDPM-100, n = 200 per run, on the brief-6 seed blocks. The flags
above are final; no variant result is pending.

## P3 — combination on the other two tasks (reuses brief 6 for the DDPM-100 cells)

### `PickCube-v1` — COMPLETE (8 cells; DDIM-5 evaluated here, DDPM-100 reused from brief 6)

Run-level means (4 runs per cell; ranges in brackets):

| arm | schedule | DDPM-100 | DDIM-5 |
|---|---|---|---|
| B0 | `fixed:8` | 60.38% [46.0-75.5] | **63.13%** [47.0-78.5] |
| B0 | adaptive | 45.63% | 50.13% [44.0-67.0] |
| V1 | `fixed:8` | 40.13% [29.5-49.5] | 41.38% [28.5-48.5] |
| V1 | adaptive | 34.25% | 34.38% [21.5-43.5] |

**Headline 1 (pre-specified):** the combination (V1, adaptive, DDIM-5) = 34.38% versus default DP
(B0, `fixed:8`, DDPM-100) = 60.38%: **-26.0 pp**, exact permutation 4 vs 4 (70 assignments,
smallest attainable two-sided p = 0.0286), **p = 0.028571** — i.e. the "improved" configuration is
significantly *worse* than doing nothing, by the largest margin recorded anywhere in this
project. On this task the two training-time changes each hurt (normalization -20.3 pp, the
adaptive rule -14.75 pp on B0 / -5.88 pp on V1 at `fixed:8`) and they do not cancel.

**Headline 2 (post hoc):** the best cell by run-level mean is not the pre-specified combination
but **plain default DP with DDIM-5** (B0, `fixed:8`, 63.13%), which is **+2.8 pp** above the
DDPM-100 default (post hoc, 4 vs 4, not significant and not pre-registered). In other words the
only change that helps on `PickCube-v1` is the cheap sampler applied to the default schedule, and
it helps by a couple of points while cutting derived policy time from 3.62 s to 0.18 s per
episode.

**DDIM-5 versus DDPM-100 per run:** 0 of 4 runs lose success significantly in each of the four
arm/schedule combinations (exact McNemar), and every cell mean is *higher* at DDIM-5, so on
`PickCube-v1` cheap sampling is cost-free in this block — the opposite of the single-task
`PushCube-v1` picture in brief 6, where one of six runs lost 8.5 pp.

### `PullCubeTool-v1` — COMPLETE (8 cells)

Run-level means (4 runs per cell):

| arm | schedule | DDPM-100 | DDIM-5 |
|---|---|---|---|
| B0 | `fixed:8` | 11.63% [9.0-15.0] | 11.00% [8.0-15.0] |
| B0 | adaptive | 11.50% | 10.25% [8.0-12.0] |
| V1 | `fixed:8` | **27.50%** [20.0-34.5] | 20.00% [10.5-30.0] |
| V1 | adaptive | 19.88% | 14.50% [5.5-21.5] |

**Headline 1 (pre-specified):** the combination (V1, adaptive, DDIM-5) = 14.50% versus default DP
(B0, `fixed:8`, DDPM-100) = 11.63%: **+2.9 pp**, exact permutation 4 vs 4 (70 assignments,
smallest attainable p = 0.0286), **p = 0.514** — not established. The best cell on this task is
**V1 with the default schedule at DDPM-100 (27.50%)**, i.e. normalization alone.

**Headline 2 (post hoc):** the best cell by run-level mean among the eight is **V1 `fixed:8` at
DDPM-100 (27.50%)**, **+15.88 pp** above default DP — the same cell headline 1 identifies (the
pre-registered normalization comparison), recorded in `results/r7_tests.csv` as the post-hoc row
(4 vs 4, smallest attainable p = 0.0286, no p-value claimed). Note that DDIM-5 *lowers* this best
cell to 20.00%.

**DDIM-5 versus DDPM-100 per run:** unlike `PickCube-v1`, cheap sampling **costs success here for
the normalized models**: exact McNemar shows **2 of 4 runs losing significantly** in the
V1 `fixed:8` comparison (27.50% -> 20.00% at the cell mean) and **2 of 4** in V1 adaptive
(19.88% -> 14.50%). For the B0 cells, 0 of 4 runs lose significantly. So the round-5 statement
"cutting sampling steps is free" holds for the co-trained checkpoint and for `PickCube-v1` here,
but is **not** true for observation-normalized `PullCubeTool-v1` models.


## P4 — co-training versus single-task, multi-run

**Design (pre-registered).** Brief 6 compared one teammate-trained CUDA checkpoint with several
single-task runs, which cannot separate co-training from device, run-to-run luck, and samples per
update. Round 7 therefore trains our own three-task models with the same trainer
(`cotrain.py`, default three-task mode), the same three-task config (200-trajectory subsets per
task, verified identical to the checkpoint's embedded entries) and 30 000 updates on MPS, seeds
0-3 (seeds 4-5 only if time permits; stopping is by time, never by result). Each run is evaluated
on **all three tasks** at `fixed:8`, DDPM-100, on the same seed blocks as the single-task runs
(`PushCube-v1` 500000-500199, `PickCube-v1` 600000-600199, `PullCubeTool-v1` 700000-700199,
n = 200 per task).

**Pre-registered comparison, per task:** our co-trained runs vs the single-task B0 runs
(6 on `PushCube-v1`, 4 on the others), exact permutation on run-level rates with the permutation
count and smallest attainable p stated (4 vs 6 = 210 -> 0.0095; 4 vs 4 = 70 -> 0.0286), the
difference with a bootstrap interval, every run's rate, and sign consistency across the three
tasks; plus the teammate's CUDA checkpoint as one descriptive number.

**Confounds that remain (listed so nothing is over-claimed).**

| aspect | our co-trained runs (round 7) | single-task B0 runs | teammate's checkpoint |
|---|---|---|---|
| trainer | `cotrain.py`, three-task mode | `cotrain.py`, one task | `cotrain.py`, three-task mode |
| samples per update | 96 (3 x 32) | 32 | 96 |
| gradient per update | mean of three per-task losses (each scaled 1/3) | the single task's loss | same as ours |
| demonstration subset | identical 200 trajectories per task | identical (verified) | identical (verified) |
| architecture | shared trunk + per-task input layer | same trunk, one input layer | same as ours |
| device | MPS | MPS | CUDA (teammate) |
| number of runs | 4 (seeds 0-3) | 4-6 | 1 |
| in-training evaluation | final only, 20 episodes x 1 env | final only, 20 x 1 | every 5 000 updates, 8 x 4 |

**CUT BY THE USER (amendment 7a, 2026-10-06 17:25): the co-training comparison is out of scope
for the final report and is not analysed.** This is a user decision, not a result-driven cut. What
ran before the decision, kept for the record but not analysed here: two of the four planned
three-task runs (seeds 0 and 1; seed 0 fully trained and evaluated on all three tasks, seed 1
trained only after the cut and left unevaluated), plus partial evaluations of a few cells. No
run-level co-training test is computed, and the parity table's co-training row is labelled
"not analysed (out of report scope)". The related P4b batch-96 control was never started.

## P5 — failure analysis of the trained models

**Design.** For `PickCube-v1` and `PullCubeTool-v1`, the run with the median `success_once` in
each arm is replayed with decision logging (up to 32 failed episodes each), plus the teammate's
co-trained checkpoint for comparison. Classification thresholds are unchanged from round 3
(`REACH` 0.05, `LIFT` 0.05, `GOAL_THRESH` 0.025; `PullCubeTool-v1` uses the harness's own phase
labels with `GRASP_EPS` 0.06 and `NEAR_D` 0.30). Median-run selection (lower median on ties):
`PickCube-v1` B0 seed 2 (57.0%), V1 seed 3 (39.5%); `PullCubeTool-v1` B0 seed 2 (9.0%), V1 seed 2
(23.5%). For `PickCube-v1` the share of episodes with `success_once` but not `success_at_end` is
reported per arm.

**RESULT (6 cells, 32 failed episodes classified per cell).**

| task | arm (median run) | `success_once` | `success_at_end` | `once` but not `at_end` | failure phases (top) |
|---|---|---|---|---|---|
| `PickCube-v1` | B0 seed 2 | 57.0% | 44.5% | 25 of 114 = **21.9%** | 30 lifted-not-placed, 2 grasped-not-lifted |
| `PickCube-v1` | V1 seed 3 | 39.5% | 17.0% | 45 of 79 = **57.0%** | 25 lifted-not-placed, 7 grasped-not-lifted |
| `PickCube-v1` | co-trained | 42.5% | 11.5% | 62 of 85 = **72.9%** | 24 lifted-not-placed, 6 grasped-not-lifted, 2 cube-at-goal-not-registered |
| `PullCubeTool-v1` | B0 seed 2 | 9.0% | 9.0% | 0 | 20 reached-cube-but-failed, 7 grasped-not-pulled, 5 never-grasped |
| `PullCubeTool-v1` | V1 seed 2 | 23.5% | 23.0% | 1 | 17 reached-cube-but-failed, 15 grasped-not-pulled, **0 never-grasped** |
| `PullCubeTool-v1` | co-trained | 7.5% | 7.5% | 0 | 18 reached-cube-but-failed, 9 grasped-not-pulled, 5 never-grasped |

**What the normalization changes.** On `PickCube-v1` the *failure* profile is broadly similar
(both arms fail mostly after lifting, before placing), but the **success** profile differs
sharply: only 21.9% of the baseline's successes are "not still at the end", versus **57.0%** for
V1 and 72.9% for the co-trained checkpoint. **Descriptive observation, not a finding:** in this
single median run per arm, a larger share of V1's (and the co-trained checkpoint's) successes were
"not still at the end" than the baseline's. That suggests a hypothesis — normalization may make the
policy less likely to be still — which this exercise does not test; it is one run per arm with 32
classified failures, it is not a test of the stillness mechanism, and it should not be read as
confirming round 4's stillness experiment. On `PullCubeTool-v1` the effect is different and benign:
V1's failures shift later in the task and it **never fails to grasp the tool** (0 of 32 failures,
versus 5 of 32 for B0 and 5 of 32 for the co-trained checkpoint), which matches its much higher
success rate. The distributional comparison is descriptive: one median run per arm, 32 failures
each, and no significance test is claimed.

## P6 — does the adaptive gain depend on policy quality?

**Not testable without retraining.** All six B0 `PushCube-v1` runs saved only final checkpoints:
`checkpoint_30000.pt`, `checkpoint_final.pt` and the two `best_eval` files all carry
`completed_updates = 30000` and, for B0s0, identical EMA weight hashes (verified), because the
runs used `--save-freq 30000` with 30 000 total updates. No 5 000 / 10 000 / 20 000 intermediates
exist, and this brief does not retrain. Skipped as pre-registered.

## P7 — compact controls

**Dropped, as pre-registered (lowest priority, dropped first).** On `PickCube-v1` the adaptive
rule is significantly *worse* than the best fixed schedule (P1a: -12.0 pp, p = 0.00027) and on
`PullCubeTool-v1` it is not better than any fixed schedule (P1a: -1.5 pp, p = 0.581), so a control
experiment for a *missing* effect was of little value; the CPU budget was used by the tasks with
pre-registered questions. Nothing was run for P7.

## P8 — consistency pass and synthesis

### Consistency pass on the round-6 report

The corrections to that report are applied in place in this release and listed in its
"Corrections applied when this file was packaged" section: the duplicated result block at the
top of the file (a slice bug in an editing step), the co-training table and its §G.1 row still
comparing a single baseline seed, §D's 4-run adaptive rows and its "still being evaluated" note,
§C's "PullCubeTool pending" heading and
its +15.82 pp / 27.44% figures (now +15.88 pp / 27.50%), §F (verified complete), and a defect in
the shipped `r6_runlevel.py` that made DDIM comparisons compare cells with themselves.

### Parity across the three tasks

Each cell says **supported**, **not established** or **contradicted**, using the pre-registered
rules of this brief and brief 6. "Best fixed vs adaptive" refers to the co-trained checkpoint
(the only policy where the adaptive rule has ever been positive).

| experiment | `PushCube-v1` | `PickCube-v1` | `PullCubeTool-v1` |
|---|---|---|---|
| best fixed vs adaptive (co-trained checkpoint) | **supported** (round 4: +14.5 pp vs `fixed:8`, p = 0.000002; round 5: +11.0 pp, p = 0.00068) | **contradicted** (fresh block: -12.0 pp vs `fixed:8`, p = 0.00027) | **not established** (fresh block: -1.5 pp, p = 0.581) |
| sampler x horizon frontier | supported (round 5/6) | **complete**: `fixed:8` dominates on both success and cost; DDPM-20/DDIM-20 within +-4.5 pp, largest compatible losses 4.3-13.6 pp | **complete**: no sampler loss except DDIM-5 for normalized models later; `fixed:8` best |
| observation normalization (V1 vs B0) | **supported** (+10.83 pp, p = 0.0043, 6v6) | **contradicted** (-20.25 pp, p = 0.057, and the "combination" is -26.0 pp at p = 0.0286) | **supported** (+15.88 pp, p = 0.0286; misses the two-task 0.025 threshold) |
| V2a shorter prediction horizon (8) | not established (round 5) | **not established**; flag suggestive (negative), -39.9 pp, every run below the baseline's worst | **not established**; flag suggestive (negative), -6.3 pp, every run below the baseline's worst |
| V2b longer prediction horizon (32) | not established (round 5, suggestive at 2 seeds, overlap at 3) | **within training noise** (flag not met; +13.6 pp, mean inside the baseline's range) | **not established**; flag suggestive (positive), +8.0 pp |
| V3 wider encoder (256) | not established (round 5; widest seed spread, 12.5 pp) | **within training noise** (flag not met; +3.9 pp, seed spread 44.5-89.5) | **not established**; flag suggestive (positive), +14.7 pp |
| adaptive on our trained models | **contradicted** (brief 6: -1.83 pp, p = 0.56) | **contradicted** (brief 6: -14.75 pp B0 / -5.88 pp V1) | **contradicted** (brief 6: -0.13 pp B0 / -7.63 pp V1) |
| combination incl. DDIM-5 on trained models | **supported** (brief 6: +9.75 pp vs default DP at 5.8x less policy time, driven by normalization) | **contradicted** (pre-specified combination -26.0 pp vs default DP, p = 0.0286; best cell is plain default DP + DDIM-5, +2.8 pp post hoc) | **not established** (pre-specified combination +2.9 pp, p = 0.514; best cell is V1 with the default schedule at DDPM-100; DDIM-5 costs 2 of 4 normalized runs) |
| co-training vs single-task | **not analysed (out of report scope — cut by the user, amendment 7a)** | same | same |

The three variant rows are 3-vs-4-run descriptive flags: the smallest attainable two-sided p is
0.057, so none can be a test at α = 0.05. "Not established" is the verdict for all six variant
cells; the flag only names the direction.

### Packaged artefacts and release verification (final)

- `results/FINAL_RESULTS.csv`: **273 rows** (up from 190 in round 6), covering every round-4 to
  round-7 cell, including the round-7 frontier, combination, variant and failure blocks. The
  `r7_cotrain_runs` rows are present for provenance even though P4 was cut by the user.
- `results/MANIFEST.md`: **5 392 raw log files / 53 942 episodes** (was 3 726 / 37 282), each with
  episode count, seed range and SHA-256.
- Round-7 raw tree: `r7_frontier_{pick,pull}` 12 analysed cells / 2 400 episodes each (each root
  also keeps a quarantined `_superseded_orphan_fixed2_*` run of 30 episodes that no analysis
  reads), `r7_combo_{pick,pull}` 16 cells / 3 200 episodes each, `r7_variants_{pick,pull}` 9 cells
  / 1 800 episodes each, `r7_failure` 6 cells / 1 200 episodes, `r7_cotrain_runs` 3 cells /
  600 episodes (P4, cut by the user; kept for provenance only) — 16 000 analysed round-7
  episodes plus the 600 co-training ones, and the 60 quarantined ones (16 660 raw per-episode
  files, all counted by the manifest).
- `scripts/r5_integrity.py` over the whole raw tree (round-5 to round-7 roots): **4 196 chunk
  files, 0 problems** (unparsable lines, duplicate seeds, wrong episode counts).
- `scripts/verify_results.py --raw-root <full raw tree>`: **3 526 values checked, 0 mismatches,
  0 problems**.
- Release tests: **21/21** pass (`tests/test_core.py` 15, `tests/test_runlevel.py` 6, including
  hand-computed permutation cases, the raw-feature rule, and the new `descriptive_flag` cases
  plus the permutation-count identities).
- New release artefacts (1.3.0): `src/adaptive_replan/runlevel.py` gained `descriptive_flag`;
  `scripts/r7_runlevel.py`, `r7_tests.py`, `r7_figures.py`, `r7_parity_figure.py`,
  `r7_p0_dimstats.py`, `r7_failure_classify.py`; seven `configs/round7_*.json`; the figures
  `fig_r7_frontier_{pick,pull}`, `fig_r7_variants_{pick,pull,pushcube}`,
  `fig_r7_combination_{pick,pull}` and `fig_r7_parity` (PNG + PDF). `pack_final_results.py` now
  discovers the `r6_*` **and** `r7_*` raw roots, and `r5_integrity.py` covers the round-7 roots.
- Fixed while preparing this round (all shipped): the DDIM comparison in `r6_runlevel.py`
  compared cells with themselves (see the corrections section of `REPORT_ROUND_6.md`, item 8), and `r7_tests.py`
  crashed on empty confidence bounds, which had silently left a stale results CSV.
- Fixed after the external reviews (all shipped): `r7_runlevel.py`'s per-call-time lookup (the
  `co_tm` arm and the DDIM-20 entry) and `pack_final_results.py`'s missing merge of the round-5
  frontier's reused sweep half — see revision note (2) above; the affected table, figures and
  `FINAL_RESULTS.csv` rows were regenerated from the raw logs and re-verified.
- Safety scan: the only matches for the searched patterns are the two documents that list the
  patterns (`docs/RELEASE_CHECK.md`, `docs/REPORT_ROUND_5.md`); no checkpoints, demonstration
  data or teammate-owned files are included; nothing is pushed and no licence is chosen.

### What was not finished (or was cut)

- **P4a/P4b (co-training): cut by the user** (amendment 7a). Two three-task runs were trained
  before the cut (seed 0 evaluated on all three tasks, seed 1 trained but unevaluated), one was
  stopped mid-training at ~6 000 updates; nothing was deleted and no co-training test is
  reported. The parity table keeps the row, labelled *not analysed (out of report scope)*.
- **P6 (does the adaptive gain depend on policy quality): not testable** — the six B0
  `PushCube-v1` runs saved only final checkpoints (verified by weight hashes).
- **P7 (controls on the new tasks): dropped** as pre-registered (lowest priority), because on both
  tasks the adaptive rule showed no advantage to control for.
- **P3's `PullCubeTool-v1` sampling cells** are run-level descriptive only (4 vs 4, smallest
  attainable p = 0.0286); the two headline comparisons are labelled pre-specified and post hoc
  as required.
- No new timing benchmark was taken; policy seconds per episode are derived from round-5 clean
  per-call measurements throughout.
