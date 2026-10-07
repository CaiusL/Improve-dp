# REPORT_ROUND_6.md — Round 6: run-level tests, adaptive transfer, combination, co-training

Formerly `REPORT_ADDENDUM_5.md`. The corrections that used to live in a separate
`REPORT_ADDENDUM_5_FIXES.md` are applied in the body below; what changed is listed in
"Corrections applied when this file was packaged" at the end.

Brief: *make the training-time claim testable, add adaptive replanning to the trained models,
and compare co-training.* Scope: Panda only; `PushCube-v1`, `PickCube-v1`, `PullCubeTool-v1`.
Nothing in an earlier report is edited by this file; the user merges.

This addendum is written incrementally; each section states its status. Every pre-registered
rule was fixed in `NOTES.md` at 2026-10-06 03:54, before the first round-6 run.

---

## 0. Method: the training run is the unit of analysis

Round 5 compared training variants with 3-4 runs per arm and a rule that demanded the variant
beat the baseline's seed spread — a threshold that turned out to be unreachable (101.0%). This
round replaces that with run-level tests and states them exactly:

- **Unit of analysis:** one success rate per training run (200 episodes each). Episodes are
  never pooled across runs to claim significance; within-run paired comparisons (adaptive vs
  `fixed:8` on identical evaluation seeds) use exact McNemar and are reported per run.
- **Exact two-sided permutation test** on two independent groups of run-level rates: all
  `C(n1+n2, n1)` label assignments are enumerated; `p = #{|diff| >= |diff_observed|}/N`. For
  6 vs 6 runs that is 924 assignments; for 4 vs 4 it is 70, and the smallest attainable
  two-sided p is 2/70 = 0.0286. Both the test and this smallest-p property are unit-tested
  against hand-computed cases (`tests/test_runlevel.py`).
- **Exact sign-flip test** for paired run-level differences (adaptive minus `fixed:8` within an
  arm): all `2**n` sign patterns; 6 runs → 64 patterns.
- **Difference in means with a 95% percentile bootstrap interval**, 20 000 resamples at fixed
  seed `20261006` (method stated wherever reported).
- **Derived policy seconds per episode** = mean policy calls × the round-5 clean ms/call
  (DDPM-100 145.0 ms, DDPM-100 with observation normalization 142.8 ms, DDIM-5 7.2 ms). No new
  timing benchmark was taken in this round; simulation time is excluded, so these numbers are
  labelled *derived* everywhere.
- **Launch guard:** every training run and every rollout phase takes an atomic lock
  (`mkdir`-based, owner PID + timestamp, stale-break only if the owner is dead *and* the age
  exceeds the phase budget); nothing can be started twice. `r5_integrity.py` is run against
  every new raw tree before analysis.
- **Training:** MPS, one run at a time, identical settings to the round-5 runs; a crashed run
  is recorded and retried once from scratch with the same seed, and no run is discarded for a
  bad result. Evaluation: CPU, `OMP_NUM_THREADS=1` per worker, at most 16 workers (18 logical
  cores = 6P+12E).

---

## A. Task A — observation normalization on `PushCube-v1`, 6 runs per arm

**Pre-registered.** V1 (observations standardized by the training-demo mean/std) vs B0 (no
normalization) on 6 vs 6 run-level success rates, exact permutation test (924), α = 0.05;
decision "supported as an improvement" iff p < 0.05 **and** the mean difference is positive;
otherwise "not established". Also reported: the difference in means with a 95% bootstrap
interval, every run's rate, and each arm's seed-to-seed sd (and whether V1 reduces it).

Runs: B0 seeds 0-2 and V1 seeds 0-3 are round-5 runs; B0 seeds 3-5 and V1 seeds 4-5 are trained
here with the identical round-5 command (only the run directory differs). Evaluation: final EMA
checkpoint, `fixed:8`, DDPM-100, seeds 500000-500199, n = 200 per run.

**RESULT (complete).** Run-level success rates, `fixed:8`, DDPM-100, n = 200 per run:

| arm | runs (seed 0-5) | mean | sd | min | max | range |
|---|---|---|---|---|---|---|
| B0 | 84.0, 92.5, 89.5, 85.0, 81.0, 79.5 | 85.25% | 4.97 | 79.5 | 92.5 | 13.0 pp |
| **V1** | 98.5, 97.5, 99.5, 91.5, 94.0, 95.5 | **96.08%** | **3.01** | 91.5 | 99.5 | 8.0 pp |

Exact permutation test (6 vs 6 runs, 924 assignments): **difference in means +10.83 pp**
(95% bootstrap interval [+6.50, +14.92]), **p = 0.004329**. Since p < 0.05 and the mean
difference is positive, **observation normalization is supported as an improvement** by the
pre-registered rule. V1 also *reduces* seed-to-seed variability (sd 4.97 → 3.01, range
13.0 → 8.0 pp), which round 5 could not establish with 3-4 runs.

This supersedes round 5's verdict on the same change. Round 5 measured +8.1 pp with 3-4 runs
and could not certify it because its rule demanded the variant beat the better baseline seed by
more than the baseline's own seed spread (a 101.0% threshold); under the run-level test the
same effect is significant at n = 6 + 6. Round 5's number and round 6's are consistent in
magnitude (the difference in the point estimate reflects the three extra runs, two of which
(B0s4 81.0, B0s5 79.5) fall below the round-5 baseline mean.

---

## B. Task B — adaptive replanning on the trained `PushCube-v1` models (transfer test)

**Pre-registered.** The round-4 primary rule exactly, with no retuning: short horizon 2 when
`d_tcp < 0.08` **or** `d_goal < 0.10`, long 8 otherwise. Its thresholds were tuned on the
teammate's co-trained checkpoint, so this is a **transfer test** and is reported as such
whatever the outcome. All 12 runs (6 B0 + 6 V1) are evaluated with the adaptive rule at
DDPM-100 on seeds 500000-500199; the round-5/round-6 `fixed:8` logs of the same runs serve as
the paired baseline.
- **B1** (within B0) and **B2** (within V1): per-run paired differences (adaptive − `fixed:8`),
  exact sign-flip test (2^6 = 64 patterns), plus per-run McNemar p-values.
- **B3** (stacking): V1-adaptive run-level rates vs B0-`fixed:8` rates (exact permutation, 924),
  the 2×2 table of run means, and whether the adaptive gain differs between the arms (exact
  permutation on the two per-run difference sets).

**Raw-feature requirement (completed).** The rule must read the *raw* environment observation,
not the normalized vector that V1 feeds to the model. Verified on two real episodes of a V1
checkpoint (one `fixed:2`, one adaptive; 191 decisions in total): the logged `d_tcp`/`d_goal`
equal, to 1e-3, the values recomputed offline from the logged raw 35-dimensional observation
rows using the release package's independent feature implementation; the horizon re-derived
from those raw features matches the logged horizon in every decision; and in all 191 decisions
the normalized-derived values differ from the raw ones, so the check is not vacuous. Full
record: `results/r6_rawfeat_check.json`; the corresponding unit test ships in the release
(`tests/test_runlevel.py::test_pushcube_rule_uses_raw_observations_only`).
**Verdict: PASSED (0 feature mismatches, 0 rule mismatches, 191/191 non-vacuous).**

**RESULT (complete, 12/12 runs).** Per-run adaptive-minus-`fixed:8` differences at DDPM-100
(n = 200 per run, exact McNemar within each run) and the pre-registered run-level tests:

| arm | per-run differences (seeds 0-5) | mean | sign-flip p | runs significantly worse | runs significantly better |
|---|---|---|---|---|---|
| B0 | +5.0, -11.0, -6.5, -5.5, +3.5, +3.5 pp | **-1.83 pp** [-6.17, +2.42] | **0.5625** | B0s1 (p = 0.00031), B0s2 (p = 0.047) | none |
| V1 | -1.5, -5.0, -1.5, -1.0, +3.0, +0.5 pp | **-0.92 pp** [-4.08, +2.17] | **0.5000** | V1s1 (p = 0.031) | none |

- **B1 (within B0):** adaptive is **not** better than `fixed:8` — mean difference -1.83 pp,
  p = 0.5625, with two of six runs *significantly worse*. The round-5 result (+11.0 pp,
  p = 0.00068) does not reproduce on models trained by our single-task pipeline.
- **B2 (within V1):** the same, -0.92 pp, p = 0.5000, one run significantly worse.
- **B3 (stacking):** V1-adaptive vs B0-`fixed:8` (the default DP baseline) = **+9.92 pp**
  [+5.50, +14.00], **p = 0.006494** — significant, but that gain is the normalization effect,
  not the rule's: the interaction test (does the adaptive gain differ between arms?) gives
  +0.92 pp [-4.25, +6.25], **p = 0.772727**, i.e. no detectable difference, and both arms'
  adaptive gains are ≈ 0.
- Cost: the adaptive rule uses 86.7-89.8 policy calls per episode versus 25.0 for `fixed:8`
  (derived policy time 12.5 s versus 3.6 s at DDPM-100) — so on these models it pays 3.5× the
  inference cost for no success gain, and in two of twelve runs a significant loss.

**Conclusion (transfer test):** the adaptive execution-horizon rule found in round 4 is
**specific to the teammate's co-trained checkpoint**. It is not a general property of this DP
policy class, and on the retrained single-task models it is neutral-to-harmful. This is a
negative result for the project's original headline claim and is reported as such.
---

## D. Task D — the combination on `PushCube-v1` (normalization × adaptive × sampling)

**Pre-registered.** All 12 runs are evaluated with DDIM-5 (deterministic sampler rebuilt from
the checkpoint's own trained betas; epsilon prediction, `eta = 0`, `clip_sample = True`) under
both `fixed:8` and the adaptive rule, on seeds 500000-500199. Together with the DDPM-100 cells
this gives eight cells: {B0, V1} × {`fixed:8`, adaptive} × {DDPM-100, DDIM-5}.

- **Headline:** default DP (B0, `fixed:8`, DDPM-100) vs the best combination (V1, adaptive,
  DDIM-5): run-level success difference by exact permutation (6 vs 6, 924) and the derived
  policy-time ratio.
- **Also:** whether DDIM-5 costs success on these single-task models — per-run paired McNemar
  against the same arm and schedule at DDPM-100 (`results/r6_runlevel_mcnemar.csv`).

**RESULT (complete, 24/24 cells).** Run-level means over all available runs. All 24 combination
cells and all 12 adaptive cells are complete; the headline test below uses the complete 6-vs-6
cells.

| arm | schedule | sampler | runs | mean | min | max | calls | derived s/ep |
|---|---|---|---|---|---|---|---|---|
| B0 | `fixed:8` | DDPM-100 | 6 | 85.2% | 79.5 | 92.5 | 25.0 | 3.62 |
| B0 | `fixed:8` | DDIM-5 | 6 | 86.3% | 81.0 | 96.5 | 25.0 | 0.18 |
| B0 | adaptive | DDPM-100 | 6 | 83.42% | 79.5 | 89.0 | 89.0 | 12.90 |
| B0 | adaptive | DDIM-5 | 6 | 80.9% | 77.5 | 84.5 | 89.2 | 0.64 |
| V1 | `fixed:8` | DDPM-100 | 6 | 96.1% | 91.5 | 99.5 | 25.0 | 3.57 |
| V1 | `fixed:8` | DDIM-5 | 6 | **96.8%** | 92.5 | 100.0 | 25.0 | 0.18 |
| V1 | adaptive | DDPM-100 | 6 | 95.17% | 90.5 | 98.0 | 87.5 | 12.50 |
| V1 | adaptive | DDIM-5 | 6 | 95.0% | 91.5 | 99.5 | 86.7 | 0.62 |

**Headline (pre-registered, complete):** default DP (B0, `fixed:8`, DDPM-100) = 85.2% at
3.62 s/episode versus the best combination (V1, adaptive, DDIM-5) = 95.0% at 0.62 s/episode:
**+9.75 pp** (95% bootstrap [+5.50, +13.83]), exact permutation **p = 0.006494**, and
**5.8× less policy time**. The improvement is driven almost entirely by observation
normalization, not by the adaptive rule (see Task B); DDIM-5 supplies the 5.8× at the same
success level.

**Does DDIM-5 cost success on these single-task models?** Per-run paired McNemar against the
same arm and schedule at DDPM-100 (`results/r6_runlevel_mcnemar.csv`), `fixed:8` cells:
B0: +1.0 (p = 0.87), +4.0 (0.096), **-8.5 (0.0137)**, +2.5 (0.52), +5.0 (0.17), +2.5 (0.53);
V1: +1.5 (0.25), +1.0 (0.73), -2.0 (0.22), +1.0 (0.84), +1.5 (0.58), +1.5 (0.51). So 11 of 12
runs are within ±5 pp and **1 of 12 loses success significantly** (B0s2), while the cell means
are +1.1 pp (B0: 86.3 vs 85.2) and +0.8 pp (V1: 96.8 vs 96.1). On average it is cost-free, but
it is **not uniformly cost-free per run**. Round 5's stronger
"no detectable loss" statement, made on the co-trained checkpoint, therefore does not transfer
unchanged to the retrained models.

---

## E. Task E — co-training versus single-task (descriptive)

**Pre-registered.** Evaluate the teammate's co-trained checkpoint `panda3_shared_a0` (EMA,
task ids 0,1,2) with `fixed:8`, DDPM-100 on `PushCube-v1` 500000-500199, `PickCube-v1`
600000-600199 and `PullCubeTool-v1` 700000-700199, and compare with the single-task B0
run-level rates per task. **Descriptive only** — one co-trained checkpoint versus several
single-task runs means no significance claim is possible for the co-trained side.

**Confounds established before running (and one correction).** Comparing the checkpoint's
embedded config, our configs and the trainer source:

| aspect | co-trained (`panda3_shared_a0`) | our single-task runs |
|---|---|---|
| trainer script | `cotrain.py` | **`cotrain.py`** (same file, invoked with `--tasks <one>`) — the brief's "train.py vs cotrain.py" confound does **not** apply |
| per-task batch / iterations | 32 per task × 3 tasks per update, 30 000 updates | 32, 30 000 updates |
| gradient per update | mean of the three task losses (each scaled by 1/3) | the single task's loss |
| samples seen per update | 96 (3 × 32) | 32 |
| demonstration subset | 200 trajectories per task | **identical 200 trajectories per task** (verified: 200/200 overlap, all three tasks) |
| architecture | shared trunk + per-task input layers (5.19 M params) | same trunk, one input layer (5.14 M) |
| device | CUDA (teammate) | MPS |
| runs | 1 (seed 0) | 6 per arm (seeds 0-5) |
| in-training evaluation | every 5 000 updates, 8 episodes × 4 envs | final only, 20 episodes × 1 env (does not affect final weights) |

**RESULT (complete).** Co-trained `panda3_shared_a0`, `fixed:8`, DDPM-100, n = 200 per task:

| task | co-trained | single-task B0 (mean) | B0 range |
|---|---|---|---|
| `PushCube-v1` | 77.0% | 85.25% (6 runs) | 79.5-92.5% |
| `PickCube-v1` | 42.5% | 60.38% (4 runs) | 46.0-75.5% |
| `PullCubeTool-v1` | 7.5% | 11.62% (4 runs) | 9.0-15.0% |

Sanity against earlier rounds on different seed blocks: the co-trained checkpoint scored 76.0%
on `PushCube-v1` `fixed:8` in round 5 (seeds 400000-400199) and 36.0% on `PickCube-v1` in
round 3 (seeds 300000-300199); 77.0% and 42.5% here are consistent with those.
**Descriptive only**: one co-trained checkpoint cannot be compared for significance against a
distribution, so no claim is made that co-training causes the difference; the confounds are
listed in the table above (gradient averaging over three tasks, 3× samples per update, device,
one run versus six, and the extra per-task input layers), with the demonstration subset
*identical* between the two setups and the trainer script the same file.

---

## C. Task C — observation normalization on the other two tasks (generality)

**Pre-registered.** For `PickCube-v1` and `PullCubeTool-v1`: single-task B0 and V1, 4 runs each
(training seeds 0-3), 30 000 updates, same code and hyperparameters as the `PushCube-v1` runs,
with configs derived from the teammate's three-task config (same 200-trajectory subsets,
`subset_rng_seed` preserved, demo paths repointed). Evaluation `fixed:8`, DDPM-100, n = 200 per
run, on fresh per-task seed blocks (`PickCube-v1` 600000-600199, `PullCubeTool-v1`
700000-700199). Primary per task: V1 vs B0 on 4 vs 4 run-level rates, exact permutation test
(70 permutations, smallest attainable two-sided p = 2/70 = 0.029); "supported" iff p < 0.05 and
the mean difference is positive; effect size reported either way; because two tasks are tested,
whether each survives α = 0.025 is also stated. `PickCube-v1` reports `success_at_end` alongside
`success_once`.

**Prerequisite verification (completed before any training run).** Each demo file's observation
dimension and control mode were checked against what the training code and environment expect:
`PickCube-v1` obs_dim 42, `PullCubeTool-v1` obs_dim 39, both with action_dim 7,
`control_mode = pd_ee_delta_pose`, and environment horizons 200 and 600 respectively — the
per-task dimensions come from the teammate's own `audit.task_spec` and are handled by the shared
model's per-task input layers. Two 2-iteration smoke runs produced checkpoints whose embedded
entries record exactly these dims and horizons.

**Cut order (pre-registered):** reduce `PullCubeTool-v1` to 3+3 runs (descriptive only, stated
as such), then drop `PullCubeTool-v1`; keep `PickCube-v1` at 4+4 if at all possible.

**RESULT (complete, 4 + 4 runs per task).** `fixed:8`, DDPM-100, n = 200 per run.

`PickCube-v1`:

| arm | runs (seeds 0-3) | mean | sd | range | mean `success_at_end` |
|---|---|---|---|---|---|
| B0 | 75.5, 63.0, 57.0, 46.0 | 60.38% | 12.30 | 29.5 pp | 40.13% |
| V1 | 42.0, 49.5, 29.5, 39.5 | 40.12% | 8.26 | 20.0 pp | 19.25% |

Exact permutation test (4 vs 4, 70 assignments): **difference -20.25 pp**
(95% bootstrap [-32.87, -7.88]), **p = 0.057143** (4/70). The pre-registered rule requires
p < 0.05 *and* a positive difference, so normalization is **not supported on `PickCube-v1`** —
and the direction is strongly *harmful*: a 20-point drop, with the bootstrap interval excluding
zero and `success_at_end` showing the same drop (40.1% -> 19.3%). It also misses the two-task
α = 0.025 threshold. V1 again reduces the spread (sd 8.26 vs 12.30), but around a much lower
mean.

`PullCubeTool-v1`:

| arm | runs (seeds 0-3) | mean | sd | range | mean `success_at_end` |
|---|---|---|---|---|---|
| B0 | 13.5, 15.0, 9.0, 9.0 | 11.62% | 3.09 | 6.0 pp | 11.62% |
| V1 | 20.0, 34.5, 23.5, 32.0 | **27.50%** | 6.87 | 14.5 pp | 27.38% |

Exact permutation test (4 vs 4, 70 assignments): **difference +15.88 pp**
(95% bootstrap [+9.63, +22.13]), **p = 0.028571** (2/70, the smallest attainable two-sided p —
the two arms separate completely). Normalization **is supported on `PullCubeTool-v1`** at
α = 0.05 and more than doubles the task's success rate (11.6% -> 27.5%); it does **not** survive
the two-task α = 0.025 threshold (0.0286 > 0.025), which is stated wherever this result is
quoted. `success_at_end` follows `success_once` almost exactly (27.38%).

**This is a task-dependent reversal of the round-5 conclusion.** Round 5 found normalization to
be the one promising training-time change (on `PushCube-v1` only); round 6 confirms it on
`PushCube-v1` (+10.83 pp, p = 0.0043) and finds it *harmful* on `PickCube-v1`. The honest
statement is per-task: normalization helps `PushCube-v1`, hurts `PickCube-v1`, and on `PullCubeTool-v1` it
*doubles* success, +15.88 pp at p = 0.0286 (which does not survive the two-task
α = 0.025). Any claim that it is a general improvement to DP is
not supported by this evidence.

---

## F. Task F (optional) — adaptive replanning on the other tasks' trained models

**Pre-registered (04:16).** Existing rules, no retuning: `PickCube-v1`
`adaptive_push:d_tcp=0,d_goal=0.05,short=2,long=8` (the round-3 test block's best adaptive
variant) and `PullCubeTool-v1` `adaptive_dist:d=0.30`. The B0 and V1 runs from Task C are
evaluated on the same seeds and compared per run against that run's `fixed:8` logs. Purpose:
do the round-3 negative result on `PickCube-v1` and the round-1 null result on
`PullCubeTool-v1` persist on single-task models?

**RESULT (complete, 16 cells).** Sign-flip exact tests on the per-run differences:

| task | arm | runs | mean difference (adaptive - `fixed:8`) | 95% bootstrap | sign-flip p |
|---|---|---|---|---|---|
| `PickCube-v1` | B0 | 4 | **-14.75 pp** | [-29.38, +1.75] | 0.125 |
| `PickCube-v1` | V1 | 4 | **-5.88 pp** | [-15.50, +3.62] | 0.125 |
| `PullCubeTool-v1` | B0 | 4 | -0.13 pp | [-4.00, +4.25] | 1.000 |
| `PullCubeTool-v1` | V1 | 4 | **-7.62 pp** | [-14.00, -1.12] | 0.125 |

With 4 runs the smallest attainable two-sided p is 2/16 = 0.125, so **none of these can reach
significance by construction** — the pre-registered F tests are descriptive by design. What they
show is directional and consistent with earlier rounds: the adaptive rule **hurts** both
`PickCube-v1` arms (as it did in round 3 on the co-trained checkpoint: 25.5% vs 36.0%) and is
null-to-slightly-negative on `PullCubeTool-v1` (as in round 1: p = 0.69). Combined with Task B,
the adaptive execution-horizon rule has not produced a success gain on any model we trained,
on any of the three tasks; its one positive result remains the teammate's co-trained checkpoint
on `PushCube-v1`.

---

## G. Synthesis and release

### G.1 Per-change summary, updated from round 5

The training run is the unit of analysis throughout; "supported" means the pre-registered
run-level rule (p < 0.05 and a positive mean difference) was met.

| change | round-6 evidence (run-level) | verdict | versus round 5 |
|---|---|---|---|
| **observation normalization, `PushCube-v1`** | 6 vs 6 runs: 96.08% vs 85.25%, **+10.83 pp** [+6.50, +14.92], **p = 0.0043**; sd 3.01 vs 4.97 | **SUPPORTED** | round 5 called it "best-supported, not certifiable" (+8.1 pp, 3-4 runs). Now certified, and it also *reduces* seed variance, which round 5 could not show. **Stronger.** |
| **observation normalization, `PickCube-v1` / `PullCubeTool-v1`** | `PickCube-v1` 4v4: 40.12% vs 60.38%, **-20.25 pp** [-32.87, -7.88], p = 0.057 (not supported; harmful); `PullCubeTool-v1` 4v4: **+15.88 pp** [+9.63, +22.13], **p = 0.0286** (supported at α = 0.05; misses α = 0.025) | **task-dependent in both directions** | round 5 tested `PushCube-v1` only and called normalization the best-supported change; the three-task evidence **contradicts a global reading** (+10.8, +15.8, -20.3 pp on PushCube, PullCubeTool, PickCube). |
| **adaptive replanning on the trained models (transfer)** | B1 (B0, 6v6): **-1.83 pp**, p = 0.5625, 2 of 6 runs significantly *worse*; B2 (V1, 6v6): **-0.92 pp**, p = 0.5000, 1 of 6 worse | **NOT SUPPORTED — no transfer** | round 5 claimed +11.0 pp vs `fixed:8` (p = 0.00068) and "composable". That gain is **specific to the co-trained checkpoint**; on our single-task models it is neutral-to-harmful. **Reversed for the general claim.** |
| **stacking (normalization + adaptive)** | V1-adaptive vs default DP (B0, `fixed:8`): +9.92 pp, p = 0.0065; interaction (does the adaptive gain differ between arms?) +0.92 pp, **p = 0.773** | the combination is better than default DP, but **the adaptive rule adds nothing**; the gain is normalization | round 5 predicted the two axes compose. They compose in *cost*, not in *value*. |
| **DDIM-5 sampling on single-task models** | cell means: B0 `fixed:8` 86.3% vs 85.2% (DDPM-100); V1 `fixed:8` 96.8% vs 96.1%; but per run: +1.0 (p = 0.87), +4.0 (p = 0.096), **-8.5 (p = 0.0137)**, +1.5 (p = 0.25) | cost-free **on average** (5.8x less policy time), **not uniformly** per run | round 5's "no detectable loss" was measured on the co-trained checkpoint; here one of six runs loses 8.5 pp significantly. **Weaker.** |
| **best combination vs default DP** | V1 + adaptive + DDIM-5 = 95.0% at 0.62 s/episode vs B0 + `fixed:8` + DDPM-100 = 85.2% at 3.62 s: **+9.75 pp** [+5.50, +13.83], **p = 0.0065**, **5.8x less policy time** | **SUPPORTED** (driven by normalization; sampling supplies the speed) | round 5 had 88.0% at 0.65 s for the co-trained model; the pattern holds, with a different driver. |
| encoder width 128 -> 256 (round 5) | not re-tested | no benefit, widest seed spread (12.5 pp) | unchanged |
| prediction horizon 16 -> 32 / 16 -> 8 (round 5) | not re-tested | V2b (32) suggestive but not significant; V2a (8) no benefit and +16% calls | unchanged |
| **co-training vs single-task** | `fixed:8`, DDPM-100: co-trained 77.0% (`PushCube-v1`, vs single-task B0 mean 85.25% over 6 runs), 42.5% (`PickCube-v1`, vs 60.38% over 4 runs) and 7.5% (`PullCubeTool-v1`, vs 11.62% over 4 runs) — single-task is ahead on all three tasks | **DESCRIPTIVE only** — one checkpoint, one training seed, different batch composition and device (all confounds listed in §E); no causal claim | not previously tested |
| action representation | not tested | outside the light-code-change bound (needs regenerated demonstrations, a different control mode and a different output head) | unchanged |

### G.2 What changes in `REPORT_ROUND_5.md` and `PROJECT_MEMO.md`

1. **`PROJECT_MEMO.md` §3, `PushCube-v1` row ("baseline 78.0%, adaptive 91.0%, better p < 0.0001").**
   The adaptive gain is real *for the teammate's co-trained checkpoint* (round 5: +11.0 pp vs
   `fixed:8`, p = 0.00068; the co-trained checkpoint scores 74.5-78.0% at `fixed:8` on three
   different seed blocks). It does **not** hold for the models our own pipeline trains: there the
   same rule is -1.83 pp (B0, p = 0.56) and -0.92 pp (V1, p = 0.50), with 3 of 12 runs
   significantly worse. The memo's "adaptive replanning helps on `PushCube-v1`" must be scoped to
   the co-trained policy, and the row's 91.0% figure belongs to the older block.
2. **`PROJECT_MEMO.md` §4 coverage table ("prediction horizon, observation encoder: not tested —
   need retraining").** No longer true: rounds 5-6 tested prediction horizon 8/32 (no benefit /
   suggestive only), encoder width 256 (no benefit) and observation normalization (a genuine,
   now-certified improvement of about +11 pp on `PushCube-v1`).
3. **`REPORT_ROUND_5.md` Task 3 verdict ("no training-time change is certifiable; V1 is the
   best-supported but not established").** Superseded: under the run-level test V1 satisfies the
   pre-registered rule on `PushCube-v1` (+10.83 pp, p = 0.0043, 6 vs 6 runs). Round 5's failure to
   certify came from a decision rule whose threshold (101.0%) was unreachable, not from a lack of
   effect.
4. **`REPORT_ROUND_5.md` Task 2 ("cutting sampling steps is effectively free on
   `PushCube-v1`")**. True on average for the retrained models as well (cell means 86.3 vs 85.2
   and 96.8 vs 96.1), but not uniformly: one of six B0 runs loses 8.5 pp significantly
   (p = 0.0137). The "free" statement should carry that per-run caveat.
5. **`REPORT_ROUND_5.md` synthesis ("the two axes are composable: adaptive + DDIM-5 = 88.0%
   at 0.65 s")**. Composition in cost is confirmed (5.8x less policy time at equal or better
   success), but the *value* of the composition comes from normalization, not from the adaptive
   rule, and the interaction test finds no differential adaptive gain (p = 0.773).
6. **Round 5's statement that "V1 does not reduce seed variance (8.0 vs 8.5 pp)"** is refined:
   with 6 runs per arm, V1's run-level sd is 3.01 versus B0's 4.97, so V1 *does* reduce
   seed-to-seed variability.
7. **`PROJECT_MEMO.md` §6 caveat "one policy and one training seed"** is no longer accurate for
   `PushCube-v1`: the round-6 comparisons use 6 trained runs per arm plus the co-trained
   checkpoint.

### G.3 Packaged artefacts and release verification

- `results/FINAL_RESULTS.csv`: **190 rows** (up from 114 in round 5); every round-4 and round-5
  row is unchanged, plus the round-6 blocks (5 new `PushCube-v1` runs, 12 adaptive cells,
  24 combination cells, 16 normalisation runs on the other two tasks, 3 co-trained cells and
  16 Task-F cells).
- `results/MANIFEST.md`: **3 726 raw log files / 37 282 episodes** (was 2 206 / 22 082), each
  with episode count, seed range and SHA-256.
- `scripts/verify_results.py --raw-root <full raw tree>`: **2 447 values checked, 0 mismatches,
  0 problems**.
- `scripts/r5_integrity.py` over the whole raw tree (round-5 and round-6 roots):
  **2 530 chunk files, 0 problems** (unparsable lines, duplicate seeds, wrong episode counts).
  The checker's patterns were extended for round 6 after the first run revealed it was only
  covering the round-5 roots.
- Release tests: **19/19** pass (`tests/test_core.py` 15, `tests/test_runlevel.py` 4 including
  hand-computed permutation cases and the raw-feature rule).
- Raw-feature requirement: `results/r6_rawfeat_check.json` — 191 decisions, **0 feature
  mismatches, 0 rule mismatches**, all non-vacuous.
- Figures: four families, PNG + PDF (`results/fig_r6_a_pushcube_runs`, `_b_combination`,
  `_c_three_tasks`, `_d_cotrain`).
- Run-level tables: `results/r6_runlevel_summary.csv` (83 cells),
  `results/r6_runlevel_arms.csv` (19 arm summaries), `results/r6_runlevel_tests.csv`
  (12 pre-registered tests), `results/r6_runlevel_mcnemar.csv` (52 per-run paired comparisons).
- Release 1.2.0: new `src/adaptive_replan/runlevel.py`, `tests/test_runlevel.py`,
  `scripts/r6_runlevel.py`, `scripts/r6_figures.py`, `scripts/r6_verify_rawfeat.py` and five
  `configs/round6_*.json`, all comment-free and free of absolute paths; `pack_final_results.py`
  discovers the `r6_*` roots; README, CHANGELOG and RELEASE_CHECK updated. Safety scan: the only
  matches for the searched patterns are the two self-referential documents that *list* the
  patterns (`docs/RELEASE_CHECK.md`, `docs/REPORT_ROUND_5.md`); no checkpoints, no
  demonstration data and no teammate-owned files are included; nothing is pushed and no licence
  is chosen.

### G.4 What was not finished

- `PegInsertionSide-v1` remains out of scope (no checkpoint).
- Task F cannot reach significance by construction at 4 runs per arm (smallest attainable
  two-sided p = 0.125); it is reported as directional evidence only, as pre-registered.
- No new timing benchmark was taken in round 6 (per the brief); all round-6 policy-time numbers
  are derived from the round-5 clean per-call measurements.
- The Task C `PullCubeTool-v1` result (p = 0.0286) does not survive the two-task α = 0.025
  threshold; both readings are stated wherever it appears.

---

## Corrections applied when this file was packaged

These corrections previously lived in a separate `REPORT_ADDENDUM_5_FIXES.md`; they are applied in
the body above and listed here for the record (release 1.3.6). No success rate, verdict or test
value changed except where a stated number was itself the error.

1. The duplicated result block that a slice bug had inserted at the top of the file (before the
   title) was removed; its content is now §C, with corrected `PullCubeTool-v1` numbers.
2. §E: the two `pending (Task C)` rows were filled in, and the single-task comparison no longer
   quotes one baseline seed (`B0s0` 75.5%) as if it were the arm mean.
3. §G.1: the same single-seed comparison in the co-training row was replaced by arm means.
4. §D: the two DDPM-100 adaptive rows were completed from 4 to 6 runs (B0 82.2% -> 83.42%,
   V1 94.5% -> 95.17%) and the "still being evaluated" note was removed; all 24 combination cells
   and all 12 adaptive cells are complete.
5. §C: the heading no longer says `PullCubeTool-v1` is pending, and that task's table is present
   with the corrected mean (27.50%, not 27.44%), sd (6.87, not 6.82) and difference
   (+15.88 pp, not +15.82 pp; CI [+9.63, +22.13], not [+9.57, +22.07]). The permutation p is
   unchanged: 2/70 = 0.028571.
6. §D: the DDIM-5 per-run cost paragraph lists all 12 runs (previously 4), so "11 of 12 runs
   within ±5 pp and 1 of 12 significantly worse" is stated rather than implied.
7. §F: checked and complete; no change needed.
8. A round-6 script defect (the DDIM baseline was matched by sampler and step count, so those
   cells were compared with themselves) was found and fixed in release 1.3.0; the corrected rows
   are in `results/r6_runlevel_mcnemar.csv`.
9. §B: the last two per-run differences in each row were stale and did not reproduce the row's own
   mean. B0 is now +5.0, -11.0, -6.5, -5.5, +3.5, +3.5 pp and V1 -1.5, -5.0, -1.5, -1.0, +3.0,
   +0.5 pp; means, sign-flip p-values and the significantly-worse runs are unchanged.
10. §F: the `PickCube-v1` V1 mean difference is -5.88 pp, not -5.87 pp.
11. Two round-7 pipeline defects found later (`r7_runlevel.py` looked the clean per-call time up
    under the wrong arm name; `pack_final_results.py` did not fold the reused round-5 sweep half
    into its `r5_front_*` rows) are fixed in release 1.3.2 and described in `REPORT_ROUND_7.md`
    revision note (2).
12. The adaptive-vs-`fixed:8` figure +11.0 pp (p = 0.00068) is a **round-5** measurement
    (`REPORT_ROUND_5.md` §2.2 frontier, seeds 400000-400199), not a round-4 one; three references
    to "round 4" were corrected. Round 4's own fresh-block comparison was +14.5 pp, p = 0.000002
    (`REPORT_ROUND_4.md` §B.2).
