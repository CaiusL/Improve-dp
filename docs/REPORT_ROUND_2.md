# REPORT_ROUND_2.md

Formerly `REPORT_ADDENDUM.md`.

Round-2 work on the adaptive-execution-horizon project: (A) a confirmatory block on fresh seeds,
(B) a competent-policy task (`PushCube-v1`), (C) a control rule, (D) fact-checks.

This file is an addendum to `REPORT_ROUND_1.md`, which is left unedited. Round-one raw results were
never overwritten; all round-2 output lives in new `results/raw/<phase>/…` folders.

Environment is unchanged from round one: Apple M5 Pro, macOS 26.6.2, CPU simulation
(`physx_cpu`, `obs_mode="state"`, `render_mode=None`, `num_envs=1`), Python 3.14.6, torch 2.14.1,
ManiSkill 3.0.1, policy `runs/panda3_shared_a0/checkpoints/final.pt` (`ema_agent`),
`OMP_NUM_THREADS=1` per worker.

---

## Task D — fact-checks (answered first; D is independent of A/B/C)

### D1. Exactly how is "success" computed, and when?

Two layers, both verified in the installed source and against 1 698 logged episodes.

**(a) The instantaneous task signal** — `info["success"]`, from each task's `evaluate()`:

- `PushCube-v1` (`mani_skill/envs/tasks/tabletop/push_cube.py`, `evaluate`, L179–192):
  `success = (||cube_xy − goal_xy|| < goal_radius) & (cube_z < cube_half_size + 5e-3)` with
  `goal_radius = 0.1` and `cube_half_size = 0.02` (L59–60). I.e. the cube is inside the goal circle
  **and** still resting on the table.
- `PullCubeTool-v1` (`…/pull_cube_tool.py`, `evaluate`, L192–216): `success = cube_pulled_close`
  (the cube's xy is within `goal_radius = 0.3` of the region in front of the robot base), alongside
  `cube_progress` and `cube_distance`.

**(b) The episode-level metric my analysis uses** — `info["episode"]["success_once"]`, produced by
`CPUGymWrapper` with `record_metrics=True`
(`mani_skill/utils/wrappers/gymnasium.py`):

```python
L47  self.success_once = False                     # on reset (and L95)
L63  if "success" in info:
L64      self.success_once = self.success_once | info["success"]   # running OR, every step
L65      episode_info["success_once"] = self.success_once
L74  if self.ignore_terminations:
L77      if "success" in info:
L78          episode_info["success_at_end"] = info["success"]     # this step's flag only
```

So:

- **`success_once` = "the task's instantaneous success was True at *any* step of the episode"**
  (a running logical OR over all steps, reset per episode) — **not** just at the end.
- **`success_at_end` = the instantaneous flag at the *final* step only.** It exists because
  `ignore_terminations=True`; without that flag it would be absent.
- **Episodes never terminate early.** The wrapper hard-sets `terminated = False`
  (`gymnasium.py` L74–75), so an episode ends only on `truncated` = time limit. Every episode
  therefore runs the full `max_episode_steps` (200 for PushCube, 600 for PullCubeTool), and
  `success_once` covers the whole window.
- My harness reads the episode-level value at the final step and also cross-checks
  `info["success"]` per step as a fallback; the fallback never triggered.

**Empirical check over 1 698 logged episodes:** `success_at_end ⇒ success_once` held in **100%** of
episodes, and there were **0 episodes that succeeded mid-episode but not at the end** — in these two
tasks success is effectively monotone (the cube, once placed/pulled, stays there). So for this
project `success_once` and `success_at_end` coincide, and the "at any time" definition does not
inflate the numbers.

> Note: `PullCubeTool-v1`'s `evaluate()` also returns its own per-step `success_once`/`success_at_end`
> keys. Those are *instantaneous* values; the episode-level numbers in `info["episode"]` come from the
> wrapper's running OR, which is what my harness reads.

### D2. Is the n = 100 block a subset of the n = 200 block? Discordance by half.

**Yes, exactly.** `results/final100_*` was produced with `analysis.py --seeds 200000-200099` and
`results/final_*` with `--seeds 200000-200199`, so the n = 100 block **is the first 100 seeds** of
the n = 200 block. (Both used the same phase `test`, same chunk files, same dedup rule.)

Paired McNemar discordance, split by half (seeds 200000–200099 vs 200100–200199):

`adaptive_dist d=0.30` **vs** `fixed:8`:

| seeds | n | adaptive | fixed:8 | adaptive-only | fixed-only | both | neither | exact p |
|---|---|---|---|---|---|---|---|---|
| 200000–200099 (first half) | 100 | 0.130 | 0.140 | 3 | 4 | 10 | 83 | **1.0000** |
| 200100–200199 (second half) | 100 | 0.180 | 0.080 | 12 | 2 | 6 | 80 | **0.0129** |
| all 200 | 200 | 0.155 | 0.110 | 15 | 6 | 16 | 163 | 0.0784 |

`adaptive_dist d=0.30` **vs** `fixed:4`:

| seeds | n | adaptive | fixed:4 | adaptive-only | fixed-only | both | neither | exact p |
|---|---|---|---|---|---|---|---|---|
| 200000–200099 | 100 | 0.130 | 0.110 | 9 | 7 | 4 | 80 | **0.8036** |
| 200100–200199 | 100 | 0.180 | 0.070 | 15 | 4 | 3 | 78 | **0.0192** |
| all 200 | 200 | 0.155 | 0.090 | 24 | 11 | 7 | 158 | 0.0410 |

**This is the most important fact-check in the addendum.** The round-one effect is **entirely
concentrated in the second half of the seed block**:

- first 100 seeds: adaptive 13.0% vs fixed:8 14.0% (p = 1.00) and vs fixed:4 11.0% (p = 0.80) —
  **no effect at all, in either comparison**;
- second 100 seeds: 18.0% vs 8.0% (p = 0.013) and vs 7.0% (p = 0.019).

So the round-one headline rests on seed heterogeneity, not on a consistent effect. That is exactly
what Task A's fresh block was designed to test, and it means the n = 100 "opposite ranking" reported
in `REPORT_ROUND_1.md` §1 was not a fluke of small samples alone — the effect genuinely varies by
seed. Task A's result is reported against this background.

### D3. Sample size for 80% power to detect 9% vs 15.5%

Script: **`power_analysis.py`** (results in `results/power_analysis.json`).

**Unpaired** (two independent arms, normal approximation to the two-proportion z-test, two-sided
α = 0.05):

$$n_{\text{per arm}}=\frac{\left(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}+z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\right)^2}{(p_2-p_1)^2}$$

→ **399 episodes per arm (798 total)**.

**Paired** (McNemar exact test on the same seeds; normal approximation, Connor 1987 / Lachin 1992).
With discordant probabilities π₁₀, π₀₁, and ψ = π₁₀+π₀₁, δ = π₁₀−π₀₁ = p₂−p₁:

$$n_{\text{pairs}}=\frac{\left(z_{1-\alpha/2}\sqrt{\psi}+z_{1-\beta}\sqrt{\psi-\delta^2}\right)^2}{\delta^2}$$

ψ is **not** determined by p₁ and p₂, so it is supplied explicitly. Using the ψ actually observed in
round one (adaptive vs fixed:4: A_only = 24, B_only = 11 of 200 → **ψ = 0.175**, δ = 0.065):

→ **323 seed pairs**.

Sensitivity to ψ:

| ψ | 0.10 | 0.15 | **0.175** | 0.25 | 0.35 | 0.50 |
|---|---|---|---|---|---|---|
| n pairs | 184 | 277 | **323** | 463 | 648 | 927 |

**Monte-Carlo validation of the paired formula** (`power_analysis.py`, 4 000 exact McNemar tests per
point, π₁₀ = 0.120, π₀₁ = 0.055): simulated power 0.729 at n = 273, **0.805 at n = 323**, 0.870 at
n = 373 — confirming the formula at the target 80%.

**Consequence:** round one's n = 200 was **below** the ~323 pairs needed. Task A's pre-registered
n = 200 is, by the same standard, also underpowered for a 6.5-point effect — so Task A can
reasonably confirm or fail to confirm a *large* effect, but a null result from it would not rule out
a real effect of the round-one magnitude. This is stated in the Task A verdict.

### D4. What random generator and seed drive the randomized schedules?

- **Generator:** `numpy.random.default_rng(...)` → **PCG64** (`RandomMean` in `adaptive_rollout.py`
  draws `self.rng.random() < p_hi` once per policy decision, with
  `p_hi = (mean − lo)/(hi − lo)` and `lo, hi = 2, 8`).
- **Seed:** re-seeded **per episode** as `np.random.default_rng(12345 + env_seed)`
  (`adaptive_rollout.py` L511). The generator built at policy construction (seed 12345) is
  overwritten before every episode, so the only thing that matters is the environment seed.
- Consequently each episode's chunk-length sequence is a deterministic function of the environment
  seed alone, and is independent of the schedule's other parameters except through `p_hi`.

**Verified by pure RNG replay** (no rollouts) — re-drawing `n_inference` values from
`default_rng(12345 + seed)` and counting draws below `p_hi` reproduced the logged `horizon_counts`
**exactly** for every episode tested:

| schedule | episodes reproduced |
|---|---|
| round one `random:mean=5.49` | 200/200 |
| round one `random:mean=7.91` | 200/200 |
| round two `random:mean=5.11` (Task A, partial) | 24/24 |
| round two pilot `random:mean=5.11` | 8/8 |

---

## Task A — confirmatory block on fresh seeds (`PullCubeTool-v1`)

Pre-registered in `NOTES.md` (2026-10-04 19:47) before any rollout: task `PullCubeTool-v1`, same
policy/checkpoint, **seeds 300000–300199** (fresh), four frozen settings — `fixed:4`, `fixed:8`,
`adaptive_dist:d=0.30,short=2,long=8` (identical rule to round one) and
`random:mean=5.11,lo=2,hi=8` (matched to the adaptive rule's round-one measured mean horizon of
5.112). Raw logs: `results/raw/confirm_pullcubetool/` (80 chunks, 800 episodes, ~40 min).

### Results (n = 200 per setting)

| schedule | n | succ | success rate | 95% Wilson CI | mean inference calls | mean horizon | cube_progress_max |
|---|---|---|---|---|---|---|---|
| **`adaptive_dist` d=0.30** | 200 | **24** | **0.120** | [0.082, 0.172] | 155.0 | 5.00 | 0.0425 |
| `fixed:4` | 200 | 17 | 0.085 | [0.054, 0.132] | 150.0 | 4.00 | 0.0368 |
| `fixed:8` | 200 | 22 | 0.110 | [0.074, 0.161] | 75.0 | 8.00 | 0.0416 |
| `random(mean=5.11)` | 200 | 17 | 0.085 | [0.054, 0.132] | 117.5 | 5.14 | 0.0361 |

The achieved mean horizons match the intent: adaptive 5.00 and the matched random 5.14.

### Primary and secondary paired comparisons (exact McNemar, same 200 seeds)

| comparison | A rate | B rate | A-only | B-only | exact p | paired difference (95% Newcombe CI) |
|---|---|---|---|---|---|---|
| **P1** adaptive vs `fixed:4` | 0.120 | 0.085 | 18 | 11 | **0.2649** | +0.035 [−0.025, +0.096] |
| **P2** adaptive vs `fixed:8` | 0.120 | 0.110 | 4 | 2 | **0.6875** | +0.010 [−0.054, +0.074] |
| **S1** adaptive vs `random(5.11)` | 0.120 | 0.085 | 18 | 11 | **0.2649** | +0.035 [−0.025, +0.096] |

Failure phases are unchanged in character: 117/176 adaptive failures never grasped the tool
(66%), 55 never brought it to the cube (31%), and only 4 (2%) reached the cube and failed the pull.

### Pre-registered verdict

| rule | outcome |
|---|---|
| P1 CONFIRMED iff adaptive > `fixed:4` **and** p < 0.05 | p = 0.265 → **NOT confirmed** (direction is right, significance is not) |
| P1 DIRECTIONAL iff adaptive > `fixed:4` **and** 0.05 ≤ p < 0.20 | p = 0.265 is outside this band too |
| P2 no-significant-loss iff p ≥ 0.05 | p = 0.688 → **no significant loss** vs `fixed:8` |
| **Task A overall = CONFIRMED iff P1 confirmed AND P2 no loss** | **NOT CONFIRMED** |

### **The round-one headline did not replicate.**

Round one reported adaptive 15.5% vs `fixed:4` 9.0% (p = 0.041) and vs `fixed:8` 11.0%
(p = 0.078). On 200 fresh seeds the same rule, same parameters and same policy give:

| schedule | round one (200000–200199) | Task A (300000–300199) | change |
|---|---|---|---|
| `adaptive_dist` d=0.30 | 15.5% | **12.0%** | **−3.5 pp** |
| `fixed:4` | 9.0% | 8.5% | −0.5 pp |
| `fixed:8` | 11.0% | **11.0%** | 0.0 pp |
| `random(5.11)` | 8.5% (mean 5.49) | 8.5% | 0.0 pp |

**The baselines replicated essentially exactly; the adaptive rule's advantage did not.** The
adaptive rule is still nominally the best schedule in the block, and it is still nominally ahead of
`fixed:4` and the matched random by +3.5 pp — but that difference is no longer distinguishable from
zero, and the effect is roughly half the size reported in round one. This is consistent with D2's
finding that round one's effect lived almost entirely in the second half of its seed block, i.e. it
was substantially a seed-heterogeneity artefact.

### What Task A can and cannot conclude

- **Can conclude:** the specific round-one claim ("adaptive beats `fixed:4` at matched cost",
  p = 0.041) **does not hold at the pre-registered significance level on fresh seeds**, and no
  significant advantage over `fixed:8` exists in either block.
- **Cannot conclude:** that adaptive replanning has *no* effect. From D3, detecting a 9% → 15.5%
  difference needs ≈ 323 paired seeds for 80% power; Task A had 200, and the observed effect here
  was only 3.5 pp, which would need on the order of 1 000+ pairs. A null verdict at n = 200 is
  therefore weak evidence of absence, not evidence of no effect.
- The paired-difference CIs are the right summary: P1 is +3.5 pp with a 95% CI of
  [−2.5, +9.6] pp — compatible with anything from a small loss to a round-one-sized gain.


---

## Task B — `PushCube-v1`, the competent-policy task

Pre-registered in `NOTES.md` before any test seed was touched. Policy unchanged
(`panda3_shared_a0/final.pt`, EMA), `task_id = 1`, `max_episode_steps = 200`, same protocol
(explicit `env.reset(seed=…)`, paired seeds, torch seed = `1000000 + env_seed`).

### Success semantics and features used

`push_cube.py::evaluate` returns `success = (||cube_xy − goal_xy|| < goal_radius) & (cube_z <
cube_half_size + 5e-3)` with `goal_radius = 0.1`, `cube_half_size = 0.02`. The goal region is
*placed* at `cube_xy + [0.1 + goal_radius, 0]`, so **the cube starts exactly 0.2 m from the goal**
and must be pushed ~0.1 m.

Only features already present in the 35-dim state observation are used
(`extra.tcp_pose[18:25]`, `extra.goal_pos[25:28]`, `extra.obj_pose[28:35]`):
`d_tcp_obj` (TCP-to-cube), `d_obj_goal_xy` (the task's own success metric), plus `obj_z`/`tcp_z`.
Decision-point quantiles measured from 1 500 calibration samples (24 episodes, `--decision-log`):
`d_tcp_obj` p25/p50/p75/p90 = 0.036/0.054/0.090/0.165; `d_obj_goal_xy` p25/p50/p75/p90 =
0.086/0.096/0.170/0.200 (the 0.20 saturation is the initial placement distance).

### The rule

`adaptive_push:d_tcp=A,d_goal=B` takes a **short (2)** chunk iff `d_tcp_obj < A` **or**
`d_obj_goal_xy < B`, else a long (8) chunk. `A=0` or `B=0` disables a disjunct (the ablations).

### Validation phase (seeds 100000–100099, n = 100)

Baselines: `fixed:2` **82%** (100 calls), `fixed:4` **77%** (50), `fixed:8` **72%** (25) —
monotone decreasing in horizon, confirming round one's finding that this task prefers short chunks.

Full tuning grid (8 configs, all pre-registered):

| config | succ | rate | mean inf | mean h |
|---|---|---|---|---|
| `d_tcp=0.05, d_goal=0` (TCP-only) | 82/100 | 0.820 | **60.4** | 3.47 |
| `d_tcp=0, d_goal=0.10` (goal-only) | 72/100 | 0.720 | 60.1 | 4.24 |
| `d_tcp=0.04, d_goal=0.10` | 79/100 | 0.790 | 74.2 | 2.87 |
| `d_tcp=0.05, d_goal=0.10` | 82/100 | 0.820 | 78.5 | 2.59 |
| `d_tcp=0.06, d_goal=0.10` | 84/100 | 0.840 | 83.2 | 2.43 |
| **`d_tcp=0.08, d_goal=0.10`** | **86/100** | **0.860** | 88.1 | 2.28 |
| `d_tcp=0.10, d_goal=0.15` | 84/100 | 0.840 | 90.9 | 2.20 |
| `d_tcp=0.15, d_goal=0.30` | 82/100 | 0.820 | 100.0 | 2.00 |

Two things stand out: the **TCP-proximity disjunct carries the benefit** (goal-only is the worst
config at 72%), and *adding* the goal disjunct to a TCP rule costs calls without adding success
(`d_tcp=0.05,d_goal=0` 82% @ 60.4 vs `d_tcp=0.05,d_goal=0.10` 82% @ 78.5).

### Held-out test block (seeds 200000–200199, n = 200 per setting)

| schedule | n | succ | success rate | 95% Wilson CI | mean inference calls | mean horizon | vs `fixed:2` calls |
|---|---|---|---|---|---|---|---|
| `fixed:2` | 200 | 152 | 0.760 | [0.696, 0.814] | 100.0 | 2.00 | — |
| `fixed:4` | 200 | 159 | 0.795 | [0.734, 0.845] | 50.0 | 4.00 | −50% |
| `fixed:8` | 200 | 156 | 0.780 | [0.718, 0.832] | 25.0 | 8.00 | −75% |
| `random(mean=2.28)` (matched to primary) | 200 | 160 | 0.800 | [0.739, 0.850] | 87.3 | 2.30 | −13% |
| `random(mean=3.47)` (matched to ablation) | 200 | 160 | 0.800 | [0.739, 0.850] | 57.2 | 3.57 | −43% |
| **`adaptive_push` d_tcp=0.08, d_goal=0.10** (primary) | 200 | **182** | **0.910** | **[0.862, 0.942]** | 88.9 | 2.25 | **−11%** |
| `adaptive_push` d_tcp=0.06, d_goal=0.10 | 200 | 172 | 0.860 | [0.805, 0.901] | 83.7 | 2.41 | −16% |
| `adaptive_push` d_tcp=0.05, d_goal=0 (TCP-only) | 200 | 172 | 0.860 | [0.805, 0.901] | **62.3** | 3.36 | **−38%** |

### **The pre-registered primary question is answered YES.**

> *Does the adaptive rule match or beat `fixed:2` success at clearly fewer inference calls?*
> ("match" = success ≥ `fixed:2`; "clearly fewer" = ≥20% fewer mean inference calls)

- **TCP-only ablation: 86.0% vs `fixed:2`'s 76.0% at 62.3 vs 100 calls → 38% fewer calls, +10 pp
  success.** This satisfies both halves of the criterion, and the paired test says the success
  gain is real (McNemar p = 0.0055).
- **Primary (d_tcp=0.08, d_goal=0.10): 91.0% vs 76.0% at 88.9 vs 100 calls → 11% fewer calls,
  +15 pp success** (p = 0.0000). It clears the success bar comfortably but only reduces cost by
  11%, so it does not by itself meet the "≥20% fewer calls" leg; the ablation does.

### Paired comparisons (exact McNemar, same 200 seeds)

| A | B | A rate | B rate | A-only | B-only | p |
|---|---|---|---|---|---|---|
| primary adaptive | `fixed:2` | 0.910 | 0.760 | 37 | 7 | **<0.0001** |
| primary adaptive | `fixed:4` | 0.910 | 0.795 | 29 | 6 | **0.0001** |
| primary adaptive | `fixed:8` | 0.910 | 0.780 | 28 | 2 | **<0.0001** |
| primary adaptive | `random(2.28)` | 0.910 | 0.800 | 30 | 8 | **0.0005** |
| primary adaptive | `random(3.47)` | 0.910 | 0.800 | 30 | 8 | **0.0005** |
| TCP-only ablation | `fixed:2` | 0.860 | 0.760 | 34 | 14 | **0.0055** |
| TCP-only ablation | `fixed:4` | 0.860 | 0.795 | 22 | 9 | **0.0294** |
| TCP-only ablation | `fixed:8` | 0.860 | 0.780 | 20 | 4 | **0.0015** |
| TCP-only ablation | `random(3.47)` | 0.860 | 0.800 | 25 | 13 | 0.073 |

**State-dependence, not just average chunk length, is what matters.** Both matched random schedules
sit at exactly 80.0%, while the adaptive rules at the *same mean horizon* reach 86–91%
(e.g. mean horizon ≈2.3: adaptive 91.0% vs random 80.0%, p = 0.0005; mean horizon ≈3.5: adaptive
86.0% vs random 80.0%, p = 0.073).

### Contrast with Task A

The two tasks disagree, and that is the headline of this addendum:

| | `PullCubeTool-v1` (Task A, fresh seeds) | `PushCube-v1` (Task B) |
|---|---|---|
| policy competence | ~6–15% success | ~76–91% success |
| adaptive vs `fixed:2` | n/a (fixed:2 not in the fresh block; vs `fixed:4` +3.5 pp, **p = 0.26**) | +10 to +15 pp, **p ≤ 0.0055** |
| adaptive vs best fixed | +1.0 pp vs `fixed:8`, p = 0.69 | +11.5 pp vs `fixed:4`, p = 0.0001 |
| verdict | **NOT CONFIRMED** | **CONFIRMED, decisively** |

The natural reading: **state-dependent replanning pays off only when the policy is competent enough
to spend most of the episode in states where the horizon choice actually matters.** On
`PullCubeTool-v1` the policy fails at the gripper and `d_tool_cube` barely varies, so the rule has
almost nothing to react to; on `PushCube-v1` the policy reliably reaches the cube, and choosing long
chunks in free space vs short chunks near the object/goal produces a large, highly significant gain.

*(The generic failure classifier in `analysis.py` mislabels PushCube failures — its thresholds are
tuned for the pull task. `pushcube_failure.py` does the task-appropriate version; see below.)*


---

## Task B (cont.) — why PushCube episodes fail

`analysis.py`'s generic failure classifier uses thresholds tuned for the PullCubeTool task, so it
mislabels PushCube failures. `pushcube_failure.py` does the task-appropriate version, using the
`--decision-log` features from 32 traced test episodes (`results/raw/pushcube_traces/`).

| schedule | n | succ | fail | fail `min d_obj_goal_xy` p50 | p90 | failures where TCP never came within 0.08 m of the cube |
|---|---|---|---|---|---|---|
| `fixed:2` | 16 | 13 | 3 | 0.1179 | 0.1737 | 0/3 |
| `adaptive_push` 0.08/0.10 | 16 | 14 | 2 | 0.1771 | 0.1786 | 0/2 |

**Every PushCube failure reached the cube** — the arm always gets there. What fails is the *fine
phase*: the cube only gets as close as 0.12–0.18 m to the goal when 0.10 m is required (it starts at
0.20 m, so a failure means it was pushed ~0.02–0.08 m instead of ~0.10 m). This is precisely the
regime in which a shorter execution horizon should help, and it is the opposite of the
PullCubeTool situation, where ~97% of failures never even grasp the tool. That mechanical
difference is the most likely reason Task B succeeds where Task A did not.

---

## Task C — control rule on an unrelated feature

**Purpose (pre-registered):** a negative control. If a rule keyed on a feature unrelated to
tool–cube geometry, with the same rate of short-horizon decisions, performs comparably to
`adaptive_dist` d=0.30, then the adaptive effect is not attributable to tool–cube distance.

**Rule:** `adaptive_thresh:feat=tcp_z,d=h,dir=lt,short=2,long=8` — short chunk when the TCP is
*low* (working down at the table). Threshold chosen to match the adaptive rule's short-decision rate.
All on **validation seeds 100000–100099** (100 seeds); no test seeds used.

### Step 1 — the adaptive rule's decision structure (83 episodes, 11 358 decisions)

| phase | decisions | short | short fraction | mean horizon |
|---|---|---|---|---|
| `not_grasped` | 4 514 | 1 919 | 0.425 | 5.45 |
| `grasped_far` | 1 929 | **0** | **0.000** | 8.00 |
| `near_cube` | 4 915 | 4 915 | **1.000** | 2.00 |

Overall short fraction **f\* = 0.6017**; `tcp_z` p50 = 0.0818. 48/83 episodes use **both** k=2 and
k=8, so the rule is genuinely reactive within episodes rather than merely picking a horizon per
episode; per-episode mean horizon p50 = 5.00.

**Round one's "frozen feature" claim is largely wrong.** Only **7/83 episodes (8%)** have
`d_tool_cube` constant across *all* decisions. Round one inferred freezing from 3 traced failed
episodes; it is a minority pattern, not the norm.

### Step 2 — the control comparison, and a matching problem I had to fix

Matching the short-decision rate with an unrelated feature turned out to be **structurally harder
than expected**: `tcp_z`'s marginal distribution depends on the policy's own trajectory, which the
control rule itself changes. Three iterations, all on the same 100 validation seeds:

| run | threshold | realised short fraction | mean inference calls | success |
|---|---|---|---|---|
| `adaptive_dist` d=0.30 (reference) | — | 0.6305 | 142.0 | 10/100 = 0.100 |
| `taskC_control` | 0.1524 (from a partial 83-episode sample) | 0.8508 | 207.7 | 15/100 = 0.150 |
| `taskC_control2` | 0.1701 (adaptive's own quantile — wrong distribution) | 0.8646 | 221.3 | 15/100 = 0.150 |
| **`taskC_control3`** | **0.0747 (control's own quantile, f = 0.6305)** | **0.7496** | 171.7 | 8/100 = 0.080 |

The first two attempts overshot badly because I matched a quantile of the *wrong* distribution.
The third used the control run's **own** `tcp_z` distribution and lands much closer
(0.750 vs the 0.631 target; 171.7 vs 142.0 calls) but is still **not exactly cost-matched** —
each iteration shifts the trajectory again, so a one-shot quantile match cannot be exact without
solving a fixed point.

### Task C verdict (pre-registered interpretation, applied to the best-matched run)

`adaptive_dist` 10.0% vs control (`tcp_z`) 8.0%; A_only = 9, B_only = 7;
**exact McNemar p = 0.8036**.

Per the pre-registered reading: **p ≥ 0.05, so the control rule is statistically indistinguishable
from `adaptive_dist`, and the (non-replicating) adaptive effect on this task is NOT specific to
tool–cube distance.** The TCP-height rule is in fact nominally *slightly worse* while using ~21%
more inference calls, so there is no evidence that the geometric feature is doing anything the
unrelated feature does not.

**Caveat, stated plainly:** the control's short fraction is 0.750 against the adaptive rule's 0.631,
so this is a comparison at roughly — not exactly — matched decision rate. Since the control used
*more* short chunks (and more compute) yet scored *lower*, the mismatch does not favour the
adaptive rule, so it does not change the conclusion; but an exactly cost-matched negative control
remains unfinished (see limitations). This is a validation-seed-only, n = 100 comparison
(10 vs 8 successes) and is therefore weak evidence either way — its main value is that it
*removes* the round-one/round-two PullCubeTool adaptive result as a positive claim about
state-dependent geometry.

---

## Not finished / limitations (round 2)

1. **Task A's n = 200 is below the ~323 paired seeds that D3 says are needed for 80% power** on a
   6.5-point effect, and the effect actually observed on fresh seeds was only 3.5 points. Task A
   therefore establishes "not confirmed at n = 200", *not* "no effect". The pre-registered seeds
   were not increased after the fact.
2. **Task C's negative control is not exactly cost-matched** (0.750 vs 0.631 short-decision rate).
   Getting an exact fixed point would need several more iterations of the full validation run, each
   ~35 minutes wall clock. Only the best-matched of three attempts is reported as the verdict; the
   other two are reported in full rather than hidden.
3. **PullCubeTool-v1's adaptive result remains unresolved in the strict sense.** Round one found
   +6.5 pp (p = 0.041) on seeds 200000–200199; Task A found +3.5 pp (p = 0.265) on new seeds
   300000–300199; D2 showed round one's effect lived entirely in the second half of its block.
   Pooling all 400 seeds would be informative but was **not** pre-registered, so it is not
   presented as a result.
4. **A transient machine-wide slowdown** (≈1 h, documented in `NOTES.md`) hit mid-run and forced
   several restarts. No chunk was re-run wastefully (`.done` markers were preserved) and no result
   was discarded, but it cost roughly an hour of the budget. Root cause was never identified;
   `ps`/`top` are blocked in this sandbox, so per-process CPU attribution was impossible.
5. **Seed duplicates** appeared in a few `pushcube_tune` chunk files where a killed chunk was
   re-run and appended. All analyses de-duplicate by seed; the driver now truncates a re-run chunk
   instead of appending, so this cannot recur. Raw files were left as written (not rewritten).
6. `panda3_shared_adapter_a0` still cannot be loaded with the released code (unchanged from round
   one), and **`PegInsertionSide-v1` still has no checkpoint**, so that main task remains
   unevaluated.
7. Both tasks used the `{2,8}` two-point randomized schedule; "random" here means "state-independent
   random on the same support with the same mean", not a draw over all horizons 1–8.
8. Only one policy variant and one training seed were available, so nothing here speaks to
   variance across training runs.

---

## Round-2 artefact index

| item | path |
|---|---|
| Task A raw logs | `results/raw/confirm_pullcubetool/` (800 episodes) |
| Task A summary / paired | `results/confirm_summary_*.csv` |
| Task B validation, calibration, tuning | `results/raw/pushcube_{val,calib,tune}/` |
| Task B test block | `results/raw/pushcube_test/` (1 600 episodes) |
| Task B summary / paired / figure | `results/pushcube_test_final_*.csv`, `results/pushcube_test_final_success_vs_cost.png` |
| Task B traces + failure analysis | `results/raw/pushcube_traces/`, `pushcube_failure.py` |
| Task C decision logs | `results/raw/taskC_{adaptive,control,control2,control3}/` |
| Task C outputs | `results/taskC_step1.json`, `results/taskC_compare.json`, `task_c_analysis.py` |
| Task D power analysis | `power_analysis.py` → `results/power_analysis.json` |
| reproducibility check | `results/raw/repro_check/` |
| every chunk-level setting run | `results/settings_run.csv` (`collect_settings.py`) |

**Round-2 rollout budget:** ~4 200 logged episodes (Task A 800, Task B 1 600 test + 856 tuning +
300 validation + 24 calibration + 32 traced, Task C ~440, plus pilots and the reproducibility
check), on top of round one's ~2 600.

