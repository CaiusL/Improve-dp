# REPORT_ADDENDUM_4 — Round 5: "Improving DP" beyond the execution horizon

Started 2026-10-05 17:50; experiment-launch cutoff 2026-10-07 18:00; report due 2026-10-08.
Framing: the course question asks which of *observation encoder, prediction horizon, action
representation, diffusion sampling* improve success or inference speed. Adaptive replanning
(execution horizon) was covered in earlier rounds; this addendum covers the new axes:
a training-speed benchmark (Task 1), a diffusion-sampling sweep including DDIM (Task 2),
and training-time variants on `PushCube-v1` gated by Task 1 (Task 3).

All rules below were pre-registered in `NOTES.md` with timestamps before the first rollout
or training run. Nothing here edits earlier reports; the user merges.

---

## Task 1 — training-speed benchmark (DONE, decision: Plan A)

**What produced the checkpoint:** the teammate's multi-task trainer `cotrain.py`
(the producing config `experiments/panda_three.json` is not on this machine; the checkpoint's
embedded `compat` block is the authoritative record: batch 32/task, 30 000 updates, AdamW
lr 1e-4 betas (0.95, 0.999) wd 1e-6, warmup 500 + cosine, EMA power 0.75, `state_normalization:
false`, sampling `official_flat_windows`, obs_horizon 2 / pred_horizon 16 / act_horizon 8,
encoder_dim 128, unet down-dims [64, 128, 256]). Single-task variant `train.py` also exists
(upstream DP style). All benchmarking was done on **copies** in `~/dp_replan/train/`; the
Desktop originals were not touched.

**Demo file:** `collect_data_train_v2/PushCube-v1/motionplanning/
trajectory.state.pd_ee_delta_pose.physx_cpu.h5` — 17.7 MB, 496 trajectory groups (494
eligible per the config audit; training uses a fixed 200-trajectory subset, subset_seed
20260929), obs_dim 35, action_dim 7, control `pd_ee_delta_pose` at 20 Hz.

**Benchmark (200 updates, single-task `PushCube-v1`, batch 32, identical hyperparameters;
warmup truncated to 150 because the script requires warmup < total-iters — compute-neutral):**

| Device | s per update | loss @1 → @200 | 30 000 updates (extrapolated) |
|---|---|---|---|
| CPU (`OMP_NUM_THREADS=1`), run 1 | 0.2012 | 1.221 → 0.549 | ≈ 1.68 h |
| CPU, run 2 (repeat) | 0.2037 | 1.221 → 0.549 | ≈ 1.70 h |
| MPS | 0.0191 | 1.179 → 0.533 | **≈ 0.16 h (10 min)** |

Raw timestamps: CPU (3.16 s → 43.19 s)/(199 updates) = 0.2012; repeat (2.86 s → 43.40 s)/199
= 0.2037; MPS (6.03 s → 9.83 s)/199 = 0.0191. CPU runs are reproducible to ~1%.
- MPS is ~10.6× faster than CPU for training. The loss trajectories agree (same start,
  same end, same shape; per-iteration values differ only at noise level because the
  diffusion noise and timesteps are drawn on the compute device, so the two runs sample
  different noise sequences). **No strong disagreement → MPS is reliable for this model.**
- Checkpoint save (41 MB): ≈ 0.1–0.15 s. Data load + model init + env start-up: ≈ 3–6 s.
- 20-episode evaluation (untrained 200-update model, DDPM-100, `physx_cpu` sim):
  3.97 s/episode with the policy on CPU vs 8.83 s/episode with the policy on MPS —
  **batch-1 DDPM inference is *slower* on MPS** (kernel-launch overhead), so the right
  combination is: *train on MPS, evaluate on CPU* (what all our rollouts already do).
- Bug found on our side (fixed in our copies, not in the originals): a **relative**
  `--run-dir` breaks checkpoint saving because ManiSkill's env creation changes the process
  CWD — the training script must be passed an absolute `--run-dir`.

**Gate: T = 0.16 h ≤ 1.0 h → PLAN A.** Task 3 runs as specified: 30 000 updates per model,
5 variants (B0, V1, V2a, V2b, V3) with two training seeds each where budget allows.
Total training ≈ 9 × 11 min ≈ 1.6 h MPS (plus evaluation, which dominates:
200 episodes ≈ 13 min CPU per model).

---

## Task 2 — diffusion sampling sweep (`REPORT` section, results as of 18:30)

### 2.1 sampler x steps, `fixed:8`, fresh seeds 400000-400099 (n = 100/cell) — COMPLETE

DDIM was configured with the checkpoint's trained betas, epsilon prediction, `eta = 0`
(diffusers `DDIMScheduler`) — the feasibility gate (10 episodes, DDIM-20 vs DDPM-20 on
identical seeds) passed 9/10 vs 10/10. Full cell table in `results/r5_sampling_summary.csv`;
paired vs DDPM-100 in `results/r5_sampling_paired.csv` (24 pairs).

Success by cell (rate %, 95% Wilson):

| steps | PushCube DDPM | PushCube DDIM | PickCube DDPM | PickCube DDIM | Pull DDPM | Pull DDIM |
|---|---|---|---|---|---|---|
| 100 | 75.0 [65.7,82.5] | — | 43.0 [33.7,52.8] | — | 13.0 [7.8,21.0] | — |
| 50 | 78.0 | 83.0 | 39.0 | 34.0 | 6.0 | 9.0 |
| 20 | 81.0 | 80.0 | 45.0 | 31.0* | 10.0 | 7.0 |
| 10 | 83.0 | 82.0 | 41.0 | 33.0* | 18.0* | 8.0 |
| 5 | 74.0 | 84.0* | 46.0 | 28.0* | 15.0 | 5.0 |

*marked = largest compatible loss exceeds, or nominal p < 0.05 test, see below.
Key paired findings (vs DDPM-100, same seeds; exact McNemar / Newcombe 95%):

- **PushCube: nothing loses.** Every reduced cell has a positive point estimate except
  DDPM-5 (-1 pt, CI [-13.0, +11.0], p = 1.0); largest compatible losses 2.2-13.0 pp.
  No reduction is *significantly better* either (best: DDIM-5 +9.0 pp, p = 0.093).
  DDIM beats DDPM at equal budget here (+9/+4/-1/+5 pts at 5/10/20/50 steps).
- **PickCube: DDIM degrades.** DDIM-5 = -15.0 pp [-27.6, -1.7], p = 0.024 — the only
  nominally significant loss in the sweep. It does **not** survive the pre-registered
  Bonferroni family (8 tests/task, alpha = 0.00625); DDIM-20 is -12.0 pp (p = 0.081).
  DDPM reductions are indistinguishable from 100 steps (all |diff| <= 4 pp, p >= 0.65).
- **PullCubeTool: noise-dominated.** Point estimates range -8 to +5 pp; none significant
  (all p >= 0.12). DDPM-5/10 happen to score above baseline (+2/+5) but intervals include
  losses up to 7.8/5.2 pp — the "excludes >10 pp" flag is met only because the base rate
  is low, not because 5 steps were shown safe.
- Fastest setting whose 95% interval excludes a loss > 10 pp (pre-registered secondary):
  PushCube **DDIM-5**; PickCube **none**; PullCubeTool **DDPM-5** (see caveat above).

**Reading.** Cutting diffusion sampling steps is effectively free on `PushCube-v1`
(the axis this project measures success-vs-cost on), mildly harmful on `PickCube-v1` for
DDIM specifically, and unresolvable at n = 100 on the weak `PullCubeTool-v1` policy.
No cell's interval excludes losses of ~10 pp at n = 100 except the DDIM-PushCube family —
the honest statement is "no detectable loss on PushCube, largest compatible loss 2.2-13 pp
depending on setting".

### 2.2 frontier — DDIM-best selection recorded BEFORE running (18:32)

Pre-registered rule: fewest DDIM steps whose PushCube 95% interval excludes a > 10 pp
loss = **DDIM-5**. Frontier cells: 4 schedules (fixed:2/4/8, adaptive 0.08/0.10) x
{DDPM-100, DDPM-20, DDIM-5} on seeds 400000-400199 (n = 200; the three fixed:8 cells reuse
2.1 logs for seeds 400000-400099 — identical specification — and add 400100-400199).
Reported inference seconds are DERIVED (mean calls x clean ms/call from 2.3) because the
frontier cells run on parallel workers; raw logged times are also kept and flagged.

### 2.2 execution-horizon x sampler frontier, PushCube-v1, n = 200 (seeds 400000-400199)

Four horizons x three sampler settings; `adaptive` = round-4 primary (short iff
d_tcp < 0.08 or d_goal < 0.10, short 2 / long 8). Cells run on parallel workers, so the
**seconds column is derived** = mean calls x clean ms/call from 2.3 (DDPM-100 144.9,
DDPM-20 28.7, DDIM-5 7.2 ms); raw logged times are kept in `r5_frontier_summary.csv` and
are inflated 1.5-3x by concurrency (marked, not used).

| horizon | DDPM-100 success | s/ep | DDPM-20 success | s/ep | DDIM-5 success | s/ep |
|---|---|---|---|---|---|---|
| adaptive 0.08/0.10 | 87.0 [81.6,91.0] | 12.88 | 87.5 [82.2,91.4] | 2.55 | **88.0 [82.8,91.8]** | **0.65** |
| fixed:2 | 83.0 | 14.49 | 83.0 | 2.87 | 81.5 | 0.72 |
| fixed:4 | 80.5 | 7.25 | 79.5 | 1.44 | 81.5 | 0.36 |
| fixed:8 | 76.0 | 3.62 | 81.5 | 0.72 | 80.5 | 0.18 |

Paired vs DDPM-100 at the same horizon (exact McNemar, Newcombe 95%): **all 12 reductions
are indistinguishable from the 100-step baseline** — every p >= 0.16, largest compatible
losses 2.5-9.0 pp (`r5_frontier_paired.csv`). The frontier is therefore flat in success and
almost two decades cheaper in seconds: e.g. adaptive at 5 DDIM steps (88.0%, 0.65 s) vs
adaptive at 100 DDPM steps (87.0%, 12.88 s) = **20x faster, +1.0 pp (p = 0.86)**.
(The CSV also carries the raw logged `inference_time_s` — 19.8/4.3/0.85 s for the adaptive
cells — which are 1.5-3x inflated by running 9-15 workers concurrently; recorded for
traceability only, not used in any statement above.)

**Combination check (does the scheduling gain survive cheap sampling?)** — adaptive vs
fixed at equal sampler setting (n = 200, our own pairing, not in the pre-registered table):

| setting | vs fixed:2 | vs fixed:4 | vs fixed:8 |
|---|---|---|---|
| DDPM-100 | +4.0 (p=0.23) | +6.5 (p=0.047) | **+11.0 (p=0.00068)** |
| DDPM-20 | +4.5 (p=0.18) | **+8.0 (p=0.0070)** | +6.0 (p=0.043) |
| DDIM-5 | +6.5 (p=0.085) | +6.5 (p=0.047) | **+7.5 (p=0.011)** |

The two axes are **composable**: the adaptive policy keeps a 6.5-7.5 pp advantage over
fixed:8 even at 5 DDIM steps (where the whole episode costs 0.22-0.85 s of policy time).
None of these survive a Bonferroni across the 9 combinations; the DDPM-100 fixed:8 and
DDIM-5 fixed:8 comparisons are the strongest.

### 2.3 clean timing (idle machine, single worker, `OMP_NUM_THREADS=1`, 10 episodes per cell)

| setting | ms/call (mean +- sd) | s/ep fixed:2 | s/ep fixed:8 | speedup vs DDPM-100 |
|---|---|---|---|---|
| DDPM-100 | 144.9 +- 0.3 | 14.50 | 3.62 | 1.0x |
| DDPM-50 | 73.2 +- 0.4 | 7.29 | 1.84 | 2.0x |
| DDPM-20 | 28.7 +- 0.3 | 2.88 | 0.71 | 5.1x |
| DDPM-10 | 14.5 +- 0.2 | 1.46 | 0.36 | 10.0x |
| DDPM-5 | 7.2 +- 0.0 | 0.72 | 0.18 | 20.2x |
| DDIM-50 | 70.7 +- 0.5 | 7.06 | 1.77 | 2.1x |
| DDIM-20 | 28.5 +- 0.1 | 2.84 | 0.71 | 5.1x |
| DDIM-10 | 14.3 +- 0.0 | 1.42 | 0.36 | 10.1x |
| DDIM-5 | 7.2 +- 0.0 | 0.72 | 0.18 | 20.0x |

Cost is linear in the number of sampling steps; DDIM costs the same as DDPM at equal step
count (same number of network evaluations per call). DDPM-100 reproduces the round-4 clean
benchmark (142.6 ms) to within 1.6%. `results/r5_timing_clean.csv`, logs in
`results/timing_r5/`.

---

## Task 3 — training-time variants on PushCube-v1 (Plan A, 30 000 updates per model)

Design (pre-registered in `NOTES.md` 18:20, before any 30k run): single-task `PushCube-v1`,
200 training demos, 30 000 updates, batch 32, lr 1e-4, warmup 500, identical settings for
every model, only the named change varies. Evaluation: our own CPU harness, `fixed:8`,
DDPM-100, **fresh seeds 500000-500199, n = 200 per model** (never the teammate's checkpoint,
never its seeds, never pooled across training runs for significance).

Primary results (success_once, Wilson 95%; calls = mean policy calls per episode):

| variant | change | seed 0 | seed 1 | mean | seed range | calls | horizon | params |
|---|---|---|---|---|---|---|---|---|
| **B0** | baseline | 84.0 [78.3,88.4] | 92.5 [88.0,95.4] | 88.25 | **8.5 pp** | 25.0 | 8.0 | 5.12 M |
| **V1** | observation normalization | **98.5** [95.7,99.5] | **97.5** [94.3,98.9] | **98.00** | **1.0 pp** | 25.0 | 8.0 | 5.12 M |
| **V2a** | pred_horizon 8 (act 7) | 88.5 [83.3,92.2] | 92.0 [87.4,95.0] | 90.25 | 3.5 pp | **29.0** | 7.0 | 5.12 M |
| **V2b** | pred_horizon 32 | 96.0 [92.3,98.0] | 95.0 [91.0,97.3] | 95.50 | 1.0 pp | 25.0 | 8.0 | 5.12 M |
| **V3** | encoder_dim 256 | 90.0 [85.1,93.4] | 84.5 [78.9,88.9] | 87.25 | 5.5 pp | 25.0 | 8.0 | 6.04 M |

Paired against the better baseline seed (B0s1 = 92.5%), exact McNemar on identical eval seeds:

| variant | seed 0 vs B0s1 | seed 1 vs B0s1 | vs B0s0 (both seeds) |
|---|---|---|---|
| V1 | **+6.0 pp [+2.0,+10.6], p = 0.0042** | **+5.0 pp [+0.7,+9.7], p = 0.041** | +14.5 / +13.5 pp, p < 1e-5 |
| V2a | -4.0 pp [-9.9,+1.8], p = 0.22 | -0.5 pp, p = 1.0 | +4.5 / +(-0.5) pp |
| V2b | +3.5 pp [-1.2,+8.4], p = 0.14 | +2.5 pp [-2.4,+7.5], p = 0.38 | +12.0 / +11.0 pp, p < 0.001 |
| V3 | -2.5 pp [-8.2,+3.2], p = 0.38 | **-8.0 pp [-14.4,-1.7], p = 0.0025** | +6.0 / -8.0 pp |

**Pre-registered decision rule outcome.** The rule requires every primary seed to exceed the
better baseline seed by more than the baseline's own seed-to-seed spread **and** to beat it
with p < 0.05. The baseline spread is 8.5 pp (84.0 vs 92.5), so the threshold is **101.0% —
unreachable at n = 200**. Therefore *no* variant is certified, including V1, whose second
seed at 97.5% misses the 101.0% bar by 3.5 pp while beating B0s1 with p = 0.041.
This is a property of the rule, not evidence against the variants: the baseline is so
seed-unstable that the margin it demands exceeds the attainable range. Two honest statements
follow, and both are reported:
- *rule-based*: no training-time change is certified as an improvement;
- *descriptive (2 seeds)*: **V1 and V2b have seed ranges that do not overlap the baseline's**
  (V1: 97.5-98.5 vs B0 84.0-92.5; V2b: 95.0-96.0 vs 84.0-92.5), and V1 is the only variant that
  is *significantly better than both baseline seeds on both training seeds*.

**Secondary seed extension (3-4 seeds per model) — it revises BOTH bullets above.**
This extension was pre-declared as characterisation only; it does not change the primary rule.
It was worth running, because it kills a conclusion that the 2-seed data suggested:

| model | seeds (n = 200 each) | mean | range |
|---|---|---|---|
| B0 (3 seeds) | 84.0 / 89.5 / 92.5 | 88.67 | 8.5 pp |
| V1 (4 seeds) | 91.5 / 97.5 / 98.5 / 99.5 | **96.75** | 8.0 pp |
| V2a (3 seeds) | 88.5 / 92.0 / 93.0 | 91.17 | 4.5 pp |
| V2b (3 seeds) | 88.5 / 95.0 / 96.0 | 93.17 | 7.5 pp |
| V3 (3 seeds) | 84.5 / 90.0 / 97.0 | 90.50 | **12.5 pp** |

Caveat recorded before the results: "the better of two baseline seeds" is itself a maximum
over two runs, which biases the comparison against variants; the spread margin compounds it.

Reading of the extended data (this supersedes the 2-seed "disjoint ranges" bullet):
- With 3-4 seeds, **every variant's range overlaps the baseline's range**, V1 (91.5-99.5 vs
  84.0-92.5) and V2b (88.5-96.0 vs 84.0-92.5) included. Both "disjoint range" statements were
  artefacts of having two seeds; they are withdrawn here explicitly.
- **V3 is the clearest demonstration of the problem**: the same 256-wide encoder produced
  84.5%, 90.0% and 97.0% (range 12.5 pp) — larger than any effect claimed for it.
- What survives every seed count is V1's **mean** shift: 96.75 vs 88.67 = **+8.1 pp**, with the
  worst of four seeds still above the baseline mean, and the pre-registered primary comparison
  (both primary seeds significantly above the better baseline seed) unchanged.
- The substantive Task-3 finding is therefore about the *measurement*: at this training budget
  (30 000 updates, 200 demos, one task) training-seed variance is 8-13 pp, i.e. as large as any
  of the effects being tested, so single- or two-seed training comparisons in this setting
  cannot establish anything; the pre-registered rule (which demanded a 101.0% threshold) is the
  formal expression of that, and the reliable wins in this project are the inference-time ones.
**Inference cost of each training-time change (clean, single worker, idle machine, 10 episodes
each at `fixed:8`, seeds 500000-500009; the training-variant evals above ran on parallel workers
so their logged `inference_time_s` is not usable for this):**

| model | ms per policy call | ref | calls/ep | inference s/ep | ref |
|---|---|---|---|---|---|
| B0 (pred 16) | 145.0 +- 0.6 | 1.00 | 25.0 | 3.62 | 1.00 |
| V1 (obs norm) | 142.8 +- 0.6 | 0.99 | 25.0 | 3.57 | 0.99 |
| V2a (pred 8, act 7) | 137.8 +- 0.3 | 0.95 | **29.0** | **4.00** | **1.10** |
| V2b (pred 32) | **156.0 +- 0.5** | **1.08** | 25.0 | **3.90** | **1.08** |
| V3 (encoder 256) | 148.7 +- 1.3 | 1.03 | 25.0 | 3.72 | 1.03 |

So the two training-time changes with a higher success mean are not free: V2b's longer horizon
costs ~8% more inference per episode, V2a's shorter horizon costs ~10% more (its per-call
saving is more than cancelled by 16% more calls), and the wider encoder ~3%. Only observation
normalization is cost-neutral (within measurement noise), which is what makes it the sensible
default change out of the four tested.

**Reading per axis (final, 3-4 seeds).** Observation normalization (V1) is the best-supported
change on this axis family: **mean 96.75 vs 88.67 (+8.1 pp)** with only four training runs, its
worst seed (91.5%) above the baseline mean, and both primary seeds significantly above the
better baseline seed — while explicitly **not** reducing seed variance (8.0 vs 8.5 pp). A
longer prediction horizon (V2b, 32) has a lower mean (93.17), overlaps the baseline range and
is not individually significant against the better baseline seed, so it is suggestive at best.
A *shorter* horizon
(V2a, 8) buys nothing and replans more often (29 vs 25 calls, +16% inference) because the
legal execution horizon shrinks to 7. Widening the encoder (V3, 128 -> 256, +0.92 M params)
does not help and is significantly *worse* than the better baseline seed on one seed
(-8.0 pp, p = 0.0025) — the encoder-width axis shows no benefit here.

---

## Task 4 — synthesis: which changes to DP help, on this project's evidence

Course question: *can changes to the observation encoder, prediction horizon, action
representation, or diffusion sampling improve success rates or inference speed?* Verified per
change, against an identically-trained baseline where training is involved.

| axis | change tested | where, n | result | verdict |
|---|---|---|---|---|
| **diffusion sampling** | DDPM 100 -> 50/20/10/5 steps | PushCube fixed:8, n=100 | 78/81/83/74% vs 75.0%; all p >= 0.13; largest compatible loss 3.3-13.0 pp | **no detectable loss; 2-20x faster** (144.9 -> 72/29/15/7 ms per call) |
| | DDIM (deterministic, same trained betas) | PushCube fixed:8, n=100 | 83/80/82/84% at 50/20/10/5; all p >= 0.09 | **no detectable loss; equal cost per step** |
| | DDIM on the other tasks | PickCube n=100 / PullCubeTool n=100 | PickCube -9 to -15 pp (5 steps: p = 0.024); PullCubeTool -4 to -8 pp (ns) | **mild harm on PickCube, inconclusive on PullCubeTool** |
| **execution horizon** (earlier rounds) | adaptive replanning vs fixed horizons | PushCube, n=200 | +11.0 pp vs fixed:8 (p = 0.00068) at DDPM-100; +7.5 pp (p = 0.011) at DDIM-5 | **confirmed and composable** with cheap sampling |
| **prediction horizon** | 16 -> 32 (V2b) | PushCube, trained 30k, n=200 x 3 seeds | 88.5/95.0/96.0% (mean 93.17) vs baseline 84.0/89.5/92.5% (mean 88.67); range overlaps the baseline's; +2.5/3.5 pp vs the better baseline seed on the two primary seeds (p = 0.14/0.38) | **suggestive, not established**; costs more per call (longer diffusion sequence) |
| | 16 -> 8, execution clamped to 7 (V2a) | PushCube, trained 30k, n=200 x 3 seeds | 88.5/92.0/93.0% (mean 91.17) vs 88.67 baseline; 29.0 calls vs 25.0 (+16% inference, and a shorter legal execution window) | **no benefit, higher cost** |
| **observation encoder** | observation normalization (V1) | PushCube, trained 30k, n=200 x 3-4 seeds | 91.5/97.5/98.5/99.5% (mean 96.75) vs baseline 84.0/89.5/92.5% (mean 88.67); both primary seeds significantly above the better baseline seed (+5.0/6.0 pp, p = 0.041/0.0042); seed spread 8.0 vs 8.5 pp | **best-supported improvement in the mean (~+8 pp)** (not certifiable by the pre-registered magnitude rule, see Task 3); does not reduce seed variance; no inference cost |
| | encoder width 128 -> 256 (V3) | PushCube, trained 30k, n=200 x 3 seeds | 84.5/90.0/97.0% (mean 90.50) vs 88.67, with the **largest seed spread of any variant (12.5 pp)**; one seed -8.0 pp vs the better baseline seed (p = 0.0025); +0.92 M params | **no benefit; seed noise swamps it** |
| **action representation** | not tested | - | - | outside the light-code-change bound: a different action space (absolute vs delta pose, different rotation parameterisation) needs regenerated demonstrations from the teammate's collection pipeline, a different environment control mode and a different policy output head; the supplied checkpoint only supports the 7-D `pd_ee_delta_pose` space it was trained on |
| **training speed** (enabler) | device/throughput benchmark | PushCube, 200 updates | MPS 0.0191 s/update (30k ~ 0.16 h) vs CPU 0.2012 / 0.2037 s/update in two runs (30k ~ 1.68-1.70 h); checkpoint save 0.1-0.15 s | **gate passed** -> full 30k-per-model plan (Plan A). Train on MPS, **evaluate on CPU** (batch-1 DDPM inference: 3.97 s/ep on CPU vs 8.83 s/ep on MPS) |

**How the two axes combine (the practical headline).** Sampling steps and execution horizon are
independent and compose: adaptive replanning + 5-step DDIM reaches 88.0% at **0.65 s** of policy
time per episode, versus the round-4 reference configuration (adaptive + 100-step DDPM) at 87.0%
and 12.88 s — a **20x inference speedup with no measurable success change**, and still +7.5 pp
over the best fixed horizon at the same sampler setting. On the training side, the analogous
"free" win is normalising the observations (V1), which is architecture-neutral, costs nothing at
inference time and moved the mean from 88.67% to 96.75% (3-4 seeds per model, +8.1 pp) — the only training-time change whose mean shift survives every seed count.

**Caveats that limit every success claim here.** (i) n = 200 per model gives +-3 to +4 pp
resolution at these success levels, so differences below ~5 pp are not resolvable. (ii) The
baseline model's seed-to-seed spread is 8.5 pp and the widest variant spread is 12.5 pp —
larger than most effects measured — so
training-time conclusions are noise-limited by construction; that is why the pre-registered
magnitude rule certifies nothing (threshold 101.0%). (iii) `PullCubeTool-v1` sits at 5-18%
success on this checkpoint, so its sampling results are uninformative at n = 100; conclusions
there need a better policy, not more seeds. (iv) All timing numbers are clean sequential,
single-worker measurements on an otherwise idle machine; the per-episode `inference_time_s`
recorded inside the parallel success runs is inflated 1.5-3x and is shipped only for
traceability.

---

## Task 4b — release folder update (`adaptive-replanning-dp`, 1.1.0)

The packaged repository was updated to carry round 5; **no checkpoints, no demonstration data,
no teammate code and no absolute paths** are included (safety scan re-run over the whole
folder: 0 hits for the teammate handle, `/Users/`, `Desktop/`, secrets or checkpoint files;
the only self-referential mention is the pattern list in `docs/RELEASE_CHECK.md`).

Added
- `scripts/r5_sampling_analysis.py`, `scripts/r5_training_analysis.py`, `scripts/r5_figures.py`,
  `scripts/r5_integrity.py` (comment-free, release-relative paths, all with `--raw-root`).
- `configs/round5_sampling_sweep.json`, `round5_frontier.json`, `round5_timing.json`,
  `round5_training_variants.json`, `model_v2a_pred8.json`, `model_v2b_pred32.json`,
  `model_v3_enc256.json`.
- `results/r5_sampling_{summary,paired}.csv`, `r5_frontier_{summary,paired}.csv`,
  `r5_timing_clean.csv`, `r5_training_{summary,paired}.csv`, the three round-5 figure families
  (PNG + PDF), and `docs/REPORT_ADDENDUM_4.md`.

Changed
- `src/adaptive_replan/envs.py` + `rollout.py` and `scripts/run_rollout.py` + `run_matrix.py`
  gained the round-5 features (DDIM sampler, checkpoints without a `compat` block,
  observation-normalisation application). These were **ported** into the package rather than
  copied, because the release is refactored while the working archive is monolithic.
- `scripts/pack_final_results.py`: nested round-5 phases discovered and packed; block discovery
  shared with the manifest writer.
- `README.md`, `docs/CHANGELOG.md` (1.1.0 section), `docs/RELEASE_CHECK.md` (round-5
  verification record).

Verification of the updated release
- `FINAL_RESULTS.csv`: **114 rows** = the 59 previously shipped rows (all byte-identical, 0
  changed) + **55 round-5 rows** across 28 blocks.
- `MANIFEST.md`: **2 206 raw files / 22 082 episodes**; verified at checksum level against the
  shipped version — **0 of the previous 1 406 checksums missing, exactly 800 new**, and the
  reported total now equals the number of listed rows (the previous file's total of 1 446
  included the 40 excluded `AMBIGUOUS_provenance` files).
- `scripts/verify_results.py --raw-root <working raw>`: **1 459 values, 0 mismatches**.
- `tests/test_core.py`: **15/15** pass.
- Equivalence of the ported code paths against the working implementation (same seeds and
  checkpoints): DDPM-100, DDIM-5 and a normalised-observation checkpoint all reproduce
  per-seed success, call counts and horizons exactly.
- `scripts/r5_integrity.py` over the round-5 raw tree: **1 010 chunk files, 0 problems**
  (this guard exists because a double-launch race tore 7 files mid-round; the affected model
  was deleted and re-run from scratch rather than patched).
- Note for the user's decision (unchanged from the previous round): the release still has **no
  `LICENSE`**, no git remote, and nothing was pushed.
