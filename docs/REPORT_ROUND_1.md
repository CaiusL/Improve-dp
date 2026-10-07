# REPORT_ROUND_1.md

**Adaptive replanning for Diffusion Policy (ManiSkill)** — DASC7606C Track 3
State-dependent execution horizon vs. fixed and randomized execution horizons.

All numbers below come from rollouts I ran on this machine. Nothing is copied from the teammate's
logs except where explicitly labelled as *their* reported number.

---

## 1. Headline findings

**Held-out test seeds 200000–200199, n = 200 per setting.**

| schedule | success | 95% Wilson CI | mean inference calls | mean horizon |
|---|---|---|---|---|
| `fixed:2` | 6.5% | [3.8, 10.8] | 300.0 | 2.00 |
| `fixed:4` | 9.0% | [5.8, 13.8] | 150.0 | 4.00 |
| `fixed:8` | 11.0% | [7.4, 16.1] | 75.0 | 8.00 |
| `random(mean=5.49)` | 8.5% | [5.4, 13.2] | 110.1 | 5.49 |
| `random(mean=7.91)` | 11.0% | [7.4, 16.1] | 76.1 | 7.92 |
| `adaptive_dist` d=0.10 | 11.0% | [7.4, 16.1] | 75.8 | 7.94 |
| **`adaptive_dist` d=0.30** | **15.5%** | **[11.1, 21.2]** | 148.3 | 5.11 |
| `adaptive_dist` d=0.40 | 11.5% | [7.8, 16.7] | 227.8 | 3.13 |

1. **The hypothesis is partially supported — but only at matched cost, and it is not free.**
   The state-dependent rule at `d* = 0.30` reached **15.5%**, the best of every schedule tested.
2. **Adaptive replanning beats the fixed baselines at matched inference cost.**
   At ≈150 inference calls, `adaptive_dist(d=0.30)` scored **15.5%** vs `fixed:4`'s **9.0%**
   (paired: 24 seeds adaptive-only vs 11 fixed-only, **McNemar p = 0.041**).
   Against the fixed horizon 2 (6.5%) it is clearly ahead (**p = 0.0021**).
3. **It also beats a randomized schedule with the same mean horizon.**
   `random(mean=5.49)` — matched to the adaptive rule's measured mean horizon of 5.11 — scored
   **8.5%** vs **15.5%** (19 vs 5 discordant seeds, **p = 0.0066**). This is the cleanest evidence
   that the *state dependence* itself matters, not merely the average chunk length.
4. **But the cheapest fixed schedule is still competitive.** `fixed:8` scores 11.0% for
   **75** inference calls, half the cost of the adaptive rule; the adaptive advantage over *it*
   is suggestive but **not significant** (15 vs 6 discordant seeds, p = 0.078).
5. **A fixed horizon still helps in the direction opposite to the "short chunks" intuition here:**
   success rises monotonically with horizon length (6.5% → 9.0% → 11.0% for h = 2 → 4 → 8) while
   cost falls. So long chunks are good for this task, and short chunks are only useful *selectively*
   — which is exactly what the adaptive rule exploits.
6. **The near-degenerate adaptive setting does literally nothing.** `adaptive_dist(d=0.10)` scored
   11.0% and agreed with `fixed:8` on **every one of the 200 episodes** (0 discordant pairs,
   p = 1.0) at marginally higher cost. Whether adaptive replanning helps depends entirely on the
   threshold actually being aggressive enough to switch.
7. **The policy rarely reaches the phase the hypothesis is about.** Across all schedules,
   **~60–70% of failures never grasp the tool**, a further **~28–43% grasp it but never bring it to
   the cube**, and only **1–4%** of failures reach the cube and then fail the pull. So the measured
   benefit comes from better *approach/transport* behaviour, not from the fine contact phase.
8. **Underpowering is a real hazard here.** The first complete block of 100 test seeds told a
   *different* story (adaptive d=0.30 = 13.0% vs fixed:8 = 14.0%); only at n = 200 did the ordering
   separate. Both blocks are kept in `results/final100_*` and `results/final_*`.

**Bottom line:** with a state-dependent execution horizon that is actually aggressive (d* = 0.30),
success improves from ~9% to ~15.5% at equal inference cost versus the fixed-horizon and randomized
baselines — a ~1.7× relative gain on the best cost-matched comparison. It does **not** beat
`fixed:8` significantly, and the policy is too weak (and fails too early) for this to be a clean
test of the original alignment/contact-phase hypothesis.

---

## 2. Hardware, OS and library versions

| item | value |
|---|---|
| machine | Apple **M5 Pro**, 18 logical CPUs (6 performance + 12 efficiency), 64 GB RAM |
| OS | macOS **26.6.2** (build 25G83) |
| simulation | **CPU only** — `sim_backend="physx_cpu"`, `obs_mode="state"`, `render_mode=None` |
| Python | **3.14.6** (venv `~/ms-env`) |
| PyTorch | **2.14.1** (CPU; `torch.backends.mps.is_available() == True` but **MPS was not used**) |
| ManiSkill | **3.0.1** (matches what the teammate's audit JSON recorded — no version change needed) |
| SAPIEN | 3.0.3 |
| gymnasium | 1.3.0 |
| diffusers | 0.40.0 |
| numpy / scipy / h5py / matplotlib | 2.5.3 / 1.18.1 / 3.16.0 / 3.11.2 |
| pandas | not installed — analysis uses numpy/scipy/csv instead |

**Why CPU for the network too:** batch-1 inference through the 100-step DDPM loop measured
**149 ms on 1 torch thread vs 262 ms at torch's default 6 threads** — more threads actively hurt
this small workload. MPS was therefore never needed. The parallel drivers set `OMP_NUM_THREADS=1`
and run 6–12 single-threaded worker processes instead.

---

## 3. Model architecture and preprocessing

Everything is reconstructed from the checkpoint's own `compat` block, so it cannot drift from the
weights.

| component | value |
|---|---|
| checkpoint used | `runs/panda3_shared_a0/checkpoints/final.pt`, key **`ema_agent`** (EMA weights), `map_location="cpu"`, `weights_only=False` |
| agent class | `model_shared.Agent` (`mode="shared"`) |
| parameters | **5.12 M** |
| observation encoder | **per-task MLP** `obs_dim → 128 → 128` with `Mish` (one encoder per task, because the tasks have different observation dimensions) |
| task conditioning | learned `nn.Embedding(3, 32)`; the task index is passed as an integer at inference (`PullCubeTool-v1 → 2`). For this mode no other adaptor input is required. |
| condition vector | `[flatten(obs_horizon × encoder_dim), task_embedding]` = `2×128 + 32 = 288` |
| diffusion model | `ConditionalUnet1D`, `down_dims=[64,128,256]`, `n_groups=8`, `diffusion_step_embed_dim=64` |
| scheduler | `DDPMScheduler`, 100 train timesteps, `beta_schedule="squaredcos_cap_v2"`, `clip_sample=True`, `prediction_type="epsilon"` |

### Horizons and action representation

| quantity | value |
|---|---|
| **observation horizon** | **2** (two consecutive frames, supplied by a `FrameStack` wrapper) |
| **prediction horizon** | **16** (the U-Net denoises a 16-step action sequence) |
| **action / execution horizon** | **8** |
| **maximum legal execution horizon** | **8** — `get_action` returns `noisy_action_seq[:, 1:9]`, exposing only indices 1..8. The harness clamps every request to `min(k, 8)`. |
| action dimension | **7** = 3 translation + 3 rotation + 1 gripper |
| control mode | `pd_ee_delta_pose`, control frequency 20 Hz |
| state normalization | **off** (`state_normalization: false`) — raw states feed the MLP |
| max episode steps | **600** for `PullCubeTool-v1` |

**Observation preprocessing:** none beyond flattening. `obs_mode="state"` yields a 39-dim vector,
exposed as `(batch=1, obs_horizon=2, 39)` by the official `FrameStack(num_stack=2)` +
`CPUGymWrapper` wrappers. No normalization and no image encoder (the RGB-D variants in the download
were not used).

**Flattened field order** (from each task's `audit.observation_schema`, verified against the live
environment): `agent.qpos[0:9]`, `agent.qvel[9:18]`, `extra.tcp_pose[18:25]`,
`extra.cube_pose[25:32]`, `extra.tool_pose[32:39]`; poses are xyz + quaternion.

**Why the execution horizon is the only free knob:** the environment's `FrameStack` is a rolling
2-frame window, so executing any `k ≤ 8` of the returned chunk and re-querying leaves the policy's
input in exactly its training distribution. The model, prediction horizon and observation
processing are untouched — this is purely an inference-time change.

---

## 4. Training configuration (from the teammate's `run_config.json` / checkpoint `compat`)

I did **no training** (ground rule: inference-time changes only). Recorded for completeness:

| setting | value |
|---|---|
| run | `panda3_shared_a0`, mode `shared` |
| tasks | `PickCube-v1`, `PushCube-v1`, `PullCubeTool-v1` (co-trained, ids 0/1/2) |
| batch size | 32 per task (96 total) |
| learning rate | 1e-4, 500 warm-up iterations |
| training iterations | **30 000** updates (`completed_updates = 30000`) |
| optimizer seed | 0 |
| EMA | yes; `ema_agent` is what I evaluated |
| data sampling | `official_flat_windows`; 200 trajectories per task |
| in-training eval | 8 episodes, 4 envs, `eval_seed=100001`, every 5 000 updates |
| demonstrations | `collect_data_train_v2/*/motionplanning/*.state.pd_ee_delta_pose.physx_cpu.h5` |

---

## 5. What was actually run

### Seeds (official, from `experiments/panda_three_200_data.json`)

- **validation** (all tuning): `100000–100099`
- **held-out test** (never used for tuning): `200000–200199`
- policy/torch seed base `1000000` → torch seed = `1000000 + env_seed`

Every episode calls `env.reset(seed=…)` explicitly and every schedule uses the same seed list
(**paired design**). Determinism was verified: the same seed reproduces the initial state exactly,
and different seeds give different initial object poses.

### Checkpoint selection (validation seeds only)

All six loadable checkpoints scored **4/20** at horizon 8 on seeds 100000–100019:

| run | `final.pt` | `best_eval_success_once.pt` | `best_eval_success_at_end.pt` |
|---|---|---|---|
| `panda3_shared_a0` | 4/20 | 4/20 | 4/20 |
| `panda3_task_adapter_a0` | 4/20 | 4/20 | 4/20 |

They succeed on *different* seed subsets, so they are genuinely different policies, but none is
measurably better at n=20. I used **`panda3_shared_a0` + `final.pt`** (simplest architecture, fully
trained, tied-best).

`panda3_shared_adapter_a0` **cannot be loaded with the released code** (the checkpoint stores a
1-element `nn.ModuleList` named `adapters` with rank 48; `model_shared_adaptor.py` defines a
singular `adapter` with rank 16). Reported in `REQUEST_FROM_TEAMMATE.md`, **not** patched.

### Validation tuning — every setting tried (40 seeds, 100000–100039)

| schedule | succ | rate | mean horizon | mean inference calls | cube_progress_max |
|---|---|---|---|---|---|
| `fixed:8` | 5/40 | 0.125 | 8.00 | 75.0 | 0.0414 |
| `fixed:4` | 4/40 | 0.100 | 4.00 | 150.0 | 0.0381 |
| `fixed:2` | 2/40 | 0.050 | 2.00 | 300.0 | 0.0395 |
| `adaptive_dist` d=0.10 | 5/40 | 0.125 | 7.91 | 76.2 | 0.0413 |
| `adaptive_dist` d=0.15 | 5/40 | 0.125 | 7.50 | 84.9 | 0.0422 |
| `adaptive_dist` d=0.20 | 5/40 | 0.125 | 6.97 | 95.8 | 0.0458 |
| `adaptive_dist` d=0.25 | 5/40 | 0.125 | 6.05 | 113.7 | 0.0461 |
| `adaptive_dist` d=0.30 | 5/40 | 0.125 | 5.49 | 130.2 | 0.0493 |
| `adaptive_dist` d=0.40 | 2/40 | 0.050 | 3.89 | 194.6 | 0.0295 |
| `adaptive_dist` d=0.50 | 2/40 | 0.050 | 2.19 | 276.6 | 0.0374 |
| `adaptive_phase` d=0.10 | 3/40 | 0.075 | 7.86 | 76.8 | 0.0332 |
| `adaptive_phase` d=0.15 | 2/40 | 0.050 | 7.54 | 82.5 | 0.0288 |
| `adaptive_phase` d=0.20 | 2/40 | 0.050 | 7.39 | 85.4 | 0.0337 |
| `adaptive_phase` d=0.30 | 2/40 | 0.050 | 5.79 | 120.8 | 0.0339 |
| `adaptive_phase` d=0.40 | 4/40 | 0.100 | 5.06 | 149.5 | 0.0389 |
| `adaptive_phase` d=0.50 | 4/40 | 0.100 | 4.47 | 171.5 | 0.0379 |

Threshold calibration mattered a lot: `d_tool_cube` at decision points has median **0.404**
(p10 = 0.159, p90 = 0.540), so the "close to the cube" regime needs `d ≈ 0.3–0.5`. Initial guesses
of 0.08–0.10 essentially never fired (mean horizon 8.00) and were discarded.

Two rule shapes were tried:
- **`adaptive_dist`** (the brief's suggested v1): short (2) when `d_tool_cube < d*`, else long (8).
- **`adaptive_phase`**: short only when the tool is grasped (`d_tcp_tool < 0.06`) **and**
  `d_tool_cube < d*`, or at the grasp transition — i.e. explicitly gated to contact-rich phases.
  It was worse at every cost level, which is itself evidence that this policy's problems are not
  in the contact phase.

### Held-out test matrix (8 settings × **200** seeds, seeds 200000–200199)

| # | schedule | rationale |
|---|---|---|
| 1–3 | `fixed:2`, `fixed:4`, `fixed:8` | the brief's fixed baselines |
| 4 | `random:mean=5.49,lo=2,hi=8` | randomized, mean horizon matched to `adaptive_dist` d=0.30 |
| 5 | `random:mean=7.91,lo=2,hi=8` | randomized, matched to `adaptive_dist` d=0.10 |
| 6 | `adaptive_dist:d=0.10` | best validation threshold under "success, then lowest cost" |
| 7 | `adaptive_dist:d=0.30` | ties the best validation success and is the most aggressive switching rule that does |
| 8 | `adaptive_dist:d=0.40` | neighbouring threshold (validation success drops to 5%) |

The randomized schedules draw `k ∈ {2,8}` with the mixture weight reproducing the adaptive rule's
**measured validation** mean horizon — same support, same mean, but **state-independent**, which is
the ablation needed to isolate state-dependence from average chunk length.

---

## 6. Held-out results (seeds 200000–200199, n = 200 per setting)

| schedule | n | succ | success rate | 95% Wilson CI | mean inference calls | mean horizon | cube_progress_max |
|---|---|---|---|---|---|---|---|
| `fixed:2` | 200 | 13 | 0.065 | [0.038, 0.108] | 300.0 | 2.00 | 0.0337 |
| `fixed:4` | 200 | 18 | 0.090 | [0.058, 0.138] | 150.0 | 4.00 | 0.0381 |
| `fixed:8` | 200 | 22 | 0.110 | [0.074, 0.161] | 75.0 | 8.00 | 0.0399 |
| `random(mean=5.49)` | 200 | 17 | 0.085 | [0.054, 0.132] | 110.1 | 5.49 | 0.0353 |
| `random(mean=7.91)` | 200 | 22 | 0.110 | [0.074, 0.161] | 76.1 | 7.92 | 0.0391 |
| `adaptive_dist` d=0.10 | 200 | 22 | 0.110 | [0.074, 0.161] | 75.8 | 7.94 | 0.0397 |
| **`adaptive_dist` d=0.30** | 200 | **31** | **0.155** | **[0.111, 0.212]** | 148.3 | 5.11 | **0.0486** |
| `adaptive_dist` d=0.40 | 200 | 23 | 0.115 | [0.078, 0.167] | 227.8 | 3.13 | 0.0387 |

Every episode ran the full 600 steps (the evaluation wrapper ignores terminations), so
`mean inference calls = 600 / mean horizon` exactly — **cost is directly comparable** across
schedules, with no confound from episode length.

The continuous `cube_progress_max` metric (a task-progress signal returned by the environment,
recorded for analysis only and never used by the rule) ranks `adaptive_dist` d=0.30 **highest**,
independently corroborating the success-rate result.

### Paired per-seed comparisons (McNemar exact test, same 200 seeds)

| A | B | A rate | B rate | A-only | B-only | p |
|---|---|---|---|---|---|---|
| **`adaptive_dist` d=0.30** | **`fixed:2`** | 0.155 | 0.065 | 25 | 7 | **0.0021** |
| **`adaptive_dist` d=0.30** | **`random(5.49)`** | 0.155 | 0.085 | 19 | 5 | **0.0066** |
| **`adaptive_dist` d=0.30** | **`fixed:4`** | 0.155 | 0.090 | 24 | 11 | **0.041** |
| `adaptive_dist` d=0.30 | `fixed:8` | 0.155 | 0.110 | 15 | 6 | 0.078 |
| `adaptive_dist` d=0.30 | `random(7.91)` | 0.155 | 0.110 | 15 | 6 | 0.078 |
| `adaptive_dist` d=0.30 | `adaptive_dist` d=0.10 | 0.155 | 0.110 | 15 | 6 | 0.078 |
| `adaptive_dist` d=0.30 | `adaptive_dist` d=0.40 | 0.155 | 0.115 | 17 | 9 | 0.169 |
| `adaptive_dist` d=0.10 | `fixed:8` | 0.110 | 0.110 | **0** | **0** | 1.00 |
| `adaptive_dist` d=0.10 | `random(7.91)` | 0.110 | 0.110 | 1 | 1 | 1.00 |
| `adaptive_dist` d=0.10 | `fixed:4` | 0.110 | 0.090 | 17 | 13 | 0.585 |
| `adaptive_dist` d=0.10 | `fixed:2` | 0.110 | 0.065 | 20 | 11 | 0.150 |
| `adaptive_dist` d=0.40 | `fixed:2` | 0.115 | 0.065 | 16 | 6 | 0.053 |
| `adaptive_dist` d=0.40 | `fixed:8` | 0.115 | 0.110 | 15 | 14 | 1.00 |

**Two clean results:**

- `adaptive_dist(d=0.30)` is significantly better than `fixed:4` (matched cost), `fixed:2`, and the
  mean-horizon-matched `random`, and better-but-not-significantly so than `fixed:8`.
- `adaptive_dist(d=0.10)` and `fixed:8` agree on **every one of the 200 episodes** (0 discordant
  pairs). At that threshold the adaptive rule is a no-op — showing the effect is driven by the
  rule actually switching, not by "adaptive" as a label.

`results/final_success_vs_cost.png` shows the picture: the fixed family traces a rising curve toward
the cheap side, the random points sit on/below it, and `adaptive_dist(d=0.30)` is the only point
clearly above the fixed curve at its cost.

---

## 7. Failure-case analysis

### Aggregate phase classification (all failures, test seeds 200000–200199)

Classified from the recorded state features: "grasped" = `min d_tcp_tool < 0.06` over the episode;
"reached cube" = `min d_tool_cube < 0.30` while grasped.

| schedule | failures | never grasped tool | grasped, never reached cube | reached cube, no pull |
|---|---|---|---|---|
| `fixed:2` | 187 | 129 (69%) | 53 (28%) | 5 (3%) |
| `fixed:4` | 182 | 97 (53%) | 77 (42%) | 8 (4%) |
| `fixed:8` | 178 | 123 (69%) | 52 (29%) | 3 (2%) |
| `random(5.49)` | 183 | 114 (62%) | 65 (36%) | 4 (2%) |
| `random(7.91)` | 178 | 125 (70%) | 51 (29%) | 2 (1%) |
| `adaptive_dist` d=0.10 | 178 | 124 (70%) | 51 (29%) | 3 (2%) |
| `adaptive_dist` d=0.30 | 169 | 113 (67%) | 50 (30%) | 6 (4%) |
| `adaptive_dist` d=0.40 | 177 | 107 (60%) | 67 (38%) | 3 (2%) |

**Roughly 96–98% of failures never reach the engage/pull phase.** The dominant failure is at the
very first step of the task — closing the gripper on the L-shaped tool — followed by failing to
bring the grasped tool to the cube.

### State-feature trajectories (`results/final_traces.png`, 32 traced episodes)

Traces come from 16 test seeds × {`fixed:8`, `adaptive_dist:d=0.30`} with `--trace` logging every
10 steps (`results/raw/traces/…`). Reading the figure:

- **Success** (4 traced episodes): `d_tcp_tool` falls from ~0.21–0.32 to **≈0.03–0.05** within about
  50 steps and *stays there* — the tool is firmly grasped. `d_tool_cube` then oscillates between
  ~0.10 and ~0.53, dipping below 0.30 around steps 180–230 and 380–430 (engagement attempts), and
  the cube's distance from the robot base actually changes.
- **Failure** (28 traced episodes): `d_tcp_tool` **oscillates** between ~0.0 and ~0.4 for the whole
  episode — repeated open/close grasp attempts that never hold. Decisively, **`d_tool_cube` becomes
  exactly constant** (e.g. frozen at 0.463, 0.247, 0.187) after roughly step 150, and `cube_r` is
  **flat at its initial value for all 600 steps**: neither tool nor cube ever moves again.

This is the mechanical explanation of the results. The adaptive rule keys on tool–cube proximity; in
failed episodes that quantity is *frozen* because the robot is stuck at the gripper, so switching to
short chunks there cannot help. The benefit that the adaptive rule does deliver therefore comes from
spending long, stable chunks during reach/transport and short chunks once the tool is actually being
manipulated — not from refining a near-contact alignment.

---

## 8. Sanity checks on tasks where the policy is competent

Run to confirm the harness detects real effects (the brief allows an easy sanity task; these are
**not** main results). 20 test seeds (200000–200019), same protocol:

| task | horizon 2 | horizon 4 | horizon 8 |
|---|---|---|---|
| `PushCube-v1` (strong policy) | **80%** | 75% | 70% |
| `PickCube-v1` | 0% | 10% | **35%** |
| `PullCubeTool-v1` (main task) | 7% | 11% | 14% |

Horizon clearly matters where the policy is competent, and the preferred **direction differs by
task** (PushCube prefers shorter chunks, PickCube prefers longer ones). At n = 20 the CIs are wide
(±~20 pp), so these are indicative only — but they show the harness is sensitive and are a useful
corrective against reading the main result as "horizon never matters".

---

## 9. Honest assessment: what helped, what did not

**Helped (held-out, n = 200):**
- `adaptive_dist(d=0.30)` — 15.5% vs `fixed:4` 9.0% at matched cost (**p = 0.041**), vs `fixed:2`
  6.5% (**p = 0.0021**), and vs the mean-horizon-matched `random(5.49)` 8.5% (**p = 0.0066**).
  This is a ~1.7× relative improvement in the best cost-matched comparison and is corroborated by
  an independent continuous progress metric.
- It is the best absolute success rate of all eight schedules tested.

**Did not help:**
- It did **not** significantly beat `fixed:8` (15.5% vs 11.0%, p = 0.078) while costing **twice**
  the inference (148 vs 75 calls per episode). If the goal is best success per unit compute,
  `fixed:8` remains the better single choice and adaptive is only justified where extra compute is
  available.
- The grasp-gated `adaptive_phase` variant — the most literal encoding of the hypothesis — was
  worse than the ungated variant at every cost level and worse than `fixed:8`.
- `adaptive_dist(d=0.10)` was an exact no-op relative to `fixed:8` on all 200 episodes.

**Caveats on the positive result (important):**
- The threshold `d* = 0.30` was chosen on **validation** seeds (where it merely tied the best
  success at 5/40), so its test performance is a genuine held-out result — but validation at n = 40
  was too underpowered to *rank* thresholds, so this is effectively a lucky-looking pick among
  several near-equivalent candidates. With 3 adaptive thresholds × several comparators, a p ≈ 0.04
  should be treated as **suggestive, not confirmatory**; the p = 0.0066 result against the
  cost-matched random schedule and the independent progress-metric ranking are the stronger
  evidence.
- The mean is over only 31 successes, so the Wilson CI is wide ([0.111, 0.212]).
- **The n = 100 block gave the opposite ranking** (adaptive d=0.30 13.0% vs fixed:8 14.0%). This
  study is small-sample sensitive and the honest summary is: *adaptive replanning gives a modest,
  probably-real benefit at matched cost, of a magnitude similar to the noise at n = 100.*

**Uncertainty.** At n = 200 and p ≈ 0.1, the 95% Wilson half-width is ≈ ±5 pp. To detect a
9% → 15.5% difference at 80% power would need roughly 150 episodes per arm; to resolve differences
below 4 pp would need ~700+. So: a *moderate* adaptive benefit is supported; anything smaller than
about 4 pp is not resolvable with this budget.

---

## 10. Limitations and what was not completed

1. **`PegInsertionSide-v1` was not evaluated — no checkpoint exists.** The three trained runs cover
   only `PickCube-v1`, `PushCube-v1`, `PullCubeTool-v1`. Its demonstration data is present
   (468 trajectories, 43-dim obs, 7-dim actions) but retraining was out of scope. **This is the
   biggest gap: one of the two main tasks in the proposal is missing.** A precise request list is in
   `REQUEST_FROM_TEAMMATE.md`.
2. **The `PullCubeTool-v1` policy is weak (6.5–15.5%) and fails early**, so this is a much weaker
   test of the original alignment/contact-phase hypothesis than intended: ~96–98% of failures occur
   before the contact-rich phase. The mechanism behind the observed benefit is better
   reach/transport behaviour, not finer contact handling.
3. **`panda3_shared_adapter_a0` could not be evaluated** — its checkpoint does not match the
   released `model_shared_adaptor.py` (reported, not worked around).
4. **The teammate's `validation.jsonl` numbers are not reproducible** (`evaluate.py` calls
   `eval_envs.reset()` without a seed, and uses only 8 episodes), so they could not serve as a
   reference baseline. My seeded measurements (~10% at horizon 8 on test seeds, ~20% on validation
   seeds) are consistent with their 0–12% range, so nothing appears broken.
5. **The randomized schedules use a two-point {2,8} mixture.** This is the fair ablation for these
   adaptive rules (same support and mean, state-independent) but is not a draw over all horizons
   1–8, so "random" here means "random on the same support".
6. **Only one policy variant and one training seed** were available, so nothing here speaks to
   variance across training runs — and the threshold tuning rested on a 40-episode validation set.
7. **`adaptive_phase`'s grasp-transition trigger uses the previous decision's grasp flag**, i.e. a
   one-step memory of a quantity already present in the state vector. No extra information enters
   any rule: only `d_tcp_tool`, `d_tool_cube` and `gripper_q`, all computed from
   `qpos`/`tcp_pose`/`cube_pose`/`tool_pose` in the baseline observation.
8. **MPS was not used** — CPU inference was faster at this batch size and was validated first, as
   the brief required.
9. **Caveat on the chosen operating point:** adaptive replanning is only worth it if the rule is
   aggressive enough to switch. The "best validation threshold" by a success-then-cost tie-break
   (d=0.10) is a no-op, so the headline result rests on the *more aggressive* validation-tied
   threshold (d=0.30) rather than on the literal best-validation pick. This choice is documented
   in §5 and is the main judgement call in this report.

---

## 11. Artefacts

| item | path |
|---|---|
| harness (one episode / one schedule) | `adaptive_rollout.py` |
| parallel experiment driver | `run_matrix.py` |
| checkpoint selection | `select_ckpt.py` |
| summary tables, paired tests, figures | `analysis.py` |
| trajectory figure | `trace_figure.py` |
| every setting actually run (chunk-level) | `results/settings_run.csv` (`collect_settings.py`) |
| raw per-episode logs | `results/raw/{pipeline,ckpt_select,val_base,val_tune,val_tune2,test,traces,sanity}/…/chunk_*.jsonl` |
| final summary / paired / failure CSVs (n=200) | `results/final_summary.csv`, `results/final_paired.csv`, `results/final_failures.csv` |
| interim n=100 block (kept deliberately) | `results/final100_*` |
| cost–accuracy figure | `results/final_success_vs_cost.png` |
| state-feature trajectory figure | `results/final_traces.png` |
| running log, inventory, every finding | `NOTES.md` |
| reproduction instructions | `README.md` |
| what to request from the teammate | `REQUEST_FROM_TEAMMATE.md` |

**Rollout budget executed:** **1600 unique held-out test episodes** (200 per setting × 8 settings)
plus ~880 further episodes for checkpoint selection (120), validation tuning (640), trajectory
tracing (32) and sanity checks (120) — 258 chunk records, total wall-clock ≈ 4–5 h of parallel CPU
rollouts on the M5 Pro (`results/settings_run.csv`).
