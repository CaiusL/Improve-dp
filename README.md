# Adaptive Replanning for Diffusion Policies on ManiSkill

**Question.** At similar inference cost, does a *state-dependent execution horizon* — how many
predicted actions are executed before the policy is queried again — improve Diffusion Policy (DP)
success over *fixed* execution horizons and *randomized* schedules?

The trained policy and its prediction horizon are never modified. Only the number of executed
actions per decision is varied, which is a pure inference-time change.

**Answer in one paragraph.** The effect depends on the task, in both directions. On
`PushCube-v1` a hand-designed geometric rule improved success from 79.5% (best fixed) to 91.0%
(p < 0.0001) on 200 held-out seeds, and beat randomized schedules with the same mean horizon
(80.0%); the replication on a second, untouched block of 200 seeds succeeded (89.0% vs 80.0% /
78.5%, both p < 0.002). On `PickCube-v1` the same style of rule is significantly *worse* than the
best fixed schedule (25.5% vs 36.0%, p < 0.001) while costing 2.4× more inference calls. On
`PullCubeTool-v1` the effect is small and did not replicate on fresh seeds. A control experiment
on `PushCube-v1` shows that *which* states receive short chunks matters (an inverted-rule control
falls back to the fixed-horizon level), but a simple "be short after step 10–30" time window
recovers most of the gain and is statistically indistinguishable from the geometric rule on the
fresh block (p = 0.25 / 0.50), so the geometric rule's edge over a well-placed schedule is modest.
Reducing DDPM sampling from 100 to 20 steps gives a 4.9× policy-inference speed-up with no
detectable success loss at n = 100 per cell (largest loss compatible with the data: 8–12 pp).

| Task | Best fixed | Best adaptive | Difference (95% CI) | p | Adaptive cost |
|---|---|---|---|---|---|
| `PushCube-v1` (block 1) | 79.5% @ 50 calls | **91.0%** @ 89 calls | **+11.5 pp** (+4.6, +18.4) | 0.00012 | 1.78× best fixed |
| `PushCube-v1` (fresh block) | 80.0% @ 50 calls | **89.0%** @ 89 calls | **+9.0 pp** (+1.9, +16.1) | 0.0014 | 1.78× best fixed |
| `PickCube-v1` | **36.0%** @ 25 calls | 25.5% @ 59 calls | **−10.5 pp** (−19.3, −1.5) | 0.00075 | 2.36× best fixed |
| `PullCubeTool-v1` | 11.0% @ 75 calls | 13.8% @ 152 calls | +2.8 pp (−1.8, +7.4) | 0.052 | 2.03× best fixed |

**Headline caveats.** (1) The `PickCube-v1` success condition requires the robot to be static, so
`success_once` and `success_at_end` diverge on 16% of episodes; the adaptive rule loses under both.
(2) The `PushCube-v1` gain was confirmed on a second, fresh block of 200 seeds (see
`docs/REPORT_ADDENDUM_3.md` for both blocks). (3) Reduced DDPM sampling (100 → 20 steps) speeds up
policy inference ~5× with no loss detected at n = 100 per cell, but that experiment is not powered
to exclude a modest loss; see the addendum for the interval on the largest compatible loss.

Full detail: `docs/REPORT_NOTES.md`, `docs/REPORT_ADDENDUM.md`, `docs/REPORT_ADDENDUM_2.md`,
`docs/REPORT_ADDENDUM_3.md`, `docs/REPORT_ADDENDUM_4.md` (sampler x horizon frontier, clean
timing, training variants), `docs/REPORT_ADDENDUM_5.md` (run-level tests, adaptive transfer, the
four-way combination) and `docs/REPORT_ADDENDUM_6.md` (round-7 parity on `PickCube-v1` and
`PullCubeTool-v1`), with `docs/REPORT_ADDENDUM_5_FIXES.md`, the verification report
`docs/STAGE_REPORT_v2_CHECK.md` and the change log `docs/CHANGELOG.md` — **this archive is
release 1.3.5**. Pre-registrations are in `configs/`; the internal working log (`NOTES.md`) and
the earlier stage reports are not part of this release.

## What a "round" is, and what each round did

A **round** is one attempt to *improve* DP or to *test* a claim about it — not a release, and not
a batch of runs. Each round fixes its change or question in advance, pre-registers the settings in
`configs/` **before** the first rollout or training run, evaluates on held-out seeds, and reports
the outcome whether or not it supports the change: round 3's `PickCube-v1` study found the adaptive
rule *worse* than the best fixed schedule, and that is the round's result. Rounds are additive — a
later round re-verifies an earlier one and never rewrites its raw logs, so `results/raw/<phase>/`
holds every round side by side.

| round | what it tried or tested | outcome | report | pre-registered in |
|---|---|---|---|---|
| 1 | Does a state-dependent execution horizon beat fixed and randomized horizons? First test, on `PullCubeTool-v1`: 8 schedules × 200 held-out seeds. | Partially supported. The aggressive rule (`adaptive_dist` d=0.30) was best at 15.5%, beating cost-matched `fixed:4` (p = 0.041) and a mean-matched random schedule (p = 0.0066), but not `fixed:8` (p = 0.078). | `REPORT_NOTES.md` | `round1_pullcubetool_test.json` |
| 2 | Confirmatory block on fresh seeds, a competent-policy task (`PushCube-v1`), a negative control rule keyed on an unrelated feature, and fact-checks. | Not confirmed. On fresh seeds P1 (adaptive vs `fixed:4`) gave p = 0.265, so the pre-registered overall rule failed; P2 (no significant loss vs `fixed:8`) held at p = 0.688. The unrelated-feature control was statistically indistinguishable from the adaptive rule (p ≥ 0.05), as a negative control should be. | `REPORT_ADDENDUM.md` | `round2_pullcubetool_confirm.json` |
| 3 | Full `PickCube-v1` study, `PushCube-v1` control experiments, cross-task analysis, first reduced-sampling-step measurements. | The decision reverses by task: on `PickCube-v1` the rule is significantly worse than `fixed:8` (25.5% vs 36.0%, p = 0.0008) at more than twice the cost; the inverted-rule control shows that *which* states get short chunks matters. | `REPORT_ADDENDUM_2.md` | `round3_pickcube_test.json`, `round3_pushcube_test.json` |
| 4 | Fresh-seed confirmation on `PushCube-v1` with a time-window search, reduced sampling steps with a clean sequential timing benchmark, exploratory `PickCube-v1` stillness check. | Confirmed. Adaptive 89.0% vs 80.0% for the best fixed (both p < 0.002), and a well-placed time window is indistinguishable from the geometric rule (p = 0.25 / 0.50); 20 steps is 4.9× faster with no loss detected (losses up to ~8–12 pp are not excluded). | `REPORT_ADDENDUM_3.md` | `round4_*.json` |
| 5 | The course question "Improving DP" beyond the execution horizon: a training-speed benchmark, a diffusion-sampling sweep including DDIM, training-time variants (prediction horizon, encoder size, observation normalization) on `PushCube-v1`, an execution-horizon × sampler frontier, clean timing. | Observation normalization (V1) is the best-supported change (mean 88.67% → 96.75%), though the pre-registered magnitude rule was not met; DDPM 100 → 20 steps is ~4.9× faster. | `REPORT_ADDENDUM_4.md` | `round5_*.json`, `model_v*.json` |
| 6 | Run-level statistics (the **training run**, not the episode, as the unit of analysis) for observation normalization and adaptive replanning; the normalization × adaptive × 5-step-DDIM combination; a descriptive co-training comparison. | Normalization as a training-time change is not supported on `PickCube-v1` (−20.25 pp, p = 0.057) and supported on `PullCubeTool-v1` (+15.82 pp, p = 0.0286). | `REPORT_ADDENDUM_5.md`, `REPORT_ADDENDUM_5_FIXES.md` | `round6_*.json` |
| 7 | Parity for `PickCube-v1` and `PullCubeTool-v1`: sampler × horizon frontiers on the co-trained checkpoint with fresh seed blocks, encoder/prediction-horizon variants on the other two tasks, the completed four-way combination, our own co-trained models against single-task runs, failure analysis, and a consistency pass over round 6. | Establishes the per-task best combinations — `PushCube-v1` V1 + `fixed:8` + DDIM-5 (96.83% / 91.25%), `PickCube-v1` B0 + `fixed:8` + DDIM-5 (63.13% / 44.25% at 0.18 s), `PullCubeTool-v1` V1 + `fixed:8` + DDPM-100 (27.50% / 27.38%). | `REPORT_ADDENDUM_6.md` | `round7_*.json` |

Three numbering conventions are worth knowing, because they make the archive look more
round-5-to-7-centric than it is:

- **Addendum numbers are offset from round numbers.** `docs/REPORT_NOTES.md` is round 1, then
  `REPORT_ADDENDUM.md` is round 2, `_2` is round 3, and so on to `_6` for round 7 — addendum *N*
  reports round *N + 1*.
- **`rN_` marks the code a round introduced, not the data it used.** `scripts/r5_*` … `scripts/r7_*`
  and `results/r5_*` … `results/r7_*` exist because those rounds needed a new method (sampler-sweep
  analysis, run-level exact tests, parity and failure classification). Rounds 1–4 needed no such
  code: they were answered with the generic toolkit (`run_rollout`, `run_matrix`,
  `analyze_results`, `make_figures`, `pack_final_results`, `verify_results`, `failure_analysis`,
  `select_checkpoint`), and the ad-hoc round-3 analysis scripts were consolidated into those at
  packaging time (see `CHANGELOG.md` 1.0.0). Hence there are no `r1_*`–`r4_*` scripts, and a
  round's data can predate the scripts named after its number.
- **Rounds group into releases.** Rounds 1–4 shipped together as 1.0.0, round 5 as 1.1.0, round 6 as
  1.2.0 and round 7 as 1.3.0; 1.3.1–1.3.5 are documentation and correction patches on round-7
  content. The round-1–4 rows live in `results/FINAL_RESULTS.csv` next to the later rounds' (59 of
  its 273 rows) rather than in separate `rN_*` tables; `docs/RELEASE_CHECK.md` §5 keeps each
  round's verification numbers as they stood when that round shipped.

## Policy and external dependencies

The policy is a **teammate-supplied, three-task co-trained Diffusion Policy** (`panda3_shared_a0`,
`final.pt`, EMA weights, 5.19 M parameters): `ConditionalUnet1D` denoiser, per-task MLP observation
encoder, learned task embedding. Checkpoints, the demonstration data and the teammate's model code
are **not redistributed here**. To run the rollouts you need:

- a directory containing the training runs, each with `checkpoints/` and `run_config.json`;
- the teammate's Python package directory (containing `model_shared.py` and the
  `diffusion_policy` package).

Point the scripts at them with `--runs-root` / `--teammate-code`, or set `DP_RUNS_ROOT` /
`DP_TEAMMATE_CODE`.

```bash
export DP_RUNS_ROOT=/path/to/training_runs
export DP_TEAMMATE_CODE=/path/to/teammate_code
```

## Environment

The experiments ran on macOS with CPU-only simulation:

| Package | Version |
|---|---|
| Python | 3.14.6 |
| torch | 2.14.1 (CPU; MPS available but not used) |
| ManiSkill | 3.0.1 |
| SAPIEN | 3.0.3 |
| gymnasium | 1.3.0 |
| diffusers | 0.40.0 |
| numpy / scipy / matplotlib | 2.5.3 / 1.18.1 / 3.11.2 |

Episodes use `sim_backend="physx_cpu"`, `obs_mode="state"`, `render_mode=None`, `num_envs=1`, and
`control_mode="pd_ee_delta_pose"`. Batch-1 diffusion inference is fastest with a single torch
thread, so the drivers set `OMP_NUM_THREADS=1` and run many worker processes instead.

## Quick start (smoke test)

```bash
pip install -r requirements.txt
export DP_RUNS_ROOT=/path/to/training_runs
export DP_TEAMMATE_CODE=/path/to/teammate_code

python scripts/run_rollout.py --task PushCube-v1 --schedule fixed:8 \
    --seeds 300000-300004 --out /tmp/smoke.jsonl
```

Five episodes, ~1–2 minutes on a laptop CPU. Each line of the output holds the seed, successes,
inference-call count and wall time.

## Reproducing the results

Every table and figure is rebuilt from the per-episode logs. **The raw logs (~248 MB across
5 392 JSONL files, 53 942 episodes) are intentionally not part of this release** — they
exceed the size the repository is meant to carry. `results/MANIFEST.md` lists every raw file with its episode count,
seed range and SHA-256 checksum, so a complete copy can be verified file by file; the summary
CSVs in `results/` are shipped in their place. Point `--raw-root` at a directory holding the
raw logs to reproduce everything below. Set `RAW` to that directory before running anything (a
full checkout keeps them in `results/raw`); in `run_matrix.py` the same flag names the directory
new logs are **written** to, so the generation examples below write under `$RAW`, which is where
the analyses then read from. The pre-registered settings for each phase are in `configs/`.

```bash
RAW=/path/to/raw_logs        # per-episode logs; a full checkout keeps them in results/raw

# final per-setting table and manifest
python scripts/pack_final_results.py --raw-root "$RAW" --with-manifest

# round 5: sampler x steps sweep (DDIM is rebuilt from the checkpoint's trained betas)
python scripts/run_matrix.py --phase ddpm20 --tasks "PushCube-v1,PickCube-v1,PullCubeTool-v1" \
    --schedules "fixed:8" --seeds 400000-400099 --chunk 10 --jobs 8 \
    --diffusion-steps 20 --raw-root "$RAW/sampling_sweep"
python scripts/run_matrix.py --phase ddim5 --tasks "PushCube-v1,PickCube-v1,PullCubeTool-v1" \
    --schedules "fixed:8" --seeds 400000-400099 --chunk 10 --jobs 8 \
    --sampler ddim --diffusion-steps 5 --raw-root "$RAW/sampling_sweep"
python scripts/r5_sampling_analysis.py \
    --phases ddpm100,ddpm50,ddpm20,ddpm10,ddpm5,ddim50,ddim20,ddim10,ddim5 \
    --schedule "fixed(k=8)" --raw-root "$RAW" --out results/r5_sampling

# round 5: clean inference timing (single worker, otherwise idle machine)
python scripts/run_rollout.py --task PushCube-v1 --schedule fixed:8 --seeds 400000-400009 \
    --sampler ddim --diffusion-steps 5 --out results/timing_r5/ddim5_fixed8.jsonl

# round 5: raw-log integrity guard before analysing
python scripts/r5_integrity.py --raw-root "$RAW"

# round 5: training-variant evaluation analysis and figures
python scripts/r5_training_analysis.py --raw-root "$RAW"
python scripts/r5_figures.py

# round 7: parity across the three tasks (frontiers on the co-trained checkpoint, variants on
# the other two tasks, the completed combination, our own co-trained models, failure analysis)
python scripts/r7_runlevel.py --raw-root "$RAW"
python scripts/r7_tests.py --raw-root "$RAW"
python scripts/r7_figures.py
python scripts/r7_parity_figure.py

# round 7 diagnostic: per-dimension statistics of the training observations V1 normalizes
python scripts/r7_p0_dimstats.py --manifest PickCube-v1=<data manifest>.json

# round 7 failure classification over episodes logged with --decision-log
python scripts/r7_failure_classify.py --spec <list of {root, task, label}>.json

# round 6: run-level tests of observation normalization, adaptive replanning and the
# normalization x adaptive x 5-step-DDIM combination (the training run is the unit of analysis)
python scripts/r6_runlevel.py --raw-root "$RAW"
python scripts/r6_figures.py

# round 6: verify that the adaptive rule read the raw observation (log one episode per schedule
# on a normalized checkpoint with --decision-log --feature-check, then:)
python scripts/r6_verify_rawfeat.py --fixed-jsonl <fixed_episode.jsonl> \
    --adaptive-jsonl <adaptive_episode.jsonl>

# per-task summary plus paired adaptive-vs-baseline tests
# (--include-adaptive-pairs also pairs the adaptive/time-window rules against each other)
python scripts/analyze_results.py --phases pushcube_fresh --task PushCube-v1 \
    --seeds 300000-300199 --raw-root "$RAW" --out results/pushcube_fresh --include-adaptive-pairs

# figures (per task and combined three-panel)
python scripts/make_figures.py --raw-root "$RAW"

# recompute the summary CSV from the raw logs and check every value
python scripts/verify_results.py --raw-root "$RAW"

# a specific experiment block (example: the fresh confirmation block)
python scripts/run_matrix.py --phase pushcube_fresh --tasks PushCube-v1 \
    --schedules "fixed:2;fixed:4;fixed:8;random:mean=2.25,lo=2,hi=8;adaptive_push:d_tcp=0.08,d_goal=0.10;adaptive_push:d_tcp=0.05,d_goal=0;adaptive_time:start=10,length=170;adaptive_time:start=30,length=170" \
    --seeds 300000-300199 --chunk 10 --jobs 8 --raw-root "$RAW"

# reduced DDPM sampling steps
python scripts/run_matrix.py --phase pushcube_steps_fresh_20 --tasks PushCube-v1 \
    --schedules "fixed:2;fixed:4;fixed:8;adaptive_push:d_tcp=0.08,d_goal=0.10" \
    --seeds 300000-300099 --chunk 10 --jobs 8 --diffusion-steps 20 --raw-root "$RAW"
```

`python <script> --help` lists all arguments for each script; `--tasks` is comma-separated and
`--schedules` is semicolon-separated.


### Run-level statistics (round 6)

Training-time comparisons never pool episodes across training runs. Each run contributes one
success rate over 200 episodes, and comparisons use exact run-level tests:

- `runlevel.permutation_test(a, b)` — exact two-sided permutation test over all
  `C(n1+n2, n1)` label assignments (6 vs 6 runs = 924; 4 vs 4 = 70, smallest attainable
  two-sided p = 2/70 = 0.0286);
- `runlevel.signflip_test(d)` — exact sign-flip test over all `2**n` sign patterns for paired
  per-run differences;
- `runlevel.bootstrap_diff_ci(a, b)` — percentile bootstrap interval for the difference in
  means (20 000 resamples, fixed seed).

Policy time per episode in round 6 is **derived** as mean policy calls x the round-5 clean
ms/call (DDPM-100 145.0 ms, DDPM-100 with observation normalization 142.8 ms, DDIM-5 7.2 ms);
no new timing benchmark was taken.

## Repository layout

```
adaptive-replanning-dp/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                 pre-registered settings for each experiment phase
├── docs/                    reports, addenda, verification and release checks
├── results/                 summary CSVs, figures, MANIFEST.md
│   └── raw/                 placeholder — raw logs are kept out (see MANIFEST.md)
├── scripts/
│   ├── run_rollout.py       one schedule for one seed list
│   ├── run_matrix.py        parallel, resumable driver over a matrix
│   ├── analyze_results.py   summaries and paired tests
│   ├── make_figures.py      success-versus-cost figures
│   ├── pack_final_results.py  FINAL_RESULTS.csv and MANIFEST.md
│   ├── verify_results.py    recompute and check the released CSV
│   ├── failure_analysis.py  task-specific failure-phase classification
│   └── select_checkpoint.py checkpoint selection on validation seeds
├── src/adaptive_replan/
│   ├── envs.py              checkpoint loading, environment construction, seeds
│   ├── features.py          observation fields, geometric features, phase labels
│   ├── policies.py          fixed, random and adaptive execution-horizon schedules
│   ├── rollout.py           single-episode rollout
│   └── statistics.py        Wilson, exact McNemar, Newcombe, power
└── tests/test_core.py       fast unit tests (run: python tests/test_core.py)
```

## Known limitations

- One policy and one training seed; the rules are hand-designed per task and hand-tuned on
  validation seeds only.
- Randomized schedules draw from the two-point support `{2, 8}`, not all horizons 1–8.
- **Not distributed:** model checkpoints, demonstrations, the teammate's training and model
  code, and the per-episode raw logs (~83 MB; see `results/MANIFEST.md` for the full file-level
  inventory and checksums). Tables and figures regenerate from the raw logs when you have them;
  re-running rollouts always requires the external checkpoint and model code.
- Reduced-DDPM-step results use n = 100 per cell; the interval on the largest compatible loss is
  reported rather than a claim of equivalence.
- No `LICENSE` file is included: the licence choice is left to the repository owner. ManiSkill and
  the Diffusion Policy baseline are separate projects with their own licences.

## Acknowledgements

Built on [ManiSkill](https://github.com/haosulab/ManiSkill) and the
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) baseline, with a co-trained policy
supplied by a project teammate.
