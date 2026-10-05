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
`docs/REPORT_ADDENDUM_3.md`, and the verification report `docs/STAGE_REPORT_v2_CHECK.md`.
Pre-registrations are in `configs/`; the internal working log (`NOTES.md`) and the earlier
stage reports are not part of this release.

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

Every table and figure is rebuilt from the per-episode logs. **The raw logs (~108 MB across
2 206 JSONL files, 22 082 episodes) are intentionally not part of this release** — they
exceed the size the repository is meant to carry. `results/MANIFEST.md` lists every raw file with its episode count,
seed range and SHA-256 checksum, so a complete copy can be verified file by file; the summary
CSVs in `results/` are shipped in their place. Point `--raw-root` at a directory holding the
raw logs to reproduce everything below. The pre-registered settings for each phase are in
`configs/`.

```bash
# final per-setting table and manifest
python scripts/pack_final_results.py --with-manifest

# round 5: sampler x steps sweep (DDIM is rebuilt from the checkpoint's trained betas)
python scripts/run_matrix.py --phase ddpm20 --tasks "PushCube-v1;PickCube-v1;PullCubeTool-v1" \
    --schedules "fixed:8" --seeds 400000-400099 --chunk 10 --jobs 8 \
    --diffusion-steps 20 --raw-root results/raw/sampling_sweep
python scripts/run_matrix.py --phase ddim5 --tasks "PushCube-v1;PickCube-v1;PullCubeTool-v1" \
    --schedules "fixed:8" --seeds 400000-400099 --chunk 10 --jobs 8 \
    --sampler ddim --diffusion-steps 5 --raw-root results/raw/sampling_sweep
python scripts/r5_sampling_analysis.py --phases ddpm100,ddpm20,ddim5 --schedule "fixed(k=8)" \
    --out results/r5_sampling

# round 5: clean inference timing (single worker, otherwise idle machine)
python scripts/run_rollout.py --task PushCube-v1 --schedule fixed:8 --seeds 400000-400009 \
    --sampler ddim --diffusion-steps 5 --out results/timing_r5/ddim5_fixed8.jsonl

# round 5: raw-log integrity guard before analysing
python scripts/r5_integrity.py --raw-root results/raw

# round 5: training-variant evaluation analysis and figures
python scripts/r5_training_analysis.py --raw-root results/raw
python scripts/r5_figures.py
```

# per-task summary plus paired adaptive-vs-baseline tests
# (--include-adaptive-pairs also pairs the adaptive/time-window rules against each other)
python scripts/analyze_results.py --phases pushcube_fresh --task PushCube-v1 \
    --seeds 300000-300199 --out results/pushcube_fresh --include-adaptive-pairs

# figures (per task and combined three-panel)
python scripts/make_figures.py

# recompute the summary CSV from the raw logs and check every value
python scripts/verify_results.py

# a specific experiment block (example: the fresh confirmation block)
python scripts/run_matrix.py --phase pushcube_fresh --tasks PushCube-v1 \
    --schedules "fixed:2;fixed:4;fixed:8;random:mean=2.25,lo=2,hi=8;adaptive_push:d_tcp=0.08,d_goal=0.10;adaptive_push:d_tcp=0.05,d_goal=0;adaptive_time:start=10,length=170;adaptive_time:start=30,length=170" \
    --seeds 300000-300199 --chunk 10 --jobs 8

# reduced DDPM sampling steps
python scripts/run_matrix.py --phase pushcube_steps_fresh_20 --tasks PushCube-v1 \
    --schedules "fixed:2;fixed:4;fixed:8;adaptive_push:d_tcp=0.08,d_goal=0.10" \
    --seeds 300000-300099 --chunk 10 --jobs 8 --diffusion-steps 20
```

`python <script> --help` lists all arguments for each script.

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
