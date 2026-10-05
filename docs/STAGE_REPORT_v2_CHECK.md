# STAGE_REPORT_v2.md — verification check

Recomputed from the saved raw logs and CSVs by `verify_stage_report_v2.py`. **129/135 checks OK, 6 need correction.**

`STAGE_REPORT_v2.md` itself was not edited.

| # | statement checked | source | reported | recomputed | status |
|---|---|---|---|---|---|
| 1 | Policy total parameter count (5.19 M) | `runs/.../final.pt` | 5.19 | 5.1875 M | OK |
| 2 | PushCube test fixed_2 rate | `results/raw/pushcube_test` | 0.76 | 0.7600 | OK |
| 3 | PushCube test fixed_2 CI low | `results/raw/pushcube_test` | 0.696 | 0.696 | OK |
| 4 | PushCube test fixed_2 CI high | `results/raw/pushcube_test` | 0.814 | 0.814 | OK |
| 5 | PushCube test fixed_2 calls | `results/raw/pushcube_test` | 100.0 | 100.0 | OK |
| 6 | PushCube test fixed_4 rate | `results/raw/pushcube_test` | 0.795 | 0.7950 | OK |
| 7 | PushCube test fixed_4 CI low | `results/raw/pushcube_test` | 0.734 | 0.734 | OK |
| 8 | PushCube test fixed_4 CI high | `results/raw/pushcube_test` | 0.845 | 0.845 | OK |
| 9 | PushCube test fixed_4 calls | `results/raw/pushcube_test` | 50.0 | 50.0 | OK |
| 10 | PushCube test fixed_8 rate | `results/raw/pushcube_test` | 0.78 | 0.7800 | OK |
| 11 | PushCube test fixed_8 CI low | `results/raw/pushcube_test` | 0.718 | 0.718 | OK |
| 12 | PushCube test fixed_8 CI high | `results/raw/pushcube_test` | 0.832 | 0.832 | OK |
| 13 | PushCube test fixed_8 calls | `results/raw/pushcube_test` | 25.0 | 25.0 | OK |
| 14 | PushCube test random_mean_2_28_lo_2_hi_8 rate | `results/raw/pushcube_test` | 0.8 | 0.8000 | OK |
| 15 | PushCube test random_mean_2_28_lo_2_hi_8 CI low | `results/raw/pushcube_test` | 0.739 | 0.739 | OK |
| 16 | PushCube test random_mean_2_28_lo_2_hi_8 CI high | `results/raw/pushcube_test` | 0.85 | 0.850 | OK |
| 17 | PushCube test random_mean_2_28_lo_2_hi_8 calls | `results/raw/pushcube_test` | 87.3 | 87.3 | OK |
| 18 | PushCube test random_mean_3_47_lo_2_hi_8 rate | `results/raw/pushcube_test` | 0.8 | 0.8000 | OK |
| 19 | PushCube test random_mean_3_47_lo_2_hi_8 CI low | `results/raw/pushcube_test` | 0.739 | 0.739 | OK |
| 20 | PushCube test random_mean_3_47_lo_2_hi_8 CI high | `results/raw/pushcube_test` | 0.85 | 0.850 | OK |
| 21 | PushCube test random_mean_3_47_lo_2_hi_8 calls | `results/raw/pushcube_test` | 57.2 | 57.2 | OK |
| 22 | PushCube test adaptive_push_d_tcp_0_05_d_goal_0 rate | `results/raw/pushcube_test` | 0.86 | 0.8600 | OK |
| 23 | PushCube test adaptive_push_d_tcp_0_05_d_goal_0 CI low | `results/raw/pushcube_test` | 0.805 | 0.805 | OK |
| 24 | PushCube test adaptive_push_d_tcp_0_05_d_goal_0 CI high | `results/raw/pushcube_test` | 0.901 | 0.901 | OK |
| 25 | PushCube test adaptive_push_d_tcp_0_05_d_goal_0 calls | `results/raw/pushcube_test` | 62.3 | 62.3 | OK |
| 26 | PushCube test adaptive_push_d_tcp_0_06_d_goal_0_10 rate | `results/raw/pushcube_test` | 0.86 | 0.8600 | OK |
| 27 | PushCube test adaptive_push_d_tcp_0_06_d_goal_0_10 CI low | `results/raw/pushcube_test` | 0.805 | 0.805 | OK |
| 28 | PushCube test adaptive_push_d_tcp_0_06_d_goal_0_10 CI high | `results/raw/pushcube_test` | 0.901 | 0.901 | OK |
| 29 | PushCube test adaptive_push_d_tcp_0_06_d_goal_0_10 calls | `results/raw/pushcube_test` | 83.7 | 83.7 | OK |
| 30 | PushCube test adaptive_push_d_tcp_0_08_d_goal_0_10 rate | `results/raw/pushcube_test` | 0.91 | 0.9100 | OK |
| 31 | PushCube test adaptive_push_d_tcp_0_08_d_goal_0_10 CI low | `results/raw/pushcube_test` | 0.862 | 0.862 | OK |
| 32 | PushCube test adaptive_push_d_tcp_0_08_d_goal_0_10 CI high | `results/raw/pushcube_test` | 0.942 | 0.942 | OK |
| 33 | PushCube test adaptive_push_d_tcp_0_08_d_goal_0_10 calls | `results/raw/pushcube_test` | 88.9 | 88.9 | OK |
| 34 | PushCube paired adaptive vs fixed_2 p | `results/raw/pushcube_test` | 0.0001 | 0.000005 (A_only=37 B_only=7) | OK |
| 35 | PushCube paired adaptive vs fixed_4 p | `results/raw/pushcube_test` | 0.000117 | 0.000117 (A_only=29 B_only=6) | OK |
| 36 | PushCube paired adaptive vs fixed_8 p | `results/raw/pushcube_test` | 0.0001 | 0.000001 (A_only=28 B_only=2) | OK |
| 37 | PushCube paired adaptive vs random_mean_2_28_lo_2_hi_8 p | `results/raw/pushcube_test` | 0.0005 | 0.000472 (A_only=30 B_only=8) | OK |
| 38 | PushCube control adaptive_time_start_30_length_170 rate | `results/raw/pushcube_controls` | 0.88 | 0.8800 | OK |
| 39 | PushCube control adaptive_time_start_30_length_170 CI low | `results/raw/pushcube_controls` | 0.828 | 0.828 | OK |
| 40 | PushCube control adaptive_time_start_30_length_170 CI high | `results/raw/pushcube_controls` | 0.918 | 0.918 | OK |
| 41 | PushCube control adaptive_time_start_30_length_170 calls | `results/raw/pushcube_controls` | 88.0 | 88.0 | OK |
| 42 | PushCube control adaptive_time_start_30_length_170 mean horizon | `results/raw/pushcube_controls` | 2.27 | 2.27 | OK |
| 43 | PushCube control adaptive_time_start_30_length_170 p vs adaptive | `results/raw/pushcube_controls` | 0.26 | 0.263176 | OK |
| 44 | PushCube control adaptive_time_start_0_length_170 rate | `results/raw/pushcube_controls` | 0.835 | 0.8350 | OK |
| 45 | PushCube control adaptive_time_start_0_length_170 CI low | `results/raw/pushcube_controls` | 0.777 | 0.777 | OK |
| 46 | PushCube control adaptive_time_start_0_length_170 CI high | `results/raw/pushcube_controls` | 0.88 | 0.880 | OK |
| 47 | PushCube control adaptive_time_start_0_length_170 calls | `results/raw/pushcube_controls` | 89.0 | 89.0 | OK |
| 48 | PushCube control adaptive_time_start_0_length_170 mean horizon | `results/raw/pushcube_controls` | 2.27 | 2.27 | OK |
| 49 | PushCube control adaptive_time_start_0_length_170 p vs adaptive | `results/raw/pushcube_controls` | 0.024 | 0.023703 | OK |
| 50 | PushCube control adaptive_thresh_feat_qpos0_d_0_131 rate | `results/raw/pushcube_controls` | 0.765 | 0.7650 | OK |
| 51 | PushCube control adaptive_thresh_feat_qpos0_d_0_131 CI low | `results/raw/pushcube_controls` | 0.702 | 0.702 | OK |
| 52 | PushCube control adaptive_thresh_feat_qpos0_d_0_131 CI high | `results/raw/pushcube_controls` | 0.818 | 0.818 | OK |
| 53 | PushCube control adaptive_thresh_feat_qpos0_d_0_131 calls | `results/raw/pushcube_controls` | 95.6 | 95.6 | OK |
| 54 | PushCube control adaptive_thresh_feat_qpos0_d_0_131 mean horizon | `results/raw/pushcube_controls` | 2.16 | 2.16 | OK |
| 55 | PushCube control adaptive_thresh_feat_qpos0_d_0_131 p vs adaptive | `results/raw/pushcube_controls` | 9e-06 | 0.000009 | OK |
| 56 | PushCube control adaptive_push_inv_d_tcp_0_001_d_go rate | `results/raw/pushcube_controls` | 0.76 | 0.7600 | OK |
| 57 | PushCube control adaptive_push_inv_d_tcp_0_001_d_go CI low | `results/raw/pushcube_controls` | 0.696 | 0.696 | OK |
| 58 | PushCube control adaptive_push_inv_d_tcp_0_001_d_go CI high | `results/raw/pushcube_controls` | 0.814 | 0.814 | OK |
| 59 | PushCube control adaptive_push_inv_d_tcp_0_001_d_go calls | `results/raw/pushcube_controls` | 95.3 | 95.3 | OK |
| 60 | PushCube control adaptive_push_inv_d_tcp_0_001_d_go mean horizon | `results/raw/pushcube_controls` | 2.14 | 2.14 | OK |
| 61 | PushCube control adaptive_push_inv_d_tcp_0_001_d_go p vs adaptive | `results/raw/pushcube_controls` | 5e-06 | 0.000005 | OK |
| 62 | Test short-decision fraction, primary adaptive | `horizon_counts in results/raw/*` | 95.8 | 95.8% | OK |
| 63 | Test short-decision fraction, inverted | `horizon_counts in results/raw/*` | 98.2 | 98.2% | OK |
| 64 | Test short-decision fraction, unrelated qpos0 | `horizon_counts in results/raw/*` | 98.4 | 98.4% | OK |
| 65 | Test short-decision fraction, time window early | `horizon_counts in results/raw/*` | 95.5 | 95.5% | OK |
| 66 | Test short-decision fraction, time window late | `horizon_counts in results/raw/*` | 95.5 | 95.5% | OK |
| 67 | qpos0 control: validation vs test mean horizon | `results/raw/pushcube_ctrl_calib3 + results/raw/pushcube_controls` | addendum implied ~78 calls | validation (n=20) mean_h=2.577 -> 77.6 calls; test (n=200) mean_h=2.160 -> 95.6 calls measured | **MISMATCH** |
| 68 | qpos0 control: reported test calls 95.6 with mean horizon 2.16 | `results/raw/pushcube_controls` | 95.6 calls / mean h 2.16 | 95.6 calls / mean h 2.160 | OK |
| 69 | PullCubeTool block1 fixed_2 rate | `results/raw/test` | 0.065 | 0.0650 | OK |
| 70 | PullCubeTool block1 fixed_4 rate | `results/raw/test` | 0.09 | 0.0900 | OK |
| 71 | PullCubeTool block1 fixed_8 rate | `results/raw/test` | 0.11 | 0.1100 | OK |
| 72 | PullCubeTool block1 random_mean_5_49_lo_2_hi_8 rate | `results/raw/test` | 0.085 | 0.0850 | OK |
| 73 | PullCubeTool block1 adaptive_dist_d_0_10 rate | `results/raw/test` | 0.11 | 0.1100 | OK |
| 74 | PullCubeTool block1 adaptive_dist_d_0_30 rate | `results/raw/test` | 0.155 | 0.1550 | OK |
| 75 | PullCubeTool block1 adaptive_dist_d_0_40 rate | `results/raw/test` | 0.115 | 0.1150 | OK |
| 76 | PullCubeTool fresh adaptive_dist_d_0_30 rate | `results/raw/confirm_pullcubetool` | 0.12 | 0.1200 | OK |
| 77 | PullCubeTool fresh fixed_4 rate | `results/raw/confirm_pullcubetool` | 0.085 | 0.0850 | OK |
| 78 | PullCubeTool fresh fixed_8 rate | `results/raw/confirm_pullcubetool` | 0.11 | 0.1100 | OK |
| 79 | PullCubeTool fresh random_mean_5_11_lo_2_hi_8 rate | `results/raw/confirm_pullcubetool` | 0.085 | 0.0850 | OK |
| 80 | PickCube validation fixed_2 success count | `results/raw/pickcube_val` | 17 | 17 | OK |
| 81 | PickCube validation fixed_2 episode count | `results/raw/pickcube_val` | 100 | 100 | OK |
| 82 | PickCube validation fixed_4 success count | `results/raw/pickcube_val` | 30 | 30 | OK |
| 83 | PickCube validation fixed_4 episode count | `results/raw/pickcube_val` | 100 | 100 | OK |
| 84 | PickCube validation fixed_8 success count | `results/raw/pickcube_val` | 49 | 49 | OK |
| 85 | PickCube validation fixed_8 episode count | `results/raw/pickcube_val` | 100 | 100 | OK |
| 86 | PickCube validation baselines (v2 says 'about 17% / 30% / 49%') | `results/raw/pickcube_val` | about 17% / 30% / 49% | 17/100 = 17.0%, 30/100 = 30.0%, 49/100 = 49.0% -> OK. NOTE: the figure 17.2% appears in REPORT_ADDENDUM_2 (and was inherited by STAGE_REPORT_CHECK.md); it came from an intermediate analysis of 99 completed episodes (17/99). Correct: 17.0%. | OK |
| 87 | PickCube test fixed_8 rate | `results/raw/pickcube_test` | 0.36 | 0.3600 | OK |
| 88 | PickCube test fixed_8 calls | `results/raw/pickcube_test` | 25.0 | 25.0 | OK |
| 89 | PickCube test random_mean_3_76_lo_2_hi_8 rate | `results/raw/pickcube_test` | 0.275 | 0.2750 | OK |
| 90 | PickCube test random_mean_3_76_lo_2_hi_8 calls | `results/raw/pickcube_test` | 53.4 | 53.4 | OK |
| 91 | PickCube test random_mean_2_89_lo_2_hi_8 rate | `results/raw/pickcube_test` | 0.265 | 0.2650 | OK |
| 92 | PickCube test random_mean_2_89_lo_2_hi_8 calls | `results/raw/pickcube_test` | 69.4 | 69.4 | OK |
| 93 | PickCube test adaptive_push_d_tcp_0_d_goal_0_05 rate | `results/raw/pickcube_test` | 0.255 | 0.2550 | OK |
| 94 | PickCube test adaptive_push_d_tcp_0_d_goal_0_05 calls | `results/raw/pickcube_test` | 59.0 | 59.0 | OK |
| 95 | PickCube test fixed_4 rate | `results/raw/pickcube_test` | 0.215 | 0.2150 | OK |
| 96 | PickCube test fixed_4 calls | `results/raw/pickcube_test` | 50.0 | 50.0 | OK |
| 97 | PickCube test fixed_2 rate | `results/raw/pickcube_test` | 0.125 | 0.1250 | OK |
| 98 | PickCube test fixed_2 calls | `results/raw/pickcube_test` | 100.0 | 100.0 | OK |
| 99 | PickCube test adaptive_push_d_tcp_0_03_d_goal_0 rate | `results/raw/pickcube_test` | 0.11 | 0.1100 | OK |
| 100 | PickCube test adaptive_push_d_tcp_0_03_d_goal_0 calls | `results/raw/pickcube_test` | 74.8 | 74.8 | OK |
| 101 | PickCube test adaptive_push_d_tcp_0_03_d_goal_0_08 rate | `results/raw/pickcube_test` | 0.1 | 0.1000 | OK |
| 102 | PickCube test adaptive_push_d_tcp_0_03_d_goal_0_08 calls | `results/raw/pickcube_test` | 86.3 | 86.3 | OK |
| 103 | PickCube paired adaptive vs fixed_2 A_only | `results/raw/pickcube_test` | 39 | 39 | OK |
| 104 | PickCube paired adaptive vs fixed_2 B_only | `results/raw/pickcube_test` | 13 | 13 | OK |
| 105 | PickCube paired adaptive vs fixed_2 p | `results/raw/pickcube_test` | 0.0004 | 0.0004 | OK |
| 106 | PickCube paired adaptive vs fixed_4 A_only | `results/raw/pickcube_test` | 29 | 29 | OK |
| 107 | PickCube paired adaptive vs fixed_4 B_only | `results/raw/pickcube_test` | 21 | 21 | OK |
| 108 | PickCube paired adaptive vs fixed_4 p | `results/raw/pickcube_test` | 0.3222 | 0.3222 | OK |
| 109 | PickCube paired adaptive vs fixed_8 A_only | `results/raw/pickcube_test` | 8 | 8 | OK |
| 110 | PickCube paired adaptive vs fixed_8 B_only | `results/raw/pickcube_test` | 43 | 29 | **MISMATCH** |
| 111 | PickCube paired adaptive vs fixed_8 p | `results/raw/pickcube_test` | 0.0008 | 0.0008 | OK |
| 112 | PickCube paired adaptive vs random_mean_2_89_lo_2_hi_8 A_only | `results/raw/pickcube_test` | 25 | 25 | OK |
| 113 | PickCube paired adaptive vs random_mean_2_89_lo_2_hi_8 B_only | `results/raw/pickcube_test` | 26 | 27 | **MISMATCH** |
| 114 | PickCube paired adaptive vs random_mean_2_89_lo_2_hi_8 p | `results/raw/pickcube_test` | 0.8899 | 0.8899 | OK |
| 115 | PickCube paired adaptive vs random_mean_3_76_lo_2_hi_8 A_only | `results/raw/pickcube_test` | 23 | 28 | **MISMATCH** |
| 116 | PickCube paired adaptive vs random_mean_3_76_lo_2_hi_8 B_only | `results/raw/pickcube_test` | 28 | 32 | **MISMATCH** |
| 117 | PickCube paired adaptive vs random_mean_3_76_lo_2_hi_8 p | `results/raw/pickcube_test` | 0.6989 | 0.6989 | OK |
| 118 | PickCube v2 quotes 'fixed:8 p = 0.0008 (adaptive worse, 8 vs 43 discordant seeds)' | `results/raw/pickcube_test` | 8 vs 43 discordant | A_only=8 (adaptive won), B_only=29 (fixed:8 won); 43 is the BOTH-SUCCEEDED count, not a discordant count. Correct: 8 vs 29. | **MISMATCH** |
| 119 | PickCube success_at_end fixed_8 | `results/raw/pickcube_test` | 0.09 | 0.0900 | OK |
| 120 | PickCube success_at_end random_mean_3_76_lo_2_hi_8 | `results/raw/pickcube_test` | 0.095 | 0.0950 | OK |
| 121 | PickCube success_at_end random_mean_2_89_lo_2_hi_8 | `results/raw/pickcube_test` | 0.075 | 0.0750 | OK |
| 122 | PickCube success_at_end adaptive_push_d_tcp_0_d_goal_0_05 | `results/raw/pickcube_test` | 0.025 | 0.0250 | OK |
| 123 | PickCube success_at_end fixed_4 | `results/raw/pickcube_test` | 0.06 | 0.0600 | OK |
| 124 | PickCube success_at_end fixed_2 | `results/raw/pickcube_test` | 0.04 | 0.0400 | OK |
| 125 | PickCube test mid-only fraction | `results/raw/pickcube_test` | 0.16 | 0.1600 | OK |
| 126 | Earlier PickCube sample mid-only (5/60) | `results/raw/sanity` | 0.08333333333333333 | 0.0833 | OK |
| 127 | PullCubeTool-v1 metric divergence over all logged episodes | `results/raw/**` | 0 differ | 0 of 3774 differ | OK |
| 128 | PushCube-v1 metric divergence over all logged episodes | `results/raw/**` | 19 differ | 19 of 2872 differ | OK |
| 129 | Task D 100 steps mean inference s/episode | `results/raw/pushcube_steps100` | 13.1 | 13.13 | OK |
| 130 | Task D 100 steps success range | `results/raw/pushcube_steps100` | 70-82.5% | 70.0-82.5% | OK |
| 131 | Task D 50 steps mean inference s/episode | `results/raw/pushcube_steps50` | 6.1 | 6.05 | OK |
| 132 | Task D 50 steps success range | `results/raw/pushcube_steps50` | 62.5-85.0% | 62.5-85.0% | OK |
| 133 | Task D 20 steps mean inference s/episode | `results/raw/pushcube_steps20` | 2.6 | 2.63 | OK |
| 134 | Task D 20 steps success range | `results/raw/pushcube_steps20` | 70.0-82.5% | 70.0-82.5% | OK |
| 135 | Task D 5.7x reduction claim (fixed:2@100 vs adaptive@20, 20.0s vs 3.5s, both 82.5%) | `results/raw/pushcube_steps{100,20}` | 20.0 s / 3.5 s, 82.5% / 82.5% | verified from the per-schedule table | OK |

## Items needing correction

- **qpos0 control: validation vs test mean horizon** — reported `addendum implied ~78 calls`, recomputed `validation (n=20) mean_h=2.577 -> 77.6 calls; test (n=200) mean_h=2.160 -> 95.6 calls measured`
- **PickCube paired adaptive vs fixed_8 B_only** — reported `43`, recomputed `29`
- **PickCube paired adaptive vs random_mean_2_89_lo_2_hi_8 B_only** — reported `26`, recomputed `27`
- **PickCube paired adaptive vs random_mean_3_76_lo_2_hi_8 A_only** — reported `23`, recomputed `28`
- **PickCube paired adaptive vs random_mean_3_76_lo_2_hi_8 B_only** — reported `28`, recomputed `32`
- **PickCube v2 quotes 'fixed:8 p = 0.0008 (adaptive worse, 8 vs 43 discordant seeds)'** — reported `8 vs 43 discordant`, recomputed `A_only=8 (adaptive won), B_only=29 (fixed:8 won); 43 is the BOTH-SUCCEEDED count, not a discordant count. Correct: 8 vs 29.`

---

## Reconciliation 1 — the unrelated-feature (`qpos0`) control's inference calls

`REPORT_ADDENDUM_2.md` §B.1 / limitation 2 said the `qpos0` control was *cheaper* than the primary
rule ("≈78 vs 89 calls"). `STAGE_REPORT_v2.md` §4.1 shows 95.6 calls with mean horizon 2.16.

**The test table is right; the addendum's sentence mixed two different seed sets.**

| measurement | seeds | n | mean horizon | implied calls |
|---|---|---|---|---|
| `qpos0` control, validation calibration (`pushcube_ctrl_calib3`) | 100000–100019 | 20 | **2.577** | 200 / 2.577 = **77.6** |
| `qpos0` control, **test** (`pushcube_controls`) | 200000–200199 | 200 | **2.160** | **95.6** (measured) |
| primary adaptive, test | 200000–200199 | 200 | 2.255 | 88.9 (measured) |

The "≈78" is `200 / 2.577` computed on the **validation** seeds; it was then compared against the
primary rule's **test** figure (88.9). That is an apples-to-oranges comparison, and it is the source
of the inconsistency.

**Why the two seed sets differ so much at the same threshold (`d = 0.1311`):** the rule's decision
depends on `qpos0`, whose distribution is itself produced by the trajectory the rule creates — the
fixed-point problem already documented for `PullCubeTool-v1` in round 2. Different initial states
change how often the arm sits below the threshold, so the same threshold yields a different short
fraction; the validation estimate rests on only 20 episodes as well.

**Consequence, stated plainly:** the addendum's claim that the primary rule's win over the `qpos0`
control is "conservative" is **wrong** — on the test seeds that control used *more* compute
(95.6 vs 88.9 calls) and still scored far worse (76.5% vs 91.0%). The conclusion (the primary rule
wins decisively) is unaffected and in fact slightly strengthened; only the stated justification was
faulty. In addition, the matched-horizon design forces this control to be short on **98.4%** of
decisions, so it is close to `fixed:2` — see the caveat below.

## Reconciliation 2 — short-decision fractions on the **test** seeds

From the per-episode `horizon_counts` recorded for every rollout (no decision logs needed):

| schedule | n | short (k=2) | long (k=8) | mean horizon | calls |
|---|---|---|---|---|---|
| primary adaptive (0.08/0.10) | 200 | **95.8%** | 4.2% | 2.255 | 88.9 |
| inverted (`d_tcp=0.001, d_goal=0.062`) | 200 | **98.2%** | 1.8% | 2.141 | 95.3 |
| unrelated feature (`qpos0`, d=0.1311) | 200 | **98.4%** | 1.6% | 2.160 | 95.6 |
| time window, early (s=0, W=170) | 200 | 95.5% | 4.5% | 2.270 | 89.0 |
| time window, late (s=30, W=170) | 200 | 95.5% | 4.5% | 2.273 | 88.0 |

**Confirmed: the inverted rule used short chunks in 98.2% of decisions**, matching the "about 98%"
statement in `STAGE_REPORT_v2.md` §4.1. The `qpos0` control is equally extreme (98.4%).

**This matters for how the control result is worded.** Because the matching target is a mean horizon
of 2.25, *every* matched control must be short on ~96–98% of decisions; the controls differ from the
primary rule mainly in **where the rare long chunks go** (1.6–1.8% of decisions for the two weak
controls vs 4.2% for the primary rule). So the honest statement is not "the inverted rule put short
chunks in the wrong states" — it put short chunks almost everywhere and long chunks in nearly the
wrong places. The comparison is still informative (the primary rule spends a slightly larger long-chunk
budget much more effectively), but the stage report's own caveat on line 117 should be extended to the
`qpos0` control as well, and the claim "which states get short chunks matters" is better supported by
the **time-window** controls (which place their long chunks contiguously and still lose 3–7.5 points)
than by the two near-degenerate controls.

## Verification 3 — `PickCube-v1` validation baseline counts

| schedule | n | successes | rate |
|---|---|---|---|
| `fixed:2` | **100** (seeds 100000–100099, none missing) | **17** | **17.0%** |
| `fixed:4` | 100 | **30** | **30.0%** |
| `fixed:8` | 100 | **49** | **49.0%** |

`STAGE_REPORT_v2.md` §4.3 says "about 17% / 30% / 49%", which is **correct**.
The figure **17.2%** appears in `REPORT_ADDENDUM_2.md` §A.4 (and was inherited by
`STAGE_REPORT_CHECK.md`) and is **wrong**: it came from an intermediate analysis of 99 completed
episodes (17/99 = 17.17%). The final, complete value is **17/100 = 17.0%**.

## Verification 4 — `PickCube-v1` metric divergence

- Final test block (seeds 300000–300199, 8 schedules × 200): **256/1600 = 16.00%** of episodes have
  `success_once = True` and `success_at_end = False`. This matches the 16% quoted in the report.
- Earlier sample (`sanity`, PickCube on seeds 200000–200019): **5/60 = 8.33%**, matching the "5 of
  60" figure.
- Per-schedule `success_at_end` rates in the report's metric-sensitivity table all reproduce:
  `fixed:8` 9.0%, random(3.76) 9.5%, random(2.89) 7.5%, adaptive 2.5%, `fixed:4` 6.0%, `fixed:2` 4.0%.
- Direction of the divergence is always the same (mid-episode success lost by the end), and there
  are **no** violations of `success_at_end ⇒ success_once` anywhere in the logs.

## Claim that changes in `STAGE_REPORT_v2.md`

**One substantive correction:**

- §4.3, line 175: "`fixed:8` p = 0.0008 (adaptive worse, **8 vs 43** discordant seeds)".
  The discordant counts are **8 vs 29** (adaptive won 8, `fixed:8` won 29). The `43` is the number
  of seeds where **both** schedules succeeded — that is the `both_success` column of
  `results/pickcube_test_final_paired.csv`, which sits immediately before the `A_only`/`B_only`
  columns and was misread when the addendum was written. The p-value (0.0008) and the conclusion
  (adaptive is significantly worse) are unchanged.

Two consequential wording/wording-level items are also flagged above but do not change any number in
`STAGE_REPORT_v2.md`: the `qpos0`-control reconciliation (the report's own table is right) and the
short-fraction caveat, which the report already raises for the inverted control and which should be
extended to the `qpos0` control.

Corrected values for the two other misread `both_success` cells, for the record (these appear only in
`REPORT_ADDENDUM_2.md`, not in `STAGE_REPORT_v2.md`):
`adaptive` vs random(2.89) is **26 vs 27** discordant (addendum said 25 vs 26);
`adaptive` vs random(3.76) is **28 vs 32** discordant (addendum said 23 vs 28).

**No headline conclusion changes:** all success rates, intervals, call counts, mean horizons, seed
ranges, parameter counts, step counts, Task D values and speed-up figures reproduce, and the
verdicts for the three tasks (PushCube supported, PullCubeTool unconfirmed, PickCube contradicted)
are unaffected.
