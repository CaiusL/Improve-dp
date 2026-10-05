"""Fast unit tests for schedules, features and statistics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_replan import (build_horizon_policy, extract_features, field_slice,  # noqa: E402
                             mcnemar_exact, newcombe_paired_diff, parse_schedule,
                             wilson_interval)


def test_parse_schedule_forms():
    assert parse_schedule("fixed:8") == ("fixed", {"k": 8})
    name, p = parse_schedule("random:mean=5.49,lo=2,hi=8")
    assert name == "random" and p == {"mean": 5.49, "lo": 2, "hi": 8}
    name, p = parse_schedule("adaptive_thresh:feat=qpos0,d=0.1,dir=lt,short=2,long=8")
    assert name == "adaptive_thresh" and p["feat"] == "qpos0" and p["dir"] == "lt"


def test_parse_schedule_rejects_unknown():
    try:
        parse_schedule("nope:1")
    except ValueError:
        return
    raise AssertionError("parse_schedule should reject an unknown schedule name")


def test_fixed_returns_requested_horizon():
    assert build_horizon_policy("fixed:2", 8)({}, 0) == 2
    assert build_horizon_policy("fixed:8", 8)({}, 0) == 8


def test_horizon_is_clamped_to_max():
    assert build_horizon_policy("fixed:8", 4)({}, 0) == 4
    hp = build_horizon_policy("fixed:16", 8)
    assert hp({}, 0) == 8


def test_adaptive_push_fires_on_either_disjunct():
    hp = build_horizon_policy("adaptive_push:d_tcp=0.08,d_goal=0.10", 8)
    assert hp({"d_tcp_obj": 0.05, "d_obj_goal_xy": 0.50}, 0) == 2
    hp = build_horizon_policy("adaptive_push:d_tcp=0.08,d_goal=0.10", 8)
    assert hp({"d_tcp_obj": 0.50, "d_obj_goal_xy": 0.05}, 0) == 2
    hp = build_horizon_policy("adaptive_push:d_tcp=0.08,d_goal=0.10", 8)
    assert hp({"d_tcp_obj": 0.50, "d_obj_goal_xy": 0.50}, 0) == 8


def test_adaptive_push_falls_back_to_3d_goal_distance():
    hp = build_horizon_policy("adaptive_push:d_tcp=0,d_goal=0.05", 8)
    assert hp({"d_tcp_obj": 9.9, "d_obj_goal": 0.01}, 0) == 2
    hp = build_horizon_policy("adaptive_push:d_tcp=0,d_goal=0.05", 8)
    assert hp({"d_tcp_obj": 9.9, "d_obj_goal": 0.90}, 0) == 8


def test_inverted_rule_is_the_complement():
    fwd = build_horizon_policy("adaptive_push:d_tcp=0.08,d_goal=0.10", 8)
    inv = build_horizon_policy("adaptive_push_inv:d_tcp=0.001,d_goal=0.01", 8)
    assert inv({"d_tcp_obj": 0.5, "d_obj_goal_xy": 0.5}, 0) == 2
    assert inv({"d_tcp_obj": 0.0, "d_obj_goal_xy": 0.5}, 0) == 8
    assert fwd({"d_tcp_obj": 0.0, "d_obj_goal_xy": 0.5}, 0) == 2


def test_time_window_boundaries():
    hp = build_horizon_policy("adaptive_time:start=30,length=170", 8)
    assert hp({"_step": 0}, 0) == 8
    assert hp({"_step": 29}, 0) == 8
    assert hp({"_step": 30}, 0) == 2
    assert hp({"_step": 199}, 0) == 2


def test_adaptive_thresh_direction():
    lt = build_horizon_policy("adaptive_thresh:feat=tcp_z,d=0.35,dir=lt,short=2,long=8", 8)
    assert lt({"tcp_z": 0.1}, 0) == 2
    assert lt({"tcp_z": 0.9}, 0) == 8
    gt = build_horizon_policy("adaptive_thresh:feat=d_obj_goal,d=0.05,dir=gt,short=2,long=8", 8)
    assert gt({"d_obj_goal": 0.02}, 0) == 8
    assert gt({"d_obj_goal": 0.50}, 0) == 2


def test_random_mean_is_reproducible_and_matched():
    a = build_horizon_policy("random:mean=5.0,lo=2,hi=8", 8, np_seed=7)
    b = build_horizon_policy("random:mean=5.0,lo=2,hi=8", 8, np_seed=7)
    sa = [a({}, i) for i in range(4000)]
    sb = [b({}, i) for i in range(4000)]
    assert sa == sb
    assert set(sa) <= {2, 8}
    assert abs(np.mean(sa) - 5.0) < 0.15


def test_wilson_interval_matches_known_values():
    lo, hi = wilson_interval(22, 200)
    assert abs(lo - 0.0738) < 0.001 and abs(hi - 0.1609) < 0.001
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and abs(hi - 0.1611) < 0.002
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and lo > 0.8


def test_mcnemar_exact_matches_known_value():
    a = [True] * 8 + [False] * 29 + [True] * 43 + [False] * 120
    b = [False] * 8 + [True] * 29 + [True] * 43 + [False] * 120
    n10, n01, p = mcnemar_exact(a, b)
    assert (n10, n01) == (8, 29)
    assert abs(p - 0.000753) < 1e-5


def test_newcombe_diff_brackets_the_point_estimate():
    a = [True] * 91 + [False] * 109
    b = [True] * 76 + [False] * 124
    diff, lo, hi = newcombe_paired_diff(a, b)
    assert abs(diff - 0.075) < 1e-9
    assert lo < diff < hi


def test_field_slice_lookup():
    assert field_slice("PushCube-v1", "extra.goal_pos") == slice(25, 28)
    assert field_slice("PullCubeTool-v1", "extra.tool_pose") == slice(32, 39)


def test_extract_features_shapes_and_quaternion_norm():
    import mani_skill.envs  # noqa: F401
    row = np.zeros(39)
    row[18:21] = [0.1, 0.2, 0.3]
    row[21:25] = [1.0, 0.0, 0.0, 0.0]
    row[25:28] = [0.4, 0.5, 0.02]
    row[28:32] = [1.0, 0.0, 0.0, 0.0]
    row[32:35] = [0.7, 0.8, 0.02]
    row[35:39] = [1.0, 0.0, 0.0, 0.0]
    f = extract_features("PullCubeTool-v1", row)
    assert abs(f["d_tcp_cube"] - np.linalg.norm([0.3, 0.3, 0.28])) < 1e-9
    assert abs(f["cube_r"] - np.linalg.norm([0.4, 0.5])) < 1e-9


def _run_all():
    """Minimal runner so the suite works without pytest installed."""
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
