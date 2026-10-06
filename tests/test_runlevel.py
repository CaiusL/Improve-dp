"""Unit tests for the run-level statistics and for the raw-feature rule.

The permutation and sign-flip tests are checked against hand-computed values, including the
smallest attainable two-sided p for a 4-vs-4 comparison (2/70). The feature tests check that
the adaptive rule's geometric features are computed from the raw observation and that they
change when the observation is normalised, which is what makes "the rule must read raw
observations" a testable property rather than a convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_replan import build_horizon_policy, extract_features  # noqa: E402
from adaptive_replan import runlevel  # noqa: E402


def test_permutation_test_hand_computed_cases():
    p, diff, n = runlevel.permutation_test([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
    assert n == 924
    assert abs(p - 2 / 924) < 1e-12
    assert diff == 1.0

    p, _, n = runlevel.permutation_test([1, 1, 0, 0], [1, 1, 0, 0])
    assert n == 70
    assert p == 1.0

    p, _, n = runlevel.permutation_test([1, 1, 1, 1], [0, 0, 0, 0])
    assert n == 70
    assert abs(p - 2 / 70) < 1e-12

    p_ab, _, n_ab = runlevel.permutation_test([1, 1, 1], [0, 0, 0])
    p_ba, _, _ = runlevel.permutation_test([0, 0, 0], [1, 1, 1])
    assert n_ab == 20
    assert p_ab == p_ba == 2 / 20


def test_signflip_hand_computed_cases():
    p, _, n = runlevel.signflip_test([0.1] * 6)
    assert n == 64
    assert abs(p - 2 / 64) < 1e-12

    assert runlevel.signflip_test([1, 0, 0, 0, 0, 0])[0] == 1.0

    p, _, n = runlevel.signflip_test([1, 1, 0, 0, 0, 0])
    assert n == 64
    assert abs(p - 0.5) < 1e-12

    assert runlevel.signflip_test([1, 1, 0, 0])[2] == 16


def test_bootstrap_interval_and_summary():
    lo, hi, obs = runlevel.bootstrap_diff_ci([0.5, 0.6, 0.4], [0.5, 0.6, 0.4], n_boot=2000)
    assert lo <= 0 <= hi
    assert obs == 0.0
    s = runlevel.summarize_arm([0.8, 0.9, 1.0])
    assert s["n_runs"] == 3
    assert abs(s["mean"] - 0.9) < 1e-12
    assert abs(s["range"] - 0.2) < 1e-12


def _pushcube_row(tcp, obj, goal, fill=0.0):
    row = np.full(35, fill, dtype=np.float64)
    row[18:21] = tcp
    row[25:28] = goal
    row[28:31] = obj
    return row


def test_pushcube_rule_uses_raw_observations_only():
    tcp = np.array([0.10, 0.00, 0.05])
    obj = np.array([0.15, 0.00, 0.05])
    goal = np.array([0.60, 0.00, 0.05])
    row = _pushcube_row(tcp, obj, goal)

    feats = extract_features("PushCube-v1", row)
    assert abs(feats["d_tcp_obj"] - 0.05) < 1e-12
    assert abs(feats["d_obj_goal_xy"] - 0.45) < 1e-12

    policy = build_horizon_policy("adaptive_push:d_tcp=0.08,d_goal=0.10,short=2,long=8", max_h=8)
    assert policy(feats, 0) == 2

    far = _pushcube_row(np.array([0.90, 0.0, 0.05]), np.array([0.15, 0.0, 0.05]),
                        np.array([0.60, 0.0, 0.05]))
    assert policy(extract_features("PushCube-v1", far), 0) == 8

    near_goal = _pushcube_row(np.array([0.90, 0.0, 0.05]), np.array([0.55, 0.0, 0.05]),
                              np.array([0.60, 0.0, 0.05]))
    assert policy(extract_features("PushCube-v1", near_goal), 0) == 2

    mean = np.full(35, 0.5)
    std = np.full(35, 0.1)
    normalised = (row - mean) / std
    feats_norm = extract_features("PushCube-v1", normalised)
    assert abs(feats_norm["d_tcp_obj"] - feats["d_tcp_obj"]) > 1e-6
    assert policy(feats_norm, 0) != policy(feats, 0)

def test_descriptive_flag_hand_computed_cases():
    base = [0.50, 0.55, 0.60]
    assert runlevel.descriptive_flag([0.90, 0.92], base) == "suggestive"
    assert runlevel.descriptive_flag([0.10, 0.12], base) == "suggestive"
    assert runlevel.descriptive_flag([0.50, 0.52], base) == "within training noise"
    assert runlevel.descriptive_flag([0.90, 0.40], base) == "within training noise"
    assert runlevel.descriptive_flag([0.65, 0.70], base) == "suggestive"
    assert runlevel.descriptive_flag([0.58, 0.59], base) == "within training noise"


def test_smallest_attainable_p_matches_permutation_counts():
    from math import comb
    assert comb(12, 6) == 924
    assert comb(10, 4) == 210
    assert comb(8, 4) == 70
    assert comb(7, 3) == 35
    assert abs(2 / comb(4, 2) - 0.333333) < 1e-6
