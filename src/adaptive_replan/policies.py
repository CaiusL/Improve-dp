"""Execution-horizon schedules: fixed, random, and state-dependent adaptive rules."""
from __future__ import annotations

import numpy as np


class HorizonPolicy:
    """Base class: maps the latest state to an execution horizon."""

    def __init__(self, name, max_h, params=None, rng=None):
        self.name = name
        self.max_h = int(max_h)
        self.params = dict(params or {})
        self.rng = rng or np.random.default_rng(0)
        self._prev = None

    def __call__(self, feats, decision_idx):
        raise NotImplementedError

    def clamp(self, k):
        return int(min(max(1, int(k)), self.max_h))


class Fixed(HorizonPolicy):
    """Always execute the same number of actions per decision."""

    def __call__(self, feats, decision_idx):
        return self.clamp(self.params["k"])


class RandomMean(HorizonPolicy):
    """State-independent two-point draw on {lo, hi} with a prescribed mean."""

    def __call__(self, feats, decision_idx):
        lo, hi = self.params["lo"], self.params["hi"]
        m = self.params["mean"]
        if hi <= lo:
            return self.clamp(lo)
        p_hi = min(max((m - lo) / (hi - lo), 0.0), 1.0)
        return self.clamp(hi if self.rng.random() < p_hi else lo)


class AdaptiveDist(HorizonPolicy):
    """Short when a single distance feature is below a threshold, long otherwise."""

    def __call__(self, feats, decision_idx):
        d = feats.get("d_tool_cube", feats.get("d_tcp_obj", np.inf))
        return self.clamp(self.params["short"] if d < self.params["d"] else self.params["long"])


class AdaptivePhase(HorizonPolicy):
    """Short only once the tool is grasped and close to the cube, or at the grasp transition."""

    def __call__(self, feats, decision_idx):
        p = self.params
        grasped = feats.get("d_tcp_tool", np.inf) < p["grasp_eps"]
        short = (grasped and feats.get("d_tool_cube", np.inf) < p["d"]) or \
                (grasped and self._prev is False)
        self._prev = grasped
        return self.clamp(p["short"] if short else p["long"])


class AdaptiveThresh(HorizonPolicy):
    """Generic single-feature threshold rule; `dir` selects the comparison direction."""

    def __call__(self, feats, decision_idx):
        p = self.params
        v = feats.get(p["feat"], np.nan)
        if v is None or not np.isfinite(v):
            return self.clamp(p["long"])
        hit = (v < p["d"]) if p.get("dir", "lt") == "lt" else (v > p["d"])
        return self.clamp(p["short"] if hit else p["long"])


class AdaptivePush(HorizonPolicy):
    """Short when the TCP is near the object or the object is near the goal."""

    def __call__(self, feats, decision_idx):
        p = self.params
        d_tcp = feats.get("d_tcp_obj", np.inf)
        d_goal = feats.get("d_obj_goal_xy", feats.get("d_obj_goal", np.inf))
        short = (p["d_tcp"] > 0 and d_tcp < p["d_tcp"]) or \
                (p["d_goal"] > 0 and d_goal < p["d_goal"])
        return self.clamp(p["short"] if short else p["long"])


class AdaptivePushInverted(HorizonPolicy):
    """Inverse state assignment: short only when the object is far from both TCP and goal."""

    def __call__(self, feats, decision_idx):
        p = self.params
        d_tcp = feats.get("d_tcp_obj", np.inf)
        d_goal = feats.get("d_obj_goal_xy", feats.get("d_obj_goal", np.inf))
        short = (d_tcp >= p["d_tcp"]) and (d_goal >= p["d_goal"])
        return self.clamp(p["short"] if short else p["long"])


class TimeWindow(HorizonPolicy):
    """Short during a fixed window of environment steps, long otherwise."""

    def __call__(self, feats, decision_idx):
        p = self.params
        step = feats.get("_step", 0)
        in_window = p["start"] <= step < (p["start"] + p["length"])
        return self.clamp(p["short"] if in_window else p["long"])


SCHEDULES = {
    "fixed": Fixed,
    "random": RandomMean,
    "adaptive_dist": AdaptiveDist,
    "adaptive_phase": AdaptivePhase,
    "adaptive_thresh": AdaptiveThresh,
    "adaptive_push": AdaptivePush,
    "adaptive_push_inv": AdaptivePushInverted,
    "adaptive_time": TimeWindow,
}

DEFAULTS = {
    "random": {"mean": 4.5, "lo": 2, "hi": 8},
    "adaptive_dist": {"d": 0.08, "short": 2, "long": 8},
    "adaptive_phase": {"d": 0.10, "short": 2, "long": 8, "grasp_eps": 0.06},
    "adaptive_thresh": {"feat": "tcp_z", "d": 0.35, "dir": "lt", "short": 2, "long": 8},
    "adaptive_push": {"d_tcp": 0.08, "d_goal": 0.15, "short": 2, "long": 8},
    "adaptive_push_inv": {"d_tcp": 0.001, "d_goal": 0.01, "short": 2, "long": 8},
    "adaptive_time": {"start": 0, "length": 170, "short": 2, "long": 8},
}

INT_PARAMS = ("k", "short", "long", "lo", "hi", "start", "length")


def parse_schedule(spec):
    """Parse ``name:key=value,key=value`` (or ``fixed:K``) into (name, params)."""
    if ":" not in spec:
        raise ValueError(f"bad schedule spec {spec!r}")
    name, _, argstr = spec.partition(":")
    if name not in SCHEDULES:
        raise ValueError(f"unknown schedule {name!r}; have {sorted(SCHEDULES)}")
    params = {}
    if argstr:
        if "=" in argstr:
            for part in argstr.split(","):
                if not part.strip():
                    continue
                k, _, v = part.partition("=")
                v = v.strip()
                try:
                    params[k.strip()] = float(v)
                except ValueError:
                    params[k.strip()] = v
        elif name == "fixed":
            params["k"] = float(argstr)
        else:
            raise ValueError(f"schedule {name!r} needs key=value params, got {argstr!r}")
    for k in INT_PARAMS:
        if k in params:
            params[k] = int(params[k])
    return name, params


def build_horizon_policy(spec, max_h, np_seed=12345):
    """Instantiate a schedule from its spec string."""
    name, params = parse_schedule(spec)
    if name == "fixed" and "k" not in params:
        raise ValueError("fixed schedule needs fixed:<k>")
    p = {**DEFAULTS.get(name, {}), **params}
    return SCHEDULES[name](name, max_h, p, rng=np.random.default_rng(np_seed))


def mean_horizon(spec, max_h=8, np_seed=12345, n=20000):
    """Monte-Carlo mean horizon of a state-independent schedule (random only)."""
    hp = build_horizon_policy(spec, max_h, np_seed)
    return float(np.mean([hp({}, i) for i in range(n)]))
