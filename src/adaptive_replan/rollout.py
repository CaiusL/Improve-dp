"""Single-episode rollout with a pluggable execution-horizon policy."""
from __future__ import annotations

import time

import numpy as np
import torch

from .envs import policy_torch_seed, set_seed
from .features import extract_features, phase_label

HORIZON_KEYS = ("horizon_counts", "mean_horizon", "n_inference", "n_env_steps")


def _scalar_from_info(info, key):
    if not isinstance(info, dict) or key not in info:
        return None
    v = info[key]
    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()
    a = np.asarray(v).reshape(-1)
    return float(a[0]) if a.size else None


def _round(v, nd=4):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return [round(float(x), nd) for x in v]
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return v


def run_episode(env, agent, task, task_id, horizon_policy, seed, max_steps,
                trace=False, trace_every=1, decide_log=False, feature_check=False,
                metric_key="success_once"):
    """Run one episode and return a per-episode record.

    ``success_once`` is the running OR of the task's instantaneous success flag;
    ``success_at_end`` is the flag on the final step. Both are always recorded.
    """
    set_seed(policy_torch_seed(seed))
    obs, info = env.reset(seed=seed)
    horizon_policy._prev = None

    n_inf = n_steps = 0
    horizons, trace_rows, decisions = [], [], []
    feats = {}
    inference_time = 0.0
    progress_max = dist_min = dist_final = None
    t0 = time.time()

    while n_steps < max_steps:
        arr = obs if not isinstance(obs, dict) else obs["state"]
        arr = np.asarray(arr, dtype=np.float32)
        last = arr.reshape(-1, arr.shape[-1])[-1]
        feats = extract_features(task, last)
        feats["_step"] = n_steps
        feats["_decision"] = n_inf

        k = horizon_policy(feats, n_inf)
        horizons.append(k)
        fc = None
        if feature_check:
            _norm = getattr(agent, "obs_norm", None)
            fc = {"raw_row": [float(x) for x in last],
                  "obs_norm_present": _norm is not None}
            if _norm is not None:
                _m, _s = _norm
                _ln = ((torch.as_tensor(last).float() - _m.detach().cpu())
                       / _s.detach().cpu()).numpy()
                _fn = extract_features(task, _ln)
                fc["d_tcp_norm"] = _round(_fn.get("d_tcp_obj"))
                fc["d_goal_norm"] = _round(_fn.get("d_obj_goal_xy"))
        if decide_log:
            decisions.append({
                "step": n_steps, "decision": n_inf, "k": int(k),
                "phase": phase_label(task, feats),
                "d_tool_cube": _round(feats.get("d_tool_cube")),
                "d_tcp_tool": _round(feats.get("d_tcp_tool")),
                "d_tcp_obj": _round(feats.get("d_tcp_obj")),
                "d_obj_goal_xy": _round(feats.get("d_obj_goal_xy")),
                "d_obj_goal": _round(feats.get("d_obj_goal")),
                "is_grasped": _round(feats.get("is_grasped")),
                "cube_z": _round(feats.get("cube_z")),
                "obj_z": _round(feats.get("obj_z")),
                "tcp_z": _round(feats.get("tcp_z")),
                "cube_r": _round(feats.get("cube_r")),
                "qpos0": _round(feats.get("qpos0")),
                "gripper_q": _round(feats.get("gripper_q")),
                **({"feature_check": fc} if fc is not None else {}),
            })

        t_inf = time.time()
        with torch.no_grad():
            obs_t = torch.as_tensor(arr).float()
            if getattr(agent, "obs_norm", None) is not None:
                mean_t, std_t = agent.obs_norm
                obs_t = (obs_t - mean_t.to(obs_t.device)) / std_t.to(obs_t.device)
            action_seq = agent.get_action(obs_t,
                                          task_id if task_id is not None else None)
        inference_time += time.time() - t_inf
        n_inf += 1
        action_seq = action_seq.cpu().numpy()
        k = min(k, action_seq.shape[1])

        truncated = False
        for i in range(k):
            obs, _rew, term, trunc, info = env.step(action_seq[:, i])
            n_steps += 1
            p = _scalar_from_info(info, "cube_progress")
            d = _scalar_from_info(info, "cube_distance")
            if p is not None:
                progress_max = p if progress_max is None else max(progress_max, p)
            if d is not None:
                dist_min = d if dist_min is None else min(dist_min, d)
                dist_final = d
            if trace and (n_steps % trace_every == 0):
                a2 = obs if not isinstance(obs, dict) else obs["state"]
                l2 = np.asarray(a2, dtype=np.float32).reshape(-1, np.asarray(a2).shape[-1])[-1]
                tr = extract_features(task, l2)
                tr.update(step=n_steps, decision=n_inf)
                trace_rows.append({kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                   for kk, vv in tr.items()})
            if bool(np.asarray(trunc).reshape(-1)[0]) or bool(np.asarray(term).reshape(-1)[0]):
                truncated = True
                break
        if truncated:
            break

    rec = {
        "task": task, "seed": int(seed), "schedule": horizon_policy.name,
        "schedule_params": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in horizon_policy.params.items()},
        "n_inference": n_inf, "n_env_steps": n_steps,
        "inference_time_s": round(inference_time, 3),
        "inference_ms_per_call": (round(1000 * inference_time / n_inf, 2) if n_inf else None),
        "wall_time_s": round(time.time() - t0, 3),
        "mean_horizon": float(np.mean(horizons)) if horizons else 0.0,
        "horizon_counts": {str(int(h)): int(horizons.count(h)) for h in sorted(set(horizons))},
        "cube_progress_max": progress_max,
        "cube_distance_min": dist_min,
        "cube_distance_final": dist_final,
    }

    ep = info.get("episode") if isinstance(info, dict) else None
    if isinstance(ep, (list, tuple)):
        ep = ep[0] if ep else None
    if isinstance(ep, dict):
        rec["success_once"] = bool(_scalar_from_info(ep, "success_once"))
        rec["success_at_end"] = bool(_scalar_from_info(ep, "success_at_end"))
        rec["return"] = _scalar_from_info(ep, "return")
        rec["episode_len"] = _scalar_from_info(ep, "episode_len")
    else:
        s = _scalar_from_info(info, "success")
        rec["success_once"] = bool(s) if s is not None else False
        rec["success_at_end"] = bool(s) if s is not None else False
        rec["return"] = None
        rec["episode_len"] = n_steps

    rec["final_features"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in feats.items()}
    if trace:
        rec["trace"] = trace_rows
    if decide_log:
        rec["decisions"] = decisions
    return rec
