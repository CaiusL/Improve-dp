"""Observation layout and hand-designed geometric features for the three tasks."""
from __future__ import annotations

import numpy as np

OBS_FIELDS = {
    "PickCube-v1": [
        ("agent.qpos", 0, 9), ("agent.qvel", 9, 18), ("extra.is_grasped", 18, 19),
        ("extra.tcp_pose", 19, 26), ("extra.goal_pos", 26, 29), ("extra.obj_pose", 29, 36),
        ("extra.tcp_to_obj_pos", 36, 39), ("extra.obj_to_goal_pos", 39, 42),
    ],
    "PushCube-v1": [
        ("agent.qpos", 0, 9), ("agent.qvel", 9, 18), ("extra.tcp_pose", 18, 25),
        ("extra.goal_pos", 25, 28), ("extra.obj_pose", 28, 35),
    ],
    "PullCubeTool-v1": [
        ("agent.qpos", 0, 9), ("agent.qvel", 9, 18), ("extra.tcp_pose", 18, 25),
        ("extra.cube_pose", 25, 32), ("extra.tool_pose", 32, 39),
    ],
}

TASK_ORDER = ["PickCube-v1", "PushCube-v1", "PullCubeTool-v1"]
TASK_INDEX = {t: i for i, t in enumerate(TASK_ORDER)}


def field_slice(task, name):
    """Return the slice of the flattened state vector holding a named field."""
    for fname, start, end in OBS_FIELDS[task]:
        if fname == name:
            return slice(start, end)
    raise KeyError(f"{task} has no field {name}")


def _norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def extract_features(task, row):
    """Geometric features computed only from the baseline state observation."""
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    f = {}

    if task in ("PullCubeTool-v1", "PushCube-v1", "PickCube-v1"):
        qpos = row[0:9]
        f["gripper_q"] = float(np.mean(qpos[7:9]))
        f["qpos0"] = float(qpos[0])

    if task == "PullCubeTool-v1":
        tcp, cube, tool = row[18:21], row[25:28], row[32:35]
        f.update(
            tcp_pos=tcp, cube_pos=cube, tool_pos=tool,
            d_tcp_tool=_norm(tcp - tool),
            d_tool_cube=_norm(tool - cube),
            d_tcp_cube=_norm(tcp - cube),
            cube_r=float(np.linalg.norm(cube[:2])),
            tcp_z=float(tcp[2]),
            tool_z=float(tool[2]),
        )
    elif task == "PickCube-v1":
        f.update(
            is_grasped=float(row[18]),
            d_tcp_obj=_norm(row[36:39]),
            d_obj_goal=_norm(row[39:42]),
            d_tcp_goal=_norm(row[19:22] - row[26:29]),
            tcp_pos=row[19:22], goal_pos=row[26:29], obj_pos=row[29:32],
            d_obj_goal_xy=_norm(row[29:31] - row[26:28]),
            cube_z=float(row[31]),
            tcp_z=float(row[21]),
            obj_goal_z=float(row[41]),
        )
    elif task == "PushCube-v1":
        tcp, obj, goal = row[18:21], row[28:31], row[25:28]
        f.update(
            tcp_pos=tcp, obj_pos=obj, goal_pos=goal,
            d_tcp_obj=_norm(tcp - obj),
            d_tcp_obj_xy=_norm(tcp[:2] - obj[:2]),
            d_obj_goal=_norm(obj - goal),
            d_obj_goal_xy=_norm(obj[:2] - goal[:2]),
            obj_z=float(obj[2]),
            tcp_z=float(tcp[2]),
        )
    return f


def phase_label(task, feats, grasp_eps=0.06, near_d=0.30):
    """Coarse phase label used for logging only; never used by a rule."""
    if task == "PullCubeTool-v1":
        grasped = feats.get("d_tcp_tool", np.inf) < grasp_eps
        if not grasped:
            return "not_grasped"
        return "near_cube" if feats.get("d_tool_cube", np.inf) < near_d else "grasped_far"
    if task == "PushCube-v1":
        return "obj_near_goal" if feats.get("d_obj_goal_xy", np.inf) < 0.10 else "obj_far_goal"
    if task == "PickCube-v1":
        return "grasped" if feats.get("is_grasped", 0.0) >= 0.5 else "not_grasped"
    return "unknown"
