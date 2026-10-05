"""Checkpoint loading and ManiSkill CPU environment construction.

The policy checkpoint and the teammate's model code are external dependencies and
are not redistributed with this repository. Point the tooling at them with
``--runs-root`` / ``--teammate-code`` or the ``DP_RUNS_ROOT`` / ``DP_TEAMMATE_CODE``
environment variables.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

POLICY_SEED_BASE = 1_000_000
TASK_ORDER = ["PickCube-v1", "PushCube-v1", "PullCubeTool-v1"]
TASK_INDEX = {t: i for i, t in enumerate(TASK_ORDER)}

_MODEL_MODULES = {
    "shared": "model_shared",
    "task_adapter": "model_task_adaptor",
    "shared_adapter": "model_shared_adaptor",
}


def teammate_code_root(explicit=None):
    """Directory containing the teammate's ``model_*.py`` and ``diffusion_policy`` package."""
    root = explicit or os.environ.get("DP_TEAMMATE_CODE")
    if not root:
        raise SystemExit(
            "teammate code root not set: pass --teammate-code or set DP_TEAMMATE_CODE "
            "to the directory holding model_shared.py and the diffusion_policy package"
        )
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"teammate code root {root} is not a directory")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def runs_root(explicit=None):
    """Directory containing the training runs (each with checkpoints/ and run_config.json)."""
    root = explicit or os.environ.get("DP_RUNS_ROOT")
    if not root:
        raise SystemExit(
            "checkpoint root not set: pass --runs-root or set DP_RUNS_ROOT to the "
            "directory containing the training runs"
        )
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"checkpoint root {root} is not a directory")
    return root


def load_run(run_name, ckpt_name="final.pt", device="cpu", code_root=None, ckpt_root=None):
    """Rebuild an Agent from a checkpoint's embedded config and load its EMA weights."""
    teammate_code_root(code_root)
    _p = Path(run_name)
    ckpt_path = _p if _p.suffix == ".pt" else runs_root(ckpt_root) / run_name / "checkpoints" / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    compat = ckpt.get("compat") or {
        "mode": ckpt.get("model_type", "shared"), "entries": ckpt["entries"],
        "settings": ckpt["settings"], "args": ckpt.get("args", {}),
        "state_normalization": False,
    }
    mode = compat["mode"]
    if mode not in _MODEL_MODULES:
        raise ValueError(f"unknown model mode {mode!r}")
    module = __import__(_MODEL_MODULES[mode])
    agent = module.Agent(compat["entries"], SimpleNamespace(**compat["settings"]))
    agent.load_state_dict(ckpt["ema_agent"], strict=True)
    agent.to(device).eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    meta = {
        "run_name": run_name, "ckpt_name": ckpt_name, "mode": mode,
        "entries": compat["entries"], "settings": compat["settings"], "args": compat["args"],
        "task_order": [e["task_id"] for e in compat["entries"]],
        "state_normalization": compat["state_normalization"],
        "completed_updates": ckpt["completed_updates"], "weights": "ema_agent",
        "obs_norm": ckpt.get("obs_norm"),
    }
    if meta["obs_norm"]:
        block = meta["obs_norm"]
        agent.obs_norm = (torch.as_tensor(block["mean"], dtype=torch.float32, device=device),
                          torch.as_tensor(block["std"], dtype=torch.float32, device=device))
    return agent, meta


def build_env(env_id, max_episode_steps, obs_horizon=2, num_envs=1, seed=0,
              sim_backend="physx_cpu", code_root=None):
    """Build a CPU evaluation environment with the official wrappers."""
    teammate_code_root(code_root)
    from diffusion_policy.make_env import make_eval_envs
    return make_eval_envs(
        env_id=env_id, num_envs=num_envs, sim_backend=sim_backend,
        env_kwargs={"obs_mode": "state", "control_mode": "pd_ee_delta_pose",
                    "max_episode_steps": max_episode_steps},
        other_kwargs={"obs_horizon": obs_horizon},
    )


def set_seed(seed):
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def policy_torch_seed(env_seed):
    """Torch seed paired with an environment seed (diffusion sampling stream)."""
    return POLICY_SEED_BASE + env_seed


def parse_seeds(spec):
    """Parse ``a-b,c,d`` (or a path to a whitespace-separated file) into a sorted list."""
    seeds = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif Path(part).exists():
            with open(part) as fh:
                seeds.extend(int(x) for x in fh.read().split() if x.strip().lstrip("-").isdigit())
        else:
            seeds.append(int(part))
    seen, out = set(), []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
