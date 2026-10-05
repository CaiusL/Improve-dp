#!/usr/bin/env python
"""Run one schedule on one task for a list of seeds, writing per-episode JSONL."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_replan import (POLICY_SEED_BASE, TASK_INDEX, build_env, build_horizon_policy,
                             load_run, parse_seeds, run_episode)

ORIG_CWD = Path.cwd()


def restore_cwd():
    """ManiSkill env creation can change the process working directory on macOS."""
    try:
        os.chdir(ORIG_CWD)
    except OSError:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="PushCube-v1", choices=sorted(TASK_INDEX))
    ap.add_argument("--run", default="panda3_shared_a0", help="training run directory name")
    ap.add_argument("--ckpt", default="final.pt", help="checkpoint file inside checkpoints/")
    ap.add_argument("--schedule", default="fixed:8", help="schedule spec, e.g. 'fixed:4'")
    ap.add_argument("--seeds", default="200000-200019", help="'a-b', 'a,b,c', or a file path")
    ap.add_argument("--max-steps", type=int, default=None, help="override episode length")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="output JSONL path")
    ap.add_argument("--append", action="store_true", help="append instead of truncating")
    ap.add_argument("--trace", action="store_true", help="store per-step feature traces")
    ap.add_argument("--trace-every", type=int, default=10)
    ap.add_argument("--decision-log", action="store_true", help="store one row per decision")
    ap.add_argument("--diffusion-steps", type=int, default=None,
                    help="override sampling steps via scheduler.set_timesteps(N)")
    ap.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm",
                    help="sampling algorithm; ddim rebuilds a deterministic scheduler from "
                         "the checkpoint's trained betas (epsilon prediction, clip_sample)")
    ap.add_argument("--runs-root", default=None, help="root holding the training runs")
    ap.add_argument("--teammate-code", default=None, help="root of the teammate's model code")
    args = ap.parse_args(argv)

    task = args.task
    agent, meta = load_run(args.run, args.ckpt, device=args.device,
                           code_root=args.teammate_code, ckpt_root=args.runs_root)
    settings = meta["settings"]
    max_h = int(settings["act_horizon"])
    obs_horizon = int(settings["obs_horizon"])
    if settings.get("state_normalization"):
        raise SystemExit("checkpoint expects state normalization; not implemented")

    if args.sampler == "ddim":
        from diffusers import DDIMScheduler
        betas = agent.noise_scheduler.betas.tolist()
        agent.noise_scheduler = DDIMScheduler(num_train_timesteps=len(betas),
                                              trained_betas=betas, clip_sample=True,
                                              prediction_type="epsilon")
    if args.diffusion_steps or args.sampler == "ddim":
        agent.noise_scheduler.set_timesteps(args.diffusion_steps or 100)
        print(f"[sampler = {args.sampler} steps = "
              f"{len(agent.noise_scheduler.timesteps)}]", flush=True)

    entries = meta["entries"]
    task_id = meta["task_order"].index(task)
    max_steps = args.max_steps or int(entries[task_id]["max_episode_steps"])

    out = Path(args.out) if args.out else Path(f"{task}_{args.schedule.replace(':', '_')}.jsonl")
    if not out.is_absolute():
        out = (ORIG_CWD / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    env = build_env(task, max_steps, obs_horizon=obs_horizon, num_envs=1, seed=0,
                    code_root=args.teammate_code)
    restore_cwd()

    hp = build_horizon_policy(args.schedule, max_h)
    print(f"[run={args.run} mode={meta['mode']} task={task} obs_horizon={obs_horizon} "
          f"pred_horizon={settings['pred_horizon']} act_horizon={max_h}]", flush=True)
    print(f"[schedule={args.schedule} params={hp.params} max_steps={max_steps} "
          f"n_seeds={len(seeds)} -> {out}]", flush=True)

    done = set()
    if args.append and out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["seed"]))

    written = 0
    with open(out, "a" if args.append else "w") as fh:
        for i, seed in enumerate(seeds):
            if seed in done:
                continue
            hp.rng = __import__("numpy").random.default_rng(12345 + seed)
            rec = run_episode(env, agent, task, task_id, hp, seed, max_steps,
                              trace=args.trace, trace_every=args.trace_every,
                              decide_log=args.decision_log)
            rec.update(run=args.run, ckpt=args.ckpt, mode=meta["mode"],
                       pred_horizon=settings["pred_horizon"], act_horizon=max_h)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            written += 1
            print(f"  [{i+1}/{len(seeds)}] seed={seed} succ={rec['success_once']} "
                  f"inf={rec['n_inference']} steps={rec['n_env_steps']} "
                  f"h={rec['mean_horizon']:.2f} t={rec['wall_time_s']:.1f}s", flush=True)

    env.close()
    print(f"[wrote {written} episodes -> {out}]", flush=True)

    meta_out = out.with_suffix(".meta.json")
    if not meta_out.exists():
        meta_out.write_text(json.dumps({
            "task": task, "run": args.run, "ckpt": args.ckpt, "mode": meta["mode"],
            "schedule": args.schedule, "schedule_class": hp.name, "schedule_params": hp.params,
            "max_legal_horizon": max_h, "obs_horizon": obs_horizon,
            "pred_horizon": settings["pred_horizon"], "act_horizon": settings["act_horizon"],
            "max_episode_steps": max_steps, "n_seeds": len(seeds), "seeds": seeds,
            "device": args.device, "torch": torch.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "sim_backend": "physx_cpu", "obs_mode": "state", "control_mode": "pd_ee_delta_pose",
            "state_normalization": False, "weights": meta["weights"],
            "completed_updates": meta["completed_updates"],
            "policy_seed_base": POLICY_SEED_BASE,
            "diffusion_steps": args.diffusion_steps or "checkpoint default",
            "sampler": args.sampler,
        }, indent=2))


if __name__ == "__main__":
    sys.exit(main())
