#!/usr/bin/env python
"""Run a schedule x seed matrix with parallel single-threaded worker processes.

Each (task, schedule, seed chunk) writes its own JSONL plus a `.done` marker, so an
interrupted run resumes without repeating completed chunks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from adaptive_replan import parse_schedule, parse_seeds  # noqa: E402

DEFAULT_RAW = HERE.parent / "results" / "raw"


def tag_of(spec):
    return re.sub(r"[^A-Za-z0-9]+", "_", spec).strip("_")


def chunk(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, help="output subdirectory name")
    ap.add_argument("--tasks", default="PushCube-v1")
    ap.add_argument("--schedules", required=True,
                    help="SEMICOLON-separated specs; a spec's own params use commas")
    ap.add_argument("--seeds", default="200000-200199")
    ap.add_argument("--chunk", type=int, default=10, help="episodes per worker process")
    ap.add_argument("--jobs", type=int, default=8, help="max concurrent workers")
    ap.add_argument("--run", default="panda3_shared_a0")
    ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--decision-log", action="store_true")
    ap.add_argument("--feature-check", action="store_true",
                    help="pass --feature-check through to run_rollout.py (implies "
                         "--decision-log rows; round-6 raw-feature audit)")
    ap.add_argument("--diffusion-steps", type=int, default=None)
    ap.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm",
                    help="sampling algorithm passed through to run_rollout.py; ddim rebuilds a "
                         "deterministic scheduler from the checkpoint's trained betas")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--teammate-code", default=None)
    ap.add_argument("--raw-root", default=str(DEFAULT_RAW))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    schedules = [s.strip() for s in args.schedules.split(";") if s.strip()]
    seeds = parse_seeds(args.seeds)
    for spec in schedules:
        parse_schedule(spec)

    raw_root = Path(args.raw_root).resolve()
    jobs = []
    for task in tasks:
        for spec in schedules:
            outdir = raw_root / args.phase / task / tag_of(spec)
            outdir.mkdir(parents=True, exist_ok=True)
            for ci, cs in enumerate(chunk(seeds, args.chunk)):
                if (outdir / f"chunk_{ci:03d}.done").exists():
                    continue
                jobs.append((task, spec, ci, cs, outdir / f"chunk_{ci:03d}.jsonl", outdir))
    jobs.sort(key=lambda j: (j[2], j[0], j[1]))

    print(f"[phase={args.phase}] {len(tasks)} task(s) x {len(schedules)} schedule(s) "
          f"x {len(seeds)} seeds -> {len(jobs)} pending chunk(s), jobs={args.jobs}")
    if args.dry_run:
        for j in jobs[:40]:
            print("   ", j[0], j[1], "chunk", j[2], f"seeds={j[3][0]}..{j[3][-1]}")
        return 0

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        env[k] = "1"
    env["PYTHONWARNINGS"] = "ignore"

    t_start = time.time()
    done = [0]

    def work(job):
        task, spec, ci, cs, cfile, outdir = job
        cmd = [sys.executable, str(HERE / "run_rollout.py"),
               "--task", task, "--schedule", spec, "--run", args.run, "--ckpt", args.ckpt,
               "--device", args.device, "--seeds", ",".join(str(s) for s in cs),
               "--out", str(cfile)]
        if args.max_steps:
            cmd += ["--max-steps", str(args.max_steps)]
        if args.trace:
            cmd += ["--trace", "--trace-every", "10"]
        if args.decision_log:
            cmd += ["--decision-log"]
        if args.feature_check:
            if not args.decision_log:
                cmd += ["--decision-log"]
            cmd += ["--feature-check"]
        if args.diffusion_steps:
            cmd += ["--diffusion-steps", str(args.diffusion_steps)]
        if args.sampler != "ddpm":
            cmd += ["--sampler", args.sampler]
        if args.runs_root:
            cmd += ["--runs-root", args.runs_root]
        if args.teammate_code:
            cmd += ["--teammate-code", args.teammate_code]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(HERE))
        ok = p.returncode == 0 and cfile.exists()
        if ok:
            (outdir / f"chunk_{ci:03d}.done").write_text("ok\n")
        else:
            (outdir / f"chunk_{ci:03d}.err").write_text(
                (p.stdout or "")[-4000:] + "\n--STDERR--\n" + (p.stderr or "")[-4000:])
        done[0] += 1
        el = time.time() - t_start
        print(f"[{done[0]}/{len(jobs)}] {task} {spec} chunk{ci} {'ok' if ok else 'FAIL'} "
              f"{time.time()-t0:.0f}s (elapsed {el/60:.1f}m, "
              f"eta {el/done[0]*(len(jobs)-done[0])/60:.1f}m)", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(work, jobs))
    ok = sum(1 for r in results if r)
    print(f"[phase={args.phase}] {ok}/{len(jobs)} chunks ok in {(time.time()-t_start)/60:.1f} min")

    for task in tasks:
        for spec in schedules:
            outdir = raw_root / args.phase / task / tag_of(spec)
            recs = []
            for f in sorted(outdir.glob("chunk_*.jsonl")):
                for line in f.read_text().splitlines():
                    if line.strip():
                        recs.append(json.loads(line))
            if recs:
                merged = outdir / "merged.jsonl"
                if merged.exists():
                    old = {json.loads(l)["seed"] for l in merged.read_text().splitlines() if l.strip()}
                    recs = [r for r in recs if r["seed"] not in old] + \
                           [json.loads(l) for l in merged.read_text().splitlines() if l.strip()]
                merged.write_text("".join(json.dumps(r) + "\n" for r in recs))
                print(f"  merged {len(recs):4d} episodes -> {merged}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
