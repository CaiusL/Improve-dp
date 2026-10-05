#!/usr/bin/env python
"""Select a checkpoint by held-out performance on validation seeds only."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from adaptive_replan import parse_seeds  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--schedule", default="fixed:8")
    ap.add_argument("--candidates", required=True,
                    help="comma list of run:checkpoint, e.g. run_a:final.pt,run_b:best.pt")
    ap.add_argument("--seeds", default="100000-100019")
    ap.add_argument("--phase", default="ckpt_select")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--teammate-code", default=None)
    ap.add_argument("--raw-root", default=str(ROOT / "results" / "raw"))
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    cands = []
    for c in args.candidates.split(","):
        run, _, ck = c.partition(":")
        cands.append((run.strip(), ck.strip() or "final.pt"))

    jobs = []
    for run, ck in cands:
        outdir = Path(args.raw_root) / args.phase / args.task / f"{run}__{ck.replace('.pt', '')}"
        outdir.mkdir(parents=True, exist_ok=True)
        for ci in range(0, len(seeds), args.chunk):
            cs = seeds[ci:ci + args.chunk]
            idx = ci // args.chunk
            if (outdir / f"chunk_{idx:03d}.done").exists():
                continue
            jobs.append((run, ck, idx, cs, outdir / f"chunk_{idx:03d}.jsonl", outdir))

    print(f"[{args.phase}] {len(cands)} candidates x {len(seeds)} seeds -> {len(jobs)} chunks")

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    t0 = time.time()
    done = [0]

    def work(job):
        run, ck, idx, cs, cfile, outdir = job
        cmd = [sys.executable, str(HERE / "run_rollout.py"), "--task", args.task,
               "--schedule", args.schedule, "--run", run, "--ckpt", ck, "--device", args.device,
               "--seeds", ",".join(map(str, cs)), "--out", str(cfile)]
        if args.runs_root:
            cmd += ["--runs-root", args.runs_root]
        if args.teammate_code:
            cmd += ["--teammate-code", args.teammate_code]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(HERE))
        ok = p.returncode == 0 and cfile.exists()
        if ok:
            (outdir / f"chunk_{idx:03d}.done").write_text("ok\n")
        else:
            (outdir / f"chunk_{idx:03d}.err").write_text(
                (p.stdout or "")[-3000:] + "\n--STDERR--\n" + (p.stderr or "")[-3000:])
        done[0] += 1
        print(f"[{done[0]}/{len(jobs)}] {run} {ck} chunk{idx} {'ok' if ok else 'FAIL'} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(work, jobs))

    print("\n=== checkpoint selection (validation seeds only) ===")
    for run, ck in cands:
        outdir = Path(args.raw_root) / args.phase / args.task / f"{run}__{ck.replace('.pt', '')}"
        merged = outdir / "merged.jsonl"
        if merged.exists():
            merged.unlink()
        recs = []
        for f in sorted(outdir.glob("chunk_*.jsonl")):
            for line in f.read_text().splitlines():
                if line.strip():
                    recs.append(json.loads(line))
        if not recs:
            print(f"  {run:<28} {ck:<30} no episodes")
            continue
        merged.write_text("".join(json.dumps(r) + "\n" for r in recs))
        k = sum(1 for r in recs if r.get("success_once"))
        print(f"  {run:<28} {ck:<30} n={len(recs):>3} succ={k:>3} rate={k/len(recs):.3f} "
              f"mean_inf={sum(r['n_inference'] for r in recs)/len(recs):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
