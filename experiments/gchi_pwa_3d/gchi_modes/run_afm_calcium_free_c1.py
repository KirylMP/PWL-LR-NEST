#!/usr/bin/env python3
"""Five parallel AFM calcium-focused fits, with independent adaptive validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy
import scipy

KNOTS = [.095, .105, .11277377008786005, .12, .135]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--nfev", type=int, default=70)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--calcium-weight", type=float, default=3.)
    ap.add_argument("--negative-c-weight", type=float, default=8.)
    args = ap.parse_args()
    if not 1 <= args.workers <= 16:
        ap.error("workers must be 1..16")
    base = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output.resolve()
    if any(out.iterdir()):
        ap.error("output directory must be empty")
    files = [
        "fit_afm_calcium_free_c1.py", "fit_modes.py", "explore_modes.py",
        "results/AFM/reference.json", "results/AFM/candidate_segments.json"]
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=base, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    manifest = dict(git_head=head, python=platform.python_version(),
                    numpy=numpy.__version__, scipy=scipy.__version__,
                    nfev=args.nfev, workers=args.workers,
                    calcium_weight=args.calcium_weight,
                    negative_c_weight=args.negative_c_weight,
                    c1_values=KNOTS,
                    sha256={p: hashlib.sha256((base/p).read_bytes()).hexdigest()
                            for p in files})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")

    def invoke(args_list, logfile):
        with (out/logfile).open("w") as log:
            return subprocess.run([sys.executable, *map(str, args_list)],
                                  cwd=base, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode == 0

    def fit_and_validate(idx, knot):
        print(f"[start] index={idx} C1={knot}", flush=True)
        cand = out/f"candidate-{idx}.json"
        fit = ["fit_afm_calcium_free_c1.py",
               "--data", base/"results/AFM/reference.json",
               "--start", base/"results/AFM/candidate_segments.json",
               "--output", cand, "--c1", knot,
               "--nfev", args.nfev, "--calcium-weight", args.calcium_weight,
               "--negative-c-weight", args.negative_c_weight]
        if not invoke(fit, f"fit-{idx}.log"):
            (out/f"status-{idx}.txt").write_text("fit_failed\n")
            print(f"[failed] fit {idx}: fit-{idx}.log", flush=True)
            return False
        tasks = [
            (["validate_modes.py", "--reference",
              base/"results/AFM/reference.json", "--candidate", cand,
              "--output", out/f"validation-{idx}.json"],
             f"validation-{idx}.log"),
            (["analyze_equilibria.py", "--candidate", cand,
              "--output", out/f"equilibria-{idx}.json"],
             f"equilibria-{idx}.log")]
        all_ok = True
        for command, logfile in tasks:
            if not invoke(command, logfile):
                all_ok = False
                print(f"[failed] {logfile}", flush=True)
        (out/f"status-{idx}.txt").write_text(
            "ok\n" if all_ok else "validation_failed\n")
        if all_ok:
            print(f"[done] index={idx} C1={knot}", flush=True)
        return all_ok

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_and_validate, i, knot)
                   for i, knot in enumerate(KNOTS)]
        finished = [f.result() for f in as_completed(futures)]
    rows = []
    for i, knot in enumerate(KNOTS):
        vpath = out/f"validation-{i}.json"
        cpath = out/f"candidate-{i}.json"
        if not vpath.exists() or not cpath.exists():
            rows.append(dict(index=i, c1=knot, status="failed_or_incomplete"))
            continue
        c = json.loads(cpath.read_text())
        v = json.loads(vpath.read_text())
        rows.append(dict(index=i, c1=knot, objective=c["objective"],
                         cycles=v["detected_cycles"], total=v["total"],
                         conditional_MAE=v["conditional_MAE"],
                         optimizer_success=c["optimizer_success"],
                         max_anchor_residual=c["max_anchor_residual"],
                         rejected=[dict(gamma=r["gamma"],
                                        status=r["candidate"]["status"])
                                   for r in v["records"]
                                   if r["candidate"]["status"] != "cycle"]))
    (out/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
    for row in rows:
        if "cycles" in row:
            e = row["conditional_MAE"]
            print(f"{row['index']} C1={row['c1']:.6f} "
                  f"cycles={row['cycles']}/{row['total']} "
                  f"Cmax_MAE={e['Cmax']} Cmin_MAE={e['Cmin']} "
                  f"period_MAE={e['period']}", flush=True)
        else:
            print(row, flush=True)
    if not all(finished):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
