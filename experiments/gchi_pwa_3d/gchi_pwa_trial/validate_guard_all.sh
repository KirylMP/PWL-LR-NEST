#!/usr/bin/env bash
# Run from this directory: bash validate_guard_all.sh runs/<run-name> [parallelism]
set -eu
run_dir=${1:?provide run directory}
parallelism=${2:-4}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
find "$run_dir" -mindepth 2 -maxdepth 2 -name candidate.json -print0 | xargs -0 -r -P "$parallelism" -I '{}' sh -c 'set -eu; f=$1; python validate_guard_candidate.py "$f" > "${f%/*}/validation.log" 2>&1' _ '{}'
python summarize_guard_runs.py "$run_dir"
