#!/usr/bin/env bash
# Run from this directory: bash run_jobs.sh [run-name] [seed-count] [parallelism] [max-nfev]
set -eu
run_name=${1:-"search-$(date -u +%Y%m%dT%H%M%SZ)"}
seed_count=${2:-12}
parallelism=${3:-4}
max_nfev=${4:-120}
case "$run_name" in *[!a-zA-Z0-9_.-]*|'') printf 'Invalid run name\n' >&2; exit 2;; esac
case "$seed_count:$parallelism:$max_nfev" in *[!0-9:]*) printf 'Counts must be positive integers\n' >&2; exit 2;; esac
if [ "$seed_count" -lt 1 ] || [ "$parallelism" -lt 1 ] || [ "$max_nfev" -lt 1 ]; then printf 'Counts must be positive\n' >&2;exit 2;fi
run_dir="runs/$run_name"
if [ -e "$run_dir" ];then printf 'Run directory already exists: %s\n' "$run_dir" >&2;exit 2;fi
mkdir -p "$run_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'run_name=%s\nseed_count=%s\nparallelism=%s\nmax_nfev=%s\ncommand=bash run_jobs.sh %s %s %s %s\n' "$run_name" "$seed_count" "$parallelism" "$max_nfev" "$run_name" "$seed_count" "$parallelism" "$max_nfev" > "$run_dir/run-info.txt"
git rev-parse HEAD > "$run_dir/code-commit.txt"
git status --short --untracked-files=normal > "$run_dir/worktree-status.txt"
python -m pip freeze > "$run_dir/environment.txt"
seq 0 "$((seed_count-1))" | xargs -P "$parallelism" -I '{}' sh -c 'set -eu; s=$1; d=$2; n=$3; python fit_multistart.py --seed "$s" --max-nfev "$n" --out "$d/$s" > "$d/$s.log" 2>&1' _ '{}' "$run_dir" "$max_nfev"
printf 'Finished optimization. Inspect candidates with: python summarize_runs.py %s\n' "$run_dir"
