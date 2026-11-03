#!/usr/bin/env bash
set -euo pipefail
run=${1:?Unique run directory}; count=${2:-8}; workers=${3:-4}; maxiter=${4:-80}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$run"
run=$(cd "$run" && pwd)
if find "$run" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then echo "run directory is not empty: $run" >&2; exit 1; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TASK_DIR="$here" TASK_RUN="$run" TASK_MAXITER="$maxiter"
seq 0 "$((count-1))" | xargs -P "$workers" -I '{}' bash -c 'python "$TASK_DIR/fit_fm_constrained.py" --data "$TASK_DIR/results/FM/reference.json" --start "$TASK_DIR/results/FM/constrained_seed.json" --output "$TASK_RUN/candidate-$1.json" --seed "$1" --maxiter "$TASK_MAXITER" --full-probes > "$TASK_RUN/fit-$1.log" 2>&1' _ '{}'
seq 0 "$((count-1))" | xargs -P "$workers" -I '{}' bash -c 'python "$TASK_DIR/validate_modes.py" --reference "$TASK_DIR/results/FM/reference.json" --candidate "$TASK_RUN/candidate-$1.json" --output "$TASK_RUN/validation-$1.json" > "$TASK_RUN/validation-$1.log" 2>&1; python "$TASK_DIR/analyze_equilibria.py" --candidate "$TASK_RUN/candidate-$1.json" --output "$TASK_RUN/equilibria-$1.json" > "$TASK_RUN/equilibria-$1.log" 2>&1' _ '{}'
python "$here/summarize_fm_constrained.py" "$run"
