#!/usr/bin/env bash
set -euo pipefail
mode=${1:?Mode FM or AFM}; run=${2:?Unique run directory}; count=${3:-12}; workers=${4:-4}; nfev=${5:-100}
case "$mode" in FM|AFM) ;; *) echo 'mode must be FM or AFM' >&2; exit 1;; esac
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$run"
run=$(cd "$run" && pwd)
if find "$run" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then echo "run directory is not empty: $run" >&2; exit 1; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TASK_DIR="$here" TASK_RUN="$run" TASK_MODE="$mode" TASK_NFEV="$nfev"
seq 0 "$((count-1))" | xargs -P "$workers" -I '{}' bash -c 'extra=(); if [[ "$TASK_MODE" == FM ]]; then extra=(--start "$TASK_DIR/results/FM/initial.json"); fi; python "$TASK_DIR/fit_modes.py" --mode "$TASK_MODE" --data "$TASK_DIR/results/$TASK_MODE/reference.json" --output "$TASK_RUN/candidate-$1.json" --seed "$1" --nfev "$TASK_NFEV" "${extra[@]}" > "$TASK_RUN/fit-$1.log" 2>&1' _ '{}'
seq 0 "$((count-1))" | xargs -P "$workers" -I '{}' bash -c 'python "$TASK_DIR/validate_modes.py" --reference "$TASK_DIR/results/$TASK_MODE/reference.json" --candidate "$TASK_RUN/candidate-$1.json" --output "$TASK_RUN/validation-$1.json" > "$TASK_RUN/validation-$1.log" 2>&1' _ '{}'
python "$here/summarize.py" "$run"
