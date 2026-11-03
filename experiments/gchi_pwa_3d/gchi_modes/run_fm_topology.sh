#!/usr/bin/env bash
set -euo pipefail
run=${1:?Unique run directory}; count=${2:-4}; workers=${3:-4}; maxiter=${4:-40}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$run"
run=$(cd "$run" && pwd)
if find "$run" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then echo "run directory is not empty: $run" >&2; exit 1; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TASK_DIR="$here" TASK_RUN="$run" TASK_MAXITER="$maxiter" TASK_COUNT="$count" TASK_WORKERS="$workers"
python - <<'PYCODE'
import hashlib,json,os,platform,subprocess
from pathlib import Path
import numpy,scipy
base=Path(os.environ['TASK_DIR']);files=['fit_fm_topology.py','fit_modes.py','explore_modes.py','results/FM/reference.json','runs/fm-constrained-01/candidate-6.json','analysis/fm-constrained-01/directional-scans.json','analysis/fm-constrained-01/reference-fold-candidate.json']
try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True,stderr=subprocess.DEVNULL).strip()
except (OSError,subprocess.CalledProcessError):commit=None
out={'git_head':commit,'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'count':int(os.environ['TASK_COUNT']),'workers':int(os.environ['TASK_WORKERS']),'maxiter':int(os.environ['TASK_MAXITER']),'file_sha256':{f:hashlib.sha256((base/f).read_bytes()).hexdigest() for f in files}}
(Path(os.environ['TASK_RUN'])/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
PYCODE
seq 0 "$((count-1))" | xargs -P "$workers" -I '{}' bash -c 'set -e; python "$TASK_DIR/fit_fm_topology.py" --data "$TASK_DIR/results/FM/reference.json" --start "$TASK_DIR/runs/fm-constrained-01/candidate-6.json" --onset-data "$TASK_DIR/analysis/fm-constrained-01/directional-scans.json" --fold-target "$TASK_DIR/analysis/fm-constrained-01/reference-fold-candidate.json" --output "$TASK_RUN/candidate-$1.json" --seed "$1" --maxiter "$TASK_MAXITER" --full-probes > "$TASK_RUN/fit-$1.log" 2>&1' _ '{}'
seq 0 "$((count-1))" | xargs -P "$workers" -I '{}' bash -c 'set -e; python "$TASK_DIR/validate_modes.py" --reference "$TASK_DIR/results/FM/reference.json" --candidate "$TASK_RUN/candidate-$1.json" --output "$TASK_RUN/validation-$1.json" > "$TASK_RUN/validation-$1.log" 2>&1; python "$TASK_DIR/analyze_equilibria.py" --candidate "$TASK_RUN/candidate-$1.json" --output "$TASK_RUN/equilibria-$1.json" > "$TASK_RUN/equilibria-$1.log" 2>&1; python "$TASK_DIR/check_fm_fold_proxy.py" --candidate "$TASK_RUN/candidate-$1.json" --output "$TASK_RUN/fold-check-$1.json" > "$TASK_RUN/fold-check-$1.log" 2>&1; python "$TASK_DIR/check_fm_onset.py" --candidate "$TASK_RUN/candidate-$1.json" --scan "$TASK_DIR/analysis/fm-constrained-01/directional-scans.json" --output "$TASK_RUN/onset-$1.json" > "$TASK_RUN/onset-$1.log" 2>&1' _ '{}'
python "$here/summarize_fm_topology.py" "$run"
