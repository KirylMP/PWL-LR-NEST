#!/usr/bin/env bash
set -euo pipefail
run=${1:?Unique run directory}; count=${2:-4}; workers=${3:-4}; maxnfev=${4:-20}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$run"
run=$(cd "$run" && pwd)
if find "$run" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then echo "run directory is not empty: $run" >&2; exit 1; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TASK_DIR="$here" TASK_RUN="$run" TASK_MAXNFEV="$maxnfev" TASK_COUNT="$count" TASK_WORKERS="$workers"
python - <<'PYCODE'
import hashlib,json,os,platform,subprocess
from pathlib import Path
import numpy,scipy
base=Path(os.environ['TASK_DIR']);files=['refine_first_return.py','poincare_fm_exact.py','fit_modes.py','explore_modes.py','runs/fm-topology-01/candidate-3.json','analysis/fm-first-return/fold.json','analysis/fm-first-return/chart.json','results/FM/reference.json','analysis/fm-constrained-01/directional-scans.json','analysis/fm-constrained-01/reference-fold-candidate.json']
try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True,stderr=subprocess.DEVNULL).strip()
except (OSError,subprocess.CalledProcessError):commit=None
out={'git_head':commit,'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'count':int(os.environ['TASK_COUNT']),'workers':int(os.environ['TASK_WORKERS']),'max_nfev':int(os.environ['TASK_MAXNFEV']),'file_sha256':{f:hashlib.sha256((base/f).read_bytes()).hexdigest() for f in files}}
(Path(os.environ['TASK_RUN'])/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
PYCODE
seq 0 "$((count-1))" | xargs -P "$workers" -I '{}' bash -c 'set -e; python "$TASK_DIR/refine_first_return.py" --candidate "$TASK_DIR/runs/fm-topology-01/candidate-3.json" --fold "$TASK_DIR/analysis/fm-first-return/fold.json" --chart "$TASK_DIR/analysis/fm-first-return/chart.json" --reference "$TASK_DIR/results/FM/reference.json" --scan "$TASK_DIR/analysis/fm-constrained-01/directional-scans.json" --output "$TASK_RUN/refinement-$1.json" --seed "$1" --max-nfev "$TASK_MAXNFEV" > "$TASK_RUN/refinement-$1.log" 2>&1; python "$TASK_DIR/export_first_return_candidate.py" --template "$TASK_DIR/runs/fm-topology-01/candidate-3.json" --fit "$TASK_RUN/refinement-$1.json" --output "$TASK_RUN/candidate-$1.json" --fold-output "$TASK_RUN/fold-$1.json"' _ '{}'
seq 0 "$((count-1))" | xargs -P "$workers" -I '{}' bash -c 'set -e; python "$TASK_DIR/validate_modes.py" --reference "$TASK_DIR/results/FM/reference.json" --candidate "$TASK_RUN/candidate-$1.json" --output "$TASK_RUN/validation-$1.json" > "$TASK_RUN/validation-$1.log" 2>&1; python "$TASK_DIR/check_fm_onset.py" --candidate "$TASK_RUN/candidate-$1.json" --scan "$TASK_DIR/analysis/fm-constrained-01/directional-scans.json" --output "$TASK_RUN/onset-$1.json" > "$TASK_RUN/onset-$1.log" 2>&1; python "$TASK_DIR/analyze_equilibria.py" --candidate "$TASK_RUN/candidate-$1.json" --output "$TASK_RUN/equilibria-$1.json" > "$TASK_RUN/equilibria-$1.log" 2>&1; python "$TASK_DIR/check_first_return_candidate.py" --candidate "$TASK_RUN/candidate-$1.json" --output "$TASK_RUN/return-check-$1.json" > "$TASK_RUN/return-check-$1.log" 2>&1; if ! python "$TASK_DIR/verify_poincare_fold.py" --candidate "$TASK_DIR/runs/fm-topology-01/candidate-3.json" --fold "$TASK_RUN/fold-$1.json" --output "$TASK_RUN/fold-branches-$1.json" > "$TASK_RUN/fold-branches-$1.log" 2>&1; then echo "Branch check failed; inspect fold-branches-$1.log" >&2; fi' _ '{}'
python "$here/summarize_first_return_refinement.py" "$run"
