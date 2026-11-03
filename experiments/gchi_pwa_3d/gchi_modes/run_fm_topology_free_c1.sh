#!/usr/bin/env bash
set -euo pipefail
run=$1
maxiter=$2
weight=$3
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$run"
run=$(cd "$run" && pwd)
if find "$run" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Output directory must be empty: $run" >&2
  exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - "$here" "$run" "$maxiter" "$weight" <<'PY'
import hashlib,json,platform,subprocess,sys
from pathlib import Path
import numpy,scipy
here=Path(sys.argv[1]);out=Path(sys.argv[2])
files=['fit_fm_topology_free_c1.py','fit_modes.py','explore_modes.py',
       'results/FM/reference.json','runs/fm-topology-01/candidate-3.json',
       'analysis/fm-constrained-01/directional-scans.json',
       'analysis/fm-constrained-01/reference-fold-candidate.json']
try: head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=here,text=True).strip()
except (OSError,subprocess.CalledProcessError):head=None
(out/'manifest.json').write_text(json.dumps({
    'git_head':head,'source_seed':3,'python':platform.python_version(),
    'numpy':numpy.__version__,'scipy':scipy.__version__,
    'maxiter':int(sys.argv[3]),'negative_calcium_weight':float(sys.argv[4]),
    'c1_values':[.033,.037,.041258585230769904,.045,.049],
    'sha256':{p:hashlib.sha256((here/p).read_bytes()).hexdigest() for p in files}
},indent=2)+'\n')
PY
for spec in "0 .033" "1 .037" "2 .041258585230769904" "3 .045" "4 .049"; do
  read -r idx knot <<< "$spec"
  echo "=== C1=$knot seed=3 index=$idx ==="
  python "$here/fit_fm_topology_free_c1.py" \
    --data "$here/results/FM/reference.json" \
    --start "$here/runs/fm-topology-01/candidate-3.json" \
    --onset-data "$here/analysis/fm-constrained-01/directional-scans.json" \
    --fold-target "$here/analysis/fm-constrained-01/reference-fold-candidate.json" \
    --output "$run/candidate-$idx.json" --seed 0 --maxiter "$maxiter" \
    --full-probes --lower-knot "$knot" --neg-c-weight "$weight" \
    > "$run/fit-$idx.log" 2>&1
  python "$here/validate_modes.py" --reference "$here/results/FM/reference.json" \
    --candidate "$run/candidate-$idx.json" \
    --output "$run/validation-$idx.json" > "$run/validation-$idx.log" 2>&1
  python "$here/analyze_equilibria.py" --candidate "$run/candidate-$idx.json" \
    --output "$run/equilibria-$idx.json" > "$run/equilibria-$idx.log" 2>&1
  python "$here/check_fm_onset.py" --candidate "$run/candidate-$idx.json" \
    --scan "$here/analysis/fm-constrained-01/directional-scans.json" \
    --output "$run/onset-$idx.json" > "$run/onset-$idx.log" 2>&1
  python "$here/check_fm_fold_proxy.py" --candidate "$run/candidate-$idx.json" \
    --output "$run/fold-check-$idx.json" > "$run/fold-check-$idx.log" 2>&1
done
python - "$run" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);rows=[]
for i in range(5):
    c=json.loads((run/f'candidate-{i}.json').read_text())
    v=json.loads((run/f'validation-{i}.json').read_text())
    rows.append(dict(index=i,c1=c['knots'][0],
                     objective=c['objective'],cycles=v['detected_cycles'],
                     total=v['total'],errors=v['conditional_MAE'],
                     solver_success=c['optimizer_success'],
                     physical_failures=[{'gamma':r['gamma'],'status':r['candidate']['status']}
                                        for r in v['records']
                                        if r['candidate']['status'].startswith(('negative','h_outside'))]))
(run/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
    print(r['index'], 'C1=',round(r['c1'],6),
          'cycles=',str(r['cycles'])+'/'+str(r['total']),
          'period_MAE=',r['errors']['period'],
          'Cmax_MAE=',r['errors']['Cmax'],
          'physical_failures=',len(r['physical_failures']))
PY
