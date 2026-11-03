# FM topology: stronger negative calcium penalty + moving first C knot

This experiment resumes the **previous FM topology fit**, not the later artificial terminal IP3 guard. It uses `runs/fm-topology-01/candidate-3.json` as the reference starting candidate. Original candidate, original `fit_fm_topology.py`, G-ChI parameters, 4-region model, input exponent, and two upper switching thresholds remain unchanged.

## Changes

- Grid of five first calcium knot positions: `0.033, 0.037, 0.041258585230769904, 0.045, 0.049` uM. The center point is the original first knot. Every choice remains strictly between the reference rest calcium `0.02866107` and the fixed second knot `0.05385610`.
- At each knot, recompute the null-space of the three pinned-equilibrium constraints, then reproject seed-3 coefficients before SLSQP. Thus all three *full-state equilibrium anchors* stay pinned algebraically.
- Increase the soft negative-C penalty coefficient from 2 to 8 (configurable via `--neg-c-weight`), consistently for short phase segments, full-period trajectories, near-onset long trajectory, and fold-proxy trajectory. The negative IP3 and h penalties remain unchanged.
- Preserve the regional spectral sign constraints stable/stable/unstable/stable. Movement of the knot may make a projected initial candidate infeasible, in which case that sweep item is logged as failed, not a successful fit.
- Keep the original finite-step/soft fold-proxy loss as an exploratory objective. Independent adaptive validation, onset, equilibria and fold-proxy checks are run after fitting. Result metrics are not presumed to improve.

## Run

From `experiments/gchi_pwa_3d/gchi_modes`:

```bash
source ../gchi_pwa_trial/.venv/bin/activate
bash run_fm_topology_free_c1.sh runs/fm-topology-free-c1-01 25 8 5 \
  | tee fm-topology-free-c1-summary.txt
```

Positional arguments: new empty output directory, SLSQP max iterations per knot, negative calcium weight, maximum parallel workers (default: 5, max: 16). Five knot cases run concurrently by default on a 16-core server. Each case writes disjoint files and runs its own independent diagnostics. BLAS, OpenMP, MKL and NumExpr threads are limited to one per process; failed cases are tracked in status-N.txt and summary.json. The runner writes manifest, fit/candidate/validation/equilibria/onset/fold-check per knot, and `summary.json`. Failed starting points are reported and skipped. This may be expensive: each knot independently integrates long onset and fold trajectories for optimizer evaluations.

For a smaller first test, run a single fixed knot (e.g. `0.037`):

```bash
python fit_fm_topology_free_c1.py \
  --data results/FM/reference.json \
  --start runs/fm-topology-01/candidate-3.json \
  --onset-data analysis/fm-constrained-01/directional-scans.json \
  --fold-target analysis/fm-constrained-01/reference-fold-candidate.json \
  --output runs/fm-topology-free-c1-smoke/candidate.json \
  --lower-knot 0.037 --neg-c-weight 8 --maxiter 1 --full-probes
```

## Compare

Use physical cycle count before conditional MAE, then onset period at `gamma=.0203`, negative-C and negative-IP3 trajectory minima, continuation of the rest branch, and upper fold diagnostics. Objective values may be compared **within this sweep only**, and the original objective of seed 3 is not directly comparable because the penalty weight changed.

Increasing penalty for `C<0` only controls sampled trajectory excursions; the optimizer's clipped integration may conceal physically invalid trajectories. A stronger penalty alone cannot prove `C>=0` is invariant. Reaching a small fold-proxy residual does not verify a fold of cycles. 12/12 checks from one initial point do not prove unique global attractors. No result is claimed until server validation runs.
