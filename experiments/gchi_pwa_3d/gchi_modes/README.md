# Exploratory FM and AFM fits of 3D G-ChI

AFM means the calcium response of the paper's AM-derived G-ChI parameter set; FM uses its FM column (KER=.05, vdelta=.05, r5=.05, vbeta=.5). IP3 is dynamic and glutamate is fixed per simulation. Reference implementation follows De Pitta et al. 2009, Table 1. The original FM reference has more equilibrium branches than this fit is required to reproduce.

Each PWA field is continuous with three horizontal C switching levels. The lowest lies halfway between the reference rest and first transition; the other two pass through the reference transition C coordinates. The zero-glutamate rest and both full-state transition points are enforced algebraically using a null-space parametrization. For FM the first target is the reference equilibrium saddle-node (Glu approximately .0201525), the second is the upper equilibrium Hopf (approximately 1.311843). AFM uses two reference Hopf points. A continuous PWA model need not have the same local bifurcation type: pinning the point does not reproduce a smooth saddle-node or Hopf, and it does not prove an HLB subtype. Coefficients in neighboring cells can differ. The zero-glutamate rest is genuinely in the lowest cell. Input terms are nonlinear functions of glutamate alone; the state field is piecewise affine for fixed glutamate. Exact anchors do not guarantee that the equilibrium branch is straight as a function of glutamate.

The optimization uses field samples, selected equilibrium samples, and finite-time reference-trajectory segments. It imposes no below-transition convergence penalty and no exhaustive branch matching. A soft spectral penalty targets stable, unstable, stable regions across the two main thresholds; stability and positivity are not hard constraints. Stability and bifurcation types must be examined separately. Clipped fixed-step integration is only an optimization proxy. The validation uses independent gamma values not used as cycle training data (but equilibrium and field data are not fully independent), adaptive integration from a point on the corresponding reference cycle, solver-node calcium checks, and reports conditional errors only for detected cycles. A detected cycle is not a full bifurcation continuation, and no-cycle/undetermined/failure is not proof of absence. Negative calcium between solver nodes is not excluded. Parameter sets, domains, bounds, and limited starting guesses are all model choices.

`results/*/reference.json` contains a reproducible scan: reference roots from multistart and forward simulations; it is not exhaustive and does not follow unstable cycles. `results/*/candidate_segments.json`, `validation_segments.json`, and the corresponding PNG are one preliminary local fit per mode. They should not be treated as a completed FM model or a verified fold diagram. FM `candidate_cycles.json` and `validation_cycles.json` retain the earlier fit without the stability penalty: 5/12 cycles detected, but the region immediately below the first transition is unstable. With the current soft stability penalty the preliminary FM fit did not pass cycle tests either; see its validation for the actual eigenvalues. Validation explicitly compares all found reference equilibria at zero, but only the lowest one was pinned in this experiment.

## Server multi-start

From repository root after `git pull --ff-only`:

```bash
cd experiments/gchi_pwa_3d/gchi_modes
source ../gchi_pwa_trial/.venv/bin/activate
bash run_multistart.sh FM runs/fm-search-01 12 4 100 | tee fm-summary.txt
bash run_multistart.sh AFM runs/afm-search-01 12 4 100 | tee afm-summary.txt
```

Arguments are mode, new or empty output directory, number of seeds, simultaneous workers, optimizer evaluation limit. Each seed is reproducible; seed 0 starts at the field fit for AFM and at the saved initial candidate for FM. Reference data are supplied; regenerate them with `explore_modes.py --mode FM --out results/FM` (or AFM) if needed. Run directories must be new/empty. Execution can take substantial time. Inspect per-run validation JSON, failed integrations, extrema and calcium/IP3/period PNGs before choosing a model. Optimizer objective values are comparable only within this same objective and mode. An unresolved orbit must not be counted as a match.

After review, stage only intended results, e.g. `git add -f runs/fm-search-01 runs/afm-search-01 fm-summary.txt afm-summary.txt`; inspect `git status --short` before commit and push.
