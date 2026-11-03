# FM topology-oriented exploratory fit

This experiment starts from `runs/fm-constrained-01/candidate-6.json`, preserves its calcium thresholds, and pins the three full-state equilibria exactly using a null-space parameterization. Adjacent fields remain continuous. The four affine-cell spectral inequalities are hard numerical constraints in the optimizer: stable, stable, unstable, stable with a 0.001/s margin. A recorded candidate is checked against those inequalities (tolerance 1e-7); this does not establish the equilibrium branch topology or HLB subtype.

The loss retains field/segment fitting and physically sampled full-period probes. It adds reference trajectory segments at gamma .021 and .0203 uM, and a full-period probe at .021 using the reference period (~88.7 s). It adds explicit C and IP3 extrema matching at five other gamma values. The fold target is the reference shooting candidate at gamma 21.2025538651 uM. Four optimized variables select a candidate initial state and period near that target; the loss softly penalizes periodic closure, Cdot phase, a second-unit-multiplier proxy computed from a finite-difference monodromy, C/IP3 extrema, and invalid sampled states. Candidate initial C, h, IP3 and period are bounded to [0.7,1.5], [0.4,0.95], [0.5,4] uM and [8,25] s. The reference fold is itself a numerical candidate; nondegeneracy and transversality remain unchecked.

All trajectory, positivity and fold conditions are **soft**. A small objective or `fold_proxy` in a candidate is not proof of a cycle or a fold: the trivial unit multiplier interpretation applies only to a closed periodic orbit. The proxy uses fixed-step RK4 (maximum nominal step .035 s), finite differences and clipped fields; it can be numerically inaccurate near switching/grazing or on stiff optimizer iterates. The best saved point may be a finite-difference evaluation rather than the final SLSQP iterate. Objective values are not comparable to earlier runs. Every candidate must be independently integrated with adaptive tolerances; stable periodic orbits should subsequently be continued, and any proposed fold should be verified for nondegeneracy and transversality. No global positivity, absence of extra attractors or correct onset mechanism is guaranteed.

A single-iteration local smoke test ran successfully and preserved the stability constraints, but did not produce a verified fold or improve the lower-onset period. In one pilot diagnostic, the fold-point shooting residual was ~0.25 in state-space norm. This is a workflow test, not a completed fit.

## Server

After committing/pushing the previous run and pulling this revision:

```bash
cd experiments/gchi_pwa_3d/gchi_modes
source ../gchi_pwa_trial/.venv/bin/activate
bash run_fm_topology.sh runs/fm-topology-01 4 4 30 | tee fm-topology-summary.txt
```

Arguments: a new/empty run directory, number of starts, simultaneous workers and SLSQP iteration limit. This is considerably slower than the earlier experiment: every evaluation includes long onset trajectories, five full-period probes and a fold proxy. The runner records Git HEAD, file checksums and library versions in `manifest.json`; each seed saves a `.checkpoint.npz` after every improvement, and a candidate when the optimizer exits. Multiple starts are small feasible perturbations; the `perturbed` column reports whether a perturbation was accepted. `solver_success=False` means no convergence criterion was met; the candidate is the best feasible point evaluated, not a demonstrated optimum.

The summary gives valid forward-detected cycles, conditional Cmax/Imax/period errors, the period at gamma=.0203 from the previous PWA-cycle initial state, and the adaptive fold shooting closure/phase residual. Fold `phase` is Cdot at the candidate initial point; neither a closure nor a phase residual alone establishes a fold. `fold_physical` refers only to points checked by the adaptive diagnostic. Files include `validation-N.json` for 12 trajectory checks, `onset-N.json` for two initial-state families at both .0203 and .021, `equilibria-N.json` for sampled threshold crossings, and `fold-check-N.json` for the adaptive shooting diagnostic. Its variational eigenvalues are not Floquet multipliers unless a genuinely closed orbit has been established. The onset points are used in training and thus not independent data, although the validation integrator is independent.

To resume an interrupted seed 0 checkpoint, keep the same code and inputs and use a separate output location:

```bash
python fit_fm_topology.py --data results/FM/reference.json \
  --start runs/fm-constrained-01/candidate-6.json \
  --onset-data analysis/fm-constrained-01/directional-scans.json \
  --fold-target analysis/fm-constrained-01/reference-fold-candidate.json \
  --output runs/fm-topology-resumed-0/candidate-0.json \
  --resume runs/fm-topology-01/candidate-0.checkpoint.npz \
  --seed 0 --maxiter 30 --full-probes
```

The checkpoint includes a fingerprint of the objective inputs, step and optimizer code; incompatible resumes are refused. Resume restores parameters, not the SLSQP internal state. Afterwards run the four diagnostics manually or use the next new run directory. Before returning results, stage only intended files, for example `git add -f runs/fm-topology-01 fm-topology-summary.txt`, inspect `git status --short`, commit and push.
