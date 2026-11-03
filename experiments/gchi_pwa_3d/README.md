# Exploratory 3D G-ChI PWA: horizontal thresholds and actual resting equilibrium

This experiment continues the G-ChI AM reference and PWA work. It is exploratory and is not a frozen paper result. C and I are in uM; h is dimensionless; time is seconds; glutamate gamma is a fixed external parameter in each run. It is a dynamical three-state model, not a 2D Li–Rinzel model with constant IP3. For fixed gamma the field is continuous and piecewise affine in (C,h,I), with thresholds C=0.1, 0.155748478..., 0.358762443... uM. The input uses three nonlinear Hill features of gamma, so it is not jointly PWA in gamma. The restoring term below 0.1 has a fitted positive coefficient k. k is bounded but no global positivity or invariance is guaranteed.

The three full-state equilibrium anchors are constrained to machine precision: gamma=0 and the two G-ChI equilibrium stability transitions. In particular, the gamma=0 equilibrium C=0.069799... uM lies strictly below the guard C=0.1, where it is a real equilibrium. The actual Jacobian eigenvalues in the lower region must still be checked for each fitted model. Exact anchor interpolation by itself does not prove HLB1, Hopf normal forms, cycles, a fold, or uniqueness.

## Motivation

The previous local model `gchi_pwa_trial/fit_actual_rest.json` produces a stable equilibrium and, from a different initial condition, a stable finite-amplitude cycle at gamma=0.0115, below the first reference transition gamma~0.01299074. This numerically observed bistability is an undesired discrepancy. `benchmarks/previous_candidate_validation.json` records it. The optimization now adds three reference-cycle trajectory targets closer to the first transition (gamma~0.0133155, 0.0155814, 0.0197232), alongside previously used reference cycles, equilibrium residuals, spectral penalties and a fold proxy. This is not a hard constraint eliminating all bistability; the separate numerical validation checks one initial condition below H1, and failure to find a cycle cannot prove absence.

One local run produced `gchi_pwa_trial/seed_candidate_low.json`, with `benchmarks/local_candidate_validation.json`: all 25 tested gamma inputs show cycles; at gamma=0.0115 the selected large initial condition relaxes to the stable equilibrium without a detected cycle. Among those 25 (which overlap training), mean absolute error vs reference is ~0.01756 uM for Cmin, ~0.00735 uM for Cmax, ~0.13541 s for period. No independent holdout, proof of HLB1 type or proof that there is no other attractor is claimed. `benchmarks/local_candidate_continuation.json` records stable and near-fold periodic orbits; the continuation stopped near grazing, and no new fold is certified. Prior continuation had a variable shadowing bug in the extra guard coefficient; `continue_candidate.py` fixes it. In that previous high-gamma branch C stayed above 0.1 and the corrected and older numerical branch happened to match.

## Server setup

From the repository root on the server:

```bash
git pull --ff-only
cd experiments/gchi_pwa_3d/gchi_pwa_trial
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib
```

Run a small *execution test* first (not a fitted result):

```bash
bash run_jobs.sh smoke-01 1 1 1
python summarize_runs.py runs/smoke-01
```

Run multiple independent starts (12 seeds, at most 4 simultaneous processes, 120 optimizer function evaluations):

```bash
bash run_jobs.sh am-search-01 12 4 120
bash validate_all.sh runs/am-search-01 4
python summarize_runs.py runs/am-search-01 | tee runs/am-search-01/summary.txt
```

Change the worker count according to the server. Run names must be unique; the script refuses to overwrite an existing directory. A job writes a log, a checkpoint, and on completion a candidate. The best evaluated point may be an optimizer finite-difference probe and must be validated. The fit uses an approximate fixed-step RK4 trajectory objective and regularization; the validation uses adaptive SciPy ODE integration independently. Runtime is not benchmarked; the number of RHS evaluations may substantially exceed `max-nfev`.

Validation records 25 sampled input values, a numerical test from one large initial condition just below H1, local equilibrium stability at that input, exact-anchor residuals and lower-cell eigenvalues. A `cycle` status below H1 with stable equilibrium confirms bistability for these starting conditions. A `no_cycle_detected` or `not_converged` status does not prove there is no other stable cycle, and error metrics cover only detected cycles. Several points occur in the fitting objective: this is not a held-out test. Do not rank a candidate only by objective or conditional error; also inspect detected cycle counts, nonconvergence, guard and anchors, IP3 where relevant, and below-H1 status.

For a selected candidate, optional pseudo-arclength continuation is:

```bash
python continue_candidate.py runs/am-search-01/0/candidate.json \
  --out runs/am-search-01/0/continuation.json
```

It begins from the default gamma 2.5 and 3.0 (override `--start-low`/`--start-high` if necessary), checks for nontrivial seed cycles, tracks Floquet multipliers, limits predictor correction and writes intermediate results. It can fail or stop near grazing, may need alternative seeds and does not itself certify fold nondegeneracy or branch completeness. No event-exponential propagation or NEST benchmark is included.

Git ignores runs/ by default. To return selected small result files to GitHub, review the files and sizes, then explicitly force-add the chosen run:

```bash
git add -f runs/am-search-01
git status --short
git commit -m "Record G-ChI PWA multistart results"
git push
```

The data in `gchi_am/output` are numerical reference outputs from the current project; the published model source is De Pittà et al. 2009, J Biol Phys, DOI 10.1007/s10867-009-9155-y, Table 1 AM and Eqs. 5, 6, 19, 20. The package does not distribute the paper. Do not overwrite paper_frozen_v1.
