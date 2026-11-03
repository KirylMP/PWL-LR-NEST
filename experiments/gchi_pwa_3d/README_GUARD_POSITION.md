# 3D G-ChI PWA: variable lower-guard position

This adds an optional new experiment to the existing G-ChI PWA workflow, preserving the previous scripts, results and frozen paper models. The same continuous horizontal-threshold family is used. For fixed glutamate the state is x=(C,h,I), the two main thresholds remain C=0.155748... and 0.358762... uM, and the additional lower boundary Cb is a single constant parameter of each fitted model. The restoring term is `(k*max(Cb-C,0), 0, 0)` with positive fitted k. It guarantees equality of the two vector fields at C=Cb for all h,I and any fixed glutamate; matrices can differ. It does not by itself guarantee positivity of the complete C derivative or physical invariance of all states.

Both Cb and k and the PWA coefficients are optimized. The explicit search interval is 0.075 <= Cb <= 0.145 uM and 0.1 <= k <= 10 s^-1; these are modeling choices, not an exhaustive range. The Cb interval stays strictly above the pinned gamma=0 calcium equilibrium C=0.069799... uM and below the lower main threshold. Thus the gamma=0 equilibrium lies in the extra lower region. The three full-state equilibria at gamma=0 and the two reference equilibrium transitions are enforced by null-space parameterization recomputed at every proposed Cb, not by soft penalty. The lower-region spectral stability is only soft-penalized in optimization and must be verified in validation.

Objective includes the prior field, trajectory, spectral and periodic/fold-proxy terms; additionally a soft pre-H1 trajectory penalty uses gamma in [.0115,.0123,.01275,.0129] and two initial conditions, comparing at 79.95,99.9,120 s with reference G-ChI equilibrium. Negative C along these probe integrations is penalized. These short fixed-step RK4 training trajectories do not prove positivity or absence of another attracting orbit, and the fold term is only a proxy. The fixed-step optimizer clips derivative values for numerical guard; validate final model with adaptive integration and physical state checks.

Start is the previously analyzed server seed 6 in `seed_guard_opt.json` from run am-search-01 (commit f5b73aa6c84d94254342439446512aee7d68ab5f). Each seed receives a reproducible perturbation in free coefficient coordinates, fold seed, guard coefficient and threshold. A best evaluated checkpoint may be a finite-difference probe, so independent validation is required. Runs can converge to local minima; a point on a search bound is not a demonstrated global optimum. Objective values include new terms, so they are not directly comparable to the previous objective.

## Server

From the repository root:

```bash
git pull --ff-only
cd experiments/gchi_pwa_3d/gchi_pwa_trial
source .venv/bin/activate   # or create it and install numpy scipy matplotlib
bash run_guard_jobs.sh guard-search-01 12 4 120
bash validate_guard_all.sh runs/guard-search-01 4
python summarize_guard_runs.py runs/guard-search-01 | tee runs/guard-search-01/summary.txt
```

Arguments are run name, number of starts, simultaneous processes, and optimizer evaluation limit. Use a unique run name. Optimizer finite-difference evaluations and the multiple validation trajectories can take substantially more time than the function-evaluation count suggests. Runner records commands, code SHA, dirty worktree status and Python packages. Logs/checkpoints are per seed; no automatic resume is implemented. The validation performs the previous 25-input cycle tests and a nine-trajectory sweep below H1 (three gamma values and three initial conditions). Each trajectory is checked for a detected cycle, return near equilibrium, domain failure, or unresolved outcome. Negative C at numerical solver nodes is flagged; unflagged does not prove positivity between nodes. Output reports probe statuses and the lower-region eigenvalues; lack of an observed cycle does not prove no other attractors. For the chosen candidate, run optional `continue_candidate.py` with the candidate path; the continuation and fold still require numerical checks and nondegeneracy analysis.

After inspection, return the run results to GitHub from this directory:

```bash
git add -f runs/guard-search-01
git status --short
git commit -m "Record variable guard PWA runs"
git push
```

Only stage the intended run: the source server worktree was previously recorded with deleted tracked files outside this experiment. Plots and validation results are exploratory, not a paper-ready conclusion. Reference model is De Pitta et al., J Biol Phys 2009, DOI 10.1007/s10867-009-9155-y, with Table 1 AM; see the parent README for details.
