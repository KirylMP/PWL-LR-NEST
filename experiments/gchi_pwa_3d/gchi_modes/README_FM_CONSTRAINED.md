# FM constrained fit

This experiment keeps the same fixed horizontal switching planes and pins the same three full-state equilibrium points exactly via a null-space parameterization. It does not require reproducing all original G-ChI equilibrium branches.

SLSQP imposes numerical eigenvalue constraints on all four affine regions: leading real parts <= -0.001 in both lower regions, >= +0.001 in the middle region and <= -0.001 in the upper region. Only evaluated candidates satisfying those inequalities to 1e-7 tolerance are saved. This makes the lower region matrices Hurwitz for each saved candidate, independent of gamma, but does not prove the rest branch reaches the intended point, rule out other branches, or guarantee any orbit. Check `equilibria-N.json`.

The loss fits reference field and trajectory segments, with soft sampled-state penalties for C<0, IP3<0 and h outside [0,1]. With `--full-probes`, it additionally integrates five training reference cycle starts for one reference period each, penalizing physical violations, closure, and Ca extrema. These are fixed-step RK4 optimization proxies with per-step bound .035 s and finite-precision clipping; they do not prove positivity or cycle stability. The old FM optimization used only 160 steps per period: for the longest training periods and fast eigenvalues, those steps can be outside the explicit RK4 stability region. The smaller step reduces that risk, but does not mathematically guarantee numerical stability for every optimizer iterate. Always inspect independent adaptive integration.

`results/FM/constrained_seed.json` is the provisional starting point from a short local calculation, with full-period probes at three training gamma values during that calculation. It pins the transitions to about 0.0201525 and 1.311843 uM Glu. The two lower-cell eigenvalue maxima are about -.0853 and -.0145; the stable rest branch is sampled as remaining physical and stable for 301 values below the lower target. These are checks, not a complete proof. A threshold intersection scan finds crossings near .01458, .02015 and 1.31184 uM Glu; the first does not change stability.

The independent validation (`analysis/fm-constrained-local/validation.json`) found physically admissible integrated cycles in 7/12 tested points, gamma about .0234 through .8561; the other five tested trajectories converged to a stable equilibrium from the chosen reference-cycle initial condition. The detected Cmax remains underestimated, mean absolute error .3633 uM, period error 3.796 s conditional on the seven detections. Initial condition dependence and other attractors remain untested. The maximum optimizer iteration count was reached, so this is a checkpoint, not a converged optimum. The saved PNG shows sampled equilibrium branches and detected cycle extrema; it does not show unstable cycles, prove their absence, or demonstrate a PWA fold.

## Server

From repository root after `git pull --ff-only`:

```bash
cd experiments/gchi_pwa_3d/gchi_modes
source ../gchi_pwa_trial/.venv/bin/activate
bash run_fm_constrained.sh runs/fm-constrained-01 8 4 80 | tee fm-constrained-summary.txt
```

Arguments: new or empty run directory, number of starts, workers, SLSQP iteration limit. Each start is a feasible small perturbation of the seed when possible; `perturbed=False` means it fell back to the unchanged seed. The server script enables five full-period probes, so this is significantly slower than the prior segment-only experiment. Different objectives are not numerically comparable. `optimizer_success=False` means SLSQP stopped without its convergence criterion; the saved best feasible point may still be assessed by validation. Each seed outputs candidate, adaptive validation, and equilibrium-threshold intersection JSON plus logs and a plot. The aggregate errors are conditional on the detected, physically admissible trajectories; unknown or failed trajectories are not counted as a match. Inspect `equilibria-N.json` to check whether the stable branch goes to the intended transition.

After inspection, stage only intended results: `git add -f runs/fm-constrained-01 fm-constrained-summary.txt`; check `git status --short`, commit and push.
