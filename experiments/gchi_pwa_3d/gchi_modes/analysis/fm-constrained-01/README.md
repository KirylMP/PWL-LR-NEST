# FM constrained server run: seed 6

Reference is the Table 1 FM G-ChI model; IP3 is dynamic and Glu is the parameter. This directory analyses server run `../../runs/fm-constrained-01`, committed at 0cb21cad. It compares seeds 0, 2, and 6 and explores seed 6 more deeply.

For the previously reported 11/12 comparison, errors are conditional on accepted detections.

| seed | Cmin MAE | Cmax MAE | Imin MAE | Imax MAE | period MAE |
|---|---:|---:|---:|---:|---:|
| 0 | .02256 | .09791 | .25492 | .44714 | .65029 |
| 2 | .02378 | .08818 | .27128 | .43683 | .81253 |
| 6 | .02422 | .08615 | .25970 | .46120 | .65153 |

Each run was stopped by the iteration limit. Candidate 6 exactly pins the full-state anchors; its adjacent leading eigenvalues around the lower main threshold are -0.001797 and +2.37927; the additional lower threshold does not change stability. Its local stability at zero glutamate is -0.06858. These calculations do not establish the HLB subtype.

All six 11/12 candidates have one `negative_ip3` trajectory at gamma approximately .02336 from the reference-cycle peak. A directional scan seeded from a neighboring PWA cycle finds an attracting periodic trajectory at .02336 with positive sampled C, IP3 and admissible h; this does not make the original invalid transient physical. The cycle near .0199 also has a nontrivial Floquet radius about .000142, while the PWA equilibrium at [0.05361318, 0.89776953, 0.54193729] is stable (spectral abscissa -.001797): this is numerical evidence of PWA bistability below the main lower threshold .0201525. A comparison trajectory of reference G-ChI at .0199 approached its stable equilibrium; this single experiment does not exclude a separate reference cycle.

Near onset at Glu=.0203, reference G-ChI cycle period is approximately 154.185 s versus 39.1185 s in PWA. G-ChI Ca range is .02934.. .87449 uM versus .00080.. .85867 uM in PWA. The small PWA Ca minimum needs attention. Very close to the lower transition, finite-time scans alone cannot establish asymptotic period scaling, a homoclinic bifurcation or a PWA fold.

A periodic shooting calculation with variational equation finds a reference G-ChI fold candidate at gamma=21.2025538651, Cmin=.0352317184, Cmax=1.1512605668, period=12.41641009 s; closure=1.47e-11. Two Floquet multipliers are within approximately 1e-5 of 1, and the third is about 9e-7. The second unit multiplier and small residual support a fold candidate; nondegeneracy and transversality are not proved. Reference forward simulation converged to a cycle at 21.2 but did not detect one at 21.3; it may select another basin.

In PWA, the sampled stable cycle continues through gamma=20, 21.20255, 30, 100, 1000, and 10000; its nontrivial Floquet spectral radius is about .070 at 21.20255. Therefore there is no evidence of a corresponding PWA fold at the reference fold point. Sampling and single-branch forward integration cannot exclude folds in gaps, isolated branches, grazing or disappearance beyond the tested range. Variational integration switches Jacobians without explicit event localization; the PWA field is continuous, but grazing and numerical errors still require care.

The plots show sampled equilibria and forward-detected cycle extrema. They do not reconstruct unstable cycles or prove the full bifurcation diagram. `bifurcation-extended.png` uses a log period axis; `bifurcation-onset.png` zooms near the lower transition. The reference fold candidate is marked as a purple diamond. `directional-scans.json` records starting states and physical checks; high/low Floquet samples and reference-fold shooting output are also preserved. `pwa-low-extra.json` is exploratory: at gamma .014 the tested PWA trajectory has negative IP3; this is not a fold classification.

From the `gchi_modes` directory, reproduce the extended plot using:

```bash
python plot_fm_extended.py --reference results/FM/reference.json \
  --candidate runs/fm-constrained-01/candidate-6.json \
  --validation runs/fm-constrained-01/validation-6.json \
  --scans analysis/fm-constrained-01/directional-scans.json \
  --reference-high analysis/fm-constrained-01/reference-high-scan.json \
  --reference-fold analysis/fm-constrained-01/reference-fold-candidate.json \
  --output fm6-plot.png
```

Use `scan_fm_candidate.py` for independent fixed-step-in-parameter, adaptive-in-time scans and `find_reference_fm_fold.py --output PATH` to recompute the reference fold candidate. `floquet_fm_cycles.py` samples multipliers from a user-supplied initial state and a list of gamma values. None of these scripts performs pseudo-arclength continuation of the unstable PWA cycle.
