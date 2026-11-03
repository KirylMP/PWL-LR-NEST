# Analysis of the FM topology search

Inputs are `../../runs/fm-topology-01` from commit 9ebd2d56. The reference is the Table 1 FM G-ChI model with dynamic IP3. Four seeds (0, 3, 28, 31) were selected based on the user's summary. `comparison.png` plots sampled equilibrium roots and cycle extrema; `onset-traces.png` compares independent phases of the original model and seeds 0 and 3 at Glu=.0203 uM after 1600 s integration. Neither plot reconstructs unstable cycles or establishes global bifurcation topology. G-ChI's purple point is a reference fold candidate (see `../fm-constrained-01/reference-fold-candidate.json`); nondegeneracy and transversality are not verified.

| Seed | Standard cycles | Mean Cmax error, uM | Mean Imax error, uM | Mean period error, s | Glu=.0203 check |
|---|---:|---:|---:|---:|---|
| 0 | 12/12 | .0800 | .3301 | 1.776 | negative IP3 |
| 3 | 12/12 | .1272 | .4312 | 2.503 | period 48.205 s; physical at checked points |
| 28 | 10/12 | .0374 | .4919 | 4.280 | period 30.126 s; physical at checked points |
| 31 | 9/12 | .1545 | .4956 | 4.226 | period 55.642 s; physical at checked points |

All mean errors are conditional on detected physical trajectories and cannot be directly compared across different accepted gamma subsets. Each seed has pinned equilibria and locally stable, stable, unstable, stable affine regions, but all four optimizations ended at the iteration limit. At Glu=.0203 the G-ChI reference period is about 154.185 s. Tests at .0203 and .021 were used by the optimizer; even though integrations use an independent adaptive solver, these are not held-out parameter values. The physical checks test a finite number of numerical states, not global invariance.

At Glu=.021 the reference period is 88.694388 s, whereas seed 3 has fundamental detected period 44.346855 s (ratio 2.000015). A closure residual after one reference period can therefore be small after *two* PWA turns. This is a plausible aliasing mechanism, not proof that it alone determined the optimizer outcome. A future fit should test the first return or constrain the absence of intermediate returns, with independent event detection.

`physical-N.json` extends selected trajectories and samples their final returned cycle: seed 0 at gamma .0203 has tail Imin ~-.02023 uM; seed 3 there has tail Imin ~.01007 uM. At gamma .2218279988, seeds 28 and 31 have tail Imin ~-.003334 and -.037995 uM respectively. Thus those examples are not merely negative transients, although other untested initial conditions might lead to other attractors. `physical-initial-N.json` saves the initial states.

At the reference fold candidate gamma=21.2025538651, direct integration finds an attracting PWA cycle in all four seeds. Peak-to-peak state closure is under 3e-7; the nontrivial Floquet spectral radii are approximately .00587, .01723, .00557 and .000362 for seeds 0, 3, 28 and 31, respectively. These cycles are far from the additional unit multiplier expected at a generic fold, so no corresponding fold is supported *on those cycles at that parameter*. The independent diagnostic of the optimized fold proxy is not itself periodic: for seed 28 its adaptive closure is .09013 and its phase residual is .32396 uM/s. Its variational eigenvalues must not be called Floquet multipliers without closure. Sampled forward scans for seeds 3 and 28 find physical checked cycles at gamma 21.20255, 30, 50, 100, 300 and 1000, but this is not pseudo-arclength continuation and cannot rule out folds in gaps, disconnected branches, or cycles beyond that range. Switching points are integrated numerically without event-localized saltation; the field is continuous, but grazing and variational error remain limitations.

Recreate the plots from `gchi_modes`:

```bash
python compare_fm_topology.py --reference results/FM/reference.json \
  --reference-scan analysis/fm-constrained-01/directional-scans.json \
  --reference-fold analysis/fm-constrained-01/reference-fold-candidate.json \
  --run runs/fm-topology-01 --output comparison.png
python plot_fm_onset_traces.py --run runs/fm-topology-01 \
  --reference-scan analysis/fm-constrained-01/directional-scans.json \
  --output onset-traces.png
```

`diagnose_fm_physical.py` separates transient and tail-cycle violations at a given gamma; `scan_fm_high_topology.py` performs the finite high-gamma forward scan; `floquet_fm_cycles.py` produces the stored multiplier samples. No solver establishes global positivity, nondegenerate PWA folds, or the absence of unobserved cycles.
