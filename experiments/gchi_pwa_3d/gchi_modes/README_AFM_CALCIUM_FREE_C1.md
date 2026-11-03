# AFM calcium-first refinement with movable lower calcium threshold

The starting reference is `results/AFM/candidate_segments.json`, with an independent held-out evaluation `results/AFM/validation_segments.json`: **12/12 physical cycles**, Cmax MAE **0.034459 µM**, Cmin MAE **0.007162 µM**, period MAE **0.720487 s**. This reference is already strong. An alternative candidate must improve calcium quality **without sacrificing physical cycles**; a lower training objective alone is not a sufficient improvement.

The reference AFM G-ChI parameters, full (C,h,IP3) state, input nonlinearity, three anchored equilibria, and four calcium-based affine regions are unchanged. The AFM family is the repository's AM-derived AFM parameterization; it is not an FM field with altered parameters.

The experiment sweeps only the first knot `C1` through five values:

```
0.095, 0.105, 0.11277377008786005 (original), 0.120, 0.135 µM
```

The other two thresholds stay at `0.15574847824214638` and `0.3587624427037859` µM. The null-space basis is recomputed for each threshold, pinning the three full-state equilibria exactly; seed coefficients are reprojected.

The objective emphasizes calcium waveform tracking, full-reference-period C extrema and closure, and negative-C penalties. IP3, h and the regional stable/stable/unstable/stable eigenvalue sign preference remain soft secondary terms. The loss uses an RK4 proxy; closure over a reference period does not establish a fundamental period or the stability of a cycle. Physical penalties are sampled and **not** invariance proofs. Numerical fit status and held-out adaptive validation must be inspected.

## Run on the 16-core server

From `experiments/gchi_pwa_3d/gchi_modes` after switching to `experiment/afm-calcium-free-c1`:

```bash
source ../gchi_pwa_trial/.venv/bin/activate
python run_afm_calcium_free_c1.py runs/afm-calcium-c1-01 \
  --nfev 70 --workers 5 --calcium-weight 3 --negative-c-weight 8 \
  | tee afm-calcium-c1-summary.txt
```

The five independent fits run concurrently with BLAS/OpenMP capped at one thread per process. Each writes its own candidate, fit log, validation, equilibrium report and plot. `summary.json` reports cycles, conditional errors, and rejected inputs. An individual failure does not prevent completion of the others; runner exit code is nonzero if any case fails. Use a new empty run directory on repeat.

**Selection rule:** first require 12/12 valid physical cycles; then compare Cmax and Cmin MAE, period MAE and onset behavior to the baseline. Conditional MAEs from fewer than 12 valid cycles must not be interpreted as improvements. Do not claim genuine Hopf bifurcations or global positivity based on these diagnostics.
