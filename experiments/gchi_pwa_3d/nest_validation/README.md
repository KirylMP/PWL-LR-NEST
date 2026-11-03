# 3D G-ChI PWA in NEST: trajectory validation

This is a separate experiment for NEST 3.10 with GSL. It generates two
extensions from the repository's NEST 3.10 `astrocyte_lr_1994` template.
The model state is `(C, h, IP3)`; the fitted continuous PWA field has three
calcium thresholds and a dynamic IP3. `IP3_0` is repurposed as **fixed
extracellular glutamate gamma**, not baseline IP3. Do not connect spikes or
currents: this experiment does not implement input events. Calcium is not
clipped. Integration uses adaptive GSL RKF45, so this is a correctness test,
not an analytical event-solver or a performance benchmark.

Defaults: AFM candidate 2 (`runs/afm-calcium-c1-01`) and FM candidate 3
(`runs/fm-topology-free-c1-01`). Rates are converted from seconds to NEST
milliseconds. The nonlinear Hill functions of fixed gamma match `fit_modes.design`.
Model class names are `gchi_am_pwa` and `gchi_fm_pwa`. AM here is the
AM-derived G-ChI set labeled `AFM` in `gchi_modes`.

From the repository root, after activating the server environment with NEST,
`nest-config`, GSL development headers, NumPy, SciPy and Matplotlib:

```bash
cd experiments/gchi_pwa_3d/nest_validation
python build.py
export LD_LIBRARY_PATH="$PWD/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python validate.py --samples 3 --duration 200 --resolution 0.1 --interval 2
```

Writes `results/results.json`, per-input compressed traces and PNGs. Each
input starts from a saved reference G-ChI cycle state; NEST and two independent
SciPy integrations use the same initial state. `nest_scipy_max_abs` gives
componentwise `(C,h,IP3)` discrepancies over the complete trace. Compare
separately to G-ChI, as close NEST/SciPy agreement only checks implementation.
Some phase drift over long cycles is expected; repeat at shorter duration and
different resolution if there is a discrepancy. Physical ranges are checked
only at recorded sample times, not proven between them. This is not a fold,
basin, global invariance, or bifurcation-continuation test. The selected FM
optimizer did not report convergence; 12/12 is a finite trajectory check.

For a quick execution test use `python validate.py --samples 1 --duration 5`.
Override `--am` and `--fm` on `build.py` to compare different candidates;
rebuild whenever candidate files change. The manifest contains checksums and
validation refuses a mismatched candidate. `build/` and `results/` are local
outputs; do not commit binaries or large trace files by default.
