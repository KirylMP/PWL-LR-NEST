# FM PWA refinement — fixed G-ChI, four zones

This is an exploratory *PWA-only* fitting experiment. The original FM G-ChI parameters are not changed:
`KER=.05, vdelta=.05, r5=.05, vbeta=.5`.

## Changes relative to fit_modes.py

- Keeps three Ca switching thresholds and a continuous PWA field with exactly pinned rest, lower transition equilibrium, and upper transition equilibrium.
- Scans the *first* C threshold between rest and the pinned lower transition instead of fixing it at the midpoint.
- Changes the cell spectral preference from stable/unstable/unstable/stable (as coded previously) to **stable/stable/unstable/stable**. The first threshold should not itself be required to trigger oscillatory onset.
- Adds finite-domain sampled inward-flux penalties at C=0, I=0, h=0 and h=1, and penalty for negative state excursions during the short trajectory fits.
- Retains reference derivative and short reference trajectory matching. Rankings are **training objectives**, not verified performance metrics.

## Execute

From `experiments/gchi_pwa_3d/gchi_modes` in the same Python environment as earlier experiments:

```bash
python fit_fm_free_knot.py \
  --reference results/FM/reference.json \
  --start runs/fm-search-01/candidate-2.json \
  --out runs/fm-free-knot-01 \
  --n-knots 7 --max-nfev 100
```

The command writes candidate JSON files and `ranking.json`. For each produced candidate, run:

```bash
python validate_modes.py \
  --reference results/FM/reference.json \
  --candidate runs/fm-free-knot-01/candidate-0.json \
  --output runs/fm-free-knot-01/validation-0.json
python analyze_equilibria.py \
  --candidate runs/fm-free-knot-01/candidate-0.json \
  --output runs/fm-free-knot-01/equilibria-0.json
```

Repeat validation for all candidates. Reject any candidate with negative IP3, negative Ca, h outside [0,1], or nonperiodic result even if the objective is low. Compare detected cycles, Cmax, periods, timing and early loss-of-stability on the held-out gamma set.

## Limitations

The anchor coordinates and gamma values come from the previously saved FM reference and are pinned as equilibria. Neither a smooth saddle-node nor a Hopf bifurcation is guaranteed by PWA pinning. The sampled flux inequalities are not proofs of global invariance. The spectral preference is soft and zone-level, not a theorem about the connected equilibrium branch. Finite-horizon return-map fit is not a test of asymptotic cycle stability. Validation from reference orbit points is not a basin search.

This experiment has **not yet been run on the server**, and makes no claim that it improves 7/12 until the independent validations succeed.