# FM sloped IP3 terminal switching region (exploratory)

**Reference G-ChI parameters unchanged.** Start from `runs/fm-topology-01/candidate-3.json` (12/12 standard cycles and a checked physical onset orbit). The lower-gamma seed-3 field is unchanged.

We introduce a single new state-space switching plane, sloped in (C,I):

```
z(C,I) = I - [2.3 + 0.5*(C - 0.8)]
f_new(x,gamma) = f_seed3(x,gamma) + a(gamma)*v*max(z,0)
a(gamma) = clip((gamma - 21.1)/(21.2025538651 - 21.1), 0, 1)
```

Here `v` is derived from pole placement on the original third calcium region (index 2), choosing nominal regional eigenvalues `[-0.2,-0.6,-2]` when the guard is fully active. The new field is continuous at the inclined plane and piecewise affine **in state for any fixed gamma**. This is a deliberate gamma-dependent high-input intervention, rather than a natural cycle-fold bifurcation. There is no claim that regional eigenvalues guarantee disappearance of every periodic orbit.

## Local screening already performed

An independent Python probe with `solve_ivp(DOP853)`, duration 380 s, common initial state `[C,h,I]=[0.9,0.7,2.8]`:

- gamma 20 and 21.1: Ca tail amplitude ~0.939 uM, oscillatory
- gamma 21.11: Ca tail amplitude near machine zero, stationary
- gamma 21.20255, 22, 30 and 50: near-zero Ca tail range, stationary
- solver-node physical checks passed for these particular inputs

This is **not** 12-point standard validation and not full basin or branch continuation. The observed shutdown occurs around 21.1–21.11, earlier than the target 21.20255; the gate becomes fully active by the target. It can be retuned with `--gate-start` and `--gate-end`.

Do not label this event as a proven cycle fold: no collision of stable and unstable cycles and no second unit Floquet multiplier were established. In particular, `M != 1` can be consistent with an abrupt artificial shutdown, but not with the standard nondegenerate saddle-node of cycles.

## Run

From `experiments/gchi_pwa_3d/gchi_modes`:

```bash
python experiment_terminal_guard.py \
  --candidate runs/fm-topology-01/candidate-3.json \
  --output analysis/fm-terminal-guard-01
```

Produces `results.json` and `cutoff.png`. Use multiple initial states and long runs to test coexistence and physicality. A stricter next step is to localize the actual extinction boundary, check continuation across it, and compare direct Poincare returns.

All original candidates and reference files remain unmodified.
