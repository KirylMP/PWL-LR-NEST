#!/usr/bin/env python3
"""Artificial terminal switching region for the *unchanged* FM topology seed 3.

The field remains continuous PWA in (C,h,I) at every fixed glutamate gamma:
  f_guard = f_seed3 + a(gamma) * v * max(0, I - I2 - slope*(C-Cpivot)).
The glutamate gate is a prescribed intervention, not a genuine cycle fold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import place_poles

from fit_modes import design

TARGET_GAMMA = 21.202553865109465


def make_guard(candidate, *, I2=2.3, slope=.5, Cpivot=.8,
               gate_start=21.1, gate_end=TARGET_GAMMA):
    co = np.asarray(candidate["coefficients"], dtype=float)
    knots = np.asarray(candidate["knots"], dtype=float)
    # Third regional matrix: seed 3 is unstable in this calcium region.
    A2 = co[1:4].T + np.column_stack(
        (np.sum(co[7:9], axis=0), np.zeros(3), np.zeros(3)))
    n = np.array([-slope, 0., 1.])
    # Observability-based feedback to assign negative real regional poles.
    K = place_poles(A2.T, n[:, None], [-.2, -.6, -2.]).gain_matrix.ravel()
    v = -K

    def gate(g):
        return float(np.clip((g - gate_start) / (gate_end - gate_start), 0, 1))

    def field(t, state, gamma, guarded=True):
        f = design(state, gamma, knots)[0] @ co
        if guarded:
            C, _, I = state
            hinge = max(0., I - (I2 + slope * (C - Cpivot)))
            f = f + gate(gamma) * hinge * v
        return f

    return field, gate, dict(I2=I2, slope=slope, Cpivot=Cpivot,
                             gate_start=gate_start, gate_end=gate_end,
                             correction_vector=v.tolist(),
                             normal=n.tolist(),
                             designed_zone_2_eigenvalues=[
                                 [float(z.real), float(z.imag)] for z in
                                 np.linalg.eigvals(A2 + np.outer(v, n))])


def simulate(field, gamma, x0, *, guarded=True, duration=380.):
    def f(t, y):
        return field(t, y, gamma, guarded)

    def peak(t, y):
        return f(t, y)[0]
    peak.direction = -1
    peak.terminal = False

    sol = solve_ivp(f, (0, duration), x0, method="DOP853",
                    rtol=3e-8, atol=3e-10, max_step=.25, events=peak)
    tail = sol.y[:, sol.t > .7 * duration]
    amp = float(np.ptp(tail[0])) if tail.shape[1] > 1 else None
    physical = (sol.success and np.min(sol.y[0]) >= -1e-8 and
                np.min(sol.y[2]) >= -1e-8 and
                np.min(sol.y[1]) >= -1e-8 and
                np.max(sol.y[1]) <= 1 + 1e-8)
    return dict(gamma=float(gamma), success=bool(sol.success),
                physical_at_nodes=bool(physical), calcium_tail_range=amp,
                end=sol.y[:, -1].tolist(),
                peaks_detected=int(len(sol.t_events[0])),
                min_C=float(np.min(sol.y[0])),
                min_I=float(np.min(sol.y[2])),
                min_h=float(np.min(sol.y[1])),
                max_h=float(np.max(sol.y[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path,
                    default=Path("runs/fm-topology-01/candidate-3.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("analysis/fm-terminal-guard-01"))
    ap.add_argument("--duration", type=float, default=380.)
    ap.add_argument("--I2", type=float, default=2.3)
    ap.add_argument("--slope", type=float, default=.5)
    ap.add_argument("--gate-start", type=float, default=21.1)
    ap.add_argument("--gate-end", type=float, default=TARGET_GAMMA)
    args = ap.parse_args()
    candidate = json.loads(args.candidate.read_text())
    field, gate, geometry = make_guard(
        candidate, I2=args.I2, slope=args.slope,
        gate_start=args.gate_start, gate_end=args.gate_end)
    gammas = [.1, 1., 10., 18., 20., 21., 21.1, 21.11, 21.12,
              21.15, 21.2, TARGET_GAMMA, 21.25, 22., 25., 30., 50.]
    x0 = [0.9, 0.7, 2.8]  # deliberately common initial condition
    records = []
    for g in gammas:
        row = simulate(field, g, x0, duration=args.duration)
        row["gate"] = gate(g)
        records.append(row)
        print(g, row["calcium_tail_range"], row["physical_at_nodes"],
              flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(parent_candidate=str(args.candidate),
                  fixed_gchi_parameters=dict(ker=.05, vdelta=.05,
                                             r5=.05, vbeta=.5),
                  geometry=geometry, initial_state=x0,
                  duration=args.duration, records=records,
                  limitations="Artificial gamma-gated terminal region; NOT "
                  "an intrinsic cycle fold. Finite initial-state/time "
                  "tests do not prove all attractors vanish or global "
                  "physical invariance. Gamma gate intentionally changes "
                  "the high-gamma field and is not a fixed autonomous "
                  "3D affine coefficient family independent of gamma.")
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    gs = [r["gamma"] for r in records if r["gamma"] >= 20]
    amps = [r["calcium_tail_range"] for r in records if r["gamma"] >= 20]
    ax.plot(gs, amps, "-o", markersize=4, label="Guarded seed 3")
    ax.axvline(TARGET_GAMMA, linestyle="--", color="gray",
               label="reference upper-fold candidate")
    ax.set_xlabel("Glutamate (uM)")
    ax.set_ylabel("Tail Ca range (uM)")
    ax.set_title("Artificial upper cutoff: not a verified fold")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "cutoff.png", dpi=170)
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
