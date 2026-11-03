#!/usr/bin/env python3
"""FM PWA refinement: scan the free first calcium knot with fixed G-ChI.

Run from gchi_modes:
 python fit_fm_free_knot.py --reference results/FM/reference.json \
   --out runs/fm-free-knot-01 --start runs/fm-search-01/candidate-2.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import least_squares

from explore_modes import P, field
from fit_modes import design


def reference_data(ref, anchors, anchor_g, knots):
    good = [r for r in ref["cycles"] if r["status"] == "cycle"]
    selected = [good[k] for k in np.unique(np.linspace(
        1, len(good) - 2, min(9, len(good) - 2), dtype=int))]
    states, gammas, derivatives, segments = [], [], [], []
    for r in selected:
        g, T = r["gamma"], r["period"]
        sol = solve_ivp(field, (0, T), r["peak"], args=(g, P["FM"]),
                        t_eval=np.linspace(0, T, 33), method="DOP853",
                        rtol=1e-9, atol=1e-11, max_step=.2)
        if not sol.success:
            raise RuntimeError("Reference cycle integration failed")
        for y in sol.y.T[:-1]:
            states.append(y)
            gammas.append(g)
            derivatives.append(field(0, y, g, P["FM"]))
        segments.append((g, T, sol.y.T))
    X = np.array(states)
    G = np.array(gammas)
    Y = np.array(derivatives)
    # First three states are pinned exactly, including the reference rest.
    D = design(anchors, anchor_g, knots)
    N = null_space(D)
    if N.shape[1] != 7:
        raise RuntimeError("Unexpected anchor-design rank")
    return X, G, Y, segments, D, N


def zone_matrix(co, zone):
    A = co[1:4].T.copy()
    if zone:
        A[:, 0] += np.sum(co[7:7 + zone], axis=0)
    return A


def boundary_residuals(co, knots):
    """Sample inward flux at C=0, I=0, h=0 and h=1 in bounded domain.

    These sampled inequalities do NOT prove global forward invariance.
    """
    bad = []
    for g in [0., .02, .1, 1., 10., 20.]:
        for c in [0., knots[0], knots[1], knots[2], .6, 1.2]:
            for i in [0., .2, .8, 2., 3.5]:
                for h in [0., 1.]:
                    y = np.array([c, h, i])
                    f = design(y, g, knots)[0] @ co
                    if c == 0.:
                        bad.append(min(f[0], 0.) / .4)
                    if i == 0.:
                        bad.append(min(f[2], 0.) / .1)
                    bad.append((min(f[1], 0.) if h == 0. else
                                max(f[1], 0.)) / .05)
    return np.asarray(bad)


def optimize(ref, anchors, anchor_g, knots, start, max_nfev):
    X, G, Y, segments, D, N = reference_data(
        ref, anchors, anchor_g, knots)
    B = design(X, G, knots) @ N
    V = np.linalg.lstsq(B, Y, rcond=None)[0]
    initial = V.ravel()
    if start is not None:
        old = np.asarray(start["coefficients"])
        if old.shape == (10, 3):
            initial = (N.T @ old).ravel()

    # The old optimizer requested instability in zone 1. That causes
    # premature onset at knot 0; require stable zone 1 instead.
    desired = np.array([-1., -1., +1., -1.])
    # Frozen reference phases; optimize finite-time return map.
    points = np.array([a for _, _, s in segments for a in s[:-1]])
    targets = np.array([a for _, _, s in segments for a in s[1:]])
    times = np.array([T / 32 for _, T, s in segments for _ in s[:-1]])
    gs = np.array([g for g, _, s in segments for _ in s[:-1]])
    scale = np.array([.4, .15, .4])

    def residual(x):
        co = N @ x.reshape(7, 3)
        def rhs(y):
            # No state clipping: unphysical excursions remain observable.
            return design(y, gs, knots) @ co
        y = points.copy()
        dt = times / 12.
        for _ in range(12):
            k1 = rhs(y)
            k2 = rhs(y + dt[:, None] * k1 / 2)
            k3 = rhs(y + dt[:, None] * k2 / 2)
            k4 = rhs(y + dt[:, None] * k3)
            y += dt[:, None] * (k1 + 2*k2 + 2*k3 + k4) / 6
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e6:
                return np.full(2500, 1e4)
        # Explicit physical-domain penalty along the short trajectories.
        phys = np.column_stack((np.minimum(y[:, 0], 0.),
                                np.minimum(y[:, 2], 0.),
                                np.minimum(y[:, 1], 0.),
                                np.maximum(y[:, 1] - 1., 0.)))
        ev = np.array([max(np.linalg.eigvals(zone_matrix(co, j)).real)
                       for j in range(4)])
        spectral = np.maximum(.015 - desired * ev, 0.) / .2
        # Exact equilibrium anchors follow algebraically from N.
        field_err = ((B @ x.reshape(7, 3) - Y) /
                     np.array([.4, .05, .1])).ravel() / np.sqrt(len(X))
        seg_err = ((y - targets) / scale).ravel() / np.sqrt(len(y))
        flux = boundary_residuals(co, knots)
        regularizer = .001 * (x - initial)
        return np.r_[field_err, 2. * seg_err, 3. * spectral,
                     4. * phys.ravel(), 4. * flux, regularizer]

    # Fixed-sized residuals are required by least_squares.
    def safe_residual(x):
        try:
            r = residual(x)
            if not np.all(np.isfinite(r)):
                raise FloatingPointError
            return r
        except (OverflowError, FloatingPointError, ValueError,
                np.linalg.LinAlgError):
            return np.full(2500, 1e4)

    # Evaluate loss safely before solving; bad initial guesses are rejected.
    base = safe_residual(initial)
    if len(base) == 2500:
        return None
    n = len(base)
    def bounded_residual(x):
        r = safe_residual(x)
        return r if len(r) == n else np.full(n, 1e4)

    sol = least_squares(bounded_residual, initial, max_nfev=max_nfev,
                        x_scale="jac", ftol=1e-5, xtol=1e-5)
    co = N @ sol.x.reshape(7, 3)
    return {"mode": "FM", "knots": knots.tolist(),
            "coefficients": co.tolist(),
            "fixed_equilibria": [
                {"gamma": float(g), "state": x.tolist()}
                for g, x in zip(anchor_g, anchors)],
            "max_anchor_residual": float(np.max(np.abs(D @ co))),
            "training_gammas": [float(x[0]) for x in segments],
            "input_exponent": .7,
            "objective": float(np.sum(sol.fun ** 2)),
            "status": str(sol.message),
            "method": "free-first-knot-grid-scan; physical-boundary-samples",
            "limitations": "No formal invariance proof, fold/Hopf classification "
                           "or validated cycle continuation. Run validate_modes."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=Path)
    ap.add_argument("--max-nfev", type=int, default=100)
    ap.add_argument("--n-knots", type=int, default=7)
    args = ap.parse_args()

    ref = json.loads(args.reference.read_text())
    if ref["mode"] != "FM":
        raise ValueError("Expected FM reference")
    if ref["parameters"] != P["FM"]:
        raise ValueError("Reference G-ChI parameters differ from fixed FM set")
    start = json.loads(args.start.read_text()) if args.start else None
    if start and start["mode"] != "FM":
        raise ValueError("Starting candidate must be FM")
    original = np.array([.04125858523076992, .05385609948006037,
                         .4241382759168485])
    # Use frozen reference anchor values from reference FM bifurcations.
    anchor_g = np.array([0., .020152517072605527, 1.3118429940002716])
    anchors = np.array([
        [.028661070981479464, .8794141767490402, .07240688915013724],
        [.05385609948006037, .8975425418917918, .5348363348718701],
        [.4241382759168485, .5856964798971739, .955256347421925]])
    args.out.mkdir(parents=True, exist_ok=True)
    candidates = []
    first_knots = np.linspace(anchors[0, 0] + .002,
                              anchors[1, 0] - .002, args.n_knots)
    for j, k1 in enumerate(first_knots):
        knots = np.array([k1, *original[1:]])
        result = optimize(ref, anchors, anchor_g, knots, start, args.max_nfev)
        if result is None:
            print("SKIP", j, "invalid starting candidate", flush=True)
            continue
        result["first_knot_grid_index"] = j
        path = args.out / f"candidate-{j}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        candidates.append((result["objective"], str(path)))
        print("FIT", j, "knot", round(k1, 6),
              "objective", round(result["objective"], 5), flush=True)
    if not candidates:
        raise RuntimeError("No fit completed")
    candidates.sort()
    (args.out / "ranking.json").write_text(
        json.dumps({"by_training_objective_only": candidates,
                    "warning": "Validate each candidate independently; "
                               "lower training loss is not proof of better FM."},
                   indent=2) + "\n")
    print("BEST TRAINING FIT", candidates[0], flush=True)


if __name__ == "__main__":
    main()
