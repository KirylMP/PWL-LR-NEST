#!/usr/bin/env python3
"""Calcium-first AFM refinement of an existing continuous four-region PWA.

The smooth AFM G-ChI field and its parameter set are never modified.
Three full-state equilibrium anchors are enforced by a knot-dependent null space.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import least_squares
from fit_modes import design
from explore_modes import P, field


def regional_eigenvalues(co):
    A = co[1:4].T.copy()
    out = [float(np.max(np.linalg.eigvals(A).real))]
    for j in range(3):
        A = A.copy()
        A[:, 0] += co[7+j]
        out.append(float(np.max(np.linalg.eigvals(A).real)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("results/AFM/reference.json"))
    ap.add_argument("--start", type=Path, default=Path("results/AFM/candidate_segments.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--c1", type=float, required=True)
    ap.add_argument("--nfev", type=int, default=70)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=.001)
    ap.add_argument("--calcium-weight", type=float, default=3.)
    ap.add_argument("--negative-c-weight", type=float, default=8.)
    args = ap.parse_args()

    ref = json.loads(args.data.read_text())
    d = json.loads(args.start.read_text())
    if ref["mode"] != "AFM" or d["mode"] != "AFM":
        raise ValueError("AFM inputs only")
    knots = np.array(d["knots"], dtype=float).copy()
    anchors = np.array([x["state"] for x in d["fixed_equilibria"]])
    ag = np.array([x["gamma"] for x in d["fixed_equilibria"]])
    if not anchors[0, 0] + 1.e-4 < args.c1 < knots[1] - 1.e-4:
        raise ValueError("C1 must remain between rest and second anchor")
    if args.nfev < 1 or args.calcium_weight <= 0 or args.negative_c_weight <= 0:
        raise ValueError("Invalid iteration or penalty parameters")
    knots[0] = args.c1
    D = design(anchors, ag, knots)
    N = null_space(D)
    shape = (N.shape[1], 3)
    if N.shape[1] != 7:
        raise ValueError("Unexpected anchor rank")
    original = np.asarray(d["coefficients"], dtype=float)
    v0 = (N.T @ original).ravel()
    co0 = N @ v0.reshape(shape)
    # Reprojection is required when C1 changes, and keeps anchors exact.
    anchor_error = float(np.max(np.abs(D @ co0)))
    if anchor_error > 1.e-9:
        raise ValueError("Anchor reprojection failed")

    good = [x for x in ref["cycles"] if x["status"] == "cycle"]
    ix = np.unique(np.linspace(1, len(good)-2, min(9, len(good)-2), dtype=int))
    train = [good[i] for i in ix]
    gg = np.array([x["gamma"] for x in train])
    T = np.array([x["period"] for x in train])
    peaks = np.array([x["peak"] for x in train])
    Cmax = np.array([x["Cmax"] for x in train])
    Cmin = np.array([x["Cmin"] for x in train])
    samples = []
    X = []
    G = []
    Y = []
    for q in train:
        g, period = q["gamma"], q["period"]
        sol = solve_ivp(field, (0, period), q["peak"], args=(g, P["AFM"]),
                        t_eval=np.linspace(0, period, 33), method="DOP853",
                        rtol=1e-9, atol=1e-11, max_step=.2)
        if not sol.success:
            raise RuntimeError("Reference cycle integration failed")
        samples.append(sol.y[:, ::4].T)
        X.extend(sol.y.T[:-1])
        G.extend([g] * (len(sol.t)-1))
        Y.extend(field(0, state, g, P["AFM"])
                 for state in sol.y.T[:-1])
    phases = np.asarray(samples)  # 9 x 9 x 3
    X, G, Y = np.asarray(X), np.asarray(G), np.asarray(Y)
    B = design(X, G, knots) @ N
    # Short segment, 4-point RK4 substeps; a smooth continuous PWA field.
    seg_dt = np.repeat(T / 32, 8)
    seg_g = np.repeat(gg, 8)
    # Full cycles: emphasize calcium minima/maxima, closure and positivity.
    full_n = int(np.ceil(float(np.max(T)) / .06))
    full_dt = T / full_n
    calls = 0
    best = [np.inf, v0.copy()]

    def rhs(states, gammas, co):
        states = np.asarray(states)
        y = np.clip(states, -3, 4)
        val = design(y, gammas, knots) @ co
        return np.clip(np.nan_to_num(val, nan=100, posinf=100,
                                     neginf=-100), -50, 50)

    def step(y, dt, gammas, co):
        k1 = rhs(y, gammas, co)
        k2 = rhs(y + dt[:, None] * k1 / 2, gammas, co)
        k3 = rhs(y + dt[:, None] * k2 / 2, gammas, co)
        k4 = rhs(y + dt[:, None] * k3, gammas, co)
        return y + dt[:, None] * (k1 + 2*k2 + 2*k3 + k4)/6

    def loss(v):
        nonlocal calls
        calls += 1
        co = N @ v.reshape(shape)
        # Short transients reproduce the calcium waveform and approximate
        # auxiliary variables, without making IP3 the optimization target.
        y = phases[:, :-1].reshape(-1, 3).copy()
        segment_min = y.copy()
        segment_max = y.copy()
        for _ in range(4):
            y = step(y, seg_dt, seg_g, co)
            segment_min = np.minimum(segment_min, y)
            segment_max = np.maximum(segment_max, y)
        err = (y.reshape(len(train), 8, 3) - phases[:, 1:])
        calcium_segment = 1.8 * args.calcium_weight * err[..., 0].ravel() / .4
        auxiliary_segment = np.r_[err[..., 1].ravel()/.15,
                                  err[..., 2].ravel()/.4]
        field_error = (B @ v.reshape(shape) - Y) / [.4, .05, .1]
        # Endpoints and extrema of the predicted orbit; the reference period
        # is a shooting length, not proof of fundamental period matching.
        yf = peaks.copy()
        ymin, ymax = yf.copy(), yf.copy()
        for _ in range(full_n):
            yf = step(yf, full_dt, gg, co)
            ymin = np.minimum(ymin, yf)
            ymax = np.maximum(ymax, yf)
        closure = (yf - peaks) / [.4, .15, .4]
        C_extrema = np.r_[(ymax[:, 0]-Cmax)/.25,
                          (ymin[:, 0]-Cmin)/.1]
        Cneg = np.r_[np.maximum(-segment_min[:, 0], 0),
                     np.maximum(-ymin[:, 0], 0)] / .05
        Ineg = np.r_[np.maximum(-segment_min[:, 2], 0),
                     np.maximum(-ymin[:, 2], 0)] / .1
        hbad = np.r_[np.maximum(-segment_min[:, 1], 0) +
                     np.maximum(segment_max[:, 1]-1, 0),
                     np.maximum(-ymin[:, 1], 0) +
                     np.maximum(ymax[:, 1]-1, 0)] / .1
        leads = regional_eigenvalues(co)
        # Original AFM was calibrated as stable / stable / unstable / stable.
        spec = np.maximum(.01 - np.array([-1., -1., 1., -1.])*leads, 0) / .2
        z = np.r_[
            field_error.ravel() / np.sqrt(field_error.size),
            calcium_segment / np.sqrt(calcium_segment.size),
            .3*auxiliary_segment / np.sqrt(auxiliary_segment.size),
            args.calcium_weight*C_extrema / np.sqrt(C_extrema.size),
            args.calcium_weight*closure[:, 0] / np.sqrt(len(train)),
            .4*closure[:, 1:].ravel() / np.sqrt(2*len(train)),
            args.negative_c_weight*Cneg / np.sqrt(len(Cneg)),
            3*Ineg / np.sqrt(len(Ineg)),
            hbad / np.sqrt(len(hbad)),
            .75*spec,
            .003*(v-v0)
        ]
        z = np.clip(np.nan_to_num(z, nan=1e4, posinf=1e4,
                                  neginf=-1e4), -1e4, 1e4)
        score = float(z @ z)
        if score < best[0]:
            best[:] = [score, v.copy()]
        if calls % 75 == 1:
            print("calls", calls, "cost", round(score, 6),
                  "best", round(best[0], 6), flush=True)
        return z

    vstart = v0.copy()
    if args.seed:
        vstart += np.random.default_rng(args.seed).normal(
            0, args.jitter, size=vstart.shape)
    initial_score = float(np.linalg.norm(loss(v0))**2)
    sol = least_squares(loss, vstart, max_nfev=args.nfev,
                        ftol=1e-6, xtol=1e-6, x_scale="jac")
    co = N @ best[1].reshape(shape)
    out = dict(mode="AFM", knots=knots.tolist(),
               coefficients=co.tolist(),
               fixed_equilibria=d["fixed_equilibria"],
               max_anchor_residual=float(np.max(np.abs(D @ co))),
               training_gammas=gg.tolist(),
               objective=best[0], initial_objective=initial_score,
               calls=calls, seed=args.seed, status=str(sol.message),
               optimizer_success=bool(sol.success),
               calcium_weight=args.calcium_weight,
               negative_calcium_weight=args.negative_c_weight,
               all_zone_leading_eigenvalues=regional_eigenvalues(co).tolist(),
               starting_candidate=str(args.start),
               limitations="Soft physical and spectral penalties; clipped RK4 "
                           "proxies may miss physical excursions. Closure at "
                           "reference period need not be a fundamental cycle. "
                           "Independent adaptive validation is required.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+"\n")
    print("DONE", args.output, "objective", best[0], flush=True)


if __name__ == "__main__":
    main()
