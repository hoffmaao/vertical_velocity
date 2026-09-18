r"""MAP inversion for Thwaites: SSA or hybrid, Weertman or regularized Coulomb.

One driver for every configuration in the SSA-vs-hybrid comparison, so the models
differ only in the stress balance (and the hybrid optionally assimilates ApRES):

    python scripts/invert.py --model ssa    --friction weertman
    python scripts/invert.py --model hybrid --friction rc
    python scripts/invert.py --model hybrid --friction weertman --no-apres

Controls are log-deviations from the physical prior means of physical_prior.py
(fenics_ice / ISMIP7 method), identical for both models:

    A = A_prior exp(theta_A),    C = C_w0 exp(theta_C * grounded).

Cost (Recinos et al. 2023 convention, no 1/N):

    J = 0.5 sum chi^2_vel  [+ 0.5 sum chi^2_ApRES]  +  R(theta_A) + R(theta_C),
    R(theta) = 0.5 (delta int theta^2 + gamma int |grad theta|^2),  gamma = delta ELL^2.

The regularization is used twice: as the prior term of J and as the optimizer
metric. L-BFGS runs in prior-whitened coordinates xi (prior_precond.py), where
R = 0.5 |xi|^2 and the regularization Hessian is the identity, starting from
xi = 0 (the physical prior mean). Convergence is declared on the relative
reduction of |dJ/dxi|, not on an iteration cap.

Writes mesh/map_<tag>.h5 (2D controls + prior means + model velocity),
results/map_<tag>.json (settings, misfits, convergence) and
results/map_<tag>_history.npz (per-evaluation cost and gradient norm).
"""
import argparse
import json
import time

import firedrake
import icepack
import numpy as np
from firedrake import (
    CheckpointFile, Constant, ExtrudedMesh, Function, FunctionSpace,
    VectorFunctionSpace, VertexOnlyMesh, conditional, dx, exp, inner, sqrt,
)
from firedrake.adjoint import *  # noqa: F401,F403  (tlm_adjoint needs the annotation hooks)
from pathlib import Path
from scipy.optimize import minimize
from tlm_adjoint.firedrake import (
    Functional, compute_gradient, reset_manager, start_manager, stop_manager,
)

from friction_laws import FRICTION_LAWS, RC_C0, flotation_factor, weertman
from observations import load_apres, load_velocity
from physical_prior import MESH_FILE, load_physical_prior
from prior_precond import Whitening

DATA = Path(__file__).resolve().parent.parent

DELTA = 1.0e-5          # prior amplitude (Recinos L-curve elbow)
ELL = 2000.0            # prior correlation length (m), L-curve elbow at VEL_STEP=1
SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 250, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


class Converged(Exception):
    pass


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    ap.add_argument("--friction", choices=sorted(FRICTION_LAWS), required=True)
    ap.add_argument("--no-apres", action="store_true", help="hybrid: velocity data only")
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--ell", type=float, default=ELL, help="prior correlation length (m)")
    ap.add_argument("--grtol", type=float, default=1e-5,
                    help="stop when |dJ/dxi| <= grtol * |dJ/dxi at the prior mean|")
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    args = ap.parse_args()
    args.apres = args.model == "hybrid" and not args.no_apres
    args.tag = f"{args.model}_{args.friction}" + ("_noapres" if args.model == "hybrid" and args.no_apres else "")
    return args


def build_problem(args):
    r"""Mesh, spaces, prior means, data and a ``forward(theta_A, theta_C) -> u``."""
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")
    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    C_w0_2d, A_prior_2d = load_physical_prior(mesh_2d)
    xs, ys, vx, vy, sx, sy = load_velocity(mesh_2d)

    if args.model == "ssa":
        mesh, Q = mesh_2d, Q_2d
        V = VectorFunctionSpace(mesh, "CG", 1)
        h, s, C_w0, A_prior = h_2d, s_2d, C_w0_2d, A_prior_2d
        u0 = Function(V).assign(u_obs_2d)
        vel_pts = np.column_stack([xs, ys])
        model_cls = icepack.models.IceStream
    else:
        mesh = ExtrudedMesh(mesh_2d, layers=1)
        Q = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
        V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
        V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)
        lift = icepack.utilities.lift3d
        h, s, C_w0, A_prior = (lift(f, Q) for f in (h_2d, s_2d, C_w0_2d, A_prior_2d))
        u0 = Function(V).project(lift(u_obs_2d, V_lift))
        vel_pts = np.column_stack([xs, ys, np.ones_like(xs)])
        model_cls = icepack.models.HybridModel
    assert Q.dim() == Q_2d.dim(), "controls must share the 2D CG1 dof ordering"

    def to_vom(space_in, space, vals):
        f_in = Function(space_in)
        f_in.dat.data[:] = vals[:len(f_in.dat.data)]
        return Function(space).interpolate(f_in)

    vom = VertexOnlyMesh(mesh, vel_pts, missing_points_behaviour="warn")
    Δv, Δv_in = FunctionSpace(vom, "DG", 0), FunctionSpace(vom.input_ordering, "DG", 0)
    obs = {"u": to_vom(Δv_in, Δv, vx), "v": to_vom(Δv_in, Δv, vy),
           "σx": to_vom(Δv_in, Δv, sx), "σy": to_vom(Δv_in, Δv, sy), "Δv": Δv}
    if args.apres:
        pts, eps, sig = load_apres(mesh_2d)
        vom_a = VertexOnlyMesh(mesh, pts, missing_points_behaviour="warn")
        Δa, Δa_in = FunctionSpace(vom_a, "DG", 0), FunctionSpace(vom_a.input_ordering, "DG", 0)
        obs.update(eps=to_vom(Δa_in, Δa, eps), σe=to_vom(Δa_in, Δa, sig), Δa=Δa)

    grounded = Function(Q).interpolate(
        conditional(flotation_factor(h, s) > 0.01, Constant(1.0), Constant(0.0)))

    def make_solver(friction):
        return icepack.solvers.FlowSolver(
            model_cls(friction=friction), dirichlet_ids=[2], ice_front_ids=[1],
            diagnostic_solver_type="petsc", diagnostic_solver_parameters=SOLVER_PARAMS)

    solver = make_solver(FRICTION_LAWS[args.friction])

    def fields(θ_A, θ_C):
        A = Function(Q).interpolate(A_prior * exp(θ_A))
        C = Function(Q).interpolate(C_w0 * exp(θ_C * grounded))
        return A, C

    def forward(θ_A, θ_C, slv=solver):
        A, C = fields(θ_A, θ_C)
        return slv.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A, friction=C)

    return dict(mesh_2d=mesh_2d, Q_2d=Q_2d, Q=Q, h=h, s=s, u0=u0, u_obs_2d=u_obs_2d,
                C_w0_2d=C_w0_2d, A_prior_2d=A_prior_2d, obs=obs, forward=forward,
                weertman_solver=make_solver(weertman), fields=fields)


def misfit(u, P, args):
    r"""Taped misfit Functional and its (velocity, ApRES) parts."""
    o = P["obs"]
    ui = Function(o["Δv"]).interpolate(u[0])
    vi = Function(o["Δv"]).interpolate(u[1])
    J = Functional(name="J")
    J.assign(0.5 * (((ui - o["u"]) / o["σx"]) ** 2 + ((vi - o["v"]) / o["σy"]) ** 2) * dx)
    J_vel = float(J)
    J_apr = 0.0
    if args.apres:
        ε = icepack.models.hybrid.horizontal_strain_rate(velocity=u, thickness=P["h"], surface=P["s"])
        εi = Function(o["Δa"]).interpolate(-(ε[0, 0] + ε[1, 1]))
        J.addto(0.5 * ((εi - o["eps"]) / o["σe"]) ** 2 * dx)
        J_apr = float(J) - J_vel
    return J, J_vel, J_apr


def main():
    args = parse_args()
    t_start = time.time()
    gamma = args.delta * args.ell ** 2
    print("=" * 64, flush=True)
    print(f"MAP inversion [{args.tag}]: model={args.model} friction={args.friction} "
          f"apres={args.apres}", flush=True)
    print(f"  prior: delta={args.delta:.1e} gamma={gamma:.1f} ELL={args.ell:.0f} m; "
          f"physical means C_w0, A_prior; whitened L-BFGS, grtol={args.grtol:.0e}", flush=True)
    print("=" * 64, flush=True)

    P = build_problem(args)
    Q, n = P["Q"], P["Q_2d"].dim()
    W = Whitening(P["Q_2d"], args.delta, gamma)
    θ_A = Function(Q, name="log_fluidity")
    θ_C = Function(Q, name="log_friction")
    print(f"Controls: 2 x {n} dofs; velocity obs {len(P['obs']['u'].dat.data)}"
          + (f", ApRES obs {len(P['obs']['eps'].dat.data)}" if args.apres else ""), flush=True)

    # Prime the velocity at the prior mean; RC is stiffer, so fall back to a
    # Weertman solve for the initial guess if it does not converge directly.
    try:
        P["u0"].assign(P["forward"](θ_A, θ_C))
    except firedrake.exceptions.ConvergenceError:
        print("  prior-mean solve failed; priming with Weertman first", flush=True)
        P["u0"].assign(P["forward"](θ_A, θ_C, slv=P["weertman_solver"]))
        P["u0"].assign(P["forward"](θ_A, θ_C))
    print("Primed forward solve at the prior mean OK", flush=True)

    ckpt = DATA / "mesh" / f"checkpoint_map_{args.tag}.npz"
    hist = {"J": [], "J_vel": [], "J_apr": [], "R": [], "gnorm": [], "accepted": []}
    state = {"good": None, "g0": None, "cache": {}, "nit": 0, "xbest": None}

    def evaluate(ξ):
        key = ξ.tobytes()
        if key in state["cache"]:
            return state["cache"][key]
        θ_A.dat.data[:] = W.to_theta(ξ[:n])
        θ_C.dat.data[:] = W.to_theta(ξ[n:])
        reset_manager(); start_manager()
        try:
            u = P["forward"](θ_A, θ_C)
            J, J_vel, J_apr = misfit(u, P, args)
            dA, dC = compute_gradient(J, (θ_A, θ_C))
        except Exception as e:   # forward or adjoint failure: reject the step
            stop_manager()
            print(f"  eval failed ({type(e).__name__}); rejecting step", flush=True)
            J0, g0 = state["good"]
            return 10.0 * J0 + 1.0, g0
        stop_manager()
        R = 0.5 * float(ξ @ ξ)
        g = ξ + np.concatenate([W.grad_to_xi(dA.dat.data_ro), W.grad_to_xi(dC.dat.data_ro)])
        f = J_vel + J_apr + R
        gn = float(np.linalg.norm(g))
        for k, v in zip(("J", "J_vel", "J_apr", "R", "gnorm", "accepted"),
                        (f, J_vel, J_apr, R, gn, False)):
            hist[k].append(v)
        P["u0"].assign(u)
        state["good"] = (f, g.copy())
        state["cache"] = {key: (f, g)}
        state["last"] = (J_vel, J_apr, R, gn)
        return f, g

    def callback(ξ):
        state["nit"] += 1
        state["xbest"] = ξ.copy()
        f, g = evaluate(ξ)
        J_vel, J_apr, R, gn = state["last"]
        hist["accepted"][-1] = True
        rel = gn / state["g0"]
        print(f"  it {state['nit']:4d}: J={f:.5e} Jv={J_vel:.4e} Ja={J_apr:.4e} R={R:.4e} "
              f"|g|={gn:.3e} rel={rel:.2e}", flush=True)
        if state["nit"] % 10 == 0:
            np.savez(ckpt, xi=ξ, nit=state["nit"], g0=state["g0"])
        if rel <= args.grtol:
            raise Converged

    ξ0 = np.zeros(2 * n)
    if ckpt.exists() and not args.fresh:
        c = np.load(ckpt)
        ξ0, state["nit"] = c["xi"], int(c["nit"])
        print(f"Resuming from {ckpt.name} at iteration {state['nit']}", flush=True)
    f0, g0 = evaluate(ξ0)
    state["g0"] = float(np.load(ckpt)["g0"]) if (ckpt.exists() and not args.fresh) else float(np.linalg.norm(g0))
    state["xbest"] = ξ0.copy()
    print(f"Start: J={f0:.5e} |g|={np.linalg.norm(g0):.3e}", flush=True)

    status = "maxiter"
    try:
        res = minimize(evaluate, ξ0, jac=True, method="L-BFGS-B", callback=callback,
                       options={"maxiter": args.maxiter - state["nit"], "maxfun": 10 * args.maxiter,
                                "maxcor": 20, "ftol": 0.0, "gtol": 0.0})
        status = res.message if isinstance(res.message, str) else res.message.decode()
    except Converged:
        status = f"converged: |g| reduced by {args.grtol:.0e}"
    print(f"\nResult: {status} after {state['nit']} iterations", flush=True)

    # ── Final state and diagnostics at the last accepted iterate ──
    ξ = state["xbest"]
    f, g = evaluate(ξ)
    J_vel, J_apr, R, gn = state["last"]
    o = P["obs"]
    N_vel, N_apr = len(o["u"].dat.data), (len(o["eps"].dat.data) if args.apres else 0)
    θ_A2 = Function(P["Q_2d"], name="log_fluidity"); θ_A2.dat.data[:] = θ_A.dat.data_ro
    θ_C2 = Function(P["Q_2d"], name="log_friction"); θ_C2.dat.data[:] = θ_C.dat.data_ro
    u = P["u0"]
    u2 = u if args.model == "ssa" else icepack.depth_average(u)
    u2.rename("velocity")
    speed = Function(P["Q_2d"], name="speed_model").interpolate(sqrt(inner(u2, u2)))
    with CheckpointFile(str(DATA / "mesh" / f"map_{args.tag}.h5"), "w") as chk:
        chk.save_mesh(P["mesh_2d"])
        for fn in (θ_A2, θ_C2, P["C_w0_2d"], P["A_prior_2d"], u2, speed):
            chk.save_function(fn)
    summary = {
        "tag": args.tag, "model": args.model, "friction": args.friction, "apres": args.apres,
        "rc_c0": RC_C0 if args.friction == "rc" else None,
        "delta": args.delta, "gamma": gamma, "ELL": args.ell,
        "prior_mean": "physical (C_w0 balance anchor, A_prior thermo)",
        "optimizer": "L-BFGS-B in prior-whitened coordinates",
        "status": status, "iterations": state["nit"], "evaluations": len(hist["J"]),
        "grad_rel_reduction": gn / state["g0"], "J": f, "J_vel": J_vel, "J_apr": J_apr, "R": R,
        "chi2_vel_per_obs": 2 * J_vel / N_vel, "N_vel": N_vel,
        "chi2_apr_per_obs": (2 * J_apr / N_apr) if N_apr else None, "N_apr": N_apr,
        "theta_A_range": [float(θ_A2.dat.data.min()), float(θ_A2.dat.data.max())],
        "theta_C_range": [float(θ_C2.dat.data.min()), float(θ_C2.dat.data.max())],
        "wall_hours": (time.time() - t_start) / 3600,
    }
    json.dump(summary, open(DATA / "results" / f"map_{args.tag}.json", "w"), indent=2)
    np.savez(DATA / "results" / f"map_{args.tag}_history.npz", **{k: np.array(v) for k, v in hist.items()})
    for k in ("status", "iterations", "grad_rel_reduction", "chi2_vel_per_obs", "chi2_apr_per_obs",
              "theta_A_range", "theta_C_range", "wall_hours"):
        print(f"  {k}: {summary[k]}", flush=True)
    print(f"Saved mesh/map_{args.tag}.h5 and results/map_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
