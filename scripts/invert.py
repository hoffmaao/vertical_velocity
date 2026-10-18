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

    J = 0.5 sum chi^2_vel  [+ 0.5 sum chi^2_ApRES]  +  R_A(theta_A) + R_C(theta_C),
    R_X(theta) = 0.5 theta^T A_X M^-1 A_X theta,   A_X = delta_X M + gamma_X K,
    gamma_X = delta_X ELL_X^2,

the fenics_ice "Laplacian" (bilaplacian-covariance) prior, Gamma = A^-1 M A^-1
(prior_precond.py). --prior-form h1 selects the 0.5 theta^T A theta form the
code used before 2026-10-14; delta means something different under each form
(see prior_precond.py), so hyperparameters are chosen per form.

The two priors are independent (no cross-covariance) and each has its own
hyperparameters: --delta-A/--ell-A for the fluidity, --delta-C/--ell-C for the
friction. Each defaults to the shared --delta/--ell, so a run that passes only
those uses one prior strength and correlation length for both controls.

--smoothness-only (h1 form only) drops the amplitude term delta_X int theta^2
from J, so R_X = 0.5 gamma_X int |grad theta|^2: the controls are smoothed but
nothing pulls them back toward the prior mean, which becomes only the starting
point. gamma_X keeps its delta_X ELL_X^2 value, and the metric below keeps the
full delta M + gamma K (the gradient-only operator is singular on constants), so
the optimizer is conditioned exactly as in a normal run.

The regularization is used twice: as the prior term of J and as the optimizer
metric. L-BFGS runs in prior-whitened coordinates xi (prior_precond.py), where
the regularization Hessian is the identity (h1) or within a factor ~2 of it
(bilaplacian), starting from xi = 0 (the physical prior mean).

Convergence follows fenics_ice (Recinos et al. 2023), which stops on whichever
of two tests fires first, after a min_iter guard:

  grtol  |dJ/dxi| <= grtol * |dJ/dxi at the prior mean|   (relative, whitened)
  ftol   (J_old - J_new)/max(|J_old|,|J_new|,1) <= ftol   (fenics_ice solver.py)

Bea's smith.toml uses ftol = 1e-8, gtol = 1e-12 and L-BFGS memory m = 200. Her
gtol is an *absolute* L-infinity norm on the theta-space gradient and so does
not transfer to our relative whitened-L2 test; ftol is dimensionless and does,
and is what actually stops her runs. At the earlier grtol = 1e-5 our MAPs
stopped with f_crit still at 1e-3..1e-6, three to six decades looser than hers,
hence the default grtol = 1e-7 and maxcor = 200 here.

Unlike fenics_ice, the ftol test must hold for ftol_patience consecutive
accepted iterations. A single stalled L-BFGS step (a line search that barely
moves) passes a one-step test while the run is still descending: the sweep had
points stop with f_crit jumping 7e-5 -> 6e-11 in one iteration, and points
stopping on the first step after a checkpoint resume, when L-BFGS has just lost
its curvature history.

Writes mesh/map_<name>.h5 (2D controls + prior means + model velocity),
results/map_<name>.json (settings, misfits, convergence) and
results/map_<name>_history.npz (per-evaluation cost and gradient norm), where
<name> defaults to the configuration tag and --name overrides it. A
regularization sweep must pass a distinct --name per (delta, ell) point: the
checkpoint is keyed on it too, so sharing a name both overwrites the production
MAP and warm-starts from another point's iterate. The checkpoint also records
both priors, and a resume under different priors is refused: its iterate lives
in the old whitened coordinates.
"""
import argparse
import json
import time

import firedrake
import icepack
import numpy as np
from firedrake import CheckpointFile, Function, inner, sqrt
from firedrake.adjoint import *  # noqa: F401,F403  (tlm_adjoint needs the annotation hooks)
from pathlib import Path
from scipy.optimize import minimize
from tlm_adjoint.firedrake import compute_gradient, reset_manager, start_manager, stop_manager

from friction_laws import ANCHOR, FRICTION_LAWS, GL_THRESH, GL_WIDTH, RC_C0, SCHOOF_U0
from physical_prior import MESH_NAME, PRIOR_FILE, TAU_D_SMOOTH
from prior_precond import DEFAULT_FORM, FORMS, Prior, pointwise_std, precision_matrix
from problem import build_problem, misfit, tag

DATA = Path(__file__).resolve().parent.parent

DELTA = 1.0e-5          # prior amplitude (Recinos L-curve elbow)
ELL = 2000.0            # prior correlation length (m), L-curve elbow at VEL_STEP=1


class Converged(Exception):
    pass


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    ap.add_argument("--friction", choices=sorted(FRICTION_LAWS), required=True)
    ap.add_argument("--no-apres", action="store_true", help="hybrid: velocity data only")
    ap.add_argument("--delta", type=float, default=DELTA,
                    help="prior strength for both controls unless overridden below")
    ap.add_argument("--ell", type=float, default=ELL,
                    help="prior correlation length (m) for both controls unless overridden")
    ap.add_argument("--delta-A", type=float, default=None, help="fluidity prior strength")
    ap.add_argument("--ell-A", type=float, default=None,
                    help="fluidity prior correlation length (m)")
    ap.add_argument("--delta-C", type=float, default=None, help="friction prior strength")
    ap.add_argument("--ell-C", type=float, default=None,
                    help="friction prior correlation length (m)")
    ap.add_argument("--prior-form", choices=FORMS, default=DEFAULT_FORM,
                    help="bilaplacian: R = 0.5 theta^T A M^-1 A theta (fenics_ice / Recinos); "
                         "h1: R = 0.5 theta^T A theta (this code before 2026-10-14)")
    ap.add_argument("--smoothness-only", action="store_true",
                    help="no prior mean: drop delta int theta^2 from J and keep only "
                         "0.5 gamma int |grad theta|^2 (delta still sets the optimizer metric)")
    ap.add_argument("--A-const-T", type=float, default=None,
                    help="replace the thermomechanical A_prior with the uniform "
                         "rate_factor(T) at this temperature (K); C_w0 is unchanged")
    ap.add_argument("--grtol", type=float, default=1e-7,
                    help="stop when |dJ/dxi| <= grtol * |dJ/dxi at the prior mean|")
    ap.add_argument("--ftol", type=float, default=1e-8,
                    help="fenics_ice functional test (solver.py l_bfgs_converged): stop "
                         "when (J_old - J_new)/max(|J_old|,|J_new|,1) <= ftol. Bea's "
                         "smith.toml uses 1e-8; pass 0 to disable")
    ap.add_argument("--ftol-patience", type=int, default=10,
                    help="the ftol test must hold for this many consecutive accepted "
                         "iterations, so one stalled line search cannot stop the run")
    ap.add_argument("--min-iter", type=int, default=20,
                    help="never declare convergence before this iteration (fenics_ice "
                         "min_iter), so one flat step cannot stop the run early")
    ap.add_argument("--maxcor", type=int, default=200,
                    help="L-BFGS memory; fenics_ice smith.toml uses m = 200")
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    ap.add_argument("--name", default=None,
                    help="output stem for the checkpoint, h5, json and history; "
                         "defaults to the configuration tag. A regularization sweep "
                         "must set this per (delta, ell) point, otherwise every point "
                         "overwrites the production MAP and resumes from its checkpoint")
    args = ap.parse_args()
    args.apres = args.model == "hybrid" and not args.no_apres
    args.tag = tag(args.model, args.friction, args.apres)
    args.name = args.name or args.tag
    args.delta_A = args.delta if args.delta_A is None else args.delta_A
    args.ell_A = args.ell if args.ell_A is None else args.ell_A
    args.delta_C = args.delta if args.delta_C is None else args.delta_C
    args.ell_C = args.ell if args.ell_C is None else args.ell_C
    args.gamma_A = args.delta_A * args.ell_A ** 2
    args.gamma_C = args.delta_C * args.ell_C ** 2
    args.tied = (args.delta_A, args.ell_A) == (args.delta_C, args.ell_C)
    if args.smoothness_only and args.prior_form != "h1":
        raise SystemExit("--smoothness-only is only defined for --prior-form h1")
    return args


def main():
    args = parse_args()
    t_start = time.time()
    print("=" * 64, flush=True)
    print(f"MAP inversion [{args.tag}]: model={args.model} friction={args.friction} "
          f"apres={args.apres}", flush=True)
    for X in ("A", "C"):
        d, g = getattr(args, "delta_" + X), getattr(args, "gamma_" + X)
        print(f"  prior {X} ({args.prior_form}): delta={d:.1e} gamma={g:g} "
              f"ELL={getattr(args, 'ell_' + X):.0f} m; pointwise prior std of theta "
              f"{pointwise_std(d, g, args.prior_form, h_min=500.0):.3f}"
              + (" (at the 500 m mesh scale)" if args.prior_form == "h1" else ""), flush=True)
    amean = ("A_prior thermo" if args.A_const_T is None else
             f"A constant = rate_factor({args.A_const_T:g} K)")
    if args.smoothness_only:
        amean += " as the START only (smoothness-only R, no amplitude term)"
    print(f"  prior means {ANCHOR[args.friction]}, {amean}; whitened L-BFGS, "
          f"grtol={args.grtol:.0e}", flush=True)
    print("=" * 64, flush=True)

    P = build_problem(args.model, args.friction, args.apres, A_const_T=args.A_const_T)
    Q, n = P["Q"], P["Q_2d"].dim()
    W_A = Prior(P["Q_2d"], args.delta_A, args.gamma_A, args.prior_form)
    W_C = W_A if args.tied else Prior(P["Q_2d"], args.delta_C, args.gamma_C, args.prior_form)
    # --smoothness-only: R_X = 0.5 |xi_X|^2 - 0.5 delta_X theta_X^T M theta_X, i.e.
    # the whitened prior with its amplitude part taken back out.
    M = precision_matrix(P["Q_2d"], 1.0, 0.0) if args.smoothness_only else None
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

    ckpt = DATA / "mesh" / f"checkpoint_map_{args.name}.npz"
    hist = {"J": [], "J_vel": [], "J_apr": [], "R": [], "R_A": [], "R_C": [],
            "gnorm": [], "accepted": []}
    priors = np.array([args.delta_A, args.gamma_A, args.delta_C, args.gamma_C])
    state = {"good": None, "g0": None, "cache": {}, "nit": 0, "xbest": None,
             "J_prev": None, "why": None, "flat": 0}
    fcrit_hist = []

    def evaluate(ξ):
        key = ξ.tobytes()
        if key in state["cache"]:
            return state["cache"][key]
        θ_A.dat.data[:] = W_A.to_theta(ξ[:n])
        θ_C.dat.data[:] = W_C.to_theta(ξ[n:])
        reset_manager(); start_manager()
        try:
            u = P["forward"](θ_A, θ_C)
            J, J_vel, J_apr = misfit(u, P, args.apres)
            dA, dC = compute_gradient(J, (θ_A, θ_C))
        except Exception as e:   # forward or adjoint failure: reject the step
            stop_manager()
            print(f"  eval failed ({type(e).__name__}); rejecting step", flush=True)
            J0, g0 = state["good"]
            return 10.0 * J0 + 1.0, g0
        stop_manager()
        R_A, gR_A = W_A.R_and_grad_xi(ξ[:n])
        R_C, gR_C = W_C.R_and_grad_xi(ξ[n:])
        g = np.concatenate([gR_A + W_A.grad_to_xi(dA.dat.data_ro),
                            gR_C + W_C.grad_to_xi(dC.dat.data_ro)])
        if M is not None:
            tA, tC = θ_A.dat.data_ro.copy(), θ_C.dat.data_ro.copy()
            MA, MC = args.delta_A * (M @ tA), args.delta_C * (M @ tC)
            R_A -= 0.5 * float(tA @ MA)
            R_C -= 0.5 * float(tC @ MC)
            g -= np.concatenate([W_A.grad_to_xi(MA), W_C.grad_to_xi(MC)])
        R = R_A + R_C
        f = J_vel + J_apr + R
        gn = float(np.linalg.norm(g))
        for k, v in zip(("J", "J_vel", "J_apr", "R", "R_A", "R_C", "gnorm", "accepted"),
                        (f, J_vel, J_apr, R, R_A, R_C, gn, False)):
            hist[k].append(v)
        P["u0"].assign(u)
        state["good"] = (f, g.copy())
        state["cache"] = {key: (f, g)}
        state["last"] = (J_vel, J_apr, R, gn)
        state["R_split"] = (R_A, R_C)
        return f, g

    def callback(ξ):
        state["nit"] += 1
        state["xbest"] = ξ.copy()
        f, g = evaluate(ξ)
        J_vel, J_apr, R, gn = state["last"]
        hist["accepted"][-1] = True
        rel = gn / state["g0"]
        # fenics_ice's functional test: the relative decrease of J across one
        # accepted iteration (fenics_ice/solver.py, l_bfgs_converged).
        Jp = state["J_prev"]
        f_crit = np.inf if Jp is None else (Jp - f) / max(abs(Jp), abs(f), 1.0)
        state["J_prev"] = f
        fcrit_hist.append(f_crit)
        print(f"  it {state['nit']:4d}: J={f:.5e} Jv={J_vel:.4e} Ja={J_apr:.4e} R={R:.4e} "
              f"|g|={gn:.3e} rel={rel:.2e} fcrit={f_crit:.2e}", flush=True)
        if state["nit"] % 10 == 0:
            np.savez(ckpt, xi=ξ, nit=state["nit"], g0=state["g0"], priors=priors,
                     smoothness_only=args.smoothness_only, prior_form=args.prior_form)
        if state["nit"] < args.min_iter:      # fenics_ice min_iter guard
            return
        if rel <= args.grtol:
            state["why"] = f"|g| reduced by {args.grtol:.0e}"
            raise Converged
        state["flat"] = state["flat"] + 1 if (args.ftol and f_crit <= args.ftol) else 0
        if state["flat"] >= args.ftol_patience:
            state["why"] = (f"ftol {args.ftol:.0e} for {args.ftol_patience} consecutive "
                            f"iterations (fenics_ice rule + patience)")
            raise Converged

    ξ0 = np.zeros(2 * n)
    if ckpt.exists() and not args.fresh:
        c = np.load(ckpt)
        # Checkpoints written before the priors were split carry no "priors";
        # they were made with one shared prior, which --delta/--ell reproduce.
        if "priors" in c.files and not np.allclose(c["priors"], priors, rtol=1e-12, atol=0):
            raise SystemExit(
                f"{ckpt.name} was written with priors (delta_A, gamma_A, delta_C, gamma_C) = "
                f"{c['priors'].tolist()}, not {priors.tolist()}; its iterate is in the other "
                f"prior's whitened coordinates. Use a different --name, or --fresh.")
        was_form = str(c["prior_form"]) if "prior_form" in c.files else "h1"
        if was_form != args.prior_form:
            raise SystemExit(f"{ckpt.name} was written with the {was_form} prior form, not "
                             f"{args.prior_form}; use a different --name, or --fresh.")
        was_smooth = bool(c["smoothness_only"]) if "smoothness_only" in c.files else False
        if was_smooth != args.smoothness_only:
            raise SystemExit(f"{ckpt.name} was written with smoothness_only={was_smooth}; "
                             f"use a different --name, or --fresh.")
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
                                "maxcor": args.maxcor, "ftol": 0.0, "gtol": 0.0})
        status = res.message if isinstance(res.message, str) else res.message.decode()
    except Converged:
        status = f"converged: {state['why']}"
    print(f"\nResult: {status} after {state['nit']} iterations", flush=True)

    # ── Final state and diagnostics at the last accepted iterate ──
    ξ = state["xbest"]
    f, g = evaluate(ξ)
    J_vel, J_apr, R, gn = state["last"]
    R_A, R_C = state["R_split"]
    o = P["obs"]
    N_vel, N_apr = len(o["u"].dat.data), (len(o["eps"].dat.data) if args.apres else 0)
    θ_A2 = Function(P["Q_2d"], name="log_fluidity"); θ_A2.dat.data[:] = θ_A.dat.data_ro
    θ_C2 = Function(P["Q_2d"], name="log_friction"); θ_C2.dat.data[:] = θ_C.dat.data_ro
    u = P["u0"]
    u2 = u if args.model == "ssa" else icepack.depth_average(u)
    u2.rename("velocity")
    speed = Function(P["Q_2d"], name="speed_model").interpolate(sqrt(inner(u2, u2)))
    with CheckpointFile(str(DATA / "mesh" / f"map_{args.name}.h5"), "w") as chk:
        chk.save_mesh(P["mesh_2d"])
        for fn in (θ_A2, θ_C2, P["C_anchor_2d"], P["A_prior_2d"], u2, speed):
            chk.save_function(fn)
    summary = {
        "tag": args.tag, "name": args.name, "model": args.model,
        "friction": args.friction, "apres": args.apres,
        "rc_c0": RC_C0 if args.friction == "rc" else None,
        "schoof_u0": SCHOOF_U0 if args.friction == "schoof" else None,
        "gl_thresh": GL_THRESH if args.friction == "schoof" else None,
        "gl_width": GL_WIDTH,
        "C_anchor": ANCHOR[args.friction],
        "mesh": MESH_NAME, "prior_file": PRIOR_FILE.name, "tau_d_smooth": TAU_D_SMOOTH,
        # delta/gamma/ELL are the shared values and are null when the two priors
        # differ; the per-control values are always recorded.
        "delta": args.delta_A if args.tied else None,
        "gamma": args.gamma_A if args.tied else None,
        "ELL": args.ell_A if args.tied else None,
        "delta_A": args.delta_A, "gamma_A": args.gamma_A, "ELL_A": args.ell_A,
        "delta_C": args.delta_C, "gamma_C": args.gamma_C, "ELL_C": args.ell_C,
        "prior_form": args.prior_form,
        "regularization": ("0.5 theta^T A M^-1 A theta (fenics_ice Laplacian prior, "
                           "covariance A^-1 M A^-1)" if args.prior_form == "bilaplacian"
                           else "0.5 theta^T A theta (covariance A^-1)"),
        "grtol": args.grtol, "ftol": args.ftol, "ftol_patience": args.ftol_patience,
        "min_iter": args.min_iter,
        "maxcor": args.maxcor, "converged_by": state["why"],
        "prior_mean": (f"physical ({ANCHOR[args.friction]} balance anchor, A_prior thermo)"
                       if args.A_const_T is None else
                       f"{ANCHOR[args.friction]} balance anchor, A constant "
                       f"rate_factor({args.A_const_T:g} K)"),
        "A_const_T": args.A_const_T,
        "smoothness_only": args.smoothness_only,
        "optimizer": "L-BFGS-B in prior-whitened coordinates",
        "status": status, "iterations": state["nit"], "evaluations": len(hist["J"]),
        "grad_rel_reduction": gn / state["g0"], "J": f, "J_vel": J_vel, "J_apr": J_apr, "R": R,
        "R_A": R_A, "R_C": R_C,
        "chi2_vel_per_obs": 2 * J_vel / N_vel, "N_vel": N_vel,
        "chi2_apr_per_obs": (2 * J_apr / N_apr) if N_apr else None, "N_apr": N_apr,
        "theta_A_range": [float(θ_A2.dat.data.min()), float(θ_A2.dat.data.max())],
        "theta_C_range": [float(θ_C2.dat.data.min()), float(θ_C2.dat.data.max())],
        "wall_hours": (time.time() - t_start) / 3600,
    }
    json.dump(summary, open(DATA / "results" / f"map_{args.name}.json", "w"), indent=2)
    np.savez(DATA / "results" / f"map_{args.name}_history.npz",
             f_crit_per_iter=np.array(fcrit_hist),
             **{k: np.array(v) for k, v in hist.items()})
    for k in ("status", "iterations", "grad_rel_reduction", "chi2_vel_per_obs", "chi2_apr_per_obs",
              "theta_A_range", "theta_C_range", "wall_hours"):
        print(f"  {k}: {summary[k]}", flush=True)
    print(f"Saved mesh/map_{args.name}.h5 and results/map_{args.name}.json", flush=True)


if __name__ == "__main__":
    main()
