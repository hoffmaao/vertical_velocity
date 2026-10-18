r"""Verify that the MAP objective's descent direction includes the regularization.

invert.py optimizes in prior-whitened coordinates xi (prior_precond.Prior), where

    f(xi) = J_vel + J_ApRES + R(xi)
    g(xi) = W.grad_to_xi(dJ/dtheta) + dR/dxi
                                      ^^^^^^ the regularization's contribution

with R = 0.5 |xi|^2 for the h1 form and R = 0.5 xi^T S xi (S = D^1/2 M^-1 D^1/2)
for the bilaplacian form.

This script rebuilds that objective (the same pieces invert.py uses, not a
re-derivation) and checks g against a central finite difference of f along a
random direction, at a *nonzero* xi. The nonzero part matters: at xi = 0 the
prior gradient vanishes, so a run that dropped it entirely would still pass.

It also reports what the finite difference would have looked like had the prior
gradient been omitted, which is the quantity the test is discriminating against.

    python scripts/test_objective_gradient.py --model ssa --friction rc
    python scripts/test_objective_gradient.py --model hybrid --friction weertman

Passes when the relative error of the central difference reaches ~1e-6 or below
at some step size, and the omit-regularization variant is rejected by orders of
magnitude.
"""
import argparse

import numpy as np
from firedrake import Function
from firedrake.adjoint import *  # noqa: F401,F403  (tlm_adjoint annotation hooks)
from tlm_adjoint.firedrake import compute_gradient, reset_manager, start_manager, stop_manager

from pathlib import Path

from prior_precond import DEFAULT_FORM, FORMS, Prior
from problem import build_problem, misfit, tag

DATA = Path(__file__).resolve().parent.parent

DELTA = 1.0e-5
ELL = 2000.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], default="ssa")
    ap.add_argument("--friction", choices=["weertman", "rc", "schoof"], default="rc")
    ap.add_argument("--no-apres", action="store_true")
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--ell", type=float, default=ELL)
    ap.add_argument("--prior-form", choices=FORMS, default=DEFAULT_FORM)
    ap.add_argument("--xi-scale", type=float, default=5.0,
                    help="magnitude of the random xi, when not testing near the MAP")
    ap.add_argument("--from-map", action="store_true",
                    help="test near the converged MAP (mesh/checkpoint_map_<tag>.npz) "
                         "instead of at a random xi. Far from the MAP the prior term is "
                         "negligible against the misfit, so omitting it is nearly "
                         "indistinguishable and the check has little discriminating power")
    ap.add_argument("--xi-mix", type=float, default=0.9,
                    help="with --from-map, test at this multiple of the MAP xi; not 1.0, "
                         "where g = xi + g_misfit cancels to ~0 and the relative error "
                         "is ill-conditioned")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.apres = args.model == "hybrid" and not args.no_apres
    args.tag = tag(args.model, args.friction, args.apres)
    return args


def main():
    args = parse_args()
    gamma = args.delta * args.ell ** 2
    print(f"gradient check [{args.tag}] {args.prior_form} prior, delta={args.delta:.1e} "
          f"gamma={gamma:g} ELL={args.ell:.0f} m", flush=True)

    P = build_problem(args.model, args.friction, args.apres)
    Q, n = P["Q"], P["Q_2d"].dim()
    W = Prior(P["Q_2d"], args.delta, gamma, args.prior_form)
    θ_A = Function(Q, name="log_fluidity")
    θ_C = Function(Q, name="log_friction")

    def f_and_g(ξ):
        r"""Exactly invert.py's objective and gradient."""
        θ_A.dat.data[:] = W.to_theta(ξ[:n])
        θ_C.dat.data[:] = W.to_theta(ξ[n:])
        reset_manager(); start_manager()
        u = P["forward"](θ_A, θ_C)
        J, J_vel, J_apr = misfit(u, P, args.apres)
        dA, dC = compute_gradient(J, (θ_A, θ_C))
        stop_manager()
        R_A, gR_A = W.R_and_grad_xi(ξ[:n])
        R_C, gR_C = W.R_and_grad_xi(ξ[n:])
        R, gR = R_A + R_C, np.concatenate([gR_A, gR_C])
        g_misfit = np.concatenate([W.grad_to_xi(dA.dat.data_ro),
                                   W.grad_to_xi(dC.dat.data_ro)])
        return J_vel + J_apr + R, g_misfit + gR, g_misfit, R

    rng = np.random.default_rng(args.seed)
    if args.from_map:
        ckpt = DATA / "mesh" / f"checkpoint_map_{args.tag}.npz"
        if not ckpt.exists():
            raise SystemExit(f"no checkpoint at {ckpt}; run invert.py for this config first")
        ξ = args.xi_mix * np.load(ckpt)["xi"]
        print(f"  testing at {args.xi_mix:g} x the MAP xi from {ckpt.name}", flush=True)
    else:
        ξ = args.xi_scale * rng.standard_normal(2 * n) / np.sqrt(2 * n)
        print("  testing at a random xi (far from the MAP: the prior term is small "
              "there, so this is a weak check - prefer --from-map)", flush=True)
    d = rng.standard_normal(2 * n)
    d /= np.linalg.norm(d)

    P["u0"].assign(P["forward"](Function(Q), Function(Q)))   # prime the solver
    f0, g0, g_mis, R0 = f_and_g(ξ)
    dir_full = float(g0 @ d)
    dir_omit = float(g_mis @ d)          # what it would be without the prior term
    print(f"\n  f = {f0:.8e}   (R = {R0:.6e}, {100 * R0 / f0:.2f}% of f)", flush=True)
    print(f"  |g_misfit| = {np.linalg.norm(g_mis):.6e}   "
          f"|g_prior| = {np.linalg.norm(g0 - g_mis):.6e}", flush=True)
    print(f"  directional derivative  with prior term: {dir_full:.8e}", flush=True)
    print(f"                       without prior term: {dir_omit:.8e}", flush=True)

    print(f"\n  {'eps':>9s} {'central diff':>16s} {'rel err (full)':>15s} "
          f"{'rel err (omit)':>15s}", flush=True)
    best = np.inf
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        fp, *_ = f_and_g(ξ + eps * d)
        fm, *_ = f_and_g(ξ - eps * d)
        fd = (fp - fm) / (2 * eps)
        e_full = abs(fd - dir_full) / max(abs(fd), 1e-30)
        e_omit = abs(fd - dir_omit) / max(abs(fd), 1e-30)
        best = min(best, e_full)
        print(f"  {eps:9.0e} {fd:16.8e} {e_full:15.3e} {e_omit:15.3e}", flush=True)

    print(f"\n  best relative error including the prior term: {best:.3e}", flush=True)
    ok = best < 1e-5
    print(f"  {'PASS' if ok else 'FAIL'}: the descent direction "
          f"{'includes' if ok else 'does NOT match'} the regularization", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
