r"""The data-misfit Hessian and prior precision at a MAP, shared by eigendec.py and uq_vaf_pcg.py.

``setup`` builds the model through problem.build_problem (so the forward
operator is the one the inversion saw), loads the MAP controls from
mesh/map_<name>.h5 and the prior they were inverted with from
results/map_<name>.json, and returns

    Hv(x)   the misfit Hessian applied to x = (theta_A, theta_C) dofs, as numpy
    Rinv(x) the prior precision Gamma^-1 applied to x, block-diagonal over the
            two controls, in the form the MAP was inverted with (prior_precond.py:
            bilaplacian A M^-1 A, or the older h1 form A), A_X = delta_X M + gamma_X K
    Rcov(x) the prior covariance Gamma applied to x

in the fenics_ice / Recinos et al. (2023) convention: posterior covariance
(H + Gamma^-1)^{-1}.

``misfit_scale="chi2"`` inflates the observation errors of each data type by
sqrt(chi2 / N) of the MAP's fit (per velocity component, i.e. half the
chi2_vel_per_obs that invert.py records, which sums x and y), never below 1:
a Birge-type model-discrepancy allowance so that data the model cannot fit to
their stated errors are not counted as more informative than they are. The
MAP itself is unchanged; only H shrinks (by exactly that factor per block).

Hessian-vector products
-----------------------
gn (default): exact Gauss-Newton products from sparse_hessian.py. The VOM
    interpolations are kept off the pyadjoint tape (they have no TLM
    annotation) and the sparse weighting is applied as linear algebra, see the
    "GN fix" section of CLAUDE.md. Symmetric positive semi-definite.
fd: finite differences of the tlm_adjoint misfit gradient, kept for validation.
    Noisy for small eigenvalues.
"""
import json
import sys
from pathlib import Path

import numpy as np
from firedrake import CheckpointFile, Function
from firedrake.adjoint import *  # noqa: F401,F403  (annotation hooks for the tape)
from firedrake.exceptions import ConvergenceError
from tlm_adjoint.firedrake import compute_gradient, reset_manager, start_manager, stop_manager

from observations import load_apres, load_velocity
from physical_prior import MESH_NAME, TAU_D_SMOOTH
from prior_precond import Prior
from problem import build_problem, misfit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sparse_hessian as sh  # noqa: E402

DATA = Path(__file__).resolve().parent.parent
EPS_FD = 1e-5           # finite-difference step of the fd method


def load_map(name, Q):
    r"""The MAP controls from mesh/map_<name>.h5, copied onto the control space ``Q``."""
    with CheckpointFile(str(DATA / "mesh" / f"map_{name}.h5"), "r") as chk:
        m = chk.load_mesh()
        θA_2d = chk.load_function(m, "log_fluidity")
        θC_2d = chk.load_function(m, "log_friction")
    θ_A = Function(Q, name="log_fluidity")
    θ_C = Function(Q, name="log_friction")
    θ_A.dat.data[:] = θA_2d.dat.data_ro
    θ_C.dat.data[:] = θC_2d.dat.data_ro
    return θ_A, θ_C


def load_chi2(name):
    r"""(chi2 per velocity component, chi2 per ApRES sample) of the MAP's fit; None if absent."""
    with open(DATA / "results" / f"map_{name}.json") as f:
        d = json.load(f)
    return d["chi2_vel_per_obs"] / 2.0, d.get("chi2_apr_per_obs")


def load_prior_form(name):
    r"""The prior form (prior_precond.FORMS) the MAP was inverted with; h1 for MAPs that predate the option."""
    with open(DATA / "results" / f"map_{name}.json") as f:
        return json.load(f).get("prior_form", "h1")


def load_prior(name, scale=1.0):
    r"""(delta_A, gamma_A, delta_C, gamma_C) the MAP was inverted with, times ``scale``."""
    with open(DATA / "results" / f"map_{name}.json") as f:
        d = json.load(f)
    if d["mesh"] != MESH_NAME or float(d["tau_d_smooth"]) != TAU_D_SMOOTH:
        raise SystemExit(f"map_{name}.json was inverted on mesh {d['mesh']} with tau_d "
                         f"smoothing {d['tau_d_smooth']:g} m, but VV_MESH={MESH_NAME} and "
                         f"VV_TAU_SMOOTH={TAU_D_SMOOTH:g}: the prior anchors would differ")
    if "delta_A" not in d:
        # Inverted before the priors were split (invert.py < 2026-09-25): one
        # shared (delta, gamma) for both controls.
        return tuple(scale * float(d[k]) for k in ("delta", "gamma", "delta", "gamma"))
    return tuple(scale * float(d[k]) for k in ("delta_A", "gamma_A", "delta_C", "gamma_C"))


def setup(model, friction, apres, map_name, method="gn", prior_scale=1.0, misfit_scale=None):
    r"""Everything needed to apply H and A at the MAP ``map_name``; see the module docstring.

    Returns a dict with P (build_problem's), n (dofs per control), theta_A,
    theta_C, Hv, Rinv, Rcov, prior_form, prior (delta_A, gamma_A, delta_C, gamma_C),
    N_vel, N_apr, sigma_scale (the factors the velocity and ApRES errors were
    multiplied by).
    """
    if misfit_scale is None:
        s_vel = s_apr = 1.0
    elif misfit_scale == "chi2":
        c_vel, c_apr = load_chi2(map_name)
        s_vel = max(1.0, np.sqrt(c_vel))
        s_apr = max(1.0, np.sqrt(c_apr)) if (apres and c_apr) else 1.0
        print(f"Observation errors inflated by sqrt(chi2/N): velocity x{s_vel:.3f} "
              f"(chi2 per component {c_vel:.2f})"
              + (f", ApRES x{s_apr:.3f} (chi2 {c_apr:.2f})" if apres else ""), flush=True)
    else:
        raise ValueError(f"unknown misfit_scale {misfit_scale!r}")
    δ_A, γ_A, δ_C, γ_C = load_prior(map_name, prior_scale)
    form = load_prior_form(map_name)
    P = build_problem(model, friction, apres)
    Q, Q_2d, mesh_2d = P["Q"], P["Q_2d"], P["mesh_2d"]
    n = Q.dim()
    θ_A, θ_C = load_map(map_name, Q)
    print(f"MAP controls: theta_A in [{θ_A.dat.data_ro.min():+.2f}, {θ_A.dat.data_ro.max():+.2f}], "
          f"theta_C in [{θ_C.dat.data_ro.min():+.2f}, {θ_C.dat.data_ro.max():+.2f}]; "
          f"{n} dofs per control", flush=True)

    # Prime the velocity at the MAP, as invert.py does, so the taped solve
    # starts from the converged state.
    try:
        P["u0"].assign(P["forward"](θ_A, θ_C))
    except ConvergenceError:
        print("  MAP solve failed from the observed velocity; priming with Weertman first",
              flush=True)
        P["u0"].assign(P["forward"](θ_A, θ_C, slv=P["weertman_solver"]))
        P["u0"].assign(P["forward"](θ_A, θ_C))
    print("Primed the velocity at the MAP", flush=True)

    # Observation points and errors, in input order; sparse_hessian reorders
    # them to the VOM-internal order itself.
    xs, ys, _, _, σx, σy = load_velocity(mesh_2d)
    σx, σy = s_vel * σx, s_vel * σy
    vel_pts = (np.column_stack([xs, ys]) if model == "ssa"
               else np.column_stack([xs, ys, np.ones_like(xs)]))
    N_vel = len(P["obs"]["u"].dat.data)
    N_apr = len(P["obs"]["eps"].dat.data) if apres else 0
    print(f"Observations: {N_vel} velocity points"
          + (f", {N_apr} ApRES strain rates" if apres else ""), flush=True)

    if method == "gn":
        taped = sh.TapedForward(lambda a, c: P["forward"](a, c), θ_A, θ_C)
        print(f"Taped theta -> u at the MAP ({len(taped.tape.get_blocks())} blocks)", flush=True)
        parts = [sh.make_Hv_gn_velocity(taped, vel_pts, σx, σy)]
        if apres:
            pts, _, σe = load_apres(mesh_2d)
            parts.append(sh.make_Hv_gn_strainrate(taped, pts, s_apr * σe, h=P["h"], s=P["s"]))
        Hv_gn = sh.combine_Hv(*parts)
        vA, vC = Function(Q), Function(Q)

        def Hv(x):
            vA.dat.data[:] = x[:n]
            vC.dat.data[:] = x[n:]
            hA, hC = Hv_gn(vA, vC)
            return np.concatenate([hA.dat.data_ro, hC.dat.data_ro])
    elif method == "fd":
        if misfit_scale is not None:
            raise ValueError("misfit_scale needs the gn method (the fd gradient uses problem.misfit)")
        θA0, θC0 = θ_A.dat.data_ro.copy(), θ_C.dat.data_ro.copy()

        def grad_misfit(a, c):
            θ_A.dat.data[:] = a
            θ_C.dat.data[:] = c
            reset_manager(); start_manager()
            J, _, _ = misfit(P["forward"](θ_A, θ_C), P, apres)
            dA, dC = compute_gradient(J, (θ_A, θ_C))
            stop_manager()
            return np.concatenate([dA.dat.data_ro, dC.dat.data_ro])

        g0 = grad_misfit(θA0, θC0)
        print(f"|grad J_misfit| at the MAP = {np.linalg.norm(g0):.4e}", flush=True)

        def Hv(x):
            return (grad_misfit(θA0 + EPS_FD * x[:n], θC0 + EPS_FD * x[n:]) - g0) / EPS_FD
    else:
        raise ValueError(f"unknown Hessian method {method!r}")

    # Prior precision and covariance, block-diagonal over (theta_A, theta_C), in
    # the form the MAP was inverted with. The controls share the 2D CG1 dof
    # ordering (build_problem asserts it), so these are the operators invert.py
    # whitened with, entry for entry.
    pr_A = Prior(Q_2d, δ_A, γ_A, form)
    pr_C = pr_A if (δ_A, γ_A) == (δ_C, γ_C) else Prior(Q_2d, δ_C, γ_C, form)
    print(f"Prior form: {form}", flush=True)

    def Rinv(x):
        return np.concatenate([pr_A.apply_precision(x[:n]), pr_C.apply_precision(x[n:])])

    def Rcov(x):
        return np.concatenate([pr_A.apply_covariance(x[:n]), pr_C.apply_covariance(x[n:])])

    return dict(P=P, n=n, theta_A=θ_A, theta_C=θ_C, Hv=Hv, Rinv=Rinv, Rcov=Rcov,
                prior_form=form, priors=(pr_A, pr_C), prior=(δ_A, γ_A, δ_C, γ_C),
                N_vel=N_vel, N_apr=N_apr, sigma_scale=(s_vel, s_apr))
