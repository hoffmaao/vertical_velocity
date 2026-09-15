r"""MISMIP+ OSSE inversion: JOINTLY recover fluidity θ_A and friction θ_C.

The 2×2 OSSE cell is (--model {ssa,hybrid}) × (--obs {surf,vert}):
  • ssa  : IceStream (depth-constant) — w is the plug-flow kinematic w = -ζ H ∇·u.
  • hybrid: HybridModel — w carries the higher-order (shear) depth structure.
  • surf : misfit = surface velocity only.
  • vert : misfit = surface velocity + ApRES-style vertical-velocity w depth profiles.

WHY JOINT: surface velocity alone cannot separate fast-because-soft-ice (θ_A) from
fast-because-slippery-bed (θ_C) — the deformation/sliding tradeoff. Vertical velocity
sees the deformation directly, so its value is in BREAKING that tradeoff. Fixing
friction to truth would hide exactly that, so we invert θ_A AND θ_C together.

Geometry (h,s) is fixed to truth (an OSSE assumes the DEM is known). Controls:
  A = A0·exp(θ_A)  (A0=20),  C = C0·exp(θ_C)  (C0=0.01, Weertman).
Prior:  R(θ) = 0.5·(δ ∫θ² + γ ∫|∇θ|²) dx per field,  γ = δ·ELL².
Writes results/mismip_inv_{model}_{obs}_d{Δ}km.{h5,npz} (θ_A,θ_C recovered + truth).

Run:  python mismip_invert.py --model hybrid --obs vert --spacing 20
"""
import argparse
import numpy as np
import firedrake
import icepack
import icepack.models.friction
import icepack.utilities
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, VertexOnlyMesh, interpolate, project,
    inner, grad, max_value, exp, ln, dx, TestFunction, assemble,
)
from tlm_adjoint.firedrake import (
    reset_manager, start_manager, stop_manager, compute_gradient, Functional,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)
from scipy.optimize import minimize as scipy_minimize
from pathlib import Path
from mismip_reference import true_fluidity, true_friction  # analytic truth (parameterless at defaults)

DATA = Path(__file__).resolve().parent.parent
A0 = Constant(20.0)
C0 = Constant(0.01)
VDEGREE = 4
ZSURF = 0.999
DELTA = 1.0e-5
BOUND = 6.0
SOLVER = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 150, "snes_rtol": 1e-6,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


def friction_weertman(**kw):
    u, h, s, C = kw["velocity"], kw["thickness"], kw["surface"], kw["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def vom_pair(mesh, pts):
    """VOM on `mesh` at `pts` + a helper mapping input-order arrays into VOM order."""
    vom = VertexOnlyMesh(mesh, pts, missing_points_behaviour="warn")
    D = FunctionSpace(vom, "DG", 0)
    Din = FunctionSpace(vom.input_ordering, "DG", 0)

    def to_vom(vals):
        fin = Function(Din)
        fin.dat.data[:len(vals)] = vals
        return Function(D).interpolate(fin)
    return vom, D, to_vom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    ap.add_argument("--obs", choices=["surf", "vert"], required=True)
    ap.add_argument("--spacing", type=float, default=20.0)
    ap.add_argument("--ell-km", type=float, default=20.0)
    ap.add_argument("--maxiter", type=int, default=150)
    args = ap.parse_args()
    GAMMA = DELTA * (args.ell_km * 1e3) ** 2
    use_w = args.obs == "vert"
    tag = f"{args.model}_{args.obs}_d{args.spacing:g}km"
    print("=" * 66, flush=True)
    print(f"MISMIP OSSE joint inversion: {tag}  (δ={DELTA:.0e}, ELL={args.ell_km}km)", flush=True)
    print("=" * 66, flush=True)

    obs = np.load(str(DATA / "results" / f"mismip_obs_d{args.spacing:g}km.npz"))
    vel_pts, vel_obs, σv = obs["vel_pts"], obs["vel_obs"], float(obs["sigma_vel"])
    apres_xyz, apres_zeta = obs["apres_xyz"], obs["apres_zeta"]
    w_obs, σw = obs["w_obs"], float(obs["sigma_w"])
    print(f"obs: {len(vel_pts)} surface-vel pts; {len(w_obs)} w pts "
          f"({'USED' if use_w else 'ignored'}); σv={σv}, σw={σw:.0e}", flush=True)

    DELTA_c, GAMMA_c = Constant(DELTA), Constant(GAMMA)

    # ── truth geometry (fixed) ──
    with CheckpointFile(str(DATA / "mesh" / "mismip_truth.h5"), "r") as c:
        m2 = c.load_mesh()
        h2 = c.load_function(m2, "thickness")
        s2 = c.load_function(m2, "surface")
        uwarm2 = c.load_function(m2, "u_surface")
    Q2 = FunctionSpace(m2, "CG", 1)
    V2 = VectorFunctionSpace(m2, "CG", 1)

    if args.model == "ssa":
        Qc, mesh_c = Q2, m2
        Q2c = Q2
        θ_A = Function(Qc, name="log_fluidity")
        θ_C = Function(Qc, name="log_friction")
        u_prev = Function(V2).assign(uwarm2)
        model = icepack.models.IceStream(friction=friction_weertman)
        solver = icepack.solvers.FlowSolver(
            model, dirichlet_ids=[1], ice_front_ids=[2], side_wall_ids=[3, 4],
            diagnostic_solver_type="petsc", diagnostic_solver_parameters=SOLVER)
        H_c = h2
        vom_v, Dv, to_v = vom_pair(m2, vel_pts)
        uo_x, uo_y = to_v(vel_obs[:, 0]), to_v(vel_obs[:, 1])
        if use_w:
            vom_e, De, to_e = vom_pair(m2, apres_xyz[:, :2])
            wo, zeta_e = to_e(w_obs), to_e(apres_zeta)
            H_e = Function(De).interpolate(h2)

        def forward():
            A = interpolate(A0 * exp(θ_A), Qc)
            C = interpolate(C0 * exp(θ_C), Qc)
            return solver.diagnostic_solve(velocity=u_prev, thickness=h2, surface=s2,
                                           fluidity=A, friction=C)

        def misfit(u):
            ux = Function(Dv).interpolate(u[0]); uy = Function(Dv).interpolate(u[1])
            forms = [0.5 * (((ux - uo_x) / σv) ** 2 + ((uy - uo_y) / σv) ** 2) * dx]   # on vom_v
            if use_w:
                divu = project(u[0].dx(0) + u[1].dx(1), Q2)         # plug-flow: w = -ζ H ∇·u
                dv = Function(De).interpolate(divu)
                w_model = -zeta_e * H_e * dv
                forms.append(0.5 * ((w_model - wo) / σw) ** 2 * dx)  # on vom_e (separate mesh)
            return forms

    else:  # hybrid
        with CheckpointFile(str(DATA / "mesh" / "mismip_truth_3d.h5"), "r") as c:
            m3 = c.load_mesh("firedrake_default_extruded")
            u3warm = c.load_function(m3, "velocity_3d")
            h3 = c.load_function(m3, "thickness_3d")
            s3 = c.load_function(m3, "surface_3d")
        Qc = FunctionSpace(m3, "CG", 1, vfamily="R", vdegree=0)        # depth-constant control
        VV = VectorFunctionSpace(m3, "CG", 1, vfamily="GLL", vdegree=VDEGREE, dim=2)
        Qw = FunctionSpace(m3, "CG", 1, vfamily="GLL", vdegree=VDEGREE)
        Q2c = FunctionSpace(m3._base_mesh, "CG", 1)
        θ_A = Function(Qc, name="log_fluidity")
        θ_C = Function(Qc, name="log_friction")
        u_prev = Function(VV).assign(u3warm)
        model = icepack.models.HybridModel(friction=friction_weertman)
        solver = icepack.solvers.FlowSolver(
            model, dirichlet_ids=[1], ice_front_ids=[2], side_wall_ids=[3, 4],
            diagnostic_solver_type="petsc", diagnostic_solver_parameters=SOLVER)
        xyz_surf = np.column_stack([vel_pts, np.full(len(vel_pts), ZSURF)])
        vom_v, Dv, to_v = vom_pair(m3, xyz_surf)
        uo_x, uo_y = to_v(vel_obs[:, 0]), to_v(vel_obs[:, 1])
        if use_w:
            vom_e, De, to_e = vom_pair(m3, apres_xyz)
            wo = to_e(w_obs)

        def forward():
            A = interpolate(A0 * exp(θ_A), Qc)
            C = interpolate(C0 * exp(θ_C), Qc)
            return solver.diagnostic_solve(velocity=u_prev, thickness=h3, surface=s3,
                                           fluidity=A, friction=C)

        def misfit(u):
            ux = Function(Dv).interpolate(u[0]); uy = Function(Dv).interpolate(u[1])
            forms = [0.5 * (((ux - uo_x) / σv) ** 2 + ((uy - uo_y) / σv) ** 2) * dx]   # on vom_v
            if use_w:
                w = icepack.utilities.vertical_velocity(velocity=u, thickness=h3,
                                                        basal_mass_balance=Constant(0.0))
                wf = project(w, Qw)
                wv = Function(De).interpolate(wf)
                forms.append(0.5 * ((wv - wo) / σw) ** 2 * dx)       # on vom_e (separate mesh)
            return forms

    # 2D-footprint mirrors for prior + diagnostics (dof-aligned with the control)
    θ2A, θ2C = Function(Q2c), Function(Q2c)
    v_testc = TestFunction(Q2c)
    θA_true = Function(Q2c).interpolate(true_fluidity(Q2c))
    # near the calving front C_true→0 (θ_C→−∞, unconstrained); floor + clip to the inversion bounds
    θC_true = Function(Q2c).interpolate(ln(max_value(true_friction(Q2c), Constant(1e-6)) / C0))
    θC_true.dat.data[:] = np.clip(θC_true.dat.data_ro, -BOUND, BOUND)

    u0 = forward(); u_prev.assign(u0)
    n = θ_A.dat.data.shape[0]
    print(f"primed forward OK (control DOFs={n} per field)", flush=True)

    def reg_and_grad(θ2):
        reg = float(assemble(0.5 * (DELTA_c * inner(θ2, θ2) + GAMMA_c * inner(grad(θ2), grad(θ2))) * dx))
        gp = assemble((DELTA_c * inner(θ2, v_testc) + GAMMA_c * inner(grad(θ2), grad(v_testc))) * dx)
        return reg, gp.dat.data_ro

    it = [0]
    state = {"J": None, "g": None}

    def obj_grad(x):
        θ_A.dat.data[:] = x[:n]; θ_C.dat.data[:] = x[n:]
        θ2A.dat.data[:] = x[:n]; θ2C.dat.data[:] = x[n:]
        reset_manager(); start_manager()
        try:
            u = forward()
        except firedrake.exceptions.ConvergenceError:
            stop_manager(); it[0] += 1
            print(f"  iter {it[0]:3d}: FWD FAIL — penalty", flush=True)
            return 10.0 * (state["J"] or 1.0), (state["g"].copy() if state["g"] is not None else np.zeros(2 * n))
        J = Functional(name="J_obs")
        for f in misfit(u):           # accumulate per-mesh terms (velocity VOM + w VOM differ)
            J.addto(f)
        Jval = float(J)
        try:
            dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
        except Exception as e:
            stop_manager(); it[0] += 1
            print(f"  iter {it[0]:3d}: ADJ FAIL {type(e).__name__} — penalty", flush=True)
            return 10.0 * (state["J"] or 1.0), (state["g"].copy() if state["g"] is not None else np.zeros(2 * n))
        stop_manager()
        rA, gA = reg_and_grad(θ2A); rC, gC = reg_and_grad(θ2C)
        gtot = np.concatenate([dJ_A.dat.data_ro + gA, dJ_C.dat.data_ro + gC])
        u_prev.assign(u)
        it[0] += 1
        print(f"  iter {it[0]:3d}: Jobs={Jval:.3e} reg={rA+rC:.3e} |g|={np.sqrt((gtot**2).sum()):.3e}", flush=True)
        state.update(J=Jval + rA + rC, g=gtot.copy())
        return Jval + rA + rC, gtot

    x0 = np.zeros(2 * n)
    bounds = [(-BOUND, BOUND)] * (2 * n)
    print(f"\nL-BFGS-B ({2*n} DOFs, max {args.maxiter})...", flush=True)
    res = scipy_minimize(obj_grad, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                         options={"maxiter": args.maxiter, "ftol": 1e-12, "gtol": 1e-8})
    print(f"Result: {res.message}", flush=True)

    θA_rec = Function(Q2c, name="theta_A_rec"); θA_rec.dat.data[:] = res.x[:n]
    θC_rec = Function(Q2c, name="theta_C_rec"); θC_rec.dat.data[:] = res.x[n:]

    def report(rec, tru, lbl):
        err = rec.dat.data_ro - tru.dat.data_ro
        rmse = float(np.sqrt((err ** 2).mean()))
        trms = float(np.sqrt((tru.dat.data_ro ** 2).mean()))
        print(f"  {lbl}: RMSE={rmse:.4f}  truth-rms={trms:.4f}  explained={100*(1-rmse/max(trms,1e-9)):.0f}%", flush=True)
        return rmse, trms

    print("recovery vs analytic truth:", flush=True)
    rmse_A, trms_A = report(θA_rec, θA_true, "θ_A (fluidity)")
    rmse_C, trms_C = report(θC_rec, θC_true, "θ_C (friction) ")

    out = DATA / "results" / f"mismip_inv_{tag}.h5"
    with CheckpointFile(str(out), "w") as c:
        c.save_mesh(Q2c.mesh())
        c.save_function(θA_rec, name="theta_A_rec")
        c.save_function(θC_rec, name="theta_C_rec")
        c.save_function(Function(Q2c, name="theta_A_true").assign(θA_true), name="theta_A_true")
        c.save_function(Function(Q2c, name="theta_C_true").assign(θC_true), name="theta_C_true")
    np.savez(str(DATA / "results" / f"mismip_inv_{tag}.npz"),
             rmse_A=rmse_A, rmse_C=rmse_C, truth_rms_A=trms_A, truth_rms_C=trms_C,
             final_J=res.fun, spacing_km=args.spacing, model=args.model, obs=args.obs)
    print(f"Saved {out}\nDone!", flush=True)


if __name__ == "__main__":
    main()
