r"""Coarse-mesh hybrid inversion with surface velocity + ApRES top-of-column.

Two cost terms:
  J_vel  : 3D surface velocity misfit (depth-averaged sense via 3D ∫)
  J_apres: ε_zz misfit at ApRES sites, top of column only

Model expressivity:
  vdeg=1 → u(x,y,ζ) linear in ζ → ε_zz(ζ) linear in ζ → matches
            a degree-1 fit to ApRES top-of-column
  vdeg=2 → u quadratic in ζ → ε_zz quadratic in ζ → matches degree-2 fit

ε_zz is computed from incompressibility: ε_zz = -tr(ε_h)
  where ε_h = icepack.models.hybrid.horizontal_strain_rate(u, h, s).

Run vdeg=1 first; rerun with VDEGREE=2 to get the degree-2 fit.
"""
import numpy as np
import h5py
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, ExtrudedMesh, VertexOnlyMesh,
    sqrt, inner, grad, max_value, exp, dx, conditional, TestFunction, assemble,
)
from firedrake.adjoint import *  # noqa
from tlm_adjoint.firedrake import (
    reset_manager, start_manager, stop_manager,
    compute_gradient, Functional,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)
from scipy.optimize import minimize as scipy_minimize
from pathlib import Path
import sys

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites_coarse.h5"
APRES_FILE = DATA_DIR / "data" / "apres_legendre_fits.h5"

VDEGREE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
HDEGREE = 1
L_REG = 5e3
MAX_ITER = 150
SIGMA_VEL = 10.0
SIGMA_EPS = 1e-3        # 1/yr — ApRES typical
APRES_WEIGHT = 1.0      # multiplier on J_apres
DEPTH_MIN = 100.0       # m — top-of-column window
DEPTH_MAX = 500.0       # m

OUTPUT_FILE = DATA_DIR / "mesh" / f"inversion_coarse_apres_vd{VDEGREE}.h5"

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 40, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

WARM_START = False  # vd=1 warm start lands on unstable point for vd=2


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def main():
    print("=" * 60, flush=True)
    print(f"Coarse hybrid inversion + ApRES (vdeg={VDEGREE})", flush=True)
    print(f"  ApRES depth window: {DEPTH_MIN:.0f}-{DEPTH_MAX:.0f} m", flush=True)
    print("=" * 60, flush=True)

    # ── Load coarse 2D fields ──────────────────────────────────────
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")
    print(f"Coarse mesh: {mesh_2d.num_vertices()} verts, "
          f"{mesh_2d.num_cells()} cells", flush=True)

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)

    # ── Extrude ────────────────────────────────────────────────────
    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", HDEGREE,
                             vfamily="GL", vdegree=VDEGREE, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1,
                                  vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    u_obs_3d = icepack.utilities.lift3d(u_obs_2d, V_lift)
    u0 = Function(V).project(u_obs_3d)

    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    A_0, C_0 = Constant(20.0), Constant(0.01)

    # Warm start from vd=1 if available (controls live on Q_lift = CG1×DG0,
    # so they are vdegree-independent; we can copy values directly).
    warm_file = DATA_DIR / "mesh" / "inversion_coarse_apres_vd1.h5"
    if WARM_START and VDEGREE > 1 and warm_file.exists():
        with CheckpointFile(str(warm_file), "r") as wchk:
            mw = wchk.load_mesh()
            θA_w = wchk.load_function(mw, "log_fluidity")
            θC_w = wchk.load_function(mw, "log_friction")
        θ_A.dat.data[:] = θA_w.dat.data_ro
        θ_C.dat.data[:] = θC_w.dat.data_ro
        print(f"Warm start from {warm_file.name}: "
              f"θ_A=[{θ_A.dat.data.min():.2f}, {θ_A.dat.data.max():.2f}], "
              f"θ_C=[{θ_C.dat.data.min():.2f}, {θ_C.dat.data.max():.2f}]",
              flush=True)

    ϕ_mask = Function(Q_lift).interpolate(
        1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h))
    grounded_mask = Function(Q_lift).interpolate(
        conditional(ϕ_mask > 0.01, Constant(1.0), Constant(0.0)))

    # ── Hybrid solver ──────────────────────────────────────────────
    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    # Prime velocity
    A_init = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_init = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    u_test = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                      fluidity=A_init, friction=C_init)
    u0.assign(u_test)
    print(f"Primed forward solve OK", flush=True)

    # ── Load ApRES top-of-column points ────────────────────────────
    with h5py.File(str(APRES_FILE), "r") as f:
        x_sites = f["summary/x"][...]
        y_sites = f["summary/y"][...]
        H_apres = f["summary/H_apres"][...]
        names = [n.decode() for n in f["summary/names"][...]]

        pts_xyz, eps_obs, eps_sig = [], [], []
        for name, xi, yi, Hi in zip(names, x_sites, y_sites, H_apres):
            depth = f[f"sites/{name}/depth"][...]
            eps = f[f"sites/{name}/eps_zz"][...]
            sig = f[f"sites/{name}/eps_sigma"][...]
            window = (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX) \
                     & np.isfinite(eps) & np.isfinite(sig) & (sig > 0)
            d_w = depth[window]
            if d_w.size == 0:
                continue
            # Sub-sample to ~5 points per site
            stride = max(1, d_w.size // 5)
            d_w = d_w[::stride]
            ζ_w = 1.0 - d_w / Hi
            pts_xyz.append(np.column_stack([
                np.full(d_w.size, xi),
                np.full(d_w.size, yi),
                ζ_w,
            ]))
            eps_obs.append(eps[window][::stride])
            eps_sig.append(sig[window][::stride])
    pts_xyz = np.vstack(pts_xyz)
    eps_obs = np.concatenate(eps_obs)
    eps_sig = np.concatenate(eps_sig)
    print(f"ApRES: {len(pts_xyz)} top-of-column points "
          f"({len(np.unique(pts_xyz[:, 0]))} sites)", flush=True)

    # Clip to mesh bbox
    coords = mesh_2d.coordinates.dat.data_ro
    x0, x1 = coords[:, 0].min(), coords[:, 0].max()
    y0, y1 = coords[:, 1].min(), coords[:, 1].max()
    inside = ((pts_xyz[:, 0] > x0) & (pts_xyz[:, 0] < x1)
              & (pts_xyz[:, 1] > y0) & (pts_xyz[:, 1] < y1))
    pts_xyz = pts_xyz[inside]
    eps_obs = eps_obs[inside]
    eps_sig = eps_sig[inside]
    print(f"  inside coarse mesh bbox: {len(pts_xyz)}", flush=True)

    vom = VertexOnlyMesh(mesh, pts_xyz, missing_points_behaviour="warn")
    Δ = FunctionSpace(vom, "DG", 0)
    Δ_in = FunctionSpace(vom.input_ordering, "DG", 0)
    print(f"  VOM: {vom.num_vertices()} vertices "
          f"(input ordering has {len(Function(Δ_in).dat.data)})", flush=True)

    f_in = Function(Δ_in)
    f_in.dat.data[:] = eps_obs[:len(f_in.dat.data)]
    eps_obs_f = Function(Δ).interpolate(f_in)
    f_in.dat.data[:] = eps_sig[:len(f_in.dat.data)]
    σ_eps_f = Function(Δ).interpolate(f_in)

    # ── Cost function ──────────────────────────────────────────────
    n_dof_local = θ_A.dat.data.shape[0]
    area = assemble(Constant(1.0) * dx(mesh))
    L_REG_c = Constant(L_REG)
    σ_u_c = Constant(SIGMA_VEL)
    apres_w = Constant(APRES_WEIGHT)
    iteration = [0]
    CHECKPOINT_EVERY = 10
    CHECKPOINT_FILE = DATA_DIR / "mesh" / f"checkpoint_coarse_apres_vd{VDEGREE}.h5"

    state = {"last_good_J": None, "last_good_g": None, "last_good_u": None}

    def obj_grad(x):
        θ_A.dat.data[:] = x[:n_dof_local]
        θ_C.dat.data[:] = x[n_dof_local:]

        reset_manager(); start_manager()

        A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
        C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
        try:
            u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                         fluidity=A, friction=C)
        except firedrake.exceptions.ConvergenceError as e:
            stop_manager()
            iteration[0] += 1
            print(f"  iter {iteration[0]:3d}: SOLVE FAILED ({e.args[0][:50]}...) "
                  f"— penalty + last-good gradient", flush=True)
            # Restore last good u for next try
            if state["last_good_u"] is not None:
                u0.assign(state["last_good_u"])
            # Return a large penalty (10× last good) and a gradient pointing
            # backwards toward the last good point so L-BFGS retreats.
            penalty = 10.0 * (state["last_good_J"] or 1.0)
            return penalty, state["last_good_g"].copy() if state["last_good_g"] is not None else np.zeros_like(x)

        # 3D velocity misfit
        J = Functional(name="J_total")
        J.assign(0.5 / area
                 * inner(u - u_obs_3d, u - u_obs_3d) / σ_u_c**2 * dx)

        # ApRES ε_zz misfit at top-of-column points
        eps_h = icepack.models.hybrid.horizontal_strain_rate(
            velocity=u, thickness=h, surface=s)
        eps_zz_ufl = -(eps_h[0, 0] + eps_h[1, 1])
        eps_model = Function(Δ).interpolate(eps_zz_ufl)

        N_pts = len(eps_obs_f.dat.data)
        J.addto(apres_w * 0.5 / Constant(float(max(N_pts, 1)))
                * ((eps_model - eps_obs_f) / σ_eps_f) ** 2 * dx)

        J_val = float(J)
        # Split parts for diagnostics (separate assembly, not annotated)
        J_vel = float(assemble(0.5 / area
                                * inner(u - u_obs_3d, u - u_obs_3d) / σ_u_c**2 * dx))
        J_apr = J_val - J_vel

        dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
        stop_manager()

        # Tikhonov regularization (not annotated)
        reg_A = float(assemble(0.5 / area
                               * L_REG_c**2 * inner(grad(θ_A), grad(θ_A)) * dx))
        reg_C = float(assemble(0.5 / area
                               * L_REG_c**2 * inner(grad(θ_C), grad(θ_C)) * dx))

        g_A = dJ_A.dat.data_ro.copy()
        g_C = dJ_C.dat.data_ro.copy()
        g_A += assemble(1.0 / area * L_REG_c**2
                        * inner(grad(θ_A), grad(TestFunction(Q_lift))) * dx).dat.data_ro
        g_C += assemble(1.0 / area * L_REG_c**2
                        * inner(grad(θ_C), grad(TestFunction(Q_lift))) * dx).dat.data_ro

        total = J_val + reg_A + reg_C
        iteration[0] += 1
        g_norm = float(np.sqrt((g_A**2).sum() + (g_C**2).sum()))
        print(f"  iter {iteration[0]:3d}: "
              f"Jv={J_vel:.3e} Ja={J_apr:.3e} reg={reg_A+reg_C:.3e} "
              f"|g|={g_norm:.3e}", flush=True)

        # Cache last good state
        state["last_good_u"] = u.copy(deepcopy=True)
        state["last_good_J"] = total
        full_g = np.concatenate([g_A, g_C])
        state["last_good_g"] = full_g.copy()
        u0.assign(u)

        # Periodic checkpoint
        if iteration[0] % CHECKPOINT_EVERY == 0:
            θ_A_chk = Function(Q_2d, name="log_fluidity")
            θ_C_chk = Function(Q_2d, name="log_friction")
            θ_A_chk.dat.data[:] = θ_A.dat.data_ro
            θ_C_chk.dat.data[:] = θ_C.dat.data_ro
            with CheckpointFile(str(CHECKPOINT_FILE), "w") as chk:
                chk.save_mesh(mesh_2d)
                chk.save_function(θ_A_chk, name="log_fluidity")
                chk.save_function(θ_C_chk, name="log_friction")
            np.savez(str(CHECKPOINT_FILE).replace(".h5", ".npz"),
                     x=x, iteration=iteration[0],
                     J_vel=J_vel, J_apr=J_apr, reg=reg_A+reg_C)
            print(f"    [checkpoint saved at iter {iteration[0]}]", flush=True)

        return total, full_g

    # Resume from checkpoint if available
    npz_file = str(CHECKPOINT_FILE).replace(".h5", ".npz")
    if Path(npz_file).exists():
        chk_data = np.load(npz_file)
        x0 = chk_data["x"]
        θ_A.dat.data[:] = x0[:n_dof_local]
        θ_C.dat.data[:] = x0[n_dof_local:]
        prev_iter = int(chk_data["iteration"])
        iteration[0] = prev_iter
        print(f"\nResuming from checkpoint at iter {prev_iter} "
              f"(Jv={chk_data['J_vel']:.3e}, Ja={chk_data['J_apr']:.3e})",
              flush=True)
        A_re = Function(Q_lift).interpolate(A_0 * exp(θ_A))
        C_re = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
        try:
            u_re = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                            fluidity=A_re, friction=C_re)
            u0.assign(u_re)
            print("  re-primed velocity OK", flush=True)
        except firedrake.exceptions.ConvergenceError:
            print("  re-prime failed, using warm-start velocity", flush=True)
    else:
        x0 = np.concatenate([θ_A.dat.data_ro.copy(), θ_C.dat.data_ro.copy()])

    print(f"\nL-BFGS-B ({2 * n_dof_local} DOFs, max {MAX_ITER} iter)...",
          flush=True)

    result = scipy_minimize(
        obj_grad, x0, method="L-BFGS-B", jac=True,
        options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": 1e-8, "disp": True},
    )
    print(f"\nResult: {result.message}", flush=True)

    θ_A.dat.data[:] = result.x[:n_dof_local]
    θ_C.dat.data[:] = result.x[n_dof_local:]
    θ_A_2d = Function(Q_2d, name="log_fluidity")
    θ_C_2d = Function(Q_2d, name="log_friction")
    θ_A_2d.dat.data[:] = θ_A.dat.data_ro
    θ_C_2d.dat.data[:] = θ_C.dat.data_ro
    print(f"θ_A: [{θ_A_2d.dat.data.min():.2f}, {θ_A_2d.dat.data.max():.2f}]",
          flush=True)
    print(f"θ_C: [{θ_C_2d.dat.data.min():.2f}, {θ_C_2d.dat.data.max():.2f}]",
          flush=True)

    # Save controls first
    with CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(θ_A_2d, name="log_fluidity")
        chk.save_function(θ_C_2d, name="log_friction")
    print(f"Saved controls to {OUTPUT_FILE}", flush=True)

    # Final forward solve
    try:
        A_opt = Function(Q_lift).interpolate(A_0 * exp(θ_A))
        C_opt = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
        u_opt = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                         fluidity=A_opt, friction=C_opt)
    except firedrake.exceptions.ConvergenceError:
        print("Final forward solve failed — using last-good velocity", flush=True)
        u_opt = u0
    u_avg = icepack.depth_average(u_opt)
    speed_model = Function(Q_2d, name="speed_model").interpolate(
        sqrt(inner(u_avg, u_avg)))
    speed_obs = Function(Q_2d, name="speed_obs").interpolate(
        sqrt(inner(u_obs_2d, u_obs_2d)))

    # Sample model ε_zz at the ApRES VOM (final values for diagnostics)
    eps_h_opt = icepack.models.hybrid.horizontal_strain_rate(
        velocity=u_opt, thickness=h, surface=s)
    eps_zz_opt = Function(Δ).interpolate(-(eps_h_opt[0, 0] + eps_h_opt[1, 1]))

    with CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(θ_A_2d, name="log_fluidity")
        chk.save_function(θ_C_2d, name="log_friction")
        chk.save_function(speed_model, name="speed_model")
        chk.save_function(speed_obs, name="speed_obs")
        chk.save_function(u_avg, name="velocity_avg")
    print(f"Saved {OUTPUT_FILE}", flush=True)

    # Compare ApRES obs vs model at sites
    mod = eps_zz_opt.dat.data_ro
    obs = eps_obs_f.dat.data_ro
    sig = σ_eps_f.dat.data_ro
    chi2 = float(np.mean(((mod - obs) / sig) ** 2))
    print(f"\nApRES diagnostics:", flush=True)
    print(f"  obs mean={obs.mean():.3e}  std={obs.std():.3e}", flush=True)
    print(f"  mod mean={mod.mean():.3e}  std={mod.std():.3e}", flush=True)
    print(f"  reduced chi^2 = {chi2:.2f}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
