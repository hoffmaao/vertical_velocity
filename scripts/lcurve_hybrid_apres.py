r"""L-curve analysis for hybrid + ApRES inversion (manual L-BFGS).

Recinos et al. 2023 / fenics_ice convention (no 1/N normalization):
- Misfit:        J = 0.5 · Σ χ² over each obs type
- Regularization R(θ) = 0.5 · (δ ∫θ² + γ ∫|∇θ|²) dx,  ELL = sqrt(γ/δ)

L is interpreted as ELL in km. δ is fixed at DELTA (matches Recinos β
L-curve elbow value); γ = δ · (L · 1000 m)² varies with L.

Runs a single L value per invocation so multiple L values can run
in parallel. Warm-starts from inversion_hires_apres_vd1.h5.

Usage:
    OMP_NUM_THREADS=1 python lcurve_hybrid_apres.py <L_km>
    e.g.  python lcurve_hybrid_apres.py 2     # ELL = 2 km
"""
import numpy as np
import h5py
import netCDF4 as nc
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, ExtrudedMesh, VertexOnlyMesh,
    sqrt, inner, grad, max_value, exp, dx, conditional, assemble,
    TestFunction,
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
import json, sys

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
APRES_FILE = DATA_DIR / "data" / "apres_legendre_fits.h5"
VEL_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/velocity/"
                "antarctica_ice_velocity_450m_v2.nc")
WARM_START_FILE = DATA_DIR / "mesh" / "inversion_hires_apres_vd1.h5"

MAX_ITER = 500
VDEGREE = 1
VEL_STEP = 1            # full 450m MEaSUREs density (~300k obs over Thwaites)
DEPTH_MIN = 100.0
DEPTH_MAX = 800.0
DELTA = 1.0e-5          # Recinos β L-curve elbow value (fixed across sweep)

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 50, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def run_single_L(L_km):
    L_m = L_km * 1000.0  # km → meters (= ELL)
    gamma_val = DELTA * L_m**2  # γ = δ · ELL²  (Matérn-1 SPDE)
    DELTA_c = Constant(DELTA)
    GAMMA_c = Constant(gamma_val)
    print("=" * 60, flush=True)
    print(f"L-curve: Hybrid + ApRES, ELL = {L_km} km ({L_m:.0f} m) "
          f"(hires, vdeg={VDEGREE}, {MAX_ITER} iters)", flush=True)
    print(f"  Recinos prior: δ={DELTA:.3e}, γ={gamma_val:.3e}  "
          f"(no 1/N normalization)", flush=True)
    print(f"  Warm-start from {WARM_START_FILE.name}", flush=True)
    print("=" * 60, flush=True)

    # ── Load mesh ──────────────────────────────────────────────────
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    print(f"Mesh: {mesh_2d.num_vertices()} verts", flush=True)

    # ── Sparse velocity observations ───────────────────────────────
    from scipy.spatial import Delaunay
    ds = nc.Dataset(str(VEL_FILE))
    x_vel = ds.variables["x"][:]; y_vel = ds.variables["y"][:]
    coords_2d = mesh_2d.coordinates.dat.data_ro
    xmin, xmax = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    ymin, ymax = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    ix0 = max(0, np.searchsorted(x_vel, xmin) - 5)
    ix1 = min(len(x_vel), np.searchsorted(x_vel, xmax) + 5)
    iy0 = max(0, np.searchsorted(-y_vel, -ymax) - 5)
    iy1 = min(len(y_vel), np.searchsorted(-y_vel, -ymin) + 5)
    sl = np.s_[iy0:iy1:VEL_STEP, ix0:ix1:VEL_STEP]
    vx = np.nan_to_num(np.array(ds.variables["VX"][sl]).astype(float), nan=0.0)
    vy = np.nan_to_num(np.array(ds.variables["VY"][sl]).astype(float), nan=0.0)
    stdx = np.nan_to_num(np.array(ds.variables["STDX"][sl]).astype(float), nan=1e6)
    stdy = np.nan_to_num(np.array(ds.variables["STDY"][sl]).astype(float), nan=1e6)
    ds.close()
    YY, XX = np.meshgrid(y_vel[iy0:iy1:VEL_STEP], x_vel[ix0:ix1:VEL_STEP], indexing="ij")
    xs_f, ys_f = XX.ravel(), YY.ravel()
    tri_d = Delaunay(coords_2d)
    inside = tri_d.find_simplex(np.column_stack([xs_f, ys_f])) >= 0
    has_data = (stdx.ravel() < 1e5) & (stdy.ravel() < 1e5) & (stdx.ravel() > 0) & (stdy.ravel() > 0)
    keep = inside & has_data
    # Store velocity data; VOM created after extrusion
    _vx_keep = vx.ravel()[keep]
    _vy_keep = vy.ravel()[keep]
    _σx_keep = np.maximum(stdx.ravel()[keep], 1.0)
    _σy_keep = np.maximum(stdy.ravel()[keep], 1.0)
    _vel_xy = np.column_stack([xs_f[keep], ys_f[keep]])
    print(f"  Sparse velocity: {keep.sum()} points (VOM after extrusion)", flush=True)

    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=VDEGREE, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    u_obs_3d = icepack.utilities.lift3d(u_obs_2d, V_lift)
    u0 = Function(V).project(u_obs_3d)

    # Velocity VOM at ζ=1 on extruded mesh
    vel_pts_3d = np.column_stack([_vel_xy, np.ones(len(_vel_xy))])
    vel_vom = VertexOnlyMesh(mesh, vel_pts_3d, missing_points_behaviour="warn")
    Δ_vel = FunctionSpace(vel_vom, "DG", 0)
    Δ_vel_in = FunctionSpace(vel_vom.input_ordering, "DG", 0)
    def obs_to_vom(vals):
        f_in = Function(Δ_vel_in); f_in.dat.data[:] = vals[:len(f_in.dat.data)]
        return Function(Δ_vel).interpolate(f_in)
    u_o = obs_to_vom(_vx_keep)
    v_o = obs_to_vom(_vy_keep)
    σ_x = obs_to_vom(_σx_keep)
    σ_y = obs_to_vom(_σy_keep)
    N_vel = len(u_o.dat.data)
    print(f"  Velocity VOM: {N_vel} points at ζ=1", flush=True)

    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    A_0, C_0 = Constant(20.0), Constant(0.01)

    # Warm-start from converged L=10km result
    if WARM_START_FILE.exists():
        with CheckpointFile(str(WARM_START_FILE), "r") as wchk:
            mw = wchk.load_mesh()
            θA_w = wchk.load_function(mw, "log_fluidity")
            θC_w = wchk.load_function(mw, "log_friction")
        θ_A.dat.data[:] = θA_w.dat.data_ro
        θ_C.dat.data[:] = θC_w.dat.data_ro
        print(f"Warm start from {WARM_START_FILE.name}: "
              f"θ_A=[{θ_A.dat.data.min():.2f}, {θ_A.dat.data.max():.2f}], "
              f"θ_C=[{θ_C.dat.data.min():.2f}, {θ_C.dat.data.max():.2f}]",
              flush=True)
    else:
        print("Cold start (no warm-start file found)", flush=True)

    grounded_mask = Function(Q_lift).interpolate(
        conditional(
            1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
            Constant(1.0), Constant(0.0)))

    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    A_init = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_init = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    u_test = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                      fluidity=A_init, friction=C_init)
    u0.assign(u_test)
    print("Primed OK", flush=True)

    # ── ApRES VOM ──────────────────────────────────────────────────
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
            w = (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX) \
                & np.isfinite(eps) & np.isfinite(sig) & (sig > 0)
            d_w = depth[w]
            if d_w.size == 0:
                continue
            stride = max(1, d_w.size // 5)
            d_w = d_w[::stride]
            pts_xyz.append(np.column_stack([
                np.full(d_w.size, xi), np.full(d_w.size, yi), 1.0 - d_w / Hi]))
            eps_obs.append(eps[w][::stride])
            eps_sig.append(sig[w][::stride])
    pts_xyz = np.vstack(pts_xyz)
    eps_obs_arr = np.concatenate(eps_obs)
    eps_sig_arr = np.concatenate(eps_sig)

    coords = mesh_2d.coordinates.dat.data_ro
    inside = ((pts_xyz[:, 0] > coords[:, 0].min()) & (pts_xyz[:, 0] < coords[:, 0].max())
              & (pts_xyz[:, 1] > coords[:, 1].min()) & (pts_xyz[:, 1] < coords[:, 1].max()))
    pts_xyz = pts_xyz[inside]
    eps_obs_arr = eps_obs_arr[inside]
    eps_sig_arr = eps_sig_arr[inside]

    vom = VertexOnlyMesh(mesh, pts_xyz, missing_points_behaviour="warn")
    Δ = FunctionSpace(vom, "DG", 0)
    Δ_in = FunctionSpace(vom.input_ordering, "DG", 0)
    f_in = Function(Δ_in)
    f_in.dat.data[:] = eps_obs_arr[:len(f_in.dat.data)]
    eps_obs_f = Function(Δ).interpolate(f_in)
    f_in.dat.data[:] = eps_sig_arr[:len(f_in.dat.data)]
    σ_eps_f = Function(Δ).interpolate(f_in)
    N_apr = vom.num_vertices()
    print(f"ApRES: {N_apr} VOM points (N_apr={N_apr})", flush=True)

    # ── Inversion ──────────────────────────────────────────────────
    n_dof = θ_A.dat.data.shape[0]
    N_apr_c = Constant(float(max(N_apr, 1)))
    iteration = [0]
    state = {"last_good_J": None, "last_good_g": None, "last_good_u": None}

    def _fail_response(label):
        iteration[0] += 1
        print(f"  iter {iteration[0]:3d}: {label} — penalty + last-good gradient",
              flush=True)
        if state["last_good_u"] is not None:
            u0.assign(state["last_good_u"])
        penalty = 10.0 * (state["last_good_J"] or 1.0)
        g_fb = state["last_good_g"].copy() if state["last_good_g"] is not None \
               else np.zeros(2 * n_dof)
        return penalty, g_fb

    def obj_grad(x):
        θ_A.dat.data[:] = x[:n_dof]
        θ_C.dat.data[:] = x[n_dof:]

        reset_manager(); start_manager()
        A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
        C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
        try:
            u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                         fluidity=A, friction=C)
        except firedrake.exceptions.ConvergenceError as e:
            stop_manager()
            return _fail_response(f"FWD FAILED ({str(e)[:40]}...)")

        # Sparse velocity misfit — evaluate u at surface (ζ=1) points
        u_interp = Function(Δ_vel).interpolate(u[0])
        v_interp = Function(Δ_vel).interpolate(u[1])
        δu, δv = u_interp - u_o, v_interp - v_o

        J = Functional(name="J")
        J.assign(0.5 * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx)

        # ApRES misfit — raw Σχ², no 1/N (Recinos convention)
        eps_h = icepack.models.hybrid.horizontal_strain_rate(
            velocity=u, thickness=h, surface=s)
        eps_mod = Function(Δ).interpolate(-(eps_h[0, 0] + eps_h[1, 1]))
        J.addto(0.5 * ((eps_mod - eps_obs_f) / σ_eps_f) ** 2 * dx)

        J_val = float(J)
        try:
            dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
        except (firedrake.exceptions.ConvergenceError, Exception) as e:
            stop_manager()
            return _fail_response(f"ADJ FAILED ({type(e).__name__}: {str(e)[:30]}...)")
        stop_manager()

        # Matérn regularization: R(θ) = 0.5 (δ ∫θ² + γ ∫|∇θ|²) dx
        v_test = TestFunction(Q_lift)
        reg_A = float(assemble(0.5 * (DELTA_c * inner(θ_A, θ_A)
                                       + GAMMA_c * inner(grad(θ_A), grad(θ_A))) * dx))
        reg_C = float(assemble(0.5 * (DELTA_c * inner(θ_C, θ_C)
                                       + GAMMA_c * inner(grad(θ_C), grad(θ_C))) * dx))
        g_A = dJ_A.dat.data_ro.copy()
        g_C = dJ_C.dat.data_ro.copy()
        g_A += assemble((DELTA_c * inner(θ_A, v_test)
                         + GAMMA_c * inner(grad(θ_A), grad(v_test))) * dx).dat.data_ro
        g_C += assemble((DELTA_c * inner(θ_C, v_test)
                         + GAMMA_c * inner(grad(θ_C), grad(v_test))) * dx).dat.data_ro

        total = J_val + reg_A + reg_C
        iteration[0] += 1
        full_g = np.concatenate([g_A, g_C])

        # Cache last good state
        state["last_good_u"] = u.copy(deepcopy=True)
        state["last_good_J"] = total
        state["last_good_g"] = full_g.copy()
        u0.assign(u)

        if iteration[0] % 20 == 0:
            J_vel = float(assemble(0.5 * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx))
            J_apr = J_val - J_vel
            g_norm = float(np.sqrt((full_g**2).sum()))
            print(f"  iter {iteration[0]:3d}: Jv={J_vel:.3e} Ja={J_apr:.3e} "
                  f"R={reg_A+reg_C:.3e} |g|={g_norm:.3e}", flush=True)

        return total, full_g

    x0 = np.concatenate([θ_A.dat.data_ro.copy(), θ_C.dat.data_ro.copy()])
    print(f"\nL-BFGS-B ({2*n_dof} DOFs, {MAX_ITER} iters)...", flush=True)

    result = scipy_minimize(
        obj_grad, x0, method="L-BFGS-B", jac=True,
        options={"maxiter": MAX_ITER, "ftol": 0, "gtol": 0, "pgtol": 0},
    )
    print(f"Result: {result.message} ({iteration[0]} evals)", flush=True)

    # ── Final evaluation ───────────────────────────────────────────
    θ_A.dat.data[:] = result.x[:n_dof]
    θ_C.dat.data[:] = result.x[n_dof:]
    A_f = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_f = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    try:
        u_f = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                       fluidity=A_f, friction=C_f)
    except firedrake.exceptions.ConvergenceError:
        u_f = u0

    u_i_f = Function(Δ_vel).interpolate(u_f[0])
    v_i_f = Function(Δ_vel).interpolate(u_f[1])
    E_vel = float(assemble(0.5 * (((u_i_f - u_o) / σ_x)**2
                                   + ((v_i_f - v_o) / σ_y)**2) * dx))
    eps_h_f = icepack.models.hybrid.horizontal_strain_rate(
        velocity=u_f, thickness=h, surface=s)
    eps_mod_f = Function(Δ).interpolate(-(eps_h_f[0, 0] + eps_h_f[1, 1]))
    E_apr = float(assemble(0.5 * ((eps_mod_f - eps_obs_f) / σ_eps_f) ** 2 * dx))
    R_A = float(assemble(0.5 * (DELTA_c * inner(θ_A, θ_A)
                                  + GAMMA_c * inner(grad(θ_A), grad(θ_A))) * dx))
    R_C = float(assemble(0.5 * (DELTA_c * inner(θ_C, θ_C)
                                  + GAMMA_c * inner(grad(θ_C), grad(θ_C))) * dx))
    R = R_A + R_C

    print(f"\nFinal: E_vel={E_vel:.4e}, E_apr={E_apr:.4e}, R={R:.4e}", flush=True)

    # Save checkpoint
    θ_A_2d = Function(Q_2d, name="log_fluidity")
    θ_C_2d = Function(Q_2d, name="log_friction")
    θ_A_2d.dat.data[:] = θ_A.dat.data_ro
    θ_C_2d.dat.data[:] = θ_C.dat.data_ro
    chk_file = DATA_DIR / "mesh" / f"lcurve_hybrid_apres_L{L_km:g}km.h5"
    with CheckpointFile(str(chk_file), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(θ_A_2d, name="log_fluidity")
        chk.save_function(θ_C_2d, name="log_friction")
    print(f"Saved {chk_file.name}", flush=True)

    # Save JSON
    result_data = {"L_km": L_km, "delta": DELTA, "gamma": gamma_val,
                    "E_vel": E_vel, "E_apr": E_apr,
                    "E_total": E_vel + E_apr, "R": R, "R_A": R_A, "R_C": R_C,
                    "iters": iteration[0],
                    "convention": "recinos"}
    json_file = DATA_DIR / "data" / f"lcurve_hybrid_apres_L{L_km:g}km.json"
    json_file.parent.mkdir(exist_ok=True)
    with open(str(json_file), "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Saved {json_file.name}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lcurve_hybrid_apres.py <L_km>")
        print(f"  L values (km) to try: 0.5, 1, 2, 3, 5, 10")
        sys.exit(1)
    run_single_L(float(sys.argv[1]))
