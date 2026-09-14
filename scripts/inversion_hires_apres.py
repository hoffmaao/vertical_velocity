r"""Hires hybrid inversion with sparse velocity + ApRES.

Uses sparse velocity observations via a 2D VertexOnlyMesh (following
icepack how-to/04-sparse-data) and ApRES vertical strain rates via a
3D VertexOnlyMesh on the extruded mesh.

Recinos et al. 2023 / fenics_ice convention (no 1/N normalization):
- Misfit:        J = 0.5 · Σ χ² over each obs type
- Regularization R(θ) = 0.5 · (δ ∫θ² + γ ∫|∇θ|²) dx,  ELL = sqrt(γ/δ)
"""
import numpy as np
import h5py
import netCDF4 as nc
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
from mpi4py import MPI
from pathlib import Path
import sys

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
WARM_FILE = DATA_DIR / "mesh" / "lcurve_hybrid_apres_L1km.h5"  # ELL=1 km elbow
APRES_FILE = DATA_DIR / "data" / "apres_legendre_fits.h5"
VEL_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/velocity/"
                "antarctica_ice_velocity_450m_v2.nc")

VDEGREE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
HDEGREE = 1
# Matérn prior R(θ) = 0.5 · (δ ∫θ² + γ ∫|∇θ|²) dx  (per Recinos et al. 2023).
# Correlation length ELL = sqrt(γ/δ). Recinos β L-curve elbow: δ=1e-5, γ=10 → ELL=1 km.
DELTA_A = 1.0e-5        # prior amplitude penalty for log-fluidity
DELTA_C = 1.0e-5        # prior amplitude penalty for log-friction
ELL_A = 1000.0          # prior correlation length for log-fluidity (m)
ELL_C = 1000.0          # prior correlation length for log-friction (m)
MAX_ITER = 500
VEL_STEP = 1            # full 450m MEaSUREs density (~300k obs over Thwaites)
APRES_WEIGHT = 1.0
DEPTH_MIN = 100.0
DEPTH_MAX = 800.0

OUTPUT_FILE = DATA_DIR / "mesh" / f"inversion_hires_apres_vd{VDEGREE}.h5"

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 50, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

WARM_START = True  # warm-start hires from existing 2D-control vd=1 result


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def main():
    print("=" * 60, flush=True)
    print(f"Hires hybrid inversion: sparse vel + ApRES (vdeg={VDEGREE})", flush=True)
    print(f"  ApRES depth window: {DEPTH_MIN:.0f}-{DEPTH_MAX:.0f} m", flush=True)
    print(f"  Velocity subsample step: {VEL_STEP} (~{VEL_STEP*450/1000:.1f} km)", flush=True)
    print("=" * 60, flush=True)

    # ── Load 2D fields ─────────────────────────────────────────────
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")
    print(f"Hires mesh: {mesh_2d.num_vertices()} verts, "
          f"{mesh_2d.num_cells()} cells", flush=True)

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)

    # ── Sparse velocity observations (icepack how-to/04-sparse-data) ─
    print("Loading sparse velocity observations...", flush=True)
    from scipy.spatial import Delaunay
    ds = nc.Dataset(str(VEL_FILE))
    x_vel = ds.variables["x"][:]
    y_vel = ds.variables["y"][:]
    coords_2d = mesh_2d.coordinates.dat.data_ro
    xmin, xmax = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    ymin, ymax = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    ix0 = max(0, np.searchsorted(x_vel, xmin) - 5)
    ix1 = min(len(x_vel), np.searchsorted(x_vel, xmax) + 5)
    iy0 = max(0, np.searchsorted(-y_vel, -ymax) - 5)
    iy1 = min(len(y_vel), np.searchsorted(-y_vel, -ymin) + 5)
    sl = np.s_[iy0:iy1:VEL_STEP, ix0:ix1:VEL_STEP]
    x_sub = x_vel[ix0:ix1:VEL_STEP]
    y_sub = y_vel[iy0:iy1:VEL_STEP]
    vx = np.nan_to_num(np.array(ds.variables["VX"][sl]).astype(float), nan=0.0)
    vy = np.nan_to_num(np.array(ds.variables["VY"][sl]).astype(float), nan=0.0)
    stdx = np.nan_to_num(np.array(ds.variables["STDX"][sl]).astype(float), nan=1e6)
    stdy = np.nan_to_num(np.array(ds.variables["STDY"][sl]).astype(float), nan=1e6)
    ds.close()
    YY, XX = np.meshgrid(y_sub, x_sub, indexing="ij")
    xs_flat, ys_flat = XX.ravel(), YY.ravel()
    vx_flat, vy_flat = vx.ravel(), vy.ravel()
    stdx_flat, stdy_flat = stdx.ravel(), stdy.ravel()
    tri = Delaunay(coords_2d)
    inside = tri.find_simplex(np.column_stack([xs_flat, ys_flat])) >= 0
    has_data = (stdx_flat < 1e5) & (stdy_flat < 1e5) & (stdx_flat > 0) & (stdy_flat > 0)
    keep = inside & has_data
    xs_obs, ys_obs = xs_flat[keep], ys_flat[keep]
    vx_obs, vy_obs = vx_flat[keep], vy_flat[keep]
    σx_obs = np.maximum(stdx_flat[keep], 1.0)
    σy_obs = np.maximum(stdy_flat[keep], 1.0)

    N_vel_raw = len(xs_obs)
    print(f"  Sparse velocity: {N_vel_raw} points (VOM created after extrusion)",
          flush=True)

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

    # Sparse velocity VOM on the 3D mesh at ζ=1 (surface)
    vel_pts_3d = np.column_stack([xs_obs, ys_obs, np.ones(N_vel_raw)])
    vel_vom = VertexOnlyMesh(mesh, vel_pts_3d, missing_points_behaviour="warn")
    Δ_vel = FunctionSpace(vel_vom, "DG", 0)
    Δ_vel_in = FunctionSpace(vel_vom.input_ordering, "DG", 0)

    def obs_to_vom(vals):
        f_in = Function(Δ_vel_in)
        f_in.dat.data[:] = vals[:len(f_in.dat.data)]
        return Function(Δ_vel).interpolate(f_in)

    u_o = obs_to_vom(vx_obs)
    v_o = obs_to_vom(vy_obs)
    σ_x = obs_to_vom(σx_obs)
    σ_y = obs_to_vom(σy_obs)
    N_vel = len(u_o.dat.data)
    print(f"  Velocity VOM: {N_vel} points at ζ=1", flush=True)

    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    A_0, C_0 = Constant(20.0), Constant(0.01)

    # Warm-start θ_A, θ_C from the existing 2D-control vd=1 result.
    # Use project() so it works under MPI with different partitions.
    from firedrake.petsc import PETSc
    rank = PETSc.COMM_WORLD.rank
    nprocs = PETSc.COMM_WORLD.size
    if WARM_START and WARM_FILE.exists() and nprocs == 1:
        # Serial warm start: direct data copy (same mesh, same partition)
        with CheckpointFile(str(WARM_FILE), "r") as wchk:
            mw = wchk.load_mesh()
            θA_w = wchk.load_function(mw, "log_fluidity")
            θC_w = wchk.load_function(mw, "log_friction")
        θ_A.dat.data[:] = np.clip(θA_w.dat.data_ro, -10, 10)
        θ_C.dat.data[:] = np.clip(θC_w.dat.data_ro, -10, 10)
        print(f"Warm start from {WARM_FILE.name} (clipped to ±10)", flush=True)
    else:
        if rank == 0:
            print(f"Cold start (θ=0, nprocs={nprocs})", flush=True)

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

    # ── MPI helpers ─────────────────────────────────────────────────
    mpi_comm = PETSc.COMM_WORLD.tompi4py()
    n_dof_local = θ_A.dat.data.shape[0]
    # Build global↔local maps for the control vector
    all_sizes = mpi_comm.allgather(n_dof_local)  # per-rank DOF count
    n_dof_global = sum(all_sizes)
    offsets = [sum(all_sizes[:i]) for i in range(len(all_sizes))]
    my_offset = offsets[rank]
    if rank == 0:
        print(f"  DOFs: {n_dof_global} per control field "
              f"({n_dof_local} local on rank 0, {nprocs} ranks)",
              flush=True)

    def local_to_global(local_A, local_C):
        """Allgather local arrays → global vector [θ_A; θ_C]."""
        gA = np.empty(n_dof_global)
        gC = np.empty(n_dof_global)
        mpi_comm.Allgatherv(local_A, [gA, all_sizes, offsets, MPI.DOUBLE])
        mpi_comm.Allgatherv(local_C, [gC, all_sizes, offsets, MPI.DOUBLE])
        return np.concatenate([gA, gC])

    def global_to_local(x_global):
        """Extract this rank's local portion from global vector."""
        gA = x_global[:n_dof_global]
        gC = x_global[n_dof_global:]
        return gA[my_offset:my_offset + n_dof_local], \
               gC[my_offset:my_offset + n_dof_local]

    # ── Cost function ──────────────────────────────────────────────
    area = assemble(Constant(1.0) * dx(mesh))
    # Matérn prior parameters (Recinos convention, no 1/N)
    gamma_A_val = DELTA_A * ELL_A**2
    gamma_C_val = DELTA_C * ELL_C**2
    DELTA_A_c = Constant(DELTA_A)
    DELTA_C_c = Constant(DELTA_C)
    GAMMA_A = Constant(gamma_A_val)
    GAMMA_C = Constant(gamma_C_val)
    if rank == 0:
        print(f"  Prior (Recinos): δ_A={DELTA_A:.3e}, γ_A={gamma_A_val:.3e}, "
              f"ELL_A={ELL_A:.0f} m", flush=True)
        print(f"  Prior (Recinos): δ_C={DELTA_C:.3e}, γ_C={gamma_C_val:.3e}, "
              f"ELL_C={ELL_C:.0f} m", flush=True)
        print(f"  Misfit: raw Σχ² (no 1/N)", flush=True)
    # σ_u_c no longer needed — per-point uncertainties via σ_x, σ_y
    apres_w = Constant(APRES_WEIGHT)
    iteration = [0]
    CHECKPOINT_EVERY = 5
    CHECKPOINT_FILE = DATA_DIR / "mesh" / f"checkpoint_hires_apres_vd{VDEGREE}.h5"

    state = {"last_good_J": None, "last_good_g": None, "last_good_u": None}

    def _fail_response(label):
        """Return penalty + last-good gradient on any solve failure."""
        iteration[0] += 1
        if rank == 0:
            print(f"  iter {iteration[0]:3d}: {label} "
                  f"— penalty + last-good gradient", flush=True)
        if state["last_good_u"] is not None:
            u0.assign(state["last_good_u"])
        penalty = 10.0 * (state["last_good_J"] or 1.0)
        g_fb = state["last_good_g"].copy() if state["last_good_g"] is not None \
               else np.zeros(2 * n_dof_global)
        return penalty, g_fb

    def obj_grad(x):
        # x is the GLOBAL control vector, identical on all ranks
        lA, lC = global_to_local(x)
        θ_A.dat.data[:] = lA
        θ_C.dat.data[:] = lC

        reset_manager(); start_manager()

        A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
        C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
        try:
            u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                         fluidity=A, friction=C)
        except firedrake.exceptions.ConvergenceError as e:
            stop_manager()
            return _fail_response(f"FWD FAILED ({e.args[0][:40]}...)")

        # Sparse velocity misfit: evaluate u at ζ=1 (surface) VOM points
        # (icepack how-to/04-sparse-data pattern, 3D VOM at surface)
        u_interp = Function(Δ_vel).interpolate(u[0])
        v_interp = Function(Δ_vel).interpolate(u[1])
        δu, δv = u_interp - u_o, v_interp - v_o

        J = Functional(name="J_total")
        J.assign(0.5 * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx)

        # ApRES ε_zz misfit — raw Σχ², no 1/N (Recinos convention)
        eps_h = icepack.models.hybrid.horizontal_strain_rate(
            velocity=u, thickness=h, surface=s)
        eps_zz_ufl = -(eps_h[0, 0] + eps_h[1, 1])
        eps_model = Function(Δ).interpolate(eps_zz_ufl)
        J.addto(apres_w * 0.5
                * ((eps_model - eps_obs_f) / σ_eps_f) ** 2 * dx)

        J_val = float(J)
        # Split parts for diagnostics
        J_vel = float(assemble(0.5 * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx))
        J_apr = J_val - J_vel

        try:
            dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
        except (firedrake.exceptions.ConvergenceError, Exception) as e:
            stop_manager()
            return _fail_response(f"ADJ FAILED ({type(e).__name__}: {str(e)[:40]}...)")
        stop_manager()

        # Matérn regularization: R(θ) = 0.5 · (δ ∫θ² + γ ∫|∇θ|²) dx
        # (Recinos et al. 2023 convention, no 1/N normalization)
        v_test = TestFunction(Q_lift)
        reg_A = float(assemble(0.5 * (DELTA_A_c * inner(θ_A, θ_A)
                                       + GAMMA_A * inner(grad(θ_A), grad(θ_A))) * dx))
        reg_C = float(assemble(0.5 * (DELTA_C_c * inner(θ_C, θ_C)
                                       + GAMMA_C * inner(grad(θ_C), grad(θ_C))) * dx))

        g_A = dJ_A.dat.data_ro.copy()
        g_C = dJ_C.dat.data_ro.copy()
        g_A += assemble((DELTA_A_c * inner(θ_A, v_test)
                         + GAMMA_A * inner(grad(θ_A), grad(v_test))) * dx).dat.data_ro
        g_C += assemble((DELTA_C_c * inner(θ_C, v_test)
                         + GAMMA_C * inner(grad(θ_C), grad(v_test))) * dx).dat.data_ro

        total = J_val + reg_A + reg_C
        iteration[0] += 1

        # Allgather local gradient → global gradient (identical on all ranks)
        full_g = local_to_global(g_A, g_C)
        g_norm = float(np.sqrt((full_g**2).sum()))

        if rank == 0:
            print(f"  iter {iteration[0]:3d}: "
                  f"Jv={J_vel:.3e} Ja={J_apr:.3e} reg={reg_A+reg_C:.3e} "
                  f"|g|={g_norm:.3e}", flush=True)

        # Cache last good state
        state["last_good_u"] = u.copy(deepcopy=True)
        state["last_good_J"] = total
        state["last_good_g"] = full_g.copy()
        u0.assign(u)

        # Periodic checkpoint (all ranks write h5; only rank 0 writes npz)
        if iteration[0] % CHECKPOINT_EVERY == 0:
            θ_A_chk = Function(Q_2d, name="log_fluidity")
            θ_C_chk = Function(Q_2d, name="log_friction")
            θ_A_chk.dat.data[:] = θ_A.dat.data_ro
            θ_C_chk.dat.data[:] = θ_C.dat.data_ro
            with CheckpointFile(str(CHECKPOINT_FILE), "w") as chk:
                chk.save_mesh(mesh_2d)
                chk.save_function(θ_A_chk, name="log_fluidity")
                chk.save_function(θ_C_chk, name="log_friction")
            if rank == 0:
                np.savez(str(CHECKPOINT_FILE).replace(".h5", ".npz"),
                         x=x, iteration=iteration[0],
                         J_vel=J_vel, J_apr=J_apr, reg=reg_A+reg_C)
                print(f"    [checkpoint saved at iter {iteration[0]}]",
                      flush=True)

        return total, full_g

    # Resume from checkpoint if available
    npz_file = str(CHECKPOINT_FILE).replace(".h5", ".npz")
    if Path(npz_file).exists():
        chk_data = np.load(npz_file)
        x0 = chk_data["x"]
        lA, lC = global_to_local(x0)
        θ_A.dat.data[:] = lA
        θ_C.dat.data[:] = lC
        prev_iter = int(chk_data["iteration"])
        iteration[0] = prev_iter
        if rank == 0:
            print(f"\nResuming from checkpoint at iter {prev_iter} "
                  f"(Jv={chk_data['J_vel']:.3e}, Ja={chk_data['J_apr']:.3e})",
                  flush=True)
        A_re = Function(Q_lift).interpolate(A_0 * exp(θ_A))
        C_re = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
        try:
            u_re = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                            fluidity=A_re, friction=C_re)
            u0.assign(u_re)
            if rank == 0:
                print("  re-primed velocity OK", flush=True)
        except firedrake.exceptions.ConvergenceError:
            if rank == 0:
                print("  re-prime failed, using warm-start velocity", flush=True)
        # Pre-populate state with one good eval so failure handler has valid data
        if rank == 0:
            print("  pre-populating state...", flush=True)
        _J0, _g0 = obj_grad(x0)
        if rank == 0:
            print(f"  state populated (J={_J0:.3e})", flush=True)
    else:
        x0 = local_to_global(θ_A.dat.data_ro.copy(), θ_C.dat.data_ro.copy())

    if rank == 0:
        print(f"\nL-BFGS-B ({2 * n_dof_global} DOFs, max {MAX_ITER} iter, "
              f"{nprocs} MPI ranks)...", flush=True)

    result = scipy_minimize(
        obj_grad, x0, method="L-BFGS-B", jac=True,
        options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": 1e-8, "disp": True},
    )
    if rank == 0:
        print(f"\nResult: {result.message}", flush=True)

    lA_final, lC_final = global_to_local(result.x)
    θ_A.dat.data[:] = lA_final
    θ_C.dat.data[:] = lC_final
    θ_A_2d = Function(Q_2d, name="log_fluidity")
    θ_C_2d = Function(Q_2d, name="log_friction")
    θ_A_2d.dat.data[:] = θ_A.dat.data_ro
    θ_C_2d.dat.data[:] = θ_C.dat.data_ro
    print(f"θ_A: [{θ_A_2d.dat.data.min():.2f}, {θ_A_2d.dat.data.max():.2f}]",
          flush=True)
    print(f"θ_C: [{θ_C_2d.dat.data.min():.2f}, {θ_C_2d.dat.data.max():.2f}]",
          flush=True)

    # Save controls first (so we don't lose the result if final solve fails)
    with CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(θ_A_2d, name="log_fluidity")
        chk.save_function(θ_C_2d, name="log_friction")
    print(f"Saved controls to {OUTPUT_FILE}", flush=True)

    # Final forward solve — use last-good u0 as initial guess
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

    # Re-save with derived fields
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
