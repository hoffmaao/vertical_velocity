r"""Gauss-Newton Hessian eigendecomposition for hybrid + ApRES inversion.

Computes leading eigenmodes of A^{-1} H_GN where:
  H_GN = Gauss-Newton Hessian of the combined velocity + ApRES misfit
  A    = delta*M + gamma*K (Laplacian prior)

Uses the finite-difference Hessian-vector product approach since
depth_average and VertexOnlyMesh interpolation are not fully
compatible with CachedHessian.

Follows Recinos et al. (2023) / fenics_ice convention.

Usage:
    OMP_NUM_THREADS=4 python run_eigendec.py [--method {fd, gn}]

Methods
-------
fd : finite-difference Hessian-vector products via tlm_adjoint compute_gradient
     (default; original behavior). Noisy small-λ tail; ~10% modes spurious.
gn : exact Gauss-Newton Hessian-vector products via sparse_hessian module
     (pyadjoint tape, VOM outside the tape). Clean spectrum, no spurious
     negatives. Recommended.
"""
import argparse
import sys
import numpy as np
import h5py
import hashlib
import netCDF4 as nc
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, ExtrudedMesh, VertexOnlyMesh,
    MixedFunctionSpace,
    sqrt, inner, grad, max_value, exp, dx, conditional, assemble,
    TestFunction, TrialFunction,
)
from firedrake.adjoint import *  # noqa
from tlm_adjoint.firedrake import (
    reset_manager, start_manager, stop_manager,
    compute_gradient, Functional,
)
from firedrake.petsc import PETSc
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)
import icepack
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
MAP_FILE = DATA_DIR / "mesh" / "inversion_hires_apres_vd1.h5"
APRES_FILE = DATA_DIR / "data" / "apres_legendre_fits.h5"
VEL_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/velocity/"
                "antarctica_ice_velocity_450m_v2.nc")

K_LEADING = 1500        # number of eigenmodes
ELL = 2000.0            # prior correlation length in meters (L-curve elbow at 2 km, VEL_STEP=1)
DELTA_PRIOR = 1.0e-5    # Recinos β L-curve elbow value (no 1/N normalization)
GAMMA_A = 1.0           # multiplier on (ELL² · δ) for θ_A
GAMMA_C = 1.0           # multiplier on (ELL² · δ) for θ_C
VEL_STEP = 1            # full 450m MEaSUREs density (~300k obs over Thwaites)
DEPTH_MIN = 100.0
DEPTH_MAX = 800.0
EPS_FD = 1e-5           # finite-difference step for Hessian-vector products

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


def main():
    global K_LEADING
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fd", "gn"], default="fd",
                    help="fd = finite-difference (legacy); gn = Gauss-Newton "
                         "via sparse_hessian (recommended, exact)")
    ap.add_argument("--K", type=int, default=None,
                    help=f"Override number of eigenmodes (default: {K_LEADING})")
    args = ap.parse_args()
    method = args.method
    if args.K is not None:
        K_LEADING = args.K

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sparse_hessian as sh

    print("=" * 60, flush=True)
    print(f"Hessian eigendecomposition: Hybrid + ApRES (method={method})",
          flush=True)
    print(f"  K={K_LEADING} modes, ELL={ELL:.0f} m, eps_fd={EPS_FD}", flush=True)
    print("=" * 60, flush=True)

    # ── Load mesh ──────────────────────────────────────────────────
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    ndof = Q_2d.dof_dset.size
    area_val = float(assemble(Constant(1.0) * dx(mesh_2d)))
    print(f"Mesh: {mesh_2d.num_vertices()} verts, ndof={ndof}, "
          f"area={area_val:.3e} m²", flush=True)

    # ── Sparse velocity VOM (2D for gradient computation) ──────────
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
    _vel_xy = np.column_stack([xs_f[keep], ys_f[keep]])
    _vx_k, _vy_k = vx.ravel()[keep], vy.ravel()[keep]
    _σx_k = np.maximum(stdx.ravel()[keep], 1.0)
    _σy_k = np.maximum(stdy.ravel()[keep], 1.0)
    print(f"Sparse velocity: {keep.sum()} points", flush=True)

    # ── Extrude ────────────────────────────────────────────────────
    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    u0 = Function(V).project(icepack.utilities.lift3d(u_obs_2d, V_lift))

    # Velocity VOM at ζ=1
    vel_pts_3d = np.column_stack([_vel_xy, np.ones(len(_vel_xy))])
    vel_vom = VertexOnlyMesh(mesh, vel_pts_3d, missing_points_behaviour="warn")
    Δ_vel = FunctionSpace(vel_vom, "DG", 0)
    Δ_vel_in = FunctionSpace(vel_vom.input_ordering, "DG", 0)
    def obs_to_vom(vals):
        f_in = Function(Δ_vel_in); f_in.dat.data[:] = vals[:len(f_in.dat.data)]
        return Function(Δ_vel).interpolate(f_in)
    u_o = obs_to_vom(_vx_k); v_o = obs_to_vom(_vy_k)
    σ_x = obs_to_vom(_σx_k); σ_y = obs_to_vom(_σy_k)
    N_vel = len(u_o.dat.data)
    print(f"Velocity VOM: {N_vel} points at ζ=1", flush=True)

    # ── ApRES VOM ──────────────────────────────────────────────────
    with h5py.File(str(APRES_FILE), "r") as f:
        x_sites = f["summary/x"][...]; y_sites = f["summary/y"][...]
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
            if d_w.size == 0: continue
            stride = max(1, d_w.size // 5)
            d_w = d_w[::stride]
            pts_xyz.append(np.column_stack([
                np.full(d_w.size, xi), np.full(d_w.size, yi), 1.0 - d_w / Hi]))
            eps_obs.append(eps[w][::stride]); eps_sig.append(sig[w][::stride])
    pts_xyz = np.vstack(pts_xyz)
    eps_obs_arr, eps_sig_arr = np.concatenate(eps_obs), np.concatenate(eps_sig)
    inside_a = ((pts_xyz[:, 0] > xmin) & (pts_xyz[:, 0] < xmax)
                & (pts_xyz[:, 1] > ymin) & (pts_xyz[:, 1] < ymax))
    pts_xyz, eps_obs_arr, eps_sig_arr = pts_xyz[inside_a], eps_obs_arr[inside_a], eps_sig_arr[inside_a]
    vom = VertexOnlyMesh(mesh, pts_xyz, missing_points_behaviour="warn")
    Δ = FunctionSpace(vom, "DG", 0)
    Δ_in = FunctionSpace(vom.input_ordering, "DG", 0)
    f_in = Function(Δ_in); f_in.dat.data[:] = eps_obs_arr[:len(f_in.dat.data)]
    eps_obs_f = Function(Δ).interpolate(f_in)
    f_in.dat.data[:] = eps_sig_arr[:len(f_in.dat.data)]
    σ_eps_f = Function(Δ).interpolate(f_in)
    N_apr = vom.num_vertices()
    print(f"ApRES VOM: {N_apr} points", flush=True)

    # ── Load MAP ───────────────────────────────────────────────────
    with CheckpointFile(str(MAP_FILE), "r") as chk:
        m_inv = chk.load_mesh()
        θA_map_loaded = chk.load_function(m_inv, "log_fluidity")
        θC_map_loaded = chk.load_function(m_inv, "log_friction")
    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    θ_A.dat.data[:] = θA_map_loaded.dat.data_ro
    θ_C.dat.data[:] = θC_map_loaded.dat.data_ro
    θA_map_vals = θ_A.dat.data_ro.copy()
    θC_map_vals = θ_C.dat.data_ro.copy()
    print(f"MAP: θ_A=[{θA_map_vals.min():.2f}, {θA_map_vals.max():.2f}], "
          f"θ_C=[{θC_map_vals.min():.2f}, {θC_map_vals.max():.2f}]", flush=True)

    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q_lift).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
                    Constant(1.0), Constant(0.0)))

    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    # Prime velocity at MAP
    A_map = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_map = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    u0 = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                  fluidity=A_map, friction=C_map)
    print("Primed at MAP", flush=True)

    # ── Misfit gradient function (Recinos convention, no 1/N) ──────
    def grad_misfit(θA_vals, θC_vals):
        """Compute gradient of the misfit J w.r.t. (θ_A, θ_C)."""
        θ_A.dat.data[:] = θA_vals
        θ_C.dat.data[:] = θC_vals
        reset_manager(); start_manager()
        A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
        C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
        u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                     fluidity=A, friction=C)
        # Sparse velocity misfit (raw Σχ²)
        u_interp = Function(Δ_vel).interpolate(u[0])
        v_interp = Function(Δ_vel).interpolate(u[1])
        δu, δv = u_interp - u_o, v_interp - v_o
        J = Functional(name="J")
        J.assign(0.5 * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx)
        # ApRES misfit (raw Σχ²)
        eps_h = icepack.models.hybrid.horizontal_strain_rate(
            velocity=u, thickness=h, surface=s)
        eps_mod = Function(Δ).interpolate(-(eps_h[0, 0] + eps_h[1, 1]))
        J.addto(0.5 * ((eps_mod - eps_obs_f) / σ_eps_f) ** 2 * dx)
        dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
        stop_manager()
        return dJ_A.dat.data_ro.copy(), dJ_C.dat.data_ro.copy()

    # ── Hessian-vector product: branch on method ──
    if method == "fd":
        print("Computing gradient at MAP (FD baseline)...", flush=True)
        g0_A, g0_C = grad_misfit(θA_map_vals, θC_map_vals)
        g0_norm = np.sqrt((g0_A**2).sum() + (g0_C**2).sum())
        print(f"  |grad| at MAP = {g0_norm:.4e}", flush=True)

        def Hv_pair(v_A, v_C):
            """Finite-difference Hessian-vector product (in cotangent space)."""
            θA_p = θA_map_vals + EPS_FD * v_A
            θC_p = θC_map_vals + EPS_FD * v_C
            gp_A, gp_C = grad_misfit(θA_p, θC_p)
            # No mass_inv: compute_gradient returns the conjugate (cotangent),
            # and the downstream Ainv operator expects cotangent input.
            hv_A = (gp_A - g0_A) / EPS_FD
            hv_C = (gp_C - g0_C) / EPS_FD
            return hv_A, hv_C
    else:  # method == "gn"
        print("Building TapedForward (pyadjoint) at MAP...", flush=True)
        θ_A.dat.data[:] = θA_map_vals
        θ_C.dat.data[:] = θC_map_vals

        def simulation(θA_in, θC_in):
            A = Function(Q_lift).interpolate(A_0 * exp(θA_in))
            C = Function(Q_lift).interpolate(C_0 * exp(θC_in * grounded_mask))
            u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                         fluidity=A, friction=C)
            return u

        taped = sh.TapedForward(simulation, θ_A, θ_C)
        print(f"  tape blocks: {len(taped.tape.get_blocks())}", flush=True)

        Hv_vel = sh.make_Hv_gn_velocity(
            taped, vel_pts_3d, _σx_k, _σy_k, N_vel)
        Hv_apr = sh.make_Hv_gn_strainrate(
            taped, pts_xyz, eps_sig_arr, N_apr, h, s)
        Hv_gn = sh.combine_Hv(Hv_vel, Hv_apr)

        def Hv_pair(v_A, v_C):
            """Exact Gauss-Newton Hv (cotangent space)."""
            vA_f = Function(Q_lift); vA_f.dat.data[:] = v_A
            vC_f = Function(Q_lift); vC_f.dat.data[:] = v_C
            hvA, hvC = Hv_gn(vA_f, vC_f)
            return hvA.dat.data_ro.copy(), hvC.dat.data_ro.copy()

    # ── Prior (Recinos convention, no 1/N) ──
    # δ controls amplitude penalty; γ = δ·ELL² gives correlation length ELL.
    # GHEP formulation: H v = λ A v with v A-orthonormal (fenics_ice convention).
    N_vel_val = float(N_vel)
    delta_val = DELTA_PRIOR
    delta = Constant(delta_val)
    gamma_A_val = GAMMA_A * ELL**2 * delta_val
    gamma_C_val = GAMMA_C * ELL**2 * delta_val
    gamma_A_eff = Constant(gamma_A_val)
    gamma_C_eff = Constant(gamma_C_val)
    print(f"Prior (Recinos, GHEP): delta={delta_val:.6e}, "
          f"gamma_A={gamma_A_val:.6e}, gamma_C={gamma_C_val:.6e}, "
          f"ELL={ELL:.0f} m", flush=True)

    def make_prior_matrix(gamma_c):
        """Assemble A = δM + γK as a scipy CSR matrix for GHEP eigsh."""
        trial_q = TrialFunction(Q_lift); test_q = TestFunction(Q_lift)
        a = (delta * inner(trial_q, test_q)
             + gamma_c * inner(grad(trial_q), grad(test_q))) * dx
        A_petsc = assemble(a).M.handle
        ai, aj, av = A_petsc.getValuesCSR()
        nrows, ncols = A_petsc.getSize()
        return sp.csr_matrix((av, aj, ai), shape=(nrows, ncols))

    A_mat_A = make_prior_matrix(gamma_A_eff)
    A_mat_C = make_prior_matrix(gamma_C_eff)
    A_block = sp.block_diag([A_mat_A, A_mat_C], format="csr")
    print(f"Built A_block sparse matrix: {A_block.shape}, nnz={A_block.nnz}",
          flush=True)
    # Pre-factorize for efficient M·y = b solves inside scipy.eigsh GHEP
    print("Factorizing A_block via LU (one-shot)...", flush=True)
    A_lu = splu(A_block.tocsc())
    print(f"  LU done, fill factor ≈ {A_lu.L.nnz / A_block.nnz:.2f}", flush=True)
    Minv_op = LinearOperator(A_block.shape,
                              matvec=lambda b: A_lu.solve(b),
                              dtype=float)

    # ── ARPACK eigensolve ──────────────────────────────────────────
    print(f"\nARPACK eigensolve for {K_LEADING} modes...", flush=True)
    n = θ_A.dat.data.shape[0]
    N = 2 * n

    # Cache fingerprint: invalidates the cache if any of these change
    # NOTE: operator is now H only (was A⁻¹MH); GHEP mode applies M=A_block externally.
    fp_data = json.dumps({
        "method": method,
        "ELL": ELL, "DELTA_PRIOR": DELTA_PRIOR,
        "GAMMA_A": GAMMA_A, "GAMMA_C": GAMMA_C,
        "EPS_FD": EPS_FD, "DEPTH_MIN": DEPTH_MIN, "DEPTH_MAX": DEPTH_MAX,
        "VEL_STEP": VEL_STEP, "n": int(n),
        "MAP_FILE": str(MAP_FILE), "MESH_FILE": str(MESH_FILE),
        "convention": "recinos_GHEP_H_only",
        "operator": "H (eigsh with M=A_block for GHEP)",
    }, sort_keys=True)
    fp = hashlib.sha256(fp_data.encode()).hexdigest()[:12]

    cache_dir = DATA_DIR / "results" / "eigendec_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"matvecs_{fp}.h5"
    v0_file = cache_dir / f"v0_{fp}.npy"

    # Deterministic initial Lanczos vector (required for cache replay)
    if v0_file.exists():
        v0 = np.load(str(v0_file))
        print(f"Loaded v0 from {v0_file.name}", flush=True)
    else:
        v0 = np.random.default_rng(42).standard_normal(N)
        np.save(str(v0_file), v0)
        print(f"Generated v0 (seed=42) → {v0_file.name}", flush=True)

    h5cache = h5py.File(str(cache_file), "a")
    if "y" not in h5cache:
        h5cache.create_dataset("x", shape=(0, N), maxshape=(None, N),
                                dtype="f8", chunks=(1, N))
        h5cache.create_dataset("y", shape=(0, N), maxshape=(None, N),
                                dtype="f8", chunks=(1, N))
        h5cache.attrs["fingerprint"] = fp_data
    n_cached = h5cache["y"].shape[0]
    print(f"Matvec cache: {n_cached} previously computed → {cache_file.name}",
          flush=True)

    call_count = [0]

    def matvec(x):
        """Apply H (GN Hessian) only. GHEP mode below handles M=A_block."""
        i = call_count[0]
        if i < n_cached:
            x_chk = h5cache["x"][i, :]
            dx = np.abs(x - x_chk).max()
            if dx > 1e-8:
                raise RuntimeError(
                    f"Cache replay mismatch at matvec {i}: max|Δx|={dx:.3e}. "
                    f"Delete {cache_file} (and {v0_file}) to start fresh.")
            y = h5cache["y"][i, :].copy()
            call_count[0] += 1
            if call_count[0] % 50 == 0 or call_count[0] == n_cached:
                print(f"  matvec {call_count[0]} (replayed from cache)",
                      flush=True)
            return y
        # Compute fresh: H · x only
        v_A = x[:n]; v_C = x[n:]
        hv_A, hv_C = Hv_pair(v_A, v_C)
        y = np.concatenate([hv_A, hv_C])
        h5cache["x"].resize(i + 1, axis=0); h5cache["x"][i, :] = x
        h5cache["y"].resize(i + 1, axis=0); h5cache["y"][i, :] = y
        h5cache.flush()
        call_count[0] += 1
        if call_count[0] % 5 == 0:
            print(f"  matvec {call_count[0]}", flush=True)
        return y

    H_op = LinearOperator((N, N), matvec=matvec, dtype=float)
    try:
        # GHEP: solve H v = λ A_block v with v A_block-orthonormal.
        # Minv reuses the pre-factorized A_lu, avoiding LU per spsolve.
        vals, vecs = eigsh(H_op, k=min(K_LEADING, N - 2), M=A_block,
                            Minv=Minv_op, which="LM", maxiter=300,
                            tol=1e-6, v0=v0)
    finally:
        h5cache.close()

    idx = np.argsort(-vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    print(f"\nLeading eigenvalues: {vals[:10]}", flush=True)
    n_pos = int(np.sum(vals > 0))
    D_pos = vals[:n_pos] / (1.0 + vals[:n_pos])
    n_eff = float(np.sum(D_pos))
    print(f"Positive modes: {n_pos} / {len(vals)}", flush=True)
    print(f"Effective constrained parameters: {n_eff:.1f}", flush=True)

    # ── Save ───────────────────────────────────────────────────────
    results_dir = DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Eigenvalues (positive modes only, matching eigenvectors.npz)
    eigval_fn = results_dir / "eigenvalues.txt"
    with open(eigval_fn, "w") as f:
        for i, lam in enumerate(vals[:n_pos]):
            f.write(f"{i} {lam:.16e}\n")
    print(f"Saved {eigval_fn}", flush=True)

    # Save eigenvectors as raw numpy (avoids HDF5 crash with many MixedFunctions)
    vecs_fn = results_dir / "eigenvectors.npz"
    np.savez(str(vecs_fn), vecs=vecs[:, :n_pos],
             eigenvalues=vals[:n_pos], n_dof=n)
    print(f"Saved {vecs_fn} ({n_pos} positive modes)", flush=True)

    # Also save the first 100 as MixedFunctions for visualization
    QQ = MixedFunctionSpace([Q_2d, Q_2d])
    eig_fn = DATA_DIR / "mesh" / "eigendec.h5"
    n_save_hdf5 = min(100, n_pos)
    with CheckpointFile(str(eig_fn), "w") as chk:
        chk.save_mesh(mesh_2d)
        for i in range(n_save_hdf5):
            mode = Function(QQ, name=f"mode_{i:04d}")
            mode.sub(0).dat.data[:] = vecs[:n, i]
            mode.sub(1).dat.data[:] = vecs[n:, i]
            chk.save_function(mode)
    print(f"Saved {eig_fn} ({n_save_hdf5} modes as MixedFunctions)", flush=True)

    # Prior hyperparameters (Recinos convention, no 1/N)
    prior_fn = results_dir / "prior_params.json"
    with open(prior_fn, "w") as f:
        json.dump({
            "delta": delta_val,
            "gamma_A": gamma_A_val,
            "gamma_C": gamma_C_val,
            "area": area_val, "N_vel": N_vel_val,
            "DELTA_PRIOR": DELTA_PRIOR,
            "GAMMA_A": GAMMA_A, "GAMMA_C": GAMMA_C,
            "ELL": ELL, "K_LEADING": K_LEADING,
            "EPS_FD": EPS_FD, "n_eff": n_eff,
            "normalization": "recinos_no_1_over_N",
        }, f, indent=2)
    print(f"Saved {prior_fn}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
