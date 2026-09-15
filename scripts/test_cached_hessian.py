r"""Test whether tlm_adjoint.CachedHessian works on our hybrid + ApRES setup.

Compares one Hessian-vector product computed two ways:
  (a) finite-difference of the gradient (current run_eigendec.py approach)
  (b) CachedHessian.action() — exact second-order via tlm_adjoint

Agreement to ~EPS_FD (1e-5 relative) means we can drop the FD path and
eliminate the spurious-negative-eigenvalue artefacts. Disagreement (or a
crash) tells us which operation along the tape lacks TLM support.

Usage:
    OMP_NUM_THREADS=4 python test_cached_hessian.py [--component vel|apres|both]
"""
import argparse
import numpy as np
import h5py
import netCDF4 as nc
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, ExtrudedMesh, VertexOnlyMesh,
    sqrt, inner, grad, max_value, exp, dx, conditional, assemble,
    TestFunction, TrialFunction,
)
from firedrake.adjoint import *  # noqa
from tlm_adjoint.firedrake import (
    reset_manager, start_manager, stop_manager,
    compute_gradient, Functional, CachedHessian,
    configure_checkpointing,
    Function as TLMFunction,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)
import icepack
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
MAP_FILE = DATA_DIR / "mesh" / "inversion_hires_apres_vd1.h5"
APRES_FILE = DATA_DIR / "data" / "apres_legendre_fits.h5"
VEL_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/velocity/"
                "antarctica_ice_velocity_450m_v2.nc")

VEL_STEP = 10
DEPTH_MIN = 100.0
DEPTH_MAX = 800.0
EPS_FD = 1e-5

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", choices=["vel", "apres", "both"], default="both",
                    help="Which misfit term(s) to include in J")
    args = ap.parse_args()
    print(f"Testing CachedHessian with misfit component(s): {args.component}",
          flush=True)

    # ── Load mesh ──────────────────────────────────────────────────
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")
    print(f"Mesh: {mesh_2d.num_vertices()} verts", flush=True)

    # ── Sparse velocity VOM ──
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

    # ── Extrude ──
    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)
    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    u0 = Function(V).project(icepack.utilities.lift3d(u_obs_2d, V_lift))

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
    print(f"Velocity VOM: {N_vel} points", flush=True)

    # ── ApRES VOM ──
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

    # ── Load MAP ──
    with CheckpointFile(str(MAP_FILE), "r") as chk:
        m_inv = chk.load_mesh()
        θA_map_loaded = chk.load_function(m_inv, "log_fluidity")
        θC_map_loaded = chk.load_function(m_inv, "log_friction")
    # static=True flags these as tape-input leaves for tlm_adjoint TLM
    θ_A = TLMFunction(Q_lift, name="log_fluidity", static=True)
    θ_C = TLMFunction(Q_lift, name="log_friction", static=True)
    θ_A.dat.data[:] = θA_map_loaded.dat.data_ro
    θ_C.dat.data[:] = θC_map_loaded.dat.data_ro
    θA_map_vals = θ_A.dat.data_ro.copy()
    θC_map_vals = θ_C.dat.data_ro.copy()
    print(f"MAP loaded", flush=True)

    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q_lift).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
                    Constant(1.0), Constant(0.0)))

    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    A_map = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_map = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    u0 = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                  fluidity=A_map, friction=C_map)
    print("Primed at MAP", flush=True)

    include_vel = args.component in ("vel", "both")
    include_apres = args.component in ("apres", "both")

    def build_misfit(θ_A_in, θ_C_in):
        """Assemble J(θ_A, θ_C). Used inside both the FD path and the tape."""
        A = Function(Q_lift).interpolate(A_0 * exp(θ_A_in))
        C = Function(Q_lift).interpolate(C_0 * exp(θ_C_in * grounded_mask))
        u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                     fluidity=A, friction=C)
        J = Functional(name="J")
        first = True
        if include_vel:
            u_interp = Function(Δ_vel).interpolate(u[0])
            v_interp = Function(Δ_vel).interpolate(u[1])
            δu, δv = u_interp - u_o, v_interp - v_o
            term = 0.5 / Constant(float(N_vel)) \
                   * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx
            (J.assign if first else J.addto)(term); first = False
        if include_apres:
            eps_h = icepack.models.hybrid.horizontal_strain_rate(
                velocity=u, thickness=h, surface=s)
            eps_mod = Function(Δ).interpolate(-(eps_h[0, 0] + eps_h[1, 1]))
            term = 0.5 / Constant(float(max(N_apr, 1))) \
                   * ((eps_mod - eps_obs_f) / σ_eps_f) ** 2 * dx
            (J.assign if first else J.addto)(term); first = False
        return J

    # ── Random test direction ──
    n = θ_A.dat.data.shape[0]
    rng = np.random.default_rng(0)
    vA_arr = rng.standard_normal(n)
    vC_arr = rng.standard_normal(n)

    # ───────────────────────────────────────────────────────────────
    # (a) FD Hessian-vector product
    # ───────────────────────────────────────────────────────────────
    def grad_at(θA_vals, θC_vals):
        θ_A.dat.data[:] = θA_vals
        θ_C.dat.data[:] = θC_vals
        reset_manager(); start_manager()
        J = build_misfit(θ_A, θ_C)
        dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
        stop_manager()
        return dJ_A.dat.data_ro.copy(), dJ_C.dat.data_ro.copy()

    print("\n[FD] gradient at MAP...", flush=True)
    g0_A, g0_C = grad_at(θA_map_vals, θC_map_vals)
    print(f"     |grad| = {np.sqrt((g0_A**2).sum() + (g0_C**2).sum()):.4e}",
          flush=True)
    print("[FD] gradient at MAP + ε·v...", flush=True)
    gp_A, gp_C = grad_at(θA_map_vals + EPS_FD * vA_arr,
                         θC_map_vals + EPS_FD * vC_arr)
    hv_A_fd = (gp_A - g0_A) / EPS_FD
    hv_C_fd = (gp_C - g0_C) / EPS_FD
    print(f"     ||Hv_A_fd|| = {np.linalg.norm(hv_A_fd):.4e}", flush=True)
    print(f"     ||Hv_C_fd|| = {np.linalg.norm(hv_C_fd):.4e}", flush=True)

    # ───────────────────────────────────────────────────────────────
    # (b) CachedHessian Hessian-vector product
    # ───────────────────────────────────────────────────────────────
    print("\n[TLM] taping at MAP (with memory checkpoint, no drop)...", flush=True)
    θ_A.dat.data[:] = θA_map_vals
    θ_C.dat.data[:] = θC_map_vals
    reset_manager()
    configure_checkpointing("memory", {"drop_references": False})
    start_manager()
    J_tape = build_misfit(θ_A, θ_C)
    stop_manager()

    print("[TLM] building CachedHessian...", flush=True)
    H = CachedHessian(J_tape)

    v_A = Function(Q_lift); v_A.dat.data[:] = vA_arr
    v_C = Function(Q_lift); v_C.dat.data[:] = vC_arr
    print("[TLM] H.action([θ_A, θ_C], [v_A, v_C])...", flush=True)
    _, _, ddJ = H.action([θ_A, θ_C], [v_A, v_C])
    hv_A_tlm = ddJ[0].riesz_representation("L2").dat.data_ro.copy()
    hv_C_tlm = ddJ[1].riesz_representation("L2").dat.data_ro.copy()
    print(f"     ||Hv_A_tlm|| = {np.linalg.norm(hv_A_tlm):.4e}", flush=True)
    print(f"     ||Hv_C_tlm|| = {np.linalg.norm(hv_C_tlm):.4e}", flush=True)

    # ── Comparison ──
    rel_A = np.linalg.norm(hv_A_tlm - hv_A_fd) / np.linalg.norm(hv_A_fd)
    rel_C = np.linalg.norm(hv_C_tlm - hv_C_fd) / np.linalg.norm(hv_C_fd)
    print(f"\nRelative diff θ_A: {rel_A:.3e}")
    print(f"Relative diff θ_C: {rel_C:.3e}")
    print(f"Expected: ~{EPS_FD:.0e} (FD truncation error)")
    if max(rel_A, rel_C) < 1e-3:
        print("RESULT: ✓ CachedHessian matches FD — TLM works on this misfit")
    else:
        print("RESULT: ✗ disagreement larger than expected — investigate")


if __name__ == "__main__":
    main()
