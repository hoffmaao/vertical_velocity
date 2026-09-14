r"""Test pyadjoint's ReducedFunctional.hessian() on hybrid + ApRES setup.

Same Hv comparison as test_cached_hessian.py but using firedrake.adjoint /
pyadjoint instead of tlm_adjoint.
"""
import argparse
import numpy as np
import h5py
import netCDF4 as nc
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, ExtrudedMesh, VertexOnlyMesh,
    inner, grad, max_value, exp, dx, conditional, assemble,
)
from firedrake.adjoint import (
    continue_annotation, pause_annotation, get_working_tape,
    ReducedFunctional, Control,
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
    ap.add_argument("--component", choices=["vel", "apres", "both"], default="vel")
    args = ap.parse_args()
    print(f"Testing pyadjoint Hessian with component: {args.component}", flush=True)

    # ── Setup (copy of test_cached_hessian.py setup) ──
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

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

    with CheckpointFile(str(MAP_FILE), "r") as chk:
        m_inv = chk.load_mesh()
        θA_map_loaded = chk.load_function(m_inv, "log_fluidity")
        θC_map_loaded = chk.load_function(m_inv, "log_friction")
    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    θ_A.dat.data[:] = θA_map_loaded.dat.data_ro
    θ_C.dat.data[:] = θC_map_loaded.dat.data_ro

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

    # ───────────────────────────────────────────────────────────────
    # Tape the forward with pyadjoint
    # ───────────────────────────────────────────────────────────────
    tape = get_working_tape()
    tape.clear_tape()
    continue_annotation()

    A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                 fluidity=A, friction=C)
    J = None
    if include_vel:
        u_interp = Function(Δ_vel).interpolate(u[0])
        v_interp = Function(Δ_vel).interpolate(u[1])
        δu, δv = u_interp - u_o, v_interp - v_o
        vel_form = 0.5 / Constant(float(N_vel)) \
                   * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx
        J = assemble(vel_form)
    if include_apres:
        eps_h = icepack.models.hybrid.horizontal_strain_rate(
            velocity=u, thickness=h, surface=s)
        eps_mod = Function(Δ).interpolate(-(eps_h[0, 0] + eps_h[1, 1]))
        apres_form = 0.5 / Constant(float(max(N_apr, 1))) \
                     * ((eps_mod - eps_obs_f) / σ_eps_f) ** 2 * dx
        J_a = assemble(apres_form)
        J = J_a if J is None else J + J_a
    pause_annotation()

    print(f"J = {float(J):.6e}", flush=True)
    print(f"Tape blocks: {len(tape.get_blocks())}", flush=True)

    controls = [Control(θ_A), Control(θ_C)]
    J_red = ReducedFunctional(J, controls)

    # Test gradient first
    print("\n[pyadjoint] gradient...", flush=True)
    dJ = J_red.derivative()
    g_A_norm = np.linalg.norm(dJ[0].dat.data_ro)
    g_C_norm = np.linalg.norm(dJ[1].dat.data_ro)
    print(f"     ||dJ/dθ_A|| = {g_A_norm:.4e}", flush=True)
    print(f"     ||dJ/dθ_C|| = {g_C_norm:.4e}", flush=True)
    print(f"     |grad| = {np.sqrt(g_A_norm**2 + g_C_norm**2):.4e}", flush=True)

    # Hessian action
    n = θ_A.dat.data.shape[0]
    rng = np.random.default_rng(0)
    v_A = Function(Q_lift); v_A.dat.data[:] = rng.standard_normal(n)
    v_C = Function(Q_lift); v_C.dat.data[:] = rng.standard_normal(n)

    print("\n[pyadjoint] Hessian action...", flush=True)
    # Need to call derivative() first to set up tlm tape evaluation
    Hv = J_red.hessian([v_A, v_C])
    hv_A_norm = np.linalg.norm(Hv[0].dat.data_ro)
    hv_C_norm = np.linalg.norm(Hv[1].dat.data_ro)
    print(f"     ||Hv_A|| = {hv_A_norm:.4e}", flush=True)
    print(f"     ||Hv_C|| = {hv_C_norm:.4e}", flush=True)

    if hv_A_norm > 1e-3 and hv_C_norm > 1e-3:
        print("RESULT: ✓ pyadjoint Hessian gives non-trivial result")
    else:
        print("RESULT: ✗ pyadjoint Hessian also near-zero — same problem")


if __name__ == "__main__":
    main()
