r"""SSA (IceStream) inversion for Thwaites, Recinos-prior convention.

Companion to inversion_hires_apres.py (Hybrid) for a clean SSA-vs-Hybrid UQ
comparison. Same mesh, controls (θ_A,θ_C on CG1), velocity data and per-point σ,
and the SAME Recinos prior + raw-Σχ² misfit convention — but the SSA/IceStream
model with Schoof regularized-Coulomb friction (no ApRES; SSA has no vertical
strain). Warm-starts from inversion_ssa_coulomb.h5.

Prior:  R(θ) = 0.5·(δ ∫θ² + γ ∫|∇θ|²) dx,  γ = δ·ELL²,  ELL=1km (matches Hybrid MAP).
Misfit: J = 0.5·Σ_q [(u_x-obs_x)²/σ_x² + (u_y-obs_y)²/σ_y²]   (VOM, no 1/N).
"""
import numpy as np
import netCDF4 as nc
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, VertexOnlyMesh,
    sqrt, inner, grad, max_value, exp, dx, conditional, TestFunction, assemble,
)
from firedrake.adjoint import *  # noqa
from tlm_adjoint.firedrake import (
    reset_manager, start_manager, stop_manager, compute_gradient, Functional,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
    weertman_sliding_law as m,
)
from scipy.optimize import minimize as scipy_minimize
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
VEL_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/velocity/"
                "antarctica_ice_velocity_450m_v2.nc")

DELTA = 1.0e-5            # prior amplitude (Recinos elbow)
ELL = 1000.0             # prior correlation length (m) — matches Hybrid MAP regularization
GAMMA = DELTA * ELL**2   # = 10
VEL_STEP = 1             # full 450 m MEaSUREs density (matches Hybrid)
MAX_ITER = 500
A_0 = Constant(20.0)
U_0 = Constant(300.0)                       # Coulomb threshold speed (m/yr)

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 250, "snes_rtol": 1e-6,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


def friction_coulomb(**kwargs):
    """Schoof regularized-Coulomb friction (icepack 04-synthetic-ice-stream-xy)."""
    u, h, s, τ_0 = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W / ρ_I * max_value(0, h - s) / h
    U = sqrt(inner(u, u))
    return τ_0 * ϕ * ((U_0 ** (1 / m + 1) + U ** (1 / m + 1)) ** (m / (m + 1)) - U_0)


def friction_weertman(**kwargs):
    """Weertman bed friction modulated by flotation (matches the Hybrid pipeline)."""
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


# friction → (function, C_0, warm-start MAP, output suffix)
FRICTION = {
    "coulomb": (friction_coulomb, 0.5, "inversion_ssa_coulomb.h5", "_recinos"),
    "weertman": (friction_weertman, 0.01, "inversion_ssa.h5", "_weertman"),
}


def load_sparse_velocity(mesh):
    from scipy.spatial import Delaunay
    ds = nc.Dataset(str(VEL_FILE))
    x_vel = ds.variables["x"][:]; y_vel = ds.variables["y"][:]
    coords = mesh.coordinates.dat.data_ro
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    ix0 = max(0, np.searchsorted(x_vel, xmin) - 5); ix1 = min(len(x_vel), np.searchsorted(x_vel, xmax) + 5)
    iy0 = max(0, np.searchsorted(-y_vel, -ymax) - 5); iy1 = min(len(y_vel), np.searchsorted(-y_vel, -ymin) + 5)
    sl = np.s_[iy0:iy1:VEL_STEP, ix0:ix1:VEL_STEP]
    x_sub, y_sub = x_vel[ix0:ix1:VEL_STEP], y_vel[iy0:iy1:VEL_STEP]
    vx = np.nan_to_num(np.array(ds.variables["VX"][sl]).astype(float), nan=0.0)
    vy = np.nan_to_num(np.array(ds.variables["VY"][sl]).astype(float), nan=0.0)
    stdx = np.nan_to_num(np.array(ds.variables["STDX"][sl]).astype(float), nan=1e6)
    stdy = np.nan_to_num(np.array(ds.variables["STDY"][sl]).astype(float), nan=1e6)
    ds.close()
    YY, XX = np.meshgrid(y_sub, x_sub, indexing="ij")
    xs, ys = XX.ravel(), YY.ravel()
    tri = Delaunay(coords)
    inside = tri.find_simplex(np.column_stack([xs, ys])) >= 0
    has = (stdx.ravel() < 1e5) & (stdy.ravel() < 1e5) & (stdx.ravel() > 0) & (stdy.ravel() > 0)
    keep = inside & has
    return (xs[keep], ys[keep], vx.ravel()[keep], vy.ravel()[keep],
            np.maximum(stdx.ravel()[keep], 1.0), np.maximum(stdy.ravel()[keep], 1.0))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--friction", choices=["coulomb", "weertman"], default="coulomb")
    args = ap.parse_args()
    fric, C0v, warm_name, suffix = FRICTION[args.friction]
    C_0 = Constant(C0v)
    WARM_FILE = DATA_DIR / "mesh" / warm_name
    OUTPUT_FILE = DATA_DIR / "mesh" / f"inversion_ssa{suffix}.h5"
    CHECKPOINT_FILE = DATA_DIR / "mesh" / f"checkpoint_ssa{suffix}.npz"
    print("=" * 60, flush=True)
    print(f"SSA (IceStream) inversion — Recinos prior, {args.friction} friction (C_0={C0v})", flush=True)
    print(f"  δ={DELTA:.1e}, ELL={ELL:.0f} m, γ={GAMMA:.1f}; VEL_STEP={VEL_STEP}", flush=True)
    print("=" * 60, flush=True)

    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        h = chk.load_function(mesh, "thickness")
        s = chk.load_function(mesh, "surface")
        u_obs = chk.load_function(mesh, "velocity")
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    print(f"Mesh: {mesh.num_vertices()} verts", flush=True)

    xs, ys, vx, vy, σx, σy = load_sparse_velocity(mesh)
    print(f"Sparse velocity: {len(xs)} points", flush=True)
    vom = VertexOnlyMesh(mesh, np.column_stack([xs, ys]), missing_points_behaviour="warn")
    Δ = FunctionSpace(vom, "DG", 0)
    Δ_in = FunctionSpace(vom.input_ordering, "DG", 0)

    def to_vom(vals):
        fin = Function(Δ_in); fin.dat.data[:] = vals[:len(fin.dat.data)]
        return Function(Δ).interpolate(fin)
    u_o, v_o, σ_x, σ_y = to_vom(vx), to_vom(vy), to_vom(σx), to_vom(σy)
    N_vel = len(u_o.dat.data)
    print(f"Velocity VOM: {N_vel} points", flush=True)

    θ_A = Function(Q, name="log_fluidity")
    θ_C = Function(Q, name="log_friction")
    u0 = Function(V)
    with CheckpointFile(str(WARM_FILE), "r") as wchk:
        mw = wchk.load_mesh()
        θ_A.dat.data[:] = np.clip(wchk.load_function(mw, "log_fluidity").dat.data_ro, -10, 10)
        θ_C.dat.data[:] = np.clip(wchk.load_function(mw, "log_friction").dat.data_ro, -10, 10)
        u0.dat.data[:] = wchk.load_function(mw, "velocity").dat.data_ro
    print(f"Warm start from {WARM_FILE.name} (θ + velocity)", flush=True)

    grounded_mask = Function(Q).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01, Constant(1.0), Constant(0.0)))

    model = icepack.models.IceStream(friction=fric)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc", diagnostic_solver_parameters=SOLVER_PARAMS)

    A_init = Function(Q).interpolate(A_0 * exp(θ_A))
    C_init = Function(Q).interpolate(C_0 * exp(θ_C * grounded_mask))
    u0.assign(solver.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A_init, friction=C_init))
    print("Primed forward solve OK", flush=True)

    DELTA_c, GAMMA_c = Constant(DELTA), Constant(GAMMA)
    v_test = TestFunction(Q)
    iteration = [0]
    state = {"J": None, "g": None, "u": None}
    n = θ_A.dat.data.shape[0]

    def obj_grad(x):
        θ_A.dat.data[:] = x[:n]; θ_C.dat.data[:] = x[n:]
        reset_manager(); start_manager()
        A = Function(Q).interpolate(A_0 * exp(θ_A))
        C = Function(Q).interpolate(C_0 * exp(θ_C * grounded_mask))
        try:
            u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A, friction=C)
        except firedrake.exceptions.ConvergenceError:
            stop_manager(); iteration[0] += 1
            print(f"  iter {iteration[0]:3d}: FWD FAIL — penalty", flush=True)
            return 10.0 * (state["J"] or 1.0), (state["g"].copy() if state["g"] is not None else np.zeros(2 * n))
        u_i = Function(Δ).interpolate(u[0]); v_i = Function(Δ).interpolate(u[1])
        J = Functional(name="J_vel")
        J.assign(0.5 * (((u_i - u_o) / σ_x) ** 2 + ((v_i - v_o) / σ_y) ** 2) * dx)
        J_vel = float(J)
        try:
            dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
        except Exception as e:
            stop_manager(); iteration[0] += 1
            print(f"  iter {iteration[0]:3d}: ADJ FAIL {type(e).__name__} — penalty", flush=True)
            return 10.0 * (state["J"] or 1.0), (state["g"].copy() if state["g"] is not None else np.zeros(2 * n))
        stop_manager()
        reg_A = float(assemble(0.5 * (DELTA_c * inner(θ_A, θ_A) + GAMMA_c * inner(grad(θ_A), grad(θ_A))) * dx))
        reg_C = float(assemble(0.5 * (DELTA_c * inner(θ_C, θ_C) + GAMMA_c * inner(grad(θ_C), grad(θ_C))) * dx))
        g_A = dJ_A.dat.data_ro.copy() + assemble(
            (DELTA_c * inner(θ_A, v_test) + GAMMA_c * inner(grad(θ_A), grad(v_test))) * dx).dat.data_ro
        g_C = dJ_C.dat.data_ro.copy() + assemble(
            (DELTA_c * inner(θ_C, v_test) + GAMMA_c * inner(grad(θ_C), grad(v_test))) * dx).dat.data_ro
        total = J_vel + reg_A + reg_C
        full_g = np.concatenate([g_A, g_C])
        iteration[0] += 1
        print(f"  iter {iteration[0]:3d}: Jvel={J_vel:.3e} reg={reg_A+reg_C:.3e} "
              f"|g|={np.sqrt((full_g**2).sum()):.3e}", flush=True)
        state.update(J=total, g=full_g.copy(), u=u.copy(deepcopy=True)); u0.assign(u)
        if iteration[0] % 10 == 0:
            np.savez(str(CHECKPOINT_FILE), x=x, iteration=iteration[0])
        return total, full_g

    x0 = np.concatenate([θ_A.dat.data_ro.copy(), θ_C.dat.data_ro.copy()])
    if CHECKPOINT_FILE.exists():
        cd = np.load(str(CHECKPOINT_FILE)); x0 = cd["x"]; iteration[0] = int(cd["iteration"])
        print(f"Resuming from checkpoint at iter {iteration[0]}", flush=True)
    print(f"\nL-BFGS-B ({2*n} DOFs, max {MAX_ITER} iter)...", flush=True)
    bounds = [(-8.0, 8.0)] * (2 * n)   # keep θ in a regime where the stiff Coulomb forward converges
    result = scipy_minimize(obj_grad, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                            options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": 1e-8, "disp": True})
    print(f"\nResult: {result.message}", flush=True)

    θ_A.dat.data[:] = result.x[:n]; θ_C.dat.data[:] = result.x[n:]
    print(f"θ_A: [{θ_A.dat.data.min():.2f}, {θ_A.dat.data.max():.2f}]", flush=True)
    print(f"θ_C: [{θ_C.dat.data.min():.2f}, {θ_C.dat.data.max():.2f}]", flush=True)
    A_opt = Function(Q).interpolate(A_0 * exp(θ_A))
    C_opt = Function(Q).interpolate(C_0 * exp(θ_C * grounded_mask))
    try:
        u_opt = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A_opt, friction=C_opt)
    except firedrake.exceptions.ConvergenceError:
        u_opt = u0
    speed_model = Function(Q, name="speed_model").interpolate(sqrt(inner(u_opt, u_opt)))
    speed_obs = Function(Q, name="speed_obs").interpolate(sqrt(inner(u_obs, u_obs)))
    u_f = Function(Δ).interpolate(u_opt[0]); v_f = Function(Δ).interpolate(u_opt[1])
    chi2 = float(np.mean(((u_f.dat.data_ro - u_o.dat.data_ro) / σ_x.dat.data_ro) ** 2
                         + ((v_f.dat.data_ro - v_o.dat.data_ro) / σ_y.dat.data_ro) ** 2))
    print(f"Final velocity χ²/N (per component avg): {chi2:.2f}", flush=True)
    with CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh)
        chk.save_function(θ_A, name="log_fluidity")
        chk.save_function(θ_C, name="log_friction")
        chk.save_function(speed_model, name="speed_model")
        chk.save_function(speed_obs, name="speed_obs")
        chk.save_function(u_opt, name="velocity")
    print(f"Saved {OUTPUT_FILE}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
