r"""L-curve analysis for SSA Weertman inversion.

Sweeps the regularization length scale L and records the misfit (E)
and regularization (R) at convergence for each L. The optimal L is
at the corner of the L-curve (log E vs log R).

Warm-starts each run from the previous L's result for efficiency.

Usage:
    python lcurve_ssa.py
"""
import numpy as np
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    sqrt, inner, grad, max_value, dx, assemble,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
    weertman_sliding_law as m,
)
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator
from pathlib import Path
import json
import sys

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
OUTPUT_DIR = DATA_DIR / "mesh"

# L-curve sweep values (km)
L_VALUES_KM = [1, 2, 5, 10, 20, 50]

MAX_ITER = 200

SOLVER_PARAMS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_max_it": 200,
    "snes_rtol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def friction(**kwargs):
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    s = kwargs["surface"]
    C = kwargs["friction"]
    p_W = ρ_W * g * max_value(0, h - s)
    p_I = ρ_I * g * h
    ϕ = 1 - p_W / p_I
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def main():
    print("=" * 60, flush=True)
    print("L-curve analysis: SSA Weertman", flush=True)
    print(f"L values: {L_VALUES_KM} km", flush=True)
    print("=" * 60, flush=True)

    # Load mesh
    with firedrake.CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        h = chk.load_function(mesh, "thickness")
        s = chk.load_function(mesh, "surface")
        u_obs = chk.load_function(mesh, "velocity")

    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)

    # Sparse observations
    import netCDF4 as nc
    from scipy.spatial import Delaunay
    ds = nc.Dataset("/media/andrew/wd1/projects/ismip7/data/velocity/antarctica_ice_velocity_450m_v2.nc")
    x_vel = ds.variables["x"][:]; y_vel = ds.variables["y"][:]
    coords = mesh.coordinates.dat.data_ro
    xmin, xmax = coords[:,0].min(), coords[:,0].max()
    ymin, ymax = coords[:,1].min(), coords[:,1].max()
    step = 10
    ix0 = max(0, np.searchsorted(x_vel, xmin) - 5)
    ix1 = min(len(x_vel), np.searchsorted(x_vel, xmax) + 5)
    iy0 = max(0, np.searchsorted(-y_vel, -ymax) - 5)
    iy1 = min(len(y_vel), np.searchsorted(-y_vel, -ymin) + 5)
    sl = np.s_[iy0:iy1:step, ix0:ix1:step]
    x_sub = x_vel[ix0:ix1:step]; y_sub = y_vel[iy0:iy1:step]
    vx = np.nan_to_num(np.array(ds.variables["VX"][sl]).astype(float), nan=0.0)
    vy = np.nan_to_num(np.array(ds.variables["VY"][sl]).astype(float), nan=0.0)
    stdx = np.nan_to_num(np.array(ds.variables["STDX"][sl]).astype(float), nan=1e6)
    stdy = np.nan_to_num(np.array(ds.variables["STDY"][sl]).astype(float), nan=1e6)
    ds.close()
    YY, XX = np.meshgrid(y_sub, x_sub, indexing="ij")
    xs_flat, ys_flat = XX.ravel(), YY.ravel()
    vx_flat, vy_flat = vx.ravel(), vy.ravel()
    stdx_flat, stdy_flat = stdx.ravel(), stdy.ravel()
    tri = Delaunay(coords)
    inside = tri.find_simplex(np.column_stack([xs_flat, ys_flat])) >= 0
    has_data = (stdx_flat < 1e5) & (stdy_flat < 1e5) & (stdx_flat > 0) & (stdy_flat > 0)
    keep = inside & has_data
    xs_obs, ys_obs = xs_flat[keep], ys_flat[keep]
    vx_obs, vy_obs = vx_flat[keep], vy_flat[keep]
    σx_obs = np.maximum(stdx_flat[keep], 1.0)
    σy_obs = np.maximum(stdy_flat[keep], 1.0)

    point_set = firedrake.VertexOnlyMesh(mesh, np.column_stack([xs_obs, ys_obs]),
                                          missing_points_behaviour="warn")
    Δ = FunctionSpace(point_set, "DG", 0)
    Δ_input = FunctionSpace(point_set.input_ordering, "DG", 0)

    def obs_to_point_set(vals):
        f_in = Function(Δ_input)
        f_in.dat.data[:] = vals[:len(f_in.dat.data)]
        f_out = Function(Δ)
        f_out.interpolate(f_in)
        return f_out

    u_o = obs_to_point_set(vx_obs)
    v_o = obs_to_point_set(vy_obs)
    σ_x = obs_to_point_set(σx_obs)
    σ_y = obs_to_point_set(σy_obs)
    N = len(u_o.dat.data)

    # Model setup
    model = icepack.models.IceStream(friction=friction)
    solver = icepack.solvers.FlowSolver(model, dirichlet_ids=[2], ice_front_ids=[1],
                                         diagnostic_solver_type="petsc",
                                         diagnostic_solver_parameters=SOLVER_PARAMS)

    u_init = Function(V).interpolate(u_obs)
    u_init.interpolate(firedrake.conditional(
        sqrt(inner(u_obs, u_obs)) < 1.0,
        firedrake.as_vector([Constant(1.0), Constant(0.0)]), u_obs))

    from firedrake import exp
    A_0 = Constant(20.0)
    C_0 = Constant(0.01)
    area = assemble(Constant(1.0) * dx(mesh))

    # Grounded mask
    ϕ_mask = Function(Q).interpolate(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h))
    grounded_mask = Function(Q).interpolate(
        firedrake.conditional(ϕ_mask > 0.01, Constant(1.0), Constant(0.0)))

    # Warm start from previous result
    θ_A = Function(Q, name="log_fluidity")
    θ_C = Function(Q, name="log_friction")
    prev = DATA_DIR / "mesh" / "inversion_ssa.h5"
    if prev.exists():
        with firedrake.CheckpointFile(str(prev), "r") as chk:
            m2 = chk.load_mesh()
            θ_A.dat.data[:] = chk.load_function(m2, "log_fluidity").dat.data_ro
            θ_C.dat.data[:] = chk.load_function(m2, "log_friction").dat.data_ro
            u_init.dat.data[:] = chk.load_function(m2, "velocity").dat.data_ro
        print(f"Warm start from {prev}", flush=True)

    results = []

    for L_km in L_VALUES_KM:
        L_REG = Constant(L_km * 1e3)
        print(f"\n{'='*40}", flush=True)
        print(f"L = {L_km} km", flush=True)
        print(f"{'='*40}", flush=True)

        # Reset controls to warm start for each L
        θ_A_run = θ_A.copy(deepcopy=True)
        θ_C_run = θ_C.copy(deepcopy=True)

        def simulation(controls):
            θ_A_c, θ_C_c = controls
            A = Function(Q).interpolate(A_0 * exp(θ_A_c))
            C = Function(Q).interpolate(C_0 * exp(θ_C_c * grounded_mask))
            return solver.diagnostic_solve(velocity=u_init, thickness=h, surface=s,
                                            fluidity=A, friction=C)

        def loss_functional(u):
            u_interp = Function(Δ).interpolate(u[0])
            v_interp = Function(Δ).interpolate(u[1])
            δu, δv = u_interp - u_o, v_interp - v_o
            return 0.5 / Constant(float(N)) * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx

        def regularization(controls):
            θ_A_c, θ_C_c = controls
            Θ = Constant(1.0)
            reg_A = 0.5 / area * (L_REG / Θ)**2 * inner(grad(θ_A_c), grad(θ_A_c)) * dx
            reg_C = 0.5 / area * (L_REG / Θ)**2 * inner(grad(θ_C_c), grad(θ_C_c)) * dx
            return reg_A + reg_C

        problem = StatisticsProblem(
            simulation=simulation, loss_functional=loss_functional,
            regularization=regularization, controls=[θ_A_run, θ_C_run])

        estimator = MaximumProbabilityEstimator(
            problem, algorithm="bfgs", memory=20,
            gradient_tolerance=1e-6, step_tolerance=1e-4,
            max_iterations=MAX_ITER, verbose=True)

        try:
            θ_A_opt, θ_C_opt = estimator.solve()
        except Exception as e:
            print(f"  Failed: {e}", flush=True)
            results.append({"L_km": L_km, "E": np.nan, "R": np.nan, "chi2_N": np.nan})
            continue

        # Evaluate misfit and regularization separately
        u_final = estimator.state
        u_f = Function(Δ).interpolate(u_final[0])
        v_f = Function(Δ).interpolate(u_final[1])
        E = float(assemble(0.5 / Constant(float(N)) * (((u_f - u_o)/σ_x)**2 + ((v_f - v_o)/σ_y)**2) * dx))

        R_A = float(assemble(0.5 / area * (L_REG)**2 * inner(grad(θ_A_opt), grad(θ_A_opt)) * dx))
        R_C = float(assemble(0.5 / area * (L_REG)**2 * inner(grad(θ_C_opt), grad(θ_C_opt)) * dx))
        R = R_A + R_C

        print(f"  E (misfit) = {E:.2f}", flush=True)
        print(f"  R (reg)    = {R:.2f} (R_A={R_A:.2f}, R_C={R_C:.2f})", flush=True)
        print(f"  χ²/N       = {E:.2f}", flush=True)
        print(f"  θ_A: [{θ_A_opt.dat.data.min():.2f}, {θ_A_opt.dat.data.max():.2f}]", flush=True)
        print(f"  θ_C: [{θ_C_opt.dat.data.min():.2f}, {θ_C_opt.dat.data.max():.2f}]", flush=True)

        results.append({"L_km": L_km, "E": E, "R": R, "chi2_N": E,
                         "R_A": R_A, "R_C": R_C})

        # Save this L's result
        with firedrake.CheckpointFile(str(OUTPUT_DIR / f"lcurve_L{L_km}km.h5"), "w") as chk:
            chk.save_mesh(mesh)
            chk.save_function(θ_A_opt, name="log_fluidity")
            chk.save_function(θ_C_opt, name="log_friction")

    # Save L-curve data
    with open(str(DATA_DIR / "data" / "lcurve_ssa.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved lcurve_ssa.json", flush=True)

    # Plot L-curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Es = [r["E"] for r in results if np.isfinite(r["E"])]
    Rs = [r["R"] for r in results if np.isfinite(r["R"])]
    Ls = [r["L_km"] for r in results if np.isfinite(r["E"])]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Rs, Es, "ko-", ms=8)
    for L, E, R in zip(Ls, Es, Rs):
        ax.annotate(f"L={L}km", (R, E), textcoords="offset points",
                    xytext=(10, 5), fontsize=10)
    ax.set_xlabel("Regularization R", fontsize=13)
    ax.set_ylabel("Misfit E (χ²/N)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("L-curve: SSA Weertman")
    ax.grid(True, alpha=0.3)
    fig.savefig(str(DATA_DIR / "figures" / "lcurve_ssa.png"), dpi=200, bbox_inches="tight")
    print("Saved figures/lcurve_ssa.png", flush=True)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
