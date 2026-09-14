r"""SSA inversion for Thwaites using sparse velocity observations with uncertainties.

Follows the icepack sparse data approach (how-to/04-sparse-data):
  - VertexOnlyMesh for sparse point evaluation
  - Per-point velocity uncertainties from MEaSUREs STDX/STDY
  - Flotation-masked Weertman friction (zero on floating ice)

Usage:
    python inversion_ssa.py
"""
import numpy as np
import firedrake
import icepack
import rasterio
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    sqrt, inner, grad, max_value, dx, assemble,
)
from firedrake.__future__ import interpolate
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    weertman_sliding_law as m,
)
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
OUTPUT_FILE = DATA_DIR / "mesh" / "inversion_ssa.h5"
BEDMACHINE = Path(
    "/media/andrew/wd1/projects/ismip7/data/bedmachine/"
    "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
)
VEL_FILE = Path(
    "/media/andrew/wd1/projects/ismip7/data/velocity/"
    "antarctica_ice_velocity_450m_v2.nc"
)

# Regularization
L_REG = Constant(10e3)
GAMMA = Constant(1.0)

# Optimization
MAX_ITER = 500
GRADIENT_TOL = 1e-4
STEP_TOL = 1e-1

# Solver
SOLVER_PARAMS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_max_it": 200,
    "snes_rtol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


# ═══════════════════════════════════════════════════════════════════
# Friction: Weertman modulated by flotation
# ═══════════════════════════════════════════════════════════════════

def friction(**kwargs):
    r"""Weertman friction with flotation ramp.

    Following icepack tutorial (04-synthetic-ice-stream-xy):
      p_W = ρ_W * g * max(0, h - s)
      p_I = ρ_I * g * h
      ϕ = 1 - p_W / p_I

    ϕ = 1 when grounded (s >> h - freeboard), ϕ → 0 at flotation.
    This provides a smooth transition rather than a hard cutoff.
    """
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
    print("SSA Inversion with sparse velocity + uncertainties", flush=True)
    print("=" * 60, flush=True)

    # ── 1. Load mesh ──
    print("\n1. Loading mesh...", flush=True)
    with firedrake.CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        h = chk.load_function(mesh, "thickness")
        b = chk.load_function(mesh, "bed")
        s = chk.load_function(mesh, "surface")
        u_obs = chk.load_function(mesh, "velocity")

    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    print(f"   Mesh: {mesh.num_vertices()} verts, {mesh.num_cells()} cells", flush=True)

    # ── 2. Build sparse velocity observation points ──
    # Subsample MEaSUREs grid to ~5km spacing over the domain
    print("\n2. Building sparse velocity observations with uncertainties...", flush=True)
    import netCDF4 as nc
    ds = nc.Dataset(str(VEL_FILE))
    x_vel = ds.variables["x"][:]
    y_vel = ds.variables["y"][:]

    # Domain bounds
    coords = mesh.coordinates.dat.data_ro
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()

    # Subsample: every ~10 pixels = ~4.5km spacing
    step = 10
    ix0 = max(0, np.searchsorted(x_vel, xmin) - 5)
    ix1 = min(len(x_vel), np.searchsorted(x_vel, xmax) + 5)
    iy0 = max(0, np.searchsorted(-y_vel, -ymax) - 5)
    iy1 = min(len(y_vel), np.searchsorted(-y_vel, -ymin) + 5)

    x_sub = x_vel[ix0:ix1:step]
    y_sub = y_vel[iy0:iy1:step]
    sl = np.s_[iy0:iy1:step, ix0:ix1:step]

    vx = np.array(ds.variables["VX"][sl]).astype(float)
    vy = np.array(ds.variables["VY"][sl]).astype(float)
    stdx = np.array(ds.variables["STDX"][sl]).astype(float)
    stdy = np.array(ds.variables["STDY"][sl]).astype(float)
    ds.close()

    # Replace NaN/masked with 0
    vx = np.nan_to_num(vx, nan=0.0)
    vy = np.nan_to_num(vy, nan=0.0)
    stdx = np.nan_to_num(stdx, nan=1e6)
    stdy = np.nan_to_num(stdy, nan=1e6)

    # Build point coordinates (y, x grid -> x, y points)
    YY, XX = np.meshgrid(y_sub, x_sub, indexing="ij")
    xs_flat = XX.ravel()
    ys_flat = YY.ravel()
    vx_flat = vx.ravel()
    vy_flat = vy.ravel()
    stdx_flat = stdx.ravel()
    stdy_flat = stdy.ravel()

    # Filter: only keep points inside the mesh domain and with valid data
    from scipy.spatial import Delaunay
    tri = Delaunay(coords)
    inside = tri.find_simplex(np.column_stack([xs_flat, ys_flat])) >= 0
    has_data = (stdx_flat < 1e5) & (stdy_flat < 1e5) & (stdx_flat > 0) & (stdy_flat > 0)
    keep = inside & has_data

    xs_obs = xs_flat[keep]
    ys_obs = ys_flat[keep]
    vx_obs = vx_flat[keep]
    vy_obs = vy_flat[keep]
    σx_obs = np.maximum(stdx_flat[keep], 1.0)  # floor at 1 m/yr
    σy_obs = np.maximum(stdy_flat[keep], 1.0)

    N_obs = len(xs_obs)
    print(f"   Sparse observations: {N_obs} points (~{step*450/1000:.1f}km spacing)", flush=True)
    print(f"   σ range: x=[{σx_obs.min():.1f}, {σx_obs.max():.1f}], "
          f"y=[{σy_obs.min():.1f}, {σy_obs.max():.1f}] m/yr", flush=True)

    # Create VertexOnlyMesh
    point_set = firedrake.VertexOnlyMesh(
        mesh, np.column_stack([xs_obs, ys_obs]),
        missing_points_behaviour="warn",
    )
    Δ = FunctionSpace(point_set, "DG", 0)

    # Map observations onto the point set (handling reordering)
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
    print(f"   Points on VertexOnlyMesh: {N}", flush=True)

    # ── 3. Set up model and solver ──
    print("\n3. Creating IceStream model...", flush=True)
    model = icepack.models.IceStream(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model,
        dirichlet_ids=[2],
        ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS,
    )

    # Initial velocity guess (with floor)
    u_init = Function(V).interpolate(u_obs)
    u_init.interpolate(
        firedrake.conditional(
            sqrt(inner(u_obs, u_obs)) < 1.0,
            firedrake.as_vector([Constant(1.0), Constant(0.0)]),
            u_obs,
        )
    )

    # ── 4. Controls ──
    print("\n4. Setting up controls...", flush=True)
    from firedrake import exp

    θ_A = Function(Q, name="log_fluidity")   # A = A₀ * exp(θ_A)
    θ_C = Function(Q, name="log_friction")   # C = C₀ * exp(θ_C)
    A_0 = Constant(20.0)
    C_0 = Constant(0.01)

    # Flotation mask for grounded ice
    p_W_mask = ρ_W * g * max_value(0, h - s)
    p_I_mask = ρ_I * g * h
    ϕ_mask = Function(Q).interpolate(1 - p_W_mask / p_I_mask)
    grounded_mask = Function(Q).interpolate(
        firedrake.conditional(ϕ_mask > 0.01, Constant(1.0), Constant(0.0))
    )
    print(f"   Grounded: {(grounded_mask.dat.data > 0.5).sum()}/{len(grounded_mask.dat.data)}", flush=True)

    # Warm-start from previous result if available
    prev_file = DATA_DIR / "mesh" / "inversion_ssa.h5"
    if prev_file.exists():
        with firedrake.CheckpointFile(str(prev_file), "r") as chk:
            mesh_prev = chk.load_mesh()
            θ_A_prev = chk.load_function(mesh_prev, "log_fluidity")
            θ_C_prev = chk.load_function(mesh_prev, "log_friction")
            u_prev = chk.load_function(mesh_prev, "velocity")
        θ_A.dat.data[:] = θ_A_prev.dat.data_ro
        θ_C.dat.data[:] = θ_C_prev.dat.data_ro
        u_init.dat.data[:] = u_prev.dat.data_ro
        print(f"   Warm start: θ_A=[{θ_A.dat.data.min():.2f}, {θ_A.dat.data.max():.2f}], "
              f"θ_C=[{θ_C.dat.data.min():.2f}, {θ_C.dat.data.max():.2f}]", flush=True)
    else:
        print("   Cold start (θ=0)", flush=True)

    # ── 5. Test forward solve ──
    print("\n5. Testing forward solve...", flush=True)
    A_test = Function(Q).interpolate(A_0 * exp(θ_A))
    C_test = Function(Q).interpolate(C_0 * exp(θ_C * grounded_mask))
    u_test = solver.diagnostic_solve(
        velocity=u_init, thickness=h, surface=s,
        fluidity=A_test, friction=C_test,
    )
    speed_test = Function(Q).interpolate(sqrt(inner(u_test, u_test)))
    print(f"   Speed: [{speed_test.dat.data.min():.0f}, {speed_test.dat.data.max():.0f}] m/yr", flush=True)

    # Check initial misfit at sparse points
    u_interp = Function(Δ).interpolate(u_test[0])
    v_interp = Function(Δ).interpolate(u_test[1])
    δu, δv = u_interp - u_o, v_interp - v_o
    init_misfit = assemble(0.5 / Constant(float(N)) * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx)
    print(f"   Initial χ²/N: {float(init_misfit):.2f}", flush=True)

    # ── 6. Define objective ──
    print("\n6. Defining objective...", flush=True)
    area = assemble(Constant(1.0) * dx(mesh))

    def simulation(controls):
        θ_A_c, θ_C_c = controls
        A = Function(Q).interpolate(A_0 * exp(θ_A_c))
        # Mask θ_C to zero on floating ice so C = C_0 there
        C = Function(Q).interpolate(C_0 * exp(θ_C_c * grounded_mask))
        return solver.diagnostic_solve(
            velocity=u_init, thickness=h, surface=s,
            fluidity=A, friction=C,
        )

    def loss_functional(u):
        u_interp = Function(Δ).interpolate(u[0])
        v_interp = Function(Δ).interpolate(u[1])
        δu = u_interp - u_o
        δv = v_interp - v_o
        return 0.5 / Constant(float(N)) * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx

    def regularization(controls):
        θ_A_c, θ_C_c = controls
        Θ = Constant(1.0)
        reg_A = 0.5 / area * (L_REG / Θ)**2 * inner(grad(θ_A_c), grad(θ_A_c)) * dx
        reg_C = 0.5 / area * (L_REG / Θ)**2 * inner(grad(θ_C_c), grad(θ_C_c)) * dx
        return reg_A + reg_C

    # ── 7. Solve ──
    print(f"\n7. Solving (max {MAX_ITER} iterations)...", flush=True)
    problem = StatisticsProblem(
        simulation=simulation,
        loss_functional=loss_functional,
        regularization=regularization,
        controls=[θ_A, θ_C],
    )

    estimator = MaximumProbabilityEstimator(
        problem,
        algorithm="bfgs",
        memory=20,
        gradient_tolerance=1e-6,
        step_tolerance=1e-4,
        max_iterations=MAX_ITER,
        verbose=True,
    )
    θ_A_opt, θ_C_opt = estimator.solve()

    # ── 8. Results ──
    print("\n8. Results...", flush=True)

    # Re-evaluate velocity at optimal controls
    A_opt = Function(Q).interpolate(A_0 * exp(θ_A_opt))
    C_opt = Function(Q).interpolate(C_0 * exp(θ_C_opt))
    u_opt = solver.diagnostic_solve(
        velocity=u_init, thickness=h, surface=s,
        fluidity=A_opt, friction=C_opt,
    )
    speed_opt = Function(Q, name="speed_model").interpolate(sqrt(inner(u_opt, u_opt)))
    speed_obs_fn = Function(Q, name="speed_obs").interpolate(sqrt(inner(u_obs, u_obs)))

    # Final misfit at sparse points
    u_f = Function(Δ).interpolate(u_opt[0])
    v_f = Function(Δ).interpolate(u_opt[1])
    final_misfit = assemble(0.5 / Constant(float(N)) * (((u_f - u_o) / σ_x)**2 + ((v_f - v_o) / σ_y)**2) * dx)

    print(f"   Model speed: [{speed_opt.dat.data.min():.0f}, {speed_opt.dat.data.max():.0f}] m/yr", flush=True)
    print(f"   Obs speed:   [{speed_obs_fn.dat.data.min():.0f}, {speed_obs_fn.dat.data.max():.0f}] m/yr", flush=True)
    print(f"   θ_A range: [{θ_A_opt.dat.data.min():.2f}, {θ_A_opt.dat.data.max():.2f}]", flush=True)
    print(f"   θ_C range: [{θ_C_opt.dat.data.min():.2f}, {θ_C_opt.dat.data.max():.2f}]", flush=True)
    print(f"   Final χ²/N: {float(final_misfit):.2f} (initial: {float(init_misfit):.2f})", flush=True)

    with firedrake.CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh)
        chk.save_function(θ_A_opt, name="log_fluidity")
        chk.save_function(θ_C_opt, name="log_friction")
        chk.save_function(speed_opt, name="speed_model")
        chk.save_function(speed_obs_fn, name="speed_obs")
        chk.save_function(u_opt, name="velocity")
    print(f"   Saved {OUTPUT_FILE}", flush=True)

    # ── 9. Plot ──
    print("\n9. Plotting...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax = axes[0, 0]
    tc = firedrake.tripcolor(speed_obs_fn, axes=ax, cmap="inferno", vmin=0, vmax=3000)
    fig.colorbar(tc, ax=ax, fraction=0.046, label="m/yr")
    ax.set_title("Observed speed"); ax.set_aspect("equal")

    ax = axes[0, 1]
    tc = firedrake.tripcolor(speed_opt, axes=ax, cmap="inferno", vmin=0, vmax=3000)
    fig.colorbar(tc, ax=ax, fraction=0.046, label="m/yr")
    ax.set_title("Model speed (SSA)"); ax.set_aspect("equal")

    ax = axes[1, 0]
    misfit_fn = Function(Q).interpolate(speed_opt - speed_obs_fn)
    tc = firedrake.tripcolor(misfit_fn, axes=ax, cmap="RdBu_r", vmin=-200, vmax=200)
    fig.colorbar(tc, ax=ax, fraction=0.046, label="m/yr")
    ax.set_title("Speed misfit"); ax.set_aspect("equal")

    ax = axes[1, 1]
    tc = firedrake.tripcolor(θ_A_opt, axes=ax, cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title("log-fluidity (θ$_A$)"); ax.set_aspect("equal")

    fig.suptitle(f"SSA Inversion (χ²/N: {float(init_misfit):.1f} → {float(final_misfit):.1f})",
                 fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(str(DATA_DIR / "figures" / "inversion_ssa.png"), dpi=150)
    print(f"   Saved figures/inversion_ssa.png", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
