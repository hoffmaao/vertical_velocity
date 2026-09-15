r"""State estimation for Thwaites: invert for fluidity and friction.

Assimilates:
  1. Surface velocity from MEaSUREs (depth-averaged comparison)
  2. ApRES vertical strain rates at sparse point locations

Controls:
  - log_fluidity (θ_A): depth-averaged Glen's law rate factor A = A₀·exp(θ_A)
  - log_friction (θ_C): friction coefficient C = C₀·exp(θ_C)

The vertical strain rate from incompressibility is ε_zz = -div(u). At each
ApRES site we evaluate the model's div(u) at the site's horizontal location
and compare against the observed vertical strain rate.

Uses icepack's StatisticsProblem + MaximumProbabilityEstimator with
the HybridModel forward solve.

Usage:
    python inversion.py
"""
import numpy as np
import h5py
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    sqrt, inner, grad, div, max_value, dx, assemble,
)
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    weertman_sliding_law as m,
)
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
APRES_FILE = DATA_DIR / "data" / "apres_legendre_fits.h5"
OUTPUT_FILE = DATA_DIR / "mesh" / "inversion.h5"

VDEGREE = 1
HDEGREE = 2

# Regularization
L_REG = Constant(10e3)     # smoothing length scale (m)
GAMMA_A = Constant(1.0)    # weight for fluidity regularization
GAMMA_C = Constant(1.0)    # weight for friction regularization
GAMMA_APRES = Constant(1.0)  # weight for ApRES strain rate misfit

# Optimization
MAX_ITER = 30
GRADIENT_TOL = 1e-4
STEP_TOL = 1e-1

# Solver parameters
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
# Custom friction
# ═══════════════════════════════════════════════════════════════════

def friction(**kwargs):
    """Regularized Coulomb friction with effective pressure."""
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    s = kwargs["surface"]
    C = kwargs["friction"]

    p_W = ρ_W * g * max_value(0, -(s - h))
    p_I = ρ_I * g * h
    N = max_value(0, p_I - p_W)
    τ_c = N / 2

    u_c = (τ_c / C) ** m
    u_b = sqrt(inner(u, u))
    return τ_c * (
        (u_c ** (1 / m + 1) + u_b ** (1 / m + 1)) ** (m / (m + 1)) - u_c
    )


# ═══════════════════════════════════════════════════════════════════
# Load ApRES observations
# ═══════════════════════════════════════════════════════════════════

def load_apres_data(mesh_2d):
    """Load ApRES site locations and observed strain rates.

    Returns site coordinates (EPSG:3031) and the depth-averaged
    vertical strain rate at each site.
    """
    with h5py.File(str(APRES_FILE), "r") as f:
        xs = f["summary/x"][:]
        ys = f["summary/y"][:]
        names = [n.decode() for n in f["summary/names"][:]]
        best_orders = f["summary/best_order"][:]

        # Extract the zeroth Legendre coefficient (depth-averaged strain rate)
        # from each site's best-order fit
        eps_obs = []
        eps_sigma = []
        for i, name in enumerate(names):
            order = best_orders[i]
            coeffs = f[f"sites/{name}/order_{order}/coeffs"][:]
            sigmas = f[f"sites/{name}/order_{order}/sigma"][:]
            # Zeroth coefficient = depth average of the strain rate
            eps_obs.append(coeffs[0])
            eps_sigma.append(sigmas[0])

    eps_obs = np.array(eps_obs)
    eps_sigma = np.array(eps_sigma)

    # Filter to sites with valid coordinates and inside the mesh
    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(eps_obs)

    # Check which sites are inside the mesh domain
    coords_mesh = mesh_2d.coordinates.dat.data_ro
    from scipy.spatial import Delaunay
    tri = Delaunay(coords_mesh)
    inside = np.array([tri.find_simplex([xs[i], ys[i]]) >= 0 for i in range(len(xs))])
    valid = valid & inside

    print(f"   {valid.sum()} ApRES sites inside mesh domain (of {len(xs)} total)")

    return (
        xs[valid], ys[valid],
        eps_obs[valid], eps_sigma[valid],
        [names[i] for i in range(len(names)) if valid[i]],
    )


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Inversion: fluidity + friction from velocity + ApRES")
    print("=" * 60)

    # ── 1. Load mesh and data ──
    print("\n1. Loading mesh and data...")
    with firedrake.CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        b_2d = chk.load_function(mesh_2d, "bed")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    print(f"   2D mesh: {mesh_2d.num_vertices()} verts, {mesh_2d.num_cells()} cells")

    # ── 2. Load ApRES data ──
    print("\n2. Loading ApRES observations...")
    apres_x, apres_y, eps_obs, eps_sigma, apres_names = load_apres_data(mesh_2d)

    # ── 3. Extrude mesh ──
    print("\n3. Extruding mesh...")
    mesh = firedrake.ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", HDEGREE, vfamily="GL",
                            vdegree=VDEGREE, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    b = icepack.utilities.lift3d(b_2d, Q_lift)

    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)
    u0_lift = icepack.utilities.lift3d(u_obs_2d, V_lift)
    u0 = Function(V, name="velocity")
    u0.project(u0_lift)

    # ── 4. Set up controls ──
    print("\n4. Setting up controls...")
    Q_2d = FunctionSpace(mesh_2d, "CG", 1)

    θ_A = Function(Q_2d, name="log_fluidity")
    θ_C = Function(Q_2d, name="log_friction")

    A_0 = Constant(20.0)
    C_0 = Constant(1e-2)

    # ── 5. Set up solver ──
    print("\n5. Creating HybridModel and solver...")
    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model,
        dirichlet_ids=[2],
        ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS,
    )

    # ── 6. Define simulation, loss, and regularization ──
    print("\n6. Defining objective functional...")
    area = assemble(Constant(1.0) * dx(mesh_2d))

    σ_u = Constant(10.0)  # velocity uncertainty (m/yr)

    def simulation(controls):
        """Forward solve: returns the hybrid velocity field."""
        from firedrake import exp
        θ_A_ctrl, θ_C_ctrl = controls
        A = icepack.utilities.lift3d(
            firedrake.interpolate(A_0 * exp(θ_A_ctrl), Q_2d), Q_lift
        )
        C = icepack.utilities.lift3d(
            firedrake.interpolate(C_0 * exp(θ_C_ctrl), Q_2d), Q_lift
        )
        return solver.diagnostic_solve(
            velocity=u0,
            thickness=h,
            surface=s,
            fluidity=A,
            friction=C,
        )

    # Create VertexOnlyMesh at ApRES site locations for differentiable
    # sparse point interpolation (following icepack how-to/04-sparse-data)
    apres_coords = np.column_stack([apres_x, apres_y])
    point_set = firedrake.VertexOnlyMesh(
        mesh_2d, apres_coords, missing_points_behaviour="warn"
    )
    Δ = FunctionSpace(point_set, "DG", 0)

    # Observed strain rate on the point set
    eps_obs_fn = Function(Δ, name="eps_obs")
    eps_obs_fn.dat.data[:] = eps_obs[:len(eps_obs_fn.dat.data)]

    # Uncertainty (σ) on the point set
    σ_eps = Function(Δ, name="sigma_eps")
    σ_eps.dat.data[:] = np.maximum(np.abs(eps_sigma[:len(σ_eps.dat.data)]), 1e-5)

    n_sites = len(eps_obs_fn.dat.data)
    print(f"   VertexOnlyMesh: {n_sites} ApRES sites")

    def loss_functional(u):
        """Combined loss: surface velocity + ApRES vertical strain rate.

        Surface velocity: depth-average u, interpolate onto ApRES sites,
        compare with MEaSUREs. (For now we use the full-field velocity misfit.)

        Vertical strain rate: ε_zz = -div(u) from incompressibility.
        Depth-average the horizontal divergence, interpolate onto ApRES
        sites via VertexOnlyMesh, compare with observed strain rates.
        """
        # 1. Surface velocity misfit (full field, not sparse)
        u_avg = icepack.depth_average(u)
        δu = u_avg - u_obs_2d
        J_vel = 0.5 / area * (δu[0] ** 2 + δu[1] ** 2) / σ_u ** 2 * dx(mesh_2d)

        # 2. Vertical strain rate misfit at ApRES sites
        # ε_zz = -div(u_avg) from incompressibility
        # Interpolate onto sparse point set (adjoint-compatible)
        div_u_at_sites = Function(Δ).interpolate(-div(u_avg))
        δeps = div_u_at_sites - eps_obs_fn
        J_apres = 0.5 * GAMMA_APRES / Constant(float(n_sites)) * \
                  (δeps / σ_eps) ** 2 * dx

        return J_vel + J_apres

    def regularization(controls):
        """Tikhonov smoothness regularization on both controls."""
        θ_A_ctrl, θ_C_ctrl = controls
        reg_A = 0.5 / area * GAMMA_A * L_REG ** 2 * inner(grad(θ_A_ctrl), grad(θ_A_ctrl)) * dx(mesh_2d)
        reg_C = 0.5 / area * GAMMA_C * L_REG ** 2 * inner(grad(θ_C_ctrl), grad(θ_C_ctrl)) * dx(mesh_2d)
        return reg_A + reg_C

    # ── 7. Solve inverse problem ──
    print(f"\n7. Setting up inverse problem...")
    problem = StatisticsProblem(
        simulation=simulation,
        loss_functional=loss_functional,
        regularization=regularization,
        controls=[θ_A, θ_C],
    )

    print(f"\n8. Solving (max {MAX_ITER} iterations)...")
    estimator = MaximumProbabilityEstimator(
        problem,
        algorithm="bfgs",
        memory=10,
        gradient_tolerance=GRADIENT_TOL,
        step_tolerance=STEP_TOL,
        max_iterations=MAX_ITER,
        verbose=True,
    )
    θ_A_opt, θ_C_opt = estimator.solve()

    # ── 9. Extract and save results ──
    print("\n9. Saving results...")
    u_opt = estimator.state
    u_avg_opt = icepack.depth_average(u_opt)

    speed_opt = Function(Q_2d, name="speed_model")
    speed_opt.interpolate(sqrt(inner(u_avg_opt, u_avg_opt)))

    speed_obs = Function(Q_2d, name="speed_obs")
    speed_obs.interpolate(sqrt(inner(u_obs_2d, u_obs_2d)))

    print(f"   Model speed: [{speed_opt.dat.data.min():.0f}, {speed_opt.dat.data.max():.0f}] m/yr")
    print(f"   Obs speed:   [{speed_obs.dat.data.min():.0f}, {speed_obs.dat.data.max():.0f}] m/yr")
    print(f"   θ_A: [{θ_A_opt.dat.data.min():.2f}, {θ_A_opt.dat.data.max():.2f}]")
    print(f"   θ_C: [{θ_C_opt.dat.data.min():.2f}, {θ_C_opt.dat.data.max():.2f}]")

    with firedrake.CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(θ_A_opt, name="log_fluidity")
        chk.save_function(θ_C_opt, name="log_friction")
        chk.save_function(speed_opt, name="speed_model")
        chk.save_function(speed_obs, name="speed_obs")
        chk.save_function(u_avg_opt, name="velocity_avg")
    print(f"   Saved {OUTPUT_FILE}")

    # ── 10. Plot ──
    print("\n10. Plotting...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    ax = axes[0, 0]
    tc = firedrake.tripcolor(speed_obs, axes=ax, cmap="inferno", vmin=0, vmax=3000)
    fig.colorbar(tc, ax=ax, fraction=0.046, label="m/yr")
    ax.scatter(apres_x, apres_y, c="cyan", s=15, edgecolors="k", linewidths=0.3, zorder=5)
    ax.set_title("Observed speed"); ax.set_aspect("equal")

    ax = axes[0, 1]
    tc = firedrake.tripcolor(speed_opt, axes=ax, cmap="inferno", vmin=0, vmax=3000)
    fig.colorbar(tc, ax=ax, fraction=0.046, label="m/yr")
    ax.scatter(apres_x, apres_y, c="cyan", s=15, edgecolors="k", linewidths=0.3, zorder=5)
    ax.set_title("Model speed (depth-avg)"); ax.set_aspect("equal")

    ax = axes[0, 2]
    misfit = Function(Q_2d, name="misfit")
    misfit.interpolate(speed_opt - speed_obs)
    tc = firedrake.tripcolor(misfit, axes=ax, cmap="RdBu_r", vmin=-500, vmax=500)
    fig.colorbar(tc, ax=ax, fraction=0.046, label="m/yr")
    ax.set_title("Speed misfit"); ax.set_aspect("equal")

    ax = axes[1, 0]
    tc = firedrake.tripcolor(θ_A_opt, axes=ax, cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title("log-fluidity (θ_A)"); ax.set_aspect("equal")

    ax = axes[1, 1]
    tc = firedrake.tripcolor(θ_C_opt, axes=ax, cmap="RdBu_r", vmin=-3, vmax=3)
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title("log-friction (θ_C)"); ax.set_aspect("equal")

    ax = axes[1, 2]
    from firedrake import exp
    A_opt = Function(Q_2d, name="fluidity")
    A_opt.interpolate(A_0 * exp(θ_A_opt))
    tc = firedrake.tripcolor(A_opt, axes=ax, cmap="viridis")
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title("Fluidity A"); ax.set_aspect("equal")

    fig.suptitle("Inversion: velocity + ApRES strain rate", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(str(DATA_DIR / "figures" / "inversion.png"), dpi=150)
    print(f"   Saved figures/inversion.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
