r"""Hybrid model inversion for Thwaites: fluidity + friction.

Uses icepack's HybridModel (extruded mesh, GL vertical basis functions)
warm-started from the SSA Weertman inversion result.

Assimilates surface velocity from MEaSUREs via sparse VertexOnlyMesh.

Usage:
    python inversion_hybrid.py
"""
import numpy as np
import sympy
from sympy import legendre as sympy_legendre
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    sqrt, inner, grad, max_value, dx, assemble,
)
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
SSA_FILE = DATA_DIR / "mesh" / "inversion_hybrid.h5"  # warm start from vdeg=1
OUTPUT_FILE = DATA_DIR / "mesh" / "inversion_hybrid_vd2.h5"

VDEGREE = 1
HDEGREE = 1

# Regularization
L_REG = Constant(10e3)

# Optimization
MAX_ITER = 500
GRADIENT_TOL = 1e-6
STEP_TOL = 1e-4

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
# Friction
# ═══════════════════════════════════════════════════════════════════

def compute_surface_velocity(q):
    """Extract surface velocity from hybrid model using Legendre weights.

    Evaluates the GL basis at ζ=1 (surface) via weighted depth averaging
    with Legendre polynomials that sum to a delta function at the surface.
    """
    def weight(n, ζ):
        norm = sympy.integrate(sympy_legendre(n, ζ) ** 2, (ζ, 0, 1))
        return sympy_legendre(n, ζ) / norm

    def legendre(n, ζ):
        return sympy_legendre(n, 2 * ζ - 1)

    Q = q.function_space()
    mesh = Q.mesh()
    ζ = firedrake.SpatialCoordinate(mesh)[mesh.geometric_dimension() - 1]
    xdegree_q, zdegree_q = q.ufl_element().degree()

    ζsym = sympy.symbols("ζsym", positive=True)
    full_weight_symbolic = sum(
        [legendre(k, 1) * weight(k, ζsym) for k in range(zdegree_q)]
    ).doit()
    weight_function_numeric = sympy.lambdify(ζsym, full_weight_symbolic, "numpy")

    q_surface = icepack.utilities.depth_average(q, weight=weight_function_numeric(ζ))
    return q_surface


def friction(**kwargs):
    """Weertman friction with flotation ramp."""
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
    print(f"Hybrid inversion (vdegree={VDEGREE})", flush=True)
    print("=" * 60, flush=True)

    # ── 1. Load mesh ──
    print("\n1. Loading mesh...", flush=True)
    with firedrake.CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        b_2d = chk.load_function(mesh_2d, "bed")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    V_2d = VectorFunctionSpace(mesh_2d, "CG", 1)
    print(f"   2D mesh: {mesh_2d.num_vertices()} verts", flush=True)

    # ── 2. Extrude mesh ──
    print("\n2. Extruding mesh...", flush=True)
    mesh = firedrake.ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", HDEGREE, vfamily="GL",
                            vdegree=VDEGREE, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)

    # Lift observed velocity and project to GL space
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)
    u0_lift = icepack.utilities.lift3d(u_obs_2d, V_lift)
    u0 = Function(V, name="velocity")
    u0.project(u0_lift)
    print(f"   V dofs: {V.dim()}", flush=True)

    # ── 3. Controls on 3D mesh (required for adjoint tracing) ──
    print("\n3. Setting up controls...", flush=True)
    from firedrake import exp

    # Controls MUST be on the extruded mesh for pyadjoint to trace
    # through the HybridModel solve. Using Q_lift (CG1 x DG0) makes
    # them constant in ζ — equivalent to depth-averaged parameters.
    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    A_0 = Constant(20.0)
    C_0 = Constant(0.01)

    # Grounded mask (on 3D mesh)
    ϕ_mask = Function(Q_lift).interpolate(
        1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h))
    grounded_mask = Function(Q_lift).interpolate(
        firedrake.conditional(ϕ_mask > 0.01, Constant(1.0), Constant(0.0)))
    print(f"   Grounded: {(grounded_mask.dat.data > 0.5).sum()}/{len(grounded_mask.dat.data)}", flush=True)

    # Warm-start from previous result
    if SSA_FILE.exists():
        with firedrake.CheckpointFile(str(SSA_FILE), "r") as chk:
            mesh_ssa = chk.load_mesh()
            # Load 2D controls and lift to 3D via data copy (same DOF count for CG1 x DG0)
            θ_A_prev = chk.load_function(mesh_ssa, "log_fluidity")
            θ_C_prev = chk.load_function(mesh_ssa, "log_friction")
            θ_A.dat.data[:] = θ_A_prev.dat.data_ro
            θ_C.dat.data[:] = θ_C_prev.dat.data_ro
            try:
                u_prev = chk.load_function(mesh_ssa, "velocity")
            except Exception:
                u_prev = chk.load_function(mesh_ssa, "velocity_avg")
        # Use previous velocity as initial guess (lifted to 3D)
        u0_prev = icepack.utilities.lift3d(u_prev, V_lift)
        u0.project(u0_prev)
        print(f"   Warm start from vdeg=1: "
              f"θ_A=[{θ_A.dat.data.min():.2f}, {θ_A.dat.data.max():.2f}], "
              f"θ_C=[{θ_C.dat.data.min():.2f}, {θ_C.dat.data.max():.2f}]", flush=True)
    else:
        print("   Cold start (no SSA result)", flush=True)

    # ── 4. Sparse velocity observations ──
    print("\n4. Building sparse observations...", flush=True)
    import netCDF4 as nc
    from scipy.spatial import Delaunay

    ds = nc.Dataset("/media/andrew/wd1/projects/ismip7/data/velocity/antarctica_ice_velocity_450m_v2.nc")
    x_vel = ds.variables["x"][:]; y_vel = ds.variables["y"][:]
    coords = mesh_2d.coordinates.dat.data_ro
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
    tri = Delaunay(coords)
    inside = tri.find_simplex(np.column_stack([xs_flat, ys_flat])) >= 0
    has_data = (stdx.ravel() < 1e5) & (stdy.ravel() < 1e5) & (stdx.ravel() > 0) & (stdy.ravel() > 0)
    keep = inside & has_data

    point_set = firedrake.VertexOnlyMesh(mesh_2d, np.column_stack([xs_flat[keep], ys_flat[keep]]),
                                          missing_points_behaviour="warn")
    Δ = FunctionSpace(point_set, "DG", 0)
    Δ_input = FunctionSpace(point_set.input_ordering, "DG", 0)

    def obs_to_point_set(vals):
        f_in = Function(Δ_input)
        f_in.dat.data[:] = vals[:len(f_in.dat.data)]
        f_out = Function(Δ)
        f_out.interpolate(f_in)
        return f_out

    u_o = obs_to_point_set(vx.ravel()[keep])
    v_o = obs_to_point_set(vy.ravel()[keep])
    σ_x = obs_to_point_set(np.maximum(stdx.ravel()[keep], 1.0))
    σ_y = obs_to_point_set(np.maximum(stdy.ravel()[keep], 1.0))
    N_obs = len(u_o.dat.data)
    print(f"   {N_obs} sparse observation points", flush=True)

    # ── 5. Model and solver ──
    print("\n5. Creating HybridModel...", flush=True)
    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS,
    )

    # Test forward solve
    print("   Forward solve...", flush=True)
    A_test = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_test = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    u_test = solver.diagnostic_solve(
        velocity=u0, thickness=h, surface=s,
        fluidity=A_test, friction=C_test,
    )
    u0.assign(u_test)
    u_avg_test = icepack.depth_average(u_test)
    speed_test = Function(Q_2d).interpolate(sqrt(inner(u_avg_test, u_avg_test)))
    print(f"   Speed: [{speed_test.dat.data.min():.0f}, {speed_test.dat.data.max():.0f}] m/yr", flush=True)

    # ── 6. Load ApRES strain rate observations ──
    print("\n6. Loading ApRES strain rate observations...", flush=True)
    import h5py
    APRES_FILE = DATA_DIR / "data" / "apres_shear_analysis.h5"
    with h5py.File(str(APRES_FILE), "r") as f:
        xs_apres = f["x"][:]
        ys_apres = f["y"][:]
        vsr_obs = f["vsr"][:]        # depth-averaged vertical strain rate (1/yr)
        vsr_err = f["vsr_err"][:]
        H_apres = f["H"][:]          # ApRES-measured thickness at each site

    valid_apres = np.isfinite(xs_apres) & np.isfinite(ys_apres) & np.isfinite(vsr_obs)
    xs_ap = xs_apres[valid_apres]
    ys_ap = ys_apres[valid_apres]
    vsr_ap = vsr_obs[valid_apres]
    H_ap = H_apres[valid_apres]

    # Convert strain rate to vertical velocity using the site thickness:
    # w = ε_zz * H (m/yr) — this is the velocity-equivalent observation
    w_obs_ap = vsr_ap * H_ap
    # Uncertainty: use a fraction of the signal or a floor
    σ_w_ap = np.maximum(np.abs(vsr_err[valid_apres]) * H_ap, 1.0)  # at least 1 m/yr

    # VertexOnlyMesh for ApRES sites
    apres_point_set = firedrake.VertexOnlyMesh(
        mesh_2d, np.column_stack([xs_ap, ys_ap]),
        missing_points_behaviour="warn")
    Δ_apres = FunctionSpace(apres_point_set, "DG", 0)
    Δ_apres_input = FunctionSpace(apres_point_set.input_ordering, "DG", 0)

    def apres_to_point_set(vals):
        f_in = Function(Δ_apres_input)
        f_in.dat.data[:] = vals[:len(f_in.dat.data)]
        f_out = Function(Δ_apres)
        f_out.interpolate(f_in)
        return f_out

    w_obs = apres_to_point_set(w_obs_ap)
    σ_w = apres_to_point_set(σ_w_ap)
    H_obs_fn = apres_to_point_set(H_ap)
    N_apres = len(w_obs.dat.data)
    print(f"   {N_apres} ApRES sites", flush=True)
    print(f"   w_obs (ε*H): [{w_obs_ap.min():.2f}, {w_obs_ap.max():.2f}] m/yr", flush=True)
    print(f"   σ_w: [{σ_w_ap.min():.2f}, {σ_w_ap.max():.2f}] m/yr", flush=True)

    GAMMA_APRES = Constant(1.0)

    # ── 7. Objective ──
    print("\n7. Defining objective...", flush=True)
    area = assemble(Constant(1.0) * dx(mesh_2d))

    def simulation(controls):
        θ_A_c, θ_C_c = controls
        A = Function(Q_lift).interpolate(A_0 * exp(θ_A_c))
        C = Function(Q_lift).interpolate(C_0 * exp(θ_C_c * grounded_mask))
        return solver.diagnostic_solve(
            velocity=u0, thickness=h, surface=s,
            fluidity=A, friction=C,
        )

    # Lift observed velocity to 3D for direct comparison (no depth_average needed)
    u_obs_3d = icepack.utilities.lift3d(u_obs_2d,
        VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2))

    def loss_velocity(u):
        """Velocity misfit integrated over the 3D domain.

        Compares the hybrid velocity directly against observations
        (lifted to 3D as constant in ζ). The ζ-integration gives the
        depth-averaged misfit. This avoids depth_average which breaks
        the adjoint tape.
        """
        δu = u - u_obs_3d
        return 0.5 / area_3d * inner(δu, δu) / Constant(10.0)**2 * dx

    def loss_apres(u):
        """Vertical velocity misfit at ApRES sites.

        From incompressibility: w = -div(u) * H at each site.
        Compare against observed w = ε_zz * H_apres.
        Both in m/yr — directly comparable to surface velocity misfit.
        """
        u_surface = compute_surface_velocity(u)
        # Model: w_model = -div(u_surface) * H at the site
        # We interpolate -div(u_surface) and multiply by H_obs on the point set
        eps_model = Function(Δ_apres).interpolate(-firedrake.div(u_surface))
        w_model = eps_model * H_obs_fn
        δw = w_model - w_obs
        return 0.5 * GAMMA_APRES / Constant(float(N_apres)) * \
              (δw / σ_w) ** 2 * dx

    area_3d = assemble(Constant(1.0) * dx(mesh))  # = area_2d * 1 (ζ range)

    def regularization(controls):
        θ_A_c, θ_C_c = controls
        Θ = Constant(1.0)
        reg_A = 0.5 / area_3d * (L_REG / Θ)**2 * inner(grad(θ_A_c), grad(θ_A_c)) * dx
        reg_C = 0.5 / area_3d * (L_REG / Θ)**2 * inner(grad(θ_C_c), grad(θ_C_c)) * dx
        return reg_A + reg_C

    # ── 8. Solve ──
    print(f"\n8. Solving (max {MAX_ITER} iterations)...", flush=True)
    problem = StatisticsProblem(
        simulation=simulation,
        loss_functional=loss_velocity,
        regularization=regularization,
        controls=[θ_A, θ_C],
    )

    estimator = MaximumProbabilityEstimator(
        problem,
        algorithm="bfgs",
        memory=20,
        gradient_tolerance=GRADIENT_TOL,
        step_tolerance=STEP_TOL,
        max_iterations=MAX_ITER,
        verbose=True,
    )
    θ_A_opt, θ_C_opt = estimator.solve()

    # ── 9. Results ──
    print("\n9. Results...", flush=True)
    A_opt = Function(Q_lift).interpolate(A_0 * exp(θ_A_opt))
    C_opt = Function(Q_lift).interpolate(C_0 * exp(θ_C_opt * grounded_mask))
    u_opt = solver.diagnostic_solve(
        velocity=u0, thickness=h, surface=s,
        fluidity=A_opt, friction=C_opt,
    )

    u_surface_opt = compute_surface_velocity(u_opt)
    u_avg_opt = icepack.depth_average(u_opt)
    speed_opt = Function(Q_2d, name="speed_model").interpolate(sqrt(inner(u_surface_opt, u_surface_opt)))
    speed_obs = Function(Q_2d, name="speed_obs").interpolate(sqrt(inner(u_obs_2d, u_obs_2d)))

    u_f = Function(Δ).interpolate(u_surface_opt[0])
    v_f = Function(Δ).interpolate(u_surface_opt[1])
    final_misfit = assemble(0.5 / Constant(float(N_obs)) * (((u_f-u_o)/σ_x)**2 + ((v_f-v_o)/σ_y)**2) * dx)

    print(f"   Model speed: [{speed_opt.dat.data.min():.0f}, {speed_opt.dat.data.max():.0f}] m/yr", flush=True)
    print(f"   θ_A: [{θ_A_opt.dat.data.min():.2f}, {θ_A_opt.dat.data.max():.2f}]", flush=True)
    print(f"   θ_C: [{θ_C_opt.dat.data.min():.2f}, {θ_C_opt.dat.data.max():.2f}]", flush=True)
    print(f"   Final χ²/N: {float(final_misfit):.2f}", flush=True)

    # Convert 3D controls to 2D for saving (same data since DG0 in vertical)
    θ_A_2d = Function(Q_2d, name="log_fluidity")
    θ_C_2d = Function(Q_2d, name="log_friction")
    θ_A_2d.dat.data[:] = θ_A_opt.dat.data_ro
    θ_C_2d.dat.data[:] = θ_C_opt.dat.data_ro

    with firedrake.CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(θ_A_2d, name="log_fluidity")
        chk.save_function(θ_C_2d, name="log_friction")
        chk.save_function(speed_opt, name="speed_model")
        chk.save_function(speed_obs, name="speed_obs")
        chk.save_function(u_avg_opt, name="velocity_avg")
    print(f"   Saved {OUTPUT_FILE}", flush=True)

    # ── 10. Plot ──
    print("\n10. Plotting...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    km = lambda x, _: f"{x/1e3:.0f}"
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    fields = [
        (speed_obs, "Observed speed", "inferno", 0, 3000),
        (speed_opt, f"Hybrid (vdeg={VDEGREE})", "inferno", 0, 3000),
        (Function(Q_2d).interpolate(speed_opt - speed_obs), "Speed misfit", "RdBu_r", -300, 300),
        (θ_A_2d, "θ_A (log-fluidity)", "RdBu_r", -5, 5),
        (θ_C_2d, "θ_C (log-friction)", "RdBu_r", -5, 5),
        (Function(Q_2d).project(firedrake.div(u_surface_opt)), "∇·u (surface)", "RdBu_r", -0.02, 0.02),
    ]
    for idx, (f, title, cmap, vmin, vmax) in enumerate(fields):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        tc = firedrake.tripcolor(f, axes=ax, cmap=cmap, vmin=vmin, vmax=vmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(tc, cax=cax)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(km))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(km))

    fig.suptitle(f"Hybrid inversion (χ²/N: {float(final_misfit):.1f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(DATA_DIR / "figures" / "inversion_hybrid_vd2.png"), dpi=200)
    print(f"   Saved figures/inversion_hybrid.png", flush=True)
    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
