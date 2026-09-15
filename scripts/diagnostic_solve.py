r"""Forward diagnostic solve for Thwaites using icepack's HybridModel.

Loads the 2D mesh from mesh/thwaites.h5, extrudes it to 3D with
terrain-following coordinates, and solves for the velocity field
using a hybrid (Blatter-Pattyn) model with effective-pressure-dependent
Coulomb friction.

The vertical basis uses Gauss-Legendre (GL) polynomials with vdegree=4,
matching the quartic Legendre polynomial fits to the ApRES vertical
velocity profiles.

Usage:
    python diagnostic_solve.py
"""
import numpy as np
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    sqrt, inner, grad, max_value, dx,
)
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    weertman_sliding_law as m,
)
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
OUTPUT_FILE = DATA_DIR / "mesh" / "diagnostic.h5"

VDEGREE = 1        # Vertical polynomial degree (start low, increase later)
HDEGREE = 2        # Horizontal CG degree

# Initial fluidity: A for Glen's flow law at ~-10C
A_INIT = Constant(20.0)  # MPa^-3 yr^-1 (icepack units)

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
# Custom friction with effective pressure
# ═══════════════════════════════════════════════════════════════════

def friction(**kwargs):
    r"""Regularized Coulomb friction with effective pressure.

    .. math::
        \tau_b = \tau_c \left[\left(\frac{u_c}{|u|}\right)^{1/m+1}
                 + 1\right]^{m/(m+1)} |u|^{1/m} \hat{u}

    where :math:`\tau_c = N/2` is the Coulomb threshold,
    :math:`N = \max(0, p_I - p_W)` is the effective pressure,
    and :math:`u_c = (\tau_c / C)^m` is the transition speed.
    """
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
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Diagnostic solve: icepack HybridModel on Thwaites")
    print(f"  vdegree={VDEGREE}, hdegree={HDEGREE}")
    print("=" * 60)

    # ── 1. Load 2D mesh and data ──
    print("\n1. Loading 2D mesh and data...")
    with firedrake.CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        b_2d = chk.load_function(mesh_2d, "bed")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    print(f"   2D mesh: {mesh_2d.num_vertices()} verts, {mesh_2d.num_cells()} cells")

    # ── 2. Extrude mesh ──
    print("\n2. Extruding mesh...")
    mesh = firedrake.ExtrudedMesh(mesh_2d, layers=1)

    # Function spaces on extruded mesh
    V = VectorFunctionSpace(mesh, "CG", HDEGREE, vfamily="GL",
                            vdegree=VDEGREE, dim=2)

    print(f"   V dofs: {V.dim()}")

    # ── 3. Lift 2D fields to 3D ──
    # The 2D data (CG1) must be lifted to 3D via icepack.utilities.lift3d,
    # which requires matching horizontal element and vdegree=0.
    print("\n3. Lifting fields to extruded mesh...")

    # Scalar fields: CG1 2D -> CG1 x DG0 3D (lift), then project to CG2 x DG0
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    b = icepack.utilities.lift3d(b_2d, Q_lift)
    h.rename("thickness")
    s.rename("surface")
    b.rename("bed")

    # Velocity: CG1 2D -> CG1 x DG0 3D (lift), then project to CG2 x GL4
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)
    u0_lift = icepack.utilities.lift3d(u_obs_2d, V_lift)
    u0 = Function(V, name="velocity")
    u0.project(u0_lift)

    print(f"   Thickness: [{h.dat.data.min():.0f}, {h.dat.data.max():.0f}] m")
    print(f"   Surface:   [{s.dat.data.min():.0f}, {s.dat.data.max():.0f}] m")

    # ── 4. Set up friction coefficient ──
    print("\n4. Setting up friction coefficient...")
    # Use a spatially uniform C; the effective pressure handles spatial variation
    C = Function(Q_lift, name="friction")
    C.interpolate(Constant(1e-2))  # will be tuned in inversion

    # ── 5. Set up model and solver ──
    print("\n5. Creating HybridModel and solver...")
    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model,
        dirichlet_ids=[2],  # inflow boundary (tag 2)
        ice_front_ids=[1],  # calving front (tag 1)
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS,
    )

    # ── 6. Solve ──
    print("\n6. Solving diagnostic problem...")
    u = solver.diagnostic_solve(
        velocity=u0,
        thickness=h,
        surface=s,
        fluidity=A_INIT,
        friction=C,
    )

    # ── 7. Extract results ──
    print("\n7. Extracting results...")
    u_avg = icepack.depth_average(u)
    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    speed_avg = Function(Q_2d, name="speed_avg")
    speed_avg.interpolate(sqrt(inner(u_avg, u_avg)))

    speed_obs = Function(Q_2d, name="speed_obs")
    speed_obs.interpolate(sqrt(inner(u_obs_2d, u_obs_2d)))

    print(f"   Model speed (depth-avg): [{speed_avg.dat.data.min():.0f}, "
          f"{speed_avg.dat.data.max():.0f}] m/yr")
    print(f"   Observed speed:          [{speed_obs.dat.data.min():.0f}, "
          f"{speed_obs.dat.data.max():.0f}] m/yr")

    # ── 8. Save ──
    print("\n8. Saving results...")
    with firedrake.CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh)
        chk.save_function(u, name="velocity")
        chk.save_function(h, name="thickness")
        chk.save_function(s, name="surface")
        chk.save_function(b, name="bed")
    # Also save 2D depth-averaged for quick inspection
    with firedrake.CheckpointFile(str(DATA_DIR / "mesh" / "diagnostic_2d.h5"), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(u_avg, name="velocity_avg")
        chk.save_function(speed_avg, name="speed_avg")
        chk.save_function(speed_obs, name="speed_obs")
    print(f"   Saved {OUTPUT_FILE}")

    # ── 9. Plot ──
    print("\n9. Plotting...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    tc = firedrake.tripcolor(speed_obs, axes=ax, cmap="inferno", vmin=0, vmax=3000)
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title("Observed speed (m/yr)")
    ax.set_aspect("equal")

    ax = axes[1]
    tc = firedrake.tripcolor(speed_avg, axes=ax, cmap="inferno", vmin=0, vmax=3000)
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title("Model speed, depth-avg (m/yr)")
    ax.set_aspect("equal")

    ax = axes[2]
    misfit = Function(Q_2d, name="misfit")
    misfit.interpolate(speed_avg - speed_obs)
    tc = firedrake.tripcolor(misfit, axes=ax, cmap="RdBu_r", vmin=-500, vmax=500)
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title("Speed misfit (m/yr)")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig_path = str(DATA_DIR / "figures" / "diagnostic.png")
    fig.savefig(fig_path, dpi=150)
    print(f"   Saved {fig_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
