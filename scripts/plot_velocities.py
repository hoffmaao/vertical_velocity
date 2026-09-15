r"""Plot observed vs modeled surface velocities for the converged inversions.

Panels:
  (a) Observed speed (MEaSUREs)
  (b) Hybrid + ApRES model speed
  (c) Speed difference (model - obs)
  (d) SSA model speed
  (e) Speed difference SSA (model - obs)

Usage:
    python plot_velocities.py
"""
import numpy as np
import firedrake
import icepack
from firedrake import (
    Function, FunctionSpace, VectorFunctionSpace, CheckpointFile,
    ExtrudedMesh, sqrt, inner,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
APRES_INV = DATA_DIR / "mesh" / "inversion_hires_apres_vd1.h5"
SSA_INV = DATA_DIR / "mesh" / "lcurve_ssa_L10km.h5"  # L-curve result, well-converged
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 100, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
from firedrake import Constant, max_value, exp, conditional


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def main():
    print("Loading mesh...", flush=True)
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    V_2d = VectorFunctionSpace(mesh_2d, "CG", 1)

    speed_obs = Function(Q_2d, name="speed_obs").interpolate(
        sqrt(inner(u_obs_2d, u_obs_2d)))
    print(f"Observed speed: [{speed_obs.dat.data.min():.0f}, "
          f"{speed_obs.dat.data.max():.0f}] m/yr", flush=True)

    # Convert coords to km for plotting
    coords = mesh_2d.coordinates.dat.data_ro
    x0, y0 = coords[:, 0].mean(), coords[:, 1].mean()

    # ── Hybrid + ApRES ─────────────────────────────────────────────
    print("Computing Hybrid+ApRES velocity...", flush=True)
    with CheckpointFile(str(APRES_INV), "r") as chk:
        m_inv = chk.load_mesh()
        θ_A_h = chk.load_function(m_inv, "log_fluidity")
        θ_C_h = chk.load_function(m_inv, "log_friction")

    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    u_obs_3d = icepack.utilities.lift3d(u_obs_2d, V_lift)
    u0 = Function(V).project(u_obs_3d)

    θ_A = Function(Q_lift)
    θ_C = Function(Q_lift)
    θ_A.dat.data[:] = θ_A_h.dat.data_ro
    θ_C.dat.data[:] = θ_C_h.dat.data_ro

    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q_lift).interpolate(
        conditional(
            1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
            Constant(1.0), Constant(0.0)))
    A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))

    model_h = icepack.models.HybridModel(friction=friction)
    solver_h = icepack.solvers.FlowSolver(
        model_h, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)
    u_hybrid = solver_h.diagnostic_solve(
        velocity=u0, thickness=h, surface=s, fluidity=A, friction=C)
    u_avg = icepack.depth_average(u_hybrid)
    speed_hybrid = Function(Q_2d, name="speed_hybrid").interpolate(
        sqrt(inner(u_avg, u_avg)))
    print(f"Hybrid speed: [{speed_hybrid.dat.data.min():.0f}, "
          f"{speed_hybrid.dat.data.max():.0f}] m/yr", flush=True)

    # ── SSA ────────────────────────────────────────────────────────
    print("Computing SSA velocity...", flush=True)
    with CheckpointFile(str(SSA_INV), "r") as chk:
        m_ssa = chk.load_mesh()
        θ_A_s = chk.load_function(m_ssa, "log_fluidity")
        θ_C_s = chk.load_function(m_ssa, "log_friction")

    θ_A_2d = Function(Q_2d)
    θ_C_2d = Function(Q_2d)
    θ_A_2d.dat.data[:] = θ_A_s.dat.data_ro
    θ_C_2d.dat.data[:] = θ_C_s.dat.data_ro

    grounded_mask_2d = Function(Q_2d).interpolate(
        conditional(
            1 - ρ_W * g * max_value(0, h_2d - s_2d) / (ρ_I * g * h_2d) > 0.01,
            Constant(1.0), Constant(0.0)))
    A_ssa = Function(Q_2d).interpolate(A_0 * exp(θ_A_2d))
    C_ssa = Function(Q_2d).interpolate(C_0 * exp(θ_C_2d * grounded_mask_2d))

    model_s = icepack.models.IceStream(friction=friction)
    solver_s = icepack.solvers.FlowSolver(
        model_s, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)
    u_init = Function(V_2d).interpolate(u_obs_2d)
    u_init.interpolate(firedrake.conditional(
        sqrt(inner(u_obs_2d, u_obs_2d)) < 1.0,
        firedrake.as_vector([Constant(1.0), Constant(0.0)]), u_obs_2d))
    u_ssa = solver_s.diagnostic_solve(
        velocity=u_init, thickness=h_2d, surface=s_2d,
        fluidity=A_ssa, friction=C_ssa)
    speed_ssa = Function(Q_2d, name="speed_ssa").interpolate(
        sqrt(inner(u_ssa, u_ssa)))
    print(f"SSA speed: [{speed_ssa.dat.data.min():.0f}, "
          f"{speed_ssa.dat.data.max():.0f}] m/yr", flush=True)

    # ── Differences ────────────────────────────────────────────────
    diff_hybrid = Function(Q_2d, name="diff_hybrid").interpolate(
        speed_hybrid - speed_obs)
    diff_ssa = Function(Q_2d, name="diff_ssa").interpolate(
        speed_ssa - speed_obs)

    # ── Plot ───────────────────────────────────────────────────────
    print("Plotting...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmax_speed = 3000
    vmax_diff = 500

    panels = [
        (axes[0, 0], speed_obs, "Observed speed", "inferno", 0, vmax_speed),
        (axes[0, 1], speed_hybrid, "Hybrid + ApRES", "inferno", 0, vmax_speed),
        (axes[0, 2], speed_ssa, "SSA", "inferno", 0, vmax_speed),
        (axes[1, 0], diff_hybrid, "Hybrid - Obs", "RdBu_r", -vmax_diff, vmax_diff),
        (axes[1, 1], diff_ssa, "SSA - Obs", "RdBu_r", -vmax_diff, vmax_diff),
    ]

    for ax, field, title, cmap, vmin, vmax in panels:
        tc = firedrake.tripcolor(field, axes=ax, cmap=cmap, vmin=vmin, vmax=vmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3%", pad=0.05)
        cb = fig.colorbar(tc, cax=cax)
        if "Obs" in title and "diff" not in title.lower():
            cb.set_label("m/yr")
        elif "diff" in title.lower() or "-" in title:
            cb.set_label("m/yr")
        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal")
        # Axes in km
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    # Panel (f): scatter obs vs model
    ax = axes[1, 2]
    s_o = speed_obs.dat.data_ro
    s_h = speed_hybrid.dat.data_ro
    s_s = speed_ssa.dat.data_ro
    # Subsample for scatter
    rng = np.random.default_rng(42)
    idx = rng.choice(len(s_o), min(5000, len(s_o)), replace=False)
    ax.scatter(s_o[idx], s_h[idx], s=1, alpha=0.3, c="C0", label="Hybrid+ApRES")
    ax.scatter(s_o[idx], s_s[idx], s=1, alpha=0.3, c="C1", label="SSA")
    ax.plot([0, vmax_speed], [0, vmax_speed], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(0, vmax_speed)
    ax.set_ylim(0, vmax_speed)
    ax.set_xlabel("Observed speed (m/yr)", fontsize=11)
    ax.set_ylabel("Model speed (m/yr)", fontsize=11)
    ax.set_title("Model vs Observed", fontsize=13)
    ax.legend(fontsize=10, markerscale=5)
    ax.set_aspect("equal")

    # RMSE stats
    rmse_h = np.sqrt(np.mean(diff_hybrid.dat.data_ro**2))
    rmse_s = np.sqrt(np.mean(diff_ssa.dat.data_ro**2))
    ax.text(0.05, 0.92, f"RMSE Hybrid: {rmse_h:.0f} m/yr\nRMSE SSA: {rmse_s:.0f} m/yr",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    out = FIG_DIR / "velocity_comparison.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"Saved {out}", flush=True)

    print(f"\nRMSE Hybrid+ApRES: {rmse_h:.1f} m/yr")
    print(f"RMSE SSA:          {rmse_s:.1f} m/yr")
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
