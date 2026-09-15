r"""Plot optimized state estimates: θ_A, θ_C, speed, and ApRES fit.

Panels:
  Row 1: θ_A (log-fluidity), θ_C (log-friction), model speed, observed speed
  Row 2: A (fluidity), C (friction), speed residual, ApRES ε_zz comparison

Usage:
    python plot_state_estimates.py
"""
import numpy as np
import h5py
import firedrake
import icepack
from firedrake import (
    Function, FunctionSpace, VectorFunctionSpace, CheckpointFile,
    ExtrudedMesh, VertexOnlyMesh, Constant,
    sqrt, inner, max_value, exp, dx, conditional,
)
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
APRES_INV = DATA_DIR / "mesh" / "inversion_hires_apres_vd1.h5"
APRES_FILE = DATA_DIR / "data" / "apres_legendre_fits.h5"
FIG_DIR = DATA_DIR / "figures"

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 100, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def main():
    print("Loading...", flush=True)
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        b_2d = chk.load_function(mesh_2d, "bed")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    V_2d = VectorFunctionSpace(mesh_2d, "CG", 1)

    with CheckpointFile(str(APRES_INV), "r") as chk:
        m_inv = chk.load_mesh()
        θ_A_loaded = chk.load_function(m_inv, "log_fluidity")
        θ_C_loaded = chk.load_function(m_inv, "log_friction")
    # Copy to current mesh to avoid cross-mesh expression issues
    θ_A_2d = Function(Q_2d, name="log_fluidity")
    θ_C_2d = Function(Q_2d, name="log_friction")
    θ_A_2d.dat.data[:] = θ_A_loaded.dat.data_ro
    θ_C_2d.dat.data[:] = θ_C_loaded.dat.data_ro

    # Derived fields on 2D mesh
    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q_2d).interpolate(
        conditional(
            1 - ρ_W * g * max_value(0, h_2d - s_2d) / (ρ_I * g * h_2d) > 0.01,
            Constant(1.0), Constant(0.0)))
    A_2d = Function(Q_2d, name="fluidity").interpolate(A_0 * exp(θ_A_2d))
    C_2d = Function(Q_2d, name="friction").interpolate(C_0 * exp(θ_C_2d * grounded_mask))
    speed_obs = Function(Q_2d).interpolate(sqrt(inner(u_obs_2d, u_obs_2d)))

    # Forward solve for model velocity
    print("Forward solve...", flush=True)
    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    u0 = Function(V).project(icepack.utilities.lift3d(u_obs_2d, V_lift))
    θ_A = Function(Q_lift); θ_A.dat.data[:] = θ_A_2d.dat.data_ro
    θ_C = Function(Q_lift); θ_C.dat.data[:] = θ_C_2d.dat.data_ro
    grounded_3d = Function(Q_lift).interpolate(
        conditional(
            1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
            Constant(1.0), Constant(0.0)))
    A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_3d))

    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)
    u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                 fluidity=A, friction=C)
    u_avg = icepack.depth_average(u)
    speed_mod = Function(Q_2d).interpolate(sqrt(inner(u_avg, u_avg)))
    diff = Function(Q_2d).interpolate(speed_mod - speed_obs)

    # ApRES data
    with h5py.File(str(APRES_FILE), "r") as f:
        x_sites = f["summary/x"][...]
        y_sites = f["summary/y"][...]
        H_apres = f["summary/H_apres"][...]
        names = [n.decode() for n in f["summary/names"][...]]

        pts_xyz, eps_obs_all, eps_sig_all = [], [], []
        for name, xi, yi, Hi in zip(names, x_sites, y_sites, H_apres):
            depth = f[f"sites/{name}/depth"][...]
            eps = f[f"sites/{name}/eps_zz"][...]
            sig = f[f"sites/{name}/eps_sigma"][...]
            w = (depth >= 100) & (depth <= 800) & np.isfinite(eps) & np.isfinite(sig) & (sig > 0)
            d_w = depth[w]
            if d_w.size == 0:
                continue
            stride = max(1, d_w.size // 5)
            d_w = d_w[::stride]
            pts_xyz.append(np.column_stack([
                np.full(d_w.size, xi), np.full(d_w.size, yi), 1.0 - d_w / Hi]))
            eps_obs_all.append(eps[w][::stride])
            eps_sig_all.append(sig[w][::stride])
    pts_xyz = np.vstack(pts_xyz)
    eps_obs_arr = np.concatenate(eps_obs_all)

    vom = VertexOnlyMesh(mesh, pts_xyz, missing_points_behaviour="warn")
    Δ = FunctionSpace(vom, "DG", 0)
    Δ_in = FunctionSpace(vom.input_ordering, "DG", 0)
    f_in = Function(Δ_in)
    f_in.dat.data[:] = eps_obs_arr[:len(f_in.dat.data)]
    eps_obs_f = Function(Δ).interpolate(f_in)

    eps_h = icepack.models.hybrid.horizontal_strain_rate(velocity=u, thickness=h, surface=s)
    eps_mod_f = Function(Δ).interpolate(-(eps_h[0, 0] + eps_h[1, 1]))

    mod_eps = eps_mod_f.dat.data_ro
    obs_eps = eps_obs_f.dat.data_ro

    # ── Plot ───────────────────────────────────────────────────────
    print("Plotting...", flush=True)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.35)

    def add_panel(ax, field, title, cmap, vmin, vmax, label=""):
        tc = firedrake.tripcolor(field, axes=ax, cmap=cmap, vmin=vmin, vmax=vmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.05)
        cb = fig.colorbar(tc, cax=cax)
        if label:
            cb.set_label(label, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        return tc

    # Row 1: θ_A, θ_C, model speed, observed speed
    ax = fig.add_subplot(gs[0, 0])
    add_panel(ax, θ_A_2d, r"$\theta_A$ (log-fluidity)", "RdBu_r", -5, 5)
    ax.scatter(x_sites, y_sites, s=5, c="lime", edgecolors="k", linewidths=0.3, zorder=5)

    ax = fig.add_subplot(gs[0, 1])
    add_panel(ax, θ_C_2d, r"$\theta_C$ (log-friction)", "RdBu_r", -5, 5)
    ax.scatter(x_sites, y_sites, s=5, c="lime", edgecolors="k", linewidths=0.3, zorder=5)

    ax = fig.add_subplot(gs[0, 2])
    add_panel(ax, speed_mod, "Model speed", "inferno", 0, 3000, "m/yr")

    ax = fig.add_subplot(gs[0, 3])
    add_panel(ax, speed_obs, "Observed speed", "inferno", 0, 3000, "m/yr")

    # Row 2: fluidity, friction coeff, speed residual, ApRES scatter
    ax = fig.add_subplot(gs[1, 0])
    add_panel(ax, A_2d, r"$A = A_0 e^{\theta_A}$ (fluidity)",
              "viridis", 0, 100, r"MPa$^{-3}$ yr$^{-1}$")

    ax = fig.add_subplot(gs[1, 1])
    C_plot = Function(Q_2d).interpolate(C_2d * Constant(1e3))  # scale for visibility
    add_panel(ax, C_plot, r"$C \cdot \varphi$ (friction $\times 10^3$)",
              "YlOrRd", 0, 50, r"$\times 10^{-3}$")

    ax = fig.add_subplot(gs[1, 2])
    add_panel(ax, diff, "Speed residual (mod - obs)", "RdBu_r", -200, 200, "m/yr")
    ax.scatter(x_sites, y_sites, s=5, c="lime", edgecolors="k", linewidths=0.3, zorder=5)
    rmse = np.sqrt(np.mean(diff.dat.data_ro**2))
    ax.text(0.02, 0.98, f"RMSE = {rmse:.0f} m/yr", transform=ax.transAxes,
            fontsize=10, va="top", bbox=dict(facecolor="white", alpha=0.8))

    # ApRES scatter
    ax = fig.add_subplot(gs[1, 3])
    emax = max(abs(obs_eps).max(), abs(mod_eps).max()) * 1.1
    ax.scatter(obs_eps * 1e3, mod_eps * 1e3, s=8, alpha=0.5, c="C0", edgecolors="none")
    ax.plot([-emax*1e3, emax*1e3], [-emax*1e3, emax*1e3], "k--", lw=0.8)
    ax.set_xlabel(r"Observed $\dot{\varepsilon}_{zz}$ ($\times 10^{-3}$ yr$^{-1}$)", fontsize=10)
    ax.set_ylabel(r"Model $\dot{\varepsilon}_{zz}$ ($\times 10^{-3}$ yr$^{-1}$)", fontsize=10)
    ax.set_title(r"ApRES $\dot{\varepsilon}_{zz}$", fontsize=12)
    ax.set_aspect("equal")
    r = np.corrcoef(obs_eps, mod_eps)[0, 1]
    rmse_eps = np.sqrt(np.mean((mod_eps - obs_eps)**2))
    ax.text(0.02, 0.98, f"r = {r:.2f}\nRMSE = {rmse_eps:.1e} /yr",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(facecolor="white", alpha=0.8))

    out = FIG_DIR / "state_estimates.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"Saved {out}", flush=True)
    print(f"Speed RMSE: {rmse:.1f} m/yr", flush=True)
    print(f"ApRES r: {r:.3f}, RMSE: {rmse_eps:.2e} /yr", flush=True)


if __name__ == "__main__":
    main()
