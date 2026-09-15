r"""Combined inversion + L-curve figure.

Left: L-curve (E_vel vs R and E_apr vs R) with optimal L marked.
Right: state estimates at optimal L (θ_A, θ_C, speed residual, ApRES scatter).

Usage:
    OMP_NUM_THREADS=4 python plot_inversion_lcurve.py
"""
import numpy as np
import h5py
import json
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
    # ── Load L-curve data ──────────────────────────────────────────
    hybrid = []
    for L in [1, 2, 5, 10, 20, 50]:
        p = DATA_DIR / "data" / f"lcurve_hybrid_apres_L{L}km.json"
        if p.exists():
            with open(p) as f:
                hybrid.append(json.load(f))

    ssa = []
    ssa_path = DATA_DIR / "data" / "lcurve_ssa_manual.json"
    if ssa_path.exists():
        with open(ssa_path) as f:
            ssa = json.load(f)

    # ── Load mesh + inversion ──────────────────────────────────────
    print("Loading mesh...", flush=True)
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)

    with CheckpointFile(str(APRES_INV), "r") as chk:
        m_inv = chk.load_mesh()
        θ_A_loaded = chk.load_function(m_inv, "log_fluidity")
        θ_C_loaded = chk.load_function(m_inv, "log_friction")
    θ_A_2d = Function(Q_2d, name="log_fluidity")
    θ_C_2d = Function(Q_2d, name="log_friction")
    θ_A_2d.dat.data[:] = θ_A_loaded.dat.data_ro
    θ_C_2d.dat.data[:] = θ_C_loaded.dat.data_ro

    speed_obs = Function(Q_2d).interpolate(sqrt(inner(u_obs_2d, u_obs_2d)))

    # Forward solve
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
    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_3d = Function(Q_lift).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
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

    # ApRES
    print("ApRES evaluation...", flush=True)
    with h5py.File(str(APRES_FILE), "r") as f:
        x_sites = f["summary/x"][...]
        y_sites = f["summary/y"][...]
        H_apres = f["summary/H_apres"][...]
        names = [n.decode() for n in f["summary/names"][...]]
        pts_xyz, eps_obs_all = [], []
        for name, xi, yi, Hi in zip(names, x_sites, y_sites, H_apres):
            depth = f[f"sites/{name}/depth"][...]
            eps = f[f"sites/{name}/eps_zz"][...]
            sig = f[f"sites/{name}/eps_sigma"][...]
            w = (depth >= 100) & (depth <= 800) & np.isfinite(eps) & np.isfinite(sig) & (sig > 0)
            d_w = depth[w]
            if d_w.size == 0: continue
            stride = max(1, d_w.size // 5)
            d_w = d_w[::stride]
            pts_xyz.append(np.column_stack([
                np.full(d_w.size, xi), np.full(d_w.size, yi), 1.0 - d_w / Hi]))
            eps_obs_all.append(eps[w][::stride])
    pts_xyz = np.vstack(pts_xyz)
    eps_obs_arr = np.concatenate(eps_obs_all)

    vom = VertexOnlyMesh(mesh, pts_xyz, missing_points_behaviour="warn")
    Δ = FunctionSpace(vom, "DG", 0)
    Δ_in = FunctionSpace(vom.input_ordering, "DG", 0)
    f_in = Function(Δ_in); f_in.dat.data[:] = eps_obs_arr[:len(f_in.dat.data)]
    eps_obs_f = Function(Δ).interpolate(f_in)
    eps_h = icepack.models.hybrid.horizontal_strain_rate(velocity=u, thickness=h, surface=s)
    eps_mod_f = Function(Δ).interpolate(-(eps_h[0, 0] + eps_h[1, 1]))
    mod_eps = eps_mod_f.dat.data_ro
    obs_eps = eps_obs_f.dat.data_ro

    # ── Figure ─────────────────────────────────────────────────────
    print("Plotting...", flush=True)
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.4)

    # ── Top-left: L-curve E_vel vs R ──────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    if hybrid:
        Ls = [d["L_km"] for d in hybrid]
        Ev = [d["E_vel"] for d in hybrid]
        Rs = [d["R"] for d in hybrid]
        ax.plot(Rs, Ev, "s-", color="C0", ms=7, lw=1.5)
        for L, E, R in zip(Ls, Ev, Rs):
            ax.annotate(f"{L}", (R, E), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="0.4")
        # Mark optimal
        i_opt = np.argmin(Ev)
        ax.plot(Rs[i_opt], Ev[i_opt], "s", color="red", ms=12, mew=2,
                mfc="none", zorder=5)
    ax.set_xlabel(r"Regularization $\mathcal{R}$", fontsize=11)
    ax.set_ylabel(r"$\mathcal{E}_{\mathrm{vel}}$ ($\chi^2/N$)", fontsize=11)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("(a) L-curve: velocity misfit", fontsize=12)
    ax.grid(True, alpha=0.2, which="both")

    # ── Top-right col 1: L-curve E_apr vs R ───────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if hybrid:
        Ea = [d["E_apr"] for d in hybrid]
        ax.plot(Rs, Ea, "^-", color="C1", ms=7, lw=1.5)
        for L, E, R in zip(Ls, Ea, Rs):
            ax.annotate(f"{L}", (R, E), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="0.4")
        ax.plot(Rs[i_opt], Ea[i_opt], "^", color="red", ms=12, mew=2,
                mfc="none", zorder=5)
    ax.set_xlabel(r"Regularization $\mathcal{R}$", fontsize=11)
    ax.set_ylabel(r"$\mathcal{E}_{\mathrm{ApRES}}$ ($\chi^2/N$)", fontsize=11)
    ax.set_xscale("log")
    ax.set_title("(b) L-curve: ApRES misfit", fontsize=12)
    ax.grid(True, alpha=0.2, which="both")

    # ── Top-right col 2: SSA L-curve ──────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    if ssa:
        Ls_s = [d["L_km"] for d in ssa]
        Es_s = [d["E"] for d in ssa]
        Rs_s = [d["R"] for d in ssa]
        ax.plot(Rs_s, Es_s, "ko-", ms=7, lw=1.5)
        for L, E, R in zip(Ls_s, Es_s, Rs_s):
            ax.annotate(f"{L}", (R, E), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="0.4")
        i_opt_s = np.argmin(Es_s)
        ax.plot(Rs_s[i_opt_s], Es_s[i_opt_s], "o", color="red", ms=12,
                mew=2, mfc="none", zorder=5)
    ax.set_xlabel(r"Regularization $\mathcal{R}$", fontsize=11)
    ax.set_ylabel(r"$\mathcal{E}$ ($\chi^2/N$)", fontsize=11)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("(c) L-curve: SSA", fontsize=12)
    ax.grid(True, alpha=0.2, which="both")

    # ── Top-right col 3: E_vel and E_apr vs L ─────────────────────
    ax = fig.add_subplot(gs[0, 3])
    if hybrid:
        ax.semilogy(Ls, Ev, "s-", color="C0", ms=6, lw=1.5,
                     label=r"$\mathcal{E}_{\mathrm{vel}}$")
        ax.semilogy(Ls, Ea, "^-", color="C1", ms=6, lw=1.5,
                     label=r"$\mathcal{E}_{\mathrm{ApRES}}$")
        ax.axvline(Ls[i_opt], color="red", ls="--", lw=1, alpha=0.7,
                    label=f"L = {Ls[i_opt]} km")
    ax.set_xlabel("$L$ (km)", fontsize=11)
    ax.set_ylabel(r"Misfit ($\chi^2/N$)", fontsize=11)
    ax.set_xscale("log")
    ax.set_title("(d) Misfit vs $L$", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls])

    # ── Bottom row: θ_A, θ_C, speed residual, ApRES scatter ──────

    def add_map(ax, field, title, cmap, vmin, vmax, label="", sites=True):
        tc = firedrake.tripcolor(field, axes=ax, cmap=cmap, vmin=vmin, vmax=vmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.05)
        cb = fig.colorbar(tc, cax=cax)
        if label: cb.set_label(label, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        if sites:
            ax.scatter(x_sites, y_sites, s=5, c="lime", edgecolors="k",
                       linewidths=0.3, zorder=5)

    ax = fig.add_subplot(gs[1, 0])
    add_map(ax, θ_A_2d, r"(e) $\theta_A$", "RdBu_r", -5, 5)

    ax = fig.add_subplot(gs[1, 1])
    add_map(ax, θ_C_2d, r"(f) $\theta_C$", "RdBu_r", -5, 5)

    ax = fig.add_subplot(gs[1, 2])
    add_map(ax, diff, "(g) Speed residual", "RdBu_r", -200, 200, "m/yr")
    rmse = np.sqrt(np.mean(diff.dat.data_ro**2))
    ax.text(0.02, 0.98, f"RMSE = {rmse:.0f} m/yr", transform=ax.transAxes,
            fontsize=9, va="top", bbox=dict(facecolor="white", alpha=0.8))

    ax = fig.add_subplot(gs[1, 3])
    ax.scatter(obs_eps * 1e3, mod_eps * 1e3, s=6, alpha=0.4, c="C0", edgecolors="none")
    emax = max(abs(obs_eps).max(), abs(mod_eps).max()) * 1.1 * 1e3
    ax.plot([-emax, emax], [-emax, emax], "k--", lw=0.8)
    ax.set_xlabel(r"Obs $\dot{\varepsilon}_{zz}$ ($\times 10^{-3}$ /yr)", fontsize=10)
    ax.set_ylabel(r"Model $\dot{\varepsilon}_{zz}$ ($\times 10^{-3}$ /yr)", fontsize=10)
    ax.set_title(r"(h) ApRES $\dot{\varepsilon}_{zz}$", fontsize=12)
    ax.set_aspect("equal")
    r = np.corrcoef(obs_eps, mod_eps)[0, 1]
    rmse_eps = np.sqrt(np.mean((mod_eps - obs_eps)**2))
    ax.text(0.02, 0.98, f"r = {r:.2f}\nRMSE = {rmse_eps:.1e} /yr",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.8))

    out = FIG_DIR / "inversion_lcurve.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
