r"""Diagnostic plot: ApRES site locations on velocity residual map,
plus ApRES misfit breakdown per site.

Checks whether the ApRES term is pulling the velocity fit in the
ApRES region.
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
    # Load mesh + obs
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)

    # Load inversion result
    with CheckpointFile(str(APRES_INV), "r") as chk:
        m_inv = chk.load_mesh()
        θ_A_2d = chk.load_function(m_inv, "log_fluidity")
        θ_C_2d = chk.load_function(m_inv, "log_friction")

    # Forward solve
    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    u_obs_3d = icepack.utilities.lift3d(u_obs_2d, V_lift)
    u0 = Function(V).project(u_obs_3d)

    θ_A = Function(Q_lift); θ_A.dat.data[:] = θ_A_2d.dat.data_ro
    θ_C = Function(Q_lift); θ_C.dat.data[:] = θ_C_2d.dat.data_ro
    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q_lift).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
                    Constant(1.0), Constant(0.0)))
    A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))

    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)
    u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                 fluidity=A, friction=C)

    u_avg = icepack.depth_average(u)
    speed_model = Function(Q_2d).interpolate(sqrt(inner(u_avg, u_avg)))
    speed_obs = Function(Q_2d).interpolate(sqrt(inner(u_obs_2d, u_obs_2d)))
    diff = Function(Q_2d).interpolate(speed_model - speed_obs)

    # Load ApRES sites
    with h5py.File(str(APRES_FILE), "r") as f:
        x_sites = f["summary/x"][...]
        y_sites = f["summary/y"][...]
        names = [n.decode() for n in f["summary/names"][...]]
        H_apres = f["summary/H_apres"][...]

    # Build VOM and evaluate model eps_zz at ApRES points
    pts_xyz, eps_obs_all, eps_sig_all, site_idx = [], [], [], []
    for i, (name, xi, yi, Hi) in enumerate(zip(names, x_sites, y_sites, H_apres)):
        with h5py.File(str(APRES_FILE), "r") as f:
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
        site_idx.extend([i] * d_w.size)

    pts_xyz = np.vstack(pts_xyz)
    eps_obs_arr = np.concatenate(eps_obs_all)
    eps_sig_arr = np.concatenate(eps_sig_all)
    site_idx = np.array(site_idx)

    vom = VertexOnlyMesh(mesh, pts_xyz, missing_points_behaviour="warn")
    Δ = FunctionSpace(vom, "DG", 0)
    Δ_in = FunctionSpace(vom.input_ordering, "DG", 0)
    f_in = Function(Δ_in)

    # Model eps_zz at VOM points
    eps_h = icepack.models.hybrid.horizontal_strain_rate(
        velocity=u, thickness=h, surface=s)
    eps_mod_f = Function(Δ).interpolate(-(eps_h[0, 0] + eps_h[1, 1]))

    # Obs eps_zz at VOM points
    f_in.dat.data[:] = eps_obs_arr[:len(f_in.dat.data)]
    eps_obs_f = Function(Δ).interpolate(f_in)
    f_in.dat.data[:] = eps_sig_arr[:len(f_in.dat.data)]
    σ_f = Function(Δ).interpolate(f_in)

    mod = eps_mod_f.dat.data_ro
    obs = eps_obs_f.dat.data_ro
    sig = σ_f.dat.data_ro

    # Per-site χ² and velocity misfit
    print(f"{'site':>12s}  {'χ²/n':>8s}  {'|ε_mod|':>10s}  {'|ε_obs|':>10s}  "
          f"{'σ_mean':>10s}  {'speed_diff':>10s}")
    print("-" * 75)
    site_chi2 = []
    site_speed_diff = []
    for i in range(len(names)):
        mask = site_idx == i
        if mask.sum() == 0:
            site_chi2.append(np.nan)
            site_speed_diff.append(np.nan)
            continue
        chi2_i = float(np.mean(((mod[mask] - obs[mask]) / sig[mask]) ** 2))
        site_chi2.append(chi2_i)
        # Speed diff at site location
        sd = float(diff.at([x_sites[i], y_sites[i]], dont_raise=True) or np.nan)
        site_speed_diff.append(sd)
        if i < 20:  # print first 20
            print(f"{names[i]:>12s}  {chi2_i:8.1f}  {np.abs(mod[mask]).mean():10.3e}  "
                  f"{np.abs(obs[mask]).mean():10.3e}  {sig[mask].mean():10.3e}  "
                  f"{sd:+10.1f}")

    site_chi2 = np.array(site_chi2)
    site_speed_diff = np.array(site_speed_diff)

    print(f"\nOverall: χ²/N = {np.nanmean(((mod-obs)/sig)**2):.1f}")
    print(f"  model ε_zz: mean={mod.mean():.3e}, std={mod.std():.3e}")
    print(f"  obs ε_zz:   mean={obs.mean():.3e}, std={obs.std():.3e}")
    print(f"  σ_ε:        mean={sig.mean():.3e}, std={sig.std():.3e}")
    print(f"  |ε_obs/σ|:  mean={np.abs(obs/sig).mean():.1f}")

    # ── Plot ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Velocity residual with ApRES sites
    ax = axes[0]
    tc = firedrake.tripcolor(diff, axes=ax, cmap="RdBu_r", vmin=-200, vmax=200)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(tc, cax=cax, label="m/yr")
    ax.scatter(x_sites, y_sites, s=15, c="lime", edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_title("(a) Hybrid - Obs speed + ApRES sites", fontsize=12)
    ax.set_aspect("equal")

    # (b) Per-site χ² at ApRES locations
    ax = axes[1]
    tc2 = firedrake.tripcolor(diff, axes=ax, cmap="Greys", vmin=-200, vmax=200, alpha=0.3)
    valid = np.isfinite(site_chi2)
    sc = ax.scatter(x_sites[valid], y_sites[valid], c=site_chi2[valid],
                    s=40, cmap="hot_r", vmin=0, vmax=100,
                    edgecolors="k", linewidths=0.5, zorder=5)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(sc, cax=cax, label="χ²/n per site")
    ax.set_title("(b) ApRES strain rate χ²/n per site", fontsize=12)
    ax.set_aspect("equal")

    # (c) Scatter: per-site speed diff vs per-site χ²
    ax = axes[2]
    valid2 = np.isfinite(site_chi2) & np.isfinite(site_speed_diff)
    ax.scatter(site_speed_diff[valid2], site_chi2[valid2], s=20, c="C0")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Speed residual at site (m/yr)", fontsize=11)
    ax.set_ylabel("ApRES χ²/n at site", fontsize=11)
    ax.set_title("(c) Velocity fit vs ApRES fit", fontsize=12)
    r = np.corrcoef(site_speed_diff[valid2], site_chi2[valid2])[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=11, va="top")

    fig.tight_layout()
    out = DATA_DIR / "figures" / "apres_diagnostic.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
