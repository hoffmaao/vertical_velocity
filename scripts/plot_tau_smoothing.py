r"""Compare MAP inversions whose friction anchor uses different tau_d smoothing.

The friction prior mean C_w0 = tau_d / (phi |u_obs|^(1/m)) takes the driving
stress tau_d = -rho_I g h grad(s) Helmholtz-smoothed over a length L
(physical_prior.py, VV_TAU_SMOOTH). This script lays the runs made with
different L side by side, one row per L:

    tau_d | C_w0 | inferred C | theta_C | speed residual (model - obs)

and adds a panel of the cost terms against L. Every run must be on the same
mesh and at the same (delta, ELL) and gradient tolerance, or the costs are not
comparable; the default list is the production-point bracket.

    VV_MESH=thwaites_fixed python scripts/plot_tau_smoothing.py

Writes figures/tau_smoothing_fields.png and figures/tau_smoothing_costs.png.
"""
import argparse
import json
from pathlib import Path

import firedrake
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from firedrake import CheckpointFile, Function, FunctionSpace, conditional, exp, inner, sqrt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from friction_laws import flotation_factor
from physical_prior import MESH_FILE, MESH_NAME

DATA = Path(__file__).resolve().parent.parent
FIG_DIR = DATA / "figures"

# (L in m, run name, prior file) for the production point (delta 1e-5, ELL 2 km).
RUNS = [
    (2000.0, "hybrid_weertman_fixed", "physical_prior_thwaites_fixed.h5"),
    (3000.0, "hybrid_weertman_fixed_tau3km", "physical_prior_thwaites_fixed_tau3km.h5"),
    (5000.0, "hybrid_weertman_fixed_tau5km", "physical_prior_thwaites_fixed_tau5km.h5"),
    (7500.0, "hybrid_weertman_fixed_tau7.5km", "physical_prior_thwaites_fixed_tau7.5km.h5"),
    (10000.0, "hybrid_weertman_fixed_tau10km", "physical_prior_thwaites_fixed_tau10km.h5"),
]

# Same limits as plot_map.py where the quantity is shared.
CLIM = {
    "tau": (0.0, 0.25),          # MPa
    "C": (1.0e-3, 1.0e0),        # MPa (m/yr)^-1/3, log scale
    "theta": (-2.5, 2.5),
    "dspeed": (-100.0, 100.0),   # m/yr
}


def panel(fig, ax, field, cmap, norm, title=None, label=None, colorbar=True):
    tc = firedrake.tripcolor(field, axes=ax, cmap=cmap, norm=norm, rasterized=True)
    if colorbar:
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.04)
        cb = fig.colorbar(tc, cax=cax)
        cb.ax.tick_params(labelsize=7)
        if label:
            cb.set_label(label, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        h = chk.load_function(mesh, "thickness")
        s = chk.load_function(mesh, "surface")
        u_obs = chk.load_function(mesh, "velocity")
    Q = FunctionSpace(mesh, "CG", 1)
    speed_obs = Function(Q).interpolate(sqrt(inner(u_obs, u_obs)))
    grounded = Function(Q).interpolate(conditional(flotation_factor(h, s) > 0.01, 1.0, 0.0))

    def load(path, names):
        with CheckpointFile(str(path), "r") as chk:
            m = chk.load_mesh()
            out = {}
            for n in names:
                f = Function(Q, name=n)
                f.dat.data[:] = chk.load_function(m, n).dat.data_ro
                out[n] = f
        return out

    rows, costs = [], []
    for L, name, prior_file in RUNS:
        pr = load(DATA / "mesh" / prior_file, ("tau_d", "C_w0"))
        mp = load(DATA / "mesh" / f"map_{name}.h5", ("log_friction", "speed_model"))
        C = Function(Q).interpolate(pr["C_w0"] * exp(mp["log_friction"] * grounded))
        dspeed = Function(Q).interpolate(mp["speed_model"] - speed_obs)
        rows.append((L, pr["tau_d"], pr["C_w0"], C, mp["log_friction"], dspeed))
        d = json.load(open(DATA / "results" / f"map_{name}.json"))
        assert d.get("mesh", MESH_NAME) == MESH_NAME, f"{name} was run on {d['mesh']}"
        rmse = float(np.sqrt(np.mean(dspeed.dat.data_ro ** 2)))
        costs.append((L, d["J"], d["J_vel"], d["J_apr"], d["R"],
                      d["chi2_vel_per_obs"], d["chi2_apr_per_obs"], rmse))
        print(f"L={L / 1000:4.1f} km  J={d['J']:.4e}  J_vel={d['J_vel']:.4e}  "
              f"R={d['R']:.4e}  chi2_vel={d['chi2_vel_per_obs']:.3f}  "
              f"chi2_apr={d['chi2_apr_per_obs']:.2f}  speed RMSE={rmse:.1f} m/yr", flush=True)

    # ── Fields: one row per smoothing length ──
    cols = [
        ("driving stress $|\\tau_d|$", "viridis", Normalize(*CLIM["tau"]), "MPa"),
        ("prior mean $C_{w0}$", "YlOrRd", LogNorm(*CLIM["C"]), "MPa (m/yr)$^{-1/3}$"),
        ("inferred $C$", "YlOrRd", LogNorm(*CLIM["C"]), "MPa (m/yr)$^{-1/3}$"),
        ("$\\theta_C = \\ln(C/C_{w0})$", "RdBu_r", Normalize(*CLIM["theta"]), ""),
        ("speed residual (model - obs)", "RdBu_r", Normalize(*CLIM["dspeed"]), "m/yr"),
    ]
    n = len(rows)
    fig, axes = plt.subplots(n, len(cols), figsize=(4.0 * len(cols), 3.3 * n))
    for i, (L, *fields) in enumerate(rows):
        for j, (field, (title, cmap, norm, label)) in enumerate(zip(fields, cols)):
            panel(fig, axes[i, j], field, cmap, norm,
                  title=title if i == 0 else None, label=label)
        axes[i, 0].set_ylabel(f"L = {L / 1000:g} km", fontsize=11)
        _, J, J_vel, _, R, chi2v, chi2a, rmse = costs[i]
        axes[i, 3].text(0.02, 0.02, f"R = {R:.3g}", transform=axes[i, 3].transAxes, fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85))
        axes[i, 4].text(0.02, 0.02, f"RMSE {rmse:.1f} m/yr\n$\\chi^2_u/N$ = {chi2v:.3f}",
                        transform=axes[i, 4].transAxes, fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85))
    fig.suptitle("Friction anchor vs driving-stress smoothing length L "
                 "(hybrid Weertman, $\\delta$ = 1e-5, ELL = 2 km)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = FIG_DIR / "tau_smoothing_fields.png"
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)

    # ── Costs against L: one panel per quantity, no shared twin axes ──
    c = np.array(costs)
    Lkm = c[:, 0] / 1000
    best = int(np.argmin(c[:, 1]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    specs = [
        (axes[0], [(c[:, 1], "total J", "#1f4e79")], "cost"),
        (axes[1], [(c[:, 2], "velocity misfit $J_{vel}$", "#1f4e79"),
                   (c[:, 4], "prior cost $R$", "#c05a17")], "cost"),
        (axes[2], [(c[:, 7], "speed RMSE", "#1f4e79")], "m/yr"),
    ]
    for ax, series, ylabel in specs:
        for y, label, color in series:
            ax.plot(Lkm, y, "-o", color=color, lw=2, ms=6)
            ax.annotate(label, (Lkm[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color="0.2")
        ax.axvline(Lkm[best], color="0.6", lw=1, ls="--")
        ax.set_xlabel("$\\tau_d$ smoothing length L (km)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(Lkm)
        ax.set_xticklabels([f"{x:g}" for x in Lkm])
        ax.set_xlim(Lkm[0] - 0.5, Lkm[-1] + 3.5)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_title("total cost (lower is better)", fontsize=10)
    axes[1].set_title("misfit and regularization", fontsize=10)
    # The RMSE moves by ~0.2 m/yr across L; give it a 5 m/yr window so the
    # panel does not dramatize a change far below the data uncertainty.
    mid = 0.5 * (c[:, 7].min() + c[:, 7].max())
    axes[2].set_ylim(mid - 2.5, mid + 2.5)
    axes[2].set_title("speed RMSE at the mesh nodes (5 m/yr window)", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "tau_smoothing_costs.png"
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
