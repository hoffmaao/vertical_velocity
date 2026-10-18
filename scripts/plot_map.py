r"""Plot the inferred state and the data misfit of a MAP inversion.

Takes the same configuration flags as invert.py, rebuilds the problem through
``problem.build_problem`` so the plotted model is exactly the one that was
inverted, reloads the controls from ``mesh/map_<tag>.h5`` and re-runs the
diagnostic solve (the checkpoint stores only the depth-averaged velocity, but
the ApRES misfit needs the depth-resolved field).

    python scripts/plot_map.py --model hybrid --friction weertman
    python scripts/plot_map.py --model ssa    --friction rc
    VV_MESH=thwaites_fixed python scripts/plot_map.py --model hybrid \
        --friction weertman --name hybrid_weertman_fixed

Writes two figures per run, where <name> is the --name the inversion was run
with (default: the configuration tag) and VV_MESH must match the mesh it ran on:

  figures/map_<name>_state.png   prior mean, inferred field and log-deviation
                                 for both the fluidity and the friction
  figures/map_<name>_misfit.png  modelled and observed speed, speed residual,
                                 normalized velocity residual at the obs points,
                                 and the ApRES eps_zz comparison

The ApRES panels are drawn for every configuration, not just the ones that
assimilated the strain rates: where the data was withheld (the ``_noapres``
hybrids and both SSA runs) it is scored as an independent validation set and
the panel is labelled as such. Colour limits are fixed defaults so the figures
can be compared across runs; pass --clim-free to rescale to each run.
"""
import argparse
import json
from pathlib import Path

import firedrake
import icepack
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from firedrake import (
    CheckpointFile, Function, FunctionSpace, VertexOnlyMesh, grad, inner, sqrt, sym,
)
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from observations import load_apres
from problem import build_problem, tag

DATA = Path(__file__).resolve().parent.parent
FIG_DIR = DATA / "figures"

# Fixed colour limits, so the six runs can be laid side by side.
CLIM = {
    "theta": (-2.5, 2.5),                 # log-deviation from the prior mean
    "A": (1.0e-1, 1.0e3),                 # MPa^-3 yr^-1, log scale
    "C": (1.0e-3, 1.0e0),                 # MPa (m/yr)^-1/3, log scale
    "speed": (0.0, 3000.0),               # m/yr
    "dspeed": (-100.0, 100.0),            # m/yr
    "chi_vel": (0.0, 5.0),                # |residual| / sigma
    "chi_apr": (-10.0, 10.0),             # signed residual / sigma
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    ap.add_argument("--friction", choices=["weertman", "rc", "schoof"], required=True)
    ap.add_argument("--no-apres", action="store_true",
                    help="the run being plotted withheld the ApRES data")
    ap.add_argument("--clim-free", action="store_true",
                    help="rescale colours to this run instead of the shared limits")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--A-const-T", type=float, default=None,
                    help="the run used invert.py --A-const-T (constant fluidity prior mean)")
    ap.add_argument("--name", default=None,
                    help="stem of the run to plot (invert.py --name); defaults to the "
                         "configuration tag")
    args = ap.parse_args()
    args.apres = args.model == "hybrid" and not args.no_apres
    args.tag = tag(args.model, args.friction, args.apres)
    args.name = args.name or args.tag
    return args


def to_2d(f, Q_2d, name=None):
    r"""Copy a control-space field onto the 2D CG1 space (shared dof ordering)."""
    out = Function(Q_2d, name=name)
    out.dat.data[:] = f.dat.data_ro
    return out


def vals(f):
    r"""Values of a VOM DG0 function, in the VOM's own point order."""
    return np.array(f.dat.data_ro, dtype=float)


def onto_vom(Δ, arr):
    r"""Put an array given in input order onto the VOM space ``Δ``.

    Points that fell outside the mesh are dropped by the VertexOnlyMesh, so
    everything is carried through the input_ordering space and read back in
    VOM order (as ``problem.build_problem`` does for the observations).
    """
    Δ_in = FunctionSpace(Δ.mesh().input_ordering, "DG", 0)
    f_in = Function(Δ_in)
    f_in.dat.data[:] = arr[:len(f_in.dat.data)]
    return Function(Δ).interpolate(f_in)


def eps_zz(u, P, model):
    r"""Vertical strain rate -(eps_xx + eps_yy) by incompressibility."""
    if model == "ssa":
        ε = sym(grad(u))
    else:
        ε = icepack.models.hybrid.horizontal_strain_rate(
            velocity=u, thickness=P["h"], surface=P["s"])
    return -(ε[0, 0] + ε[1, 1])


def panel(fig, ax, field, title, cmap, norm, label=""):
    tc = firedrake.tripcolor(field, axes=ax, cmap=cmap, norm=norm)
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(tc, cax=cax)
    if label:
        cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return tc


def scatter_panel(fig, ax, bg, x, y, c, title, cmap, norm, label="", s=0.4):
    firedrake.tripcolor(bg, axes=ax, cmap="Greys", alpha=0.30, rasterized=True)
    sc = ax.scatter(x, y, c=c, s=s, cmap=cmap, norm=norm, linewidths=0)
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(sc, cax=cax)
    if label:
        cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def note(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7", pad=3))


def main():
    args = parse_args()
    FIG_DIR.mkdir(exist_ok=True)
    ckpt = DATA / "mesh" / f"map_{args.name}.h5"
    if not ckpt.exists():
        raise SystemExit(f"no MAP checkpoint at {ckpt}; run invert.py for this configuration first")
    print(f"[{args.name}] building problem", flush=True)

    P = build_problem(args.model, args.friction, args.apres, A_const_T=args.A_const_T)
    mesh_2d, Q_2d, Q = P["mesh_2d"], P["Q_2d"], P["Q"]

    with CheckpointFile(str(ckpt), "r") as chk:
        m = chk.load_mesh()
        θ_A_2d = to_2d(chk.load_function(m, "log_fluidity"), Q_2d, "log_fluidity")
        θ_C_2d = to_2d(chk.load_function(m, "log_friction"), Q_2d, "log_friction")
    θ_A, θ_C = Function(Q), Function(Q)
    θ_A.dat.data[:] = θ_A_2d.dat.data_ro
    θ_C.dat.data[:] = θ_C_2d.dat.data_ro

    print(f"[{args.tag}] diagnostic solve at the MAP", flush=True)
    u = P["forward"](θ_A, θ_C)
    A, C = P["fields"](θ_A, θ_C)
    A_2d, C_2d = to_2d(A, Q_2d, "fluidity"), to_2d(C, Q_2d, "friction")

    # The observations are SURFACE velocities, so the hybrid is compared at the
    # surface (zeta = 1), as the misfit is. Its depth average differs wherever
    # the ice deforms, by up to 60 m/yr over sticky patches in the trunk.
    speed_obs = Function(Q_2d).interpolate(sqrt(inner(P["u_obs_2d"], P["u_obs_2d"])))
    if args.model == "ssa":
        speed_mod = Function(Q_2d).interpolate(sqrt(inner(u, u)))
    else:
        xy = mesh_2d.coordinates.dat.data_ro
        vom_s = VertexOnlyMesh(P["mesh"], np.column_stack([xy, np.ones(len(xy))]),
                               missing_points_behaviour="error", redundant=False)
        Δs, Δs_in = FunctionSpace(vom_s, "DG", 0), FunctionSpace(vom_s.input_ordering, "DG", 0)
        us = [Function(Δs_in).interpolate(Function(Δs).interpolate(u[i])).dat.data_ro
              for i in range(2)]
        speed_mod = Function(Q_2d)
        speed_mod.dat.data[:] = np.hypot(*us)
    dspeed = Function(Q_2d).interpolate(speed_mod - speed_obs)

    # ── Velocity misfit at the observation points ──────────────────────────
    # u lives on the (extruded) flow mesh and the observations on the
    # VertexOnlyMesh, so the model is interpolated onto the VOM first and the
    # residual is formed there, where both operands share a domain.
    o = P["obs"]
    Δv = o["Δv"]
    xy_v = np.array(Δv.mesh().coordinates.dat.data_ro, dtype=float)
    ui = Function(Δv).interpolate(u[0])
    vi = Function(Δv).interpolate(u[1])
    χx = vals(Function(Δv).interpolate((ui - o["u"]) / o["σx"]))
    χy = vals(Function(Δv).interpolate((vi - o["v"]) / o["σy"]))
    χ_vel = np.hypot(χx, χy)
    chi2_vel = float(np.mean(χ_vel ** 2))

    # ── ApRES strain rates: assimilated, or withheld as validation ─────────
    pts, eps_obs_in, eps_sig_in = load_apres(mesh_2d)
    a_pts = pts if args.model == "hybrid" else pts[:, :2]
    vom_a = VertexOnlyMesh(Q.mesh(), a_pts, missing_points_behaviour="warn")
    Δa = FunctionSpace(vom_a, "DG", 0)
    xy_a = np.array(vom_a.coordinates.dat.data_ro, dtype=float)
    eps_o, sig_o = onto_vom(Δa, eps_obs_in), onto_vom(Δa, eps_sig_in)
    ζ_a = vals(onto_vom(Δa, pts[:, 2]))
    eps_m = Function(Δa).interpolate(eps_zz(u, P, args.model))
    eps_obs, eps_mod = vals(eps_o), vals(eps_m)
    χ_apr = vals(Function(Δa).interpolate((eps_m - eps_o) / sig_o))
    chi2_apr = float(np.mean(χ_apr ** 2))
    role = "assimilated" if args.apres else "withheld (validation)"
    print(f"[{args.tag}] chi2_vel/obs = {chi2_vel:.3f} over {len(χ_vel)} obs", flush=True)
    print(f"[{args.tag}] chi2_apr/obs = {chi2_apr:.3f} over {len(χ_apr)} obs [{role}]", flush=True)

    # Per-station aggregate of the ApRES residual.
    stations, inv = np.unique(xy_a[:, :2], axis=0, return_inverse=True)
    st_chi = np.array([np.mean(χ_apr[inv == i]) for i in range(len(stations))])

    def lim(key, v, log=False):
        r"""Shared colour limits, or robust per-run ones under --clim-free."""
        if not args.clim_free:
            return CLIM[key]
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
        if v.size == 0:
            return CLIM[key]
        return float(np.percentile(v, 1)), float(np.percentile(v, 99))

    # ── Figure 1: the inferred state ───────────────────────────────────────
    print(f"[{args.tag}] plotting state", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    n_th = Normalize(*lim("theta", np.r_[θ_A_2d.dat.data_ro, θ_C_2d.dat.data_ro]))
    n_A = LogNorm(*lim("A", A_2d.dat.data_ro, log=True))
    n_C = LogNorm(*lim("C", C_2d.dat.data_ro, log=True))

    a_src = ("thermal" if args.A_const_T is None
             else f"constant, rate_factor({args.A_const_T:g} K)")
    panel(fig, axes[0, 0], P["A_prior_2d"], rf"prior mean $A_{{\rm prior}}$ ({a_src})",
          "viridis", n_A, r"MPa$^{-3}$ yr$^{-1}$")
    panel(fig, axes[0, 1], A_2d, r"inferred $A = A_{\rm prior}\,e^{\theta_A}$",
          "viridis", n_A, r"MPa$^{-3}$ yr$^{-1}$")
    panel(fig, axes[0, 2], θ_A_2d, r"$\theta_A = \ln(A/A_{\rm prior})$", "RdBu_r", n_th)
    if P["anchor_name"] == "tau_c0":
        c_sym, c_anchor, c_unit = r"\tau_c", r"\tau_{c0}", "MPa (yield stress)"
    else:
        c_sym, c_anchor, c_unit = "C", "C_{w0}", r"MPa (m/yr)$^{-1/3}$"
    panel(fig, axes[1, 0], P["C_anchor_2d"], rf"prior mean ${c_anchor}$ (driving-stress anchor)",
          "YlOrRd", n_C, c_unit)
    panel(fig, axes[1, 1], C_2d, rf"inferred ${c_sym} = {c_anchor}\,e^{{\theta_C\,\mathrm{{gnd}}}}$",
          "YlOrRd", n_C, c_unit)
    panel(fig, axes[1, 2], θ_C_2d, rf"$\theta_C = \ln({c_sym}/{c_anchor})$", "RdBu_r", n_th)
    for ax in (axes[0, 2], axes[1, 2]):
        ax.scatter(stations[:, 0], stations[:, 1], s=6, c="lime",
                   edgecolors="k", linewidths=0.3, zorder=5)
    note(axes[0, 2], rf"$\theta_A \in$ [{θ_A_2d.dat.data_ro.min():.2f},"
                     rf" {θ_A_2d.dat.data_ro.max():.2f}]")
    note(axes[1, 2], rf"$\theta_C \in$ [{θ_C_2d.dat.data_ro.min():.2f},"
                     rf" {θ_C_2d.dat.data_ro.max():.2f}]")
    fig.suptitle(f"MAP state estimate - {args.name}"
                 f"   (model={args.model}, friction={args.friction}, ApRES {role})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / f"map_{args.name}_state.png"
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)

    # ── Figure 2: the misfit ───────────────────────────────────────────────
    print(f"[{args.tag}] plotting misfit", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    n_sp = Normalize(*lim("speed", speed_obs.dat.data_ro))
    n_ds = Normalize(*lim("dspeed", dspeed.dat.data_ro))
    n_cv = Normalize(*lim("chi_vel", χ_vel))
    n_ca = Normalize(*lim("chi_apr", st_chi))

    panel(fig, axes[0, 0], speed_obs, "observed speed (MEaSUREs)", "inferno", n_sp, "m/yr")
    panel(fig, axes[0, 1], speed_mod, "modelled surface speed at the MAP", "inferno", n_sp, "m/yr")
    panel(fig, axes[0, 2], dspeed, "speed residual (model - obs)", "RdBu_r", n_ds, "m/yr")
    note(axes[0, 2], f"RMSE = {np.sqrt(np.mean(dspeed.dat.data_ro ** 2)):.0f} m/yr")

    scatter_panel(fig, axes[1, 0], speed_obs, xy_v[:, 0], xy_v[:, 1], χ_vel,
                  r"velocity residual $|\chi_u|$ at the obs points", "magma", n_cv,
                  r"$\sqrt{\chi_x^2 + \chi_y^2}$")
    note(axes[1, 0], rf"$\chi^2/N$ = {chi2_vel:.2f}   N = {len(χ_vel):,}")

    scatter_panel(fig, axes[1, 1], speed_obs, stations[:, 0], stations[:, 1], st_chi,
                  r"ApRES $\dot\varepsilon_{zz}$ residual per station", "RdBu_r", n_ca,
                  r"mean $\chi_\varepsilon$", s=34)
    note(axes[1, 1], rf"$\chi^2/N$ = {chi2_apr:.1f}   N = {len(χ_apr)}" "\n" f"{role}")

    ax = axes[1, 2]
    sc = ax.scatter(eps_obs * 1e3, eps_mod * 1e3, c=(1.0 - ζ_a) * 100,
                    s=12, cmap="cividis", linewidths=0, alpha=0.85)
    e = np.r_[eps_obs, eps_mod] * 1e3
    span = [min(e.min(), 0) * 1.05, max(e.max(), 0) * 1.05]
    ax.plot(span, span, "k--", lw=0.8, zorder=0)
    ax.set_xlim(span)
    ax.set_ylim(span)
    ax.set_aspect("equal")
    ax.set_xlabel(r"observed $\dot\varepsilon_{zz}$ ($10^{-3}$ yr$^{-1}$)", fontsize=10)
    ax.set_ylabel(r"modelled $\dot\varepsilon_{zz}$ ($10^{-3}$ yr$^{-1}$)", fontsize=10)
    ax.set_title(r"ApRES $\dot\varepsilon_{zz}$, model vs observed", fontsize=11)
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"depth fraction $d/H$ (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    r = float(np.corrcoef(eps_obs, eps_mod)[0, 1])
    note(ax, f"r = {r:.2f}\nRMSE = {np.sqrt(np.mean((eps_mod - eps_obs) ** 2)):.1e} /yr")

    fig.suptitle(f"MAP data misfit - {args.name}"
                 f"   (model={args.model}, friction={args.friction}, ApRES {role})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / f"map_{args.name}_misfit.png"
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)

    summary = {"tag": args.tag, "name": args.name, "apres_role": role,
               "chi2_vel_per_obs": chi2_vel, "N_vel": int(len(χ_vel)),
               "chi2_apr_per_obs": chi2_apr, "N_apr": int(len(χ_apr)),
               "eps_corr": r, "speed_rmse": float(np.sqrt(np.mean(dspeed.dat.data_ro ** 2)))}
    json.dump(summary, open(DATA / "results" / f"map_{args.name}_diagnostics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
