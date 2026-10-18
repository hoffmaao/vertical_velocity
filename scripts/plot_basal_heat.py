r"""Basal shear stress and heat sources: the thermo prior's versus the MAP's.

The fluidity prior (physical_prior.py -> thermo_model.py) heats the ice with

    q_fric = grounded_frac * C_heat |u_obs|^(1/m + 1),   C_heat = C_w0 phi,

so its basal shear stress is the anchor's, tau_b = C_w0 phi |u_obs|^(1/m) =
|tau_d| (the 5 km Helmholtz-smoothed driving stress wherever phi >= PHI_MIN),
not the inverted one, and the sliding speed is the observed *surface* speed.
The strain heating it uses is h q_strain amplified by shear_amp. This script
puts those next to the production hybrid MAP's own basal stress and frictional
heat, tau_b = C phi |u_b|^(1/m) with u_b the MAP's basal velocity, on shared
colour scales, with the flow direction drawn on the stress panels.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/plot_basal_heat.py \
        --friction rc --map sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km

--map is the --name the inversion ran with (default: the production MAP
map_hybrid_<friction>). For the regularized Coulomb law the MAP's basal stress
is tau_b = C phi |u_b|^(1/m) / (1 + X), X = C |u_b|^(1/m) / (c0 rho_I g h), the
derivative of friction_laws.make_regularized_coulomb's dissipation.
Writes figures/basal_heat_prior_vs_map_<name>.png and prints domain-integrated
heat (GW) for each source.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
from firedrake import (CheckpointFile, Constant, Function, FunctionSpace, VertexOnlyMesh,
                       assemble, dx, grad, inner, max_value, sqrt, sym)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.tri import Triangulation

import thermo_model as tm
from friction_laws import RC_C0, U_MIN, flotation_factor
from icepack.constants import gravity as g, ice_density as ρ_I, weertman_sliding_law as m
from physical_prior import load_geometry
from problem import build_problem

DATA = Path(__file__).resolve().parent.parent
TO_W = 1e6 / 3.15576e7          # MPa m/yr -> W/m^2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--friction", choices=["weertman", "rc", "schoof"], default="weertman")
    ap.add_argument("--map", default=None, help="the --name the inversion ran with")
    args = ap.parse_args()
    map_name = args.map or f"hybrid_{args.friction}"
    P = build_problem("hybrid", args.friction, True)
    mesh_2d, Q_2d, Q = P["mesh_2d"], P["Q_2d"], P["Q"]
    with CheckpointFile(str(DATA / "mesh" / f"map_{map_name}.h5"), "r") as chk:
        mm = chk.load_mesh()
        th_A = chk.load_function(mm, "log_fluidity").dat.data_ro.copy()
        th_C = chk.load_function(mm, "log_friction").dat.data_ro.copy()
    θ_A, θ_C = Function(Q), Function(Q)
    θ_A.dat.data[:] = th_A
    θ_C.dat.data[:] = th_C
    print("diagnostic solve at the MAP", flush=True)
    u = P["forward"](θ_A, θ_C)
    _, C3 = P["fields"](θ_A, θ_C)

    # Basal velocity: evaluate the 3D field at zeta = 0 below every 2D node.
    xy = mesh_2d.coordinates.dat.data_ro
    vom = VertexOnlyMesh(P["mesh"], np.column_stack([xy, np.zeros(len(xy))]),
                         missing_points_behaviour="error", redundant=False)
    Δ = FunctionSpace(vom, "DG", 0)
    Δin = FunctionSpace(vom.input_ordering, "DG", 0)
    ub = np.column_stack([Function(Δin).interpolate(Function(Δ).interpolate(u[i])).dat.data_ro
                          for i in range(2)])
    Ub = np.sqrt((ub ** 2).sum(axis=1) + U_MIN ** 2)

    _, G = load_geometry()
    hQ, sQ, bQ = (Function(Q_2d) for _ in range(3))
    for f, k in zip((hQ, sQ, bQ), ("thickness", "surface", "bed")):
        f.dat.data[:] = G[k].dat.data_ro      # same 2D CG1 dof order, separate mesh object
    phi = Function(Q_2d).interpolate(flotation_factor(hQ, sQ)).dat.data_ro
    phi_pos = np.maximum(phi, 0.0)
    u_obs = P["u_obs_2d"]
    Uo = Function(Q_2d).interpolate(sqrt(inner(u_obs, u_obs))).dat.data_ro
    C_w0 = P["C_w0_2d"].dat.data_ro
    C_map = C3.dat.data_ro                        # same 2D CG1 dof order as Q_2d

    # What the thermo prior sees (physical_prior.py:156 and solve_energy).
    gfrac = Function(Q_2d).interpolate(
        tm.grounded_frac(hQ, sQ, bQ, tm.DEFAULTS)).dat.data_ro
    tau_prior = C_w0 * phi_pos * np.maximum(Uo, 1e-6) ** (1 / m)
    q_prior = gfrac * tau_prior * np.maximum(Uo, 1e-6)
    # The MAP's own basal stress and frictional heat: d(dissipation)/d|u_b| of
    # the friction law the MAP was inverted with (friction_laws.py).
    tau_map = C_map * phi_pos * Ub ** (1 / m)
    if args.friction == "rc":
        X = C_map * Ub ** (1 / m) / (RC_C0 * ρ_I * g * np.maximum(hQ.dat.data_ro, 1.0))
        tau_map = tau_map / (1 + X)
    q_map = tau_map * Ub
    # Strain heating as the thermo model uses it, at A_prior and u_obs, amplified.
    A_pr = P["A_prior_2d"]
    eps = sym(grad(u_obs))
    eps_e = sqrt(0.5 * inner(eps, eps) + Constant(1e-16))
    q_strain = Function(Q_2d).interpolate(
        max_value(hQ, Constant(10.0)) * 2 * A_pr ** (-1.0 / 3) * eps_e ** (1.0 / 3 + 1)
    ).dat.data_ro * tm.DEFAULTS["shear_amp"]

    grounded = phi > 0.05
    print(f"\ngrounded_frac on grounded ice (phi > 0.05): median {np.median(gfrac[grounded]):.3f}, "
          f"p5 {np.percentile(gfrac[grounded], 5):.3f}, p95 {np.percentile(gfrac[grounded], 95):.3f}")
    print(f"|u_b| / |u_obs| on grounded ice: median {np.median(Ub[grounded] / np.maximum(Uo[grounded], 1)):.2f}")
    print(f"tau_b [kPa] grounded median: prior {1e3 * np.median(tau_prior[grounded]):.0f}, "
          f"MAP {1e3 * np.median(tau_map[grounded]):.0f}")

    def integral(vals):
        f = Function(Q_2d); f.dat.data[:] = vals
        return assemble(f * dx) * TO_W / 1e9        # GW
    for name, q in [("thermo prior friction (as applied, with grounded_frac)", q_prior),
                    ("thermo prior friction without grounded_frac", tau_prior * Uo),
                    ("MAP friction tau_b u_b", q_map),
                    (f"thermo prior strain heating x{tm.DEFAULTS['shear_amp']:g}", q_strain),
                    ("strain heating unamplified", q_strain / tm.DEFAULTS["shear_amp"])]:
        print(f"  {name:55s} {integral(q):8.2f} GW")

    tri = Triangulation(xy[:, 0] / 1e3, xy[:, 1] / 1e3, mesh_2d.coordinates.cell_node_map().values)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    stride = max(1, len(xy) // 900)
    sel = np.arange(0, len(xy), stride)
    sel = sel[Uo[sel] > 30]
    uo = u_obs.dat.data_ro

    def draw(ax, vals, title, cmap, norm, label, arrows=False):
        im = ax.tripcolor(tri, vals, cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
        if arrows:
            d = uo[sel] / np.linalg.norm(uo[sel], axis=1)[:, None]
            ax.quiver(xy[sel, 0] / 1e3, xy[sel, 1] / 1e3, d[:, 0], d[:, 1], color="k",
                      alpha=0.45, scale=45, width=0.0018, headwidth=3, pivot="mid")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=label)

    tnorm = LogNorm(10, 400)
    draw(axes[0, 0], np.clip(1e3 * tau_prior, 10, 400),
         "thermo prior $\\tau_b$ = anchor = smoothed $|\\tau_d|$", "viridis", tnorm, "kPa", True)
    draw(axes[0, 1], np.clip(1e3 * tau_map, 10, 400),
         "MAP $\\tau_b = C\\,\\phi\\,|u_b|^{1/m}$" + (" / (1 + X)" if args.friction == "rc" else ""),
         "viridis", tnorm, "kPa", True)
    ratio = np.log(np.maximum(tau_map, 1e-6) / np.maximum(tau_prior, 1e-6))
    ratio[~grounded] = 0.0
    draw(axes[0, 2], np.clip(ratio, -3, 3), r"$\ln(\tau_{b,\rm MAP}/\tau_{b,\rm prior})$ (grounded)",
         "PuOr_r", TwoSlopeNorm(0, -3, 3), "", True)
    qnorm = LogNorm(0.01, 10)
    draw(axes[1, 0], np.clip(q_prior * TO_W, 0.01, 10),
         "thermo prior friction heat (as applied)\n"
         r"grounded_frac $\cdot\,\tau_b\,|u_{\rm obs}|$", "inferno", qnorm, "W m$^{-2}$")
    draw(axes[1, 1], np.clip(q_map * TO_W, 0.01, 10), r"MAP friction heat $\tau_b\,|u_b|$",
         "inferno", qnorm, "W m$^{-2}$")
    draw(axes[1, 2], np.clip(q_strain * TO_W, 0.01, 10),
         f"thermo prior strain heat $h\\,q_{{\\rm strain}}$ x {tm.DEFAULTS['shear_amp']:g}",
         "inferno", qnorm, "W m$^{-2}$")
    fig.suptitle(f"Basal shear stress and heat sources: thermo fluidity prior vs the hybrid "
                 f"{args.friction} MAP {map_name} (arrows: observed flow direction)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = DATA / "figures" / f"basal_heat_prior_vs_map_{map_name}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
