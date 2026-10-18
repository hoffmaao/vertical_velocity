r"""SSA vs hybrid at the end of a forward run: speed, grounding lines, thickness difference.

Reads the final states that forward.py saved, results/forward_<map>_<T>yr.h5
(thickness_final, velocity_final; the hybrid one depth-averaged), for one SSA and
one hybrid run on the same mesh, and draws three panels on the same zoom:

  (a) SSA speed at T,
  (b) hybrid speed at T,
  (c) hybrid - SSA thickness at T,

each with the grounding line at T of both runs and the initial grounding line.
The grounding line is the zero contour of the height above flotation
h - (rho_W/rho_I) max(0, -b), the same flotation criterion the forward uses.

    VV_MESH=thwaites_fixed python scripts/plot_gz_compare.py --T 50 \
        sweep_ssa_rc_d1e-05_L2km_thwaites_fixed_tau5km \
        sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km

With --tag the tagged runs forward_<map>_<T>yr_<tag>.h5 are read.

Writes figures/gz_compare_<T>yr<suffix>.png.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from firedrake import CheckpointFile
from icepack.constants import ice_density as ρ_I, water_density as ρ_W

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from physical_prior import MESH_FILE  # noqa: E402

DATA = Path(__file__).resolve().parent.parent
FIG_DIR = DATA / "figures"
ZOOM_PAD = 40e3     # m around the grounding lines
GL_COLOUR = {"ssa": "C3", "hybrid": "C0", "init": "k"}


def load_final(name, T, tag=""):
    stem = f"forward_{name}_{T:g}yr" + (f"_{tag}" if tag else "")
    with CheckpointFile(str(DATA / "results" / f"{stem}.h5"), "r") as chk:
        m = chk.load_mesh()
        h = chk.load_function(m, "thickness_final").dat.data_ro.copy()
        u = chk.load_function(m, "velocity_final").dat.data_ro.copy()
    return h, np.sqrt((u ** 2).sum(axis=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ssa", help="the SSA run's forward.py --map name")
    ap.add_argument("hybrid", help="the hybrid run's forward.py --map name")
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--tag", default="",
                    help="forward file tag: read forward_<map>_<T>yr_<tag>.h5")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        h0 = chk.load_function(mesh, "thickness").dat.data_ro.copy()
        b = chk.load_function(mesh, "bed").dat.data_ro.copy()
    xy = mesh.coordinates.dat.data_ro
    cells = mesh.coordinates.function_space().cell_node_list
    tri = Triangulation(xy[:, 0], xy[:, 1], cells)

    h_s, U_s = load_final(args.ssa, args.T, args.tag)
    h_h, U_h = load_final(args.hybrid, args.T, args.tag)
    flot = (ρ_W / ρ_I) * np.maximum(0.0, -b)
    haf = {"init": h0 - flot, "ssa": h_s - flot, "hybrid": h_h - flot}

    # Zoom on the grounding zone: every node near any of the three grounding lines.
    near = np.zeros(len(b), dtype=bool)
    for v in haf.values():
        near |= np.abs(v) < 50.0
    x0, x1 = xy[near, 0].min() - ZOOM_PAD, xy[near, 0].max() + ZOOM_PAD
    y0, y1 = xy[near, 1].min() - ZOOM_PAD, xy[near, 1].max() + ZOOM_PAD

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), constrained_layout=True)
    in_box = (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
    vmax_u = np.percentile(np.r_[U_s[in_box], U_h[in_box]], 99.5)
    unorm = LogNorm(vmin=10.0, vmax=vmax_u)
    dh = h_h - h_s
    lim = np.percentile(np.abs(dh[in_box]), 99)
    panels = [
        (U_s, "magma", unorm, "speed (m/yr)", f"(a) SSA speed at t = {args.T:g} yr"),
        (U_h, "magma", unorm, "speed (m/yr)", f"(b) hybrid speed at t = {args.T:g} yr"),
        (dh, "RdBu", TwoSlopeNorm(0.0, -lim, lim), "thickness difference (m)",
         f"(c) hybrid $-$ SSA thickness at t = {args.T:g} yr"),
    ]
    for ax, (val, cmap, norm, clabel, title) in zip(axes, panels):
        pc = ax.tripcolor(tri, np.maximum(val, 10.0) if cmap == "magma" else val,
                          cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
        fig.colorbar(pc, ax=ax, shrink=0.85, label=clabel)
        for key, lw in (("init", 1.0), ("ssa", 1.6), ("hybrid", 1.6)):
            ax.tricontour(tri, haf[key], levels=[0.0], colors=GL_COLOUR[key], linewidths=lw,
                          linestyles="--" if key == "init" else "-")
        ax.set(xlim=(x0, x1), ylim=(y0, y1), title=title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        # Scale bar in the lower-left corner, which lies outside the mesh.
        ax.plot([x0 + 0.05 * (x1 - x0), x0 + 0.05 * (x1 - x0) + 20e3],
                [y0 + 0.05 * (y1 - y0)] * 2, color="k", lw=3)
        ax.text(x0 + 0.05 * (x1 - x0) + 10e3, y0 + 0.07 * (y1 - y0), "20 km", ha="center",
                color="k", fontsize=9)
    handles = [Line2D([], [], color=GL_COLOUR["init"], ls="--", lw=1, label="grounding line, t = 0"),
               Line2D([], [], color=GL_COLOUR["ssa"], lw=1.6, label=f"SSA, t = {args.T:g} yr"),
               Line2D([], [], color=GL_COLOUR["hybrid"], lw=1.6, label=f"hybrid, t = {args.T:g} yr")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=11)

    grounded = {k: float(np.sum((v > 0) & in_box)) for k, v in haf.items()}
    print(f"grounded nodes in the zoom: t=0 {grounded['init']:.0f}, SSA {grounded['ssa']:.0f}, "
          f"hybrid {grounded['hybrid']:.0f}", flush=True)
    print(f"hybrid - SSA thickness in the zoom: median {np.median(dh[in_box]):+.1f} m, "
          f"p1 {np.percentile(dh[in_box], 1):+.1f}, p99 {np.percentile(dh[in_box], 99):+.1f} m", flush=True)
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"gz_compare_{args.T:g}yr{args.suffix}.png"
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
