r"""Where SSA and the hybrid model dissipate their energy: ice or bed.

Plots results/paper/dissipation_<model>.npz (dissipation_partition.py) in the
paper's figure grammar (figstyle.py):

  (a, b)  share of the grounded-ice dissipation that happens at the bed,
          Phi_bed / (Phi_bed + Phi_ice), for SSA and the hybrid model;
  (c)     share of the hybrid model's ice dissipation carried by vertical
          shear rather than membrane stretching (zero by construction in SSA);
  (d)     the partition of the integrated power on all grounded ice and within
          5 km of an ApRES station: bed, vertical shear and stretching, as a
          percentage of the total, which is printed at the end of each bar.

Floating ice is left blank on the maps: it has no bed term.

    python scripts/plot_dissipation.py

Writes figures/paper/F_dissipation.{png,pdf} and dissipation_numbers.json.
"""
import json

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

import figstyle as fs
import paper_figures as pf
from figstyle import BLACK

NEAR_M = 5e3                     # "near the line": within this of an ApRES station
# Partition categories (validated: lightness, chroma, CVD and normal-vision separation),
# kept clear of the model red and blue. Every segment also carries a printed share.
PART = (("bed", "basal friction", "#a0561a"),
        ("shear", "vertical shear", "#e3a01a"),
        ("membrane", "stretching", "#1e9c86"))


def load_partition(ex):
    r"""Per-triangle powers (W) of both models on the 2D mesh of ``ex``."""
    xy, c = ex["xy"], ex["cells"]
    centroid = xy[c].mean(axis=1)
    tree = cKDTree(centroid)
    out = {}
    for m in pf.MODELS:
        z = dict(np.load(str(pf.RES / "paper" / f"dissipation_{m}.npz")))
        dist, k = tree.query(z["xy"])
        assert dist.max() < 1.0 and len(np.unique(k)) == len(c), "cells do not match the mesh"
        out[m] = {key: np.empty(len(c)) for key in z if key != "xy"}
        for key in out[m]:
            out[m][key][k] = z[key]
    return out, centroid


def to_nodes(tri_vals, cells, n_nodes):
    r"""Sum per-triangle values onto their nodes (each node gets a third of each)."""
    out = np.zeros(n_nodes)
    for j in range(3):
        np.add.at(out, cells[:, j], tri_vals / 3)
    return out


def main():
    d, tri, area = pf.load()
    ex = d["ssa"]["ex"]
    cells, nn = ex["cells"], len(ex["xy"])
    part, centroid = load_partition(ex)
    grounded = part["ssa"]["phi"] > 0
    st, _, _ = pf.stations(d["hybrid"]["ex"])
    near = grounded & (cKDTree(st).query(centroid)[0] < NEAR_M)

    # Nodal power densities (W/m^2) for the maps; floating triangles masked.
    a_node = to_nodes(part["ssa"]["area"], cells, nn)
    dens = {m: {k: to_nodes(part[m][k], cells, nn) / a_node for k in ("ice", "bed", "shear")}
            for m in pf.MODELS}
    tri_g = Triangulation(tri.x, tri.y, tri.triangles, mask=~grounded)

    box = pf.full_box(d)
    fig, axs, caxs = fs.map_grid(1, [2, 1], pf.aspect(box), bottom=1.95)
    share = Normalize(0, 1)
    for c, m in enumerate(pf.MODELS):
        q = dens[m]
        v = q["bed"] / np.maximum(q["bed"] + q["ice"], 1e-12)
        mp = fs.tri_field(axs[0][c], tri_g, v, "share", share)
    fs.colorbar(fig, mp, caxs[0][0], r"Bed share $\Phi_b/(\Phi_b+\Phi_i)$")
    q = dens["hybrid"]
    mp = fs.tri_field(axs[0][2], tri_g, q["shear"] / np.maximum(q["ice"], 1e-12), "share", share)
    fs.colorbar(fig, mp, caxs[0][1], "Vertical-shear share\nof ice dissipation")
    pf.finish_maps(axs, box, d, tri, stations=True)

    # (d) integrated partition.
    W = fig.get_figwidth()
    H = fig.get_figheight()
    ax = fig.add_axes([0.30 / W + 0.085, 0.40 / H, 0.62, 1.25 / H])
    rows = [(m, mask, where) for where, mask in (("grounded ice", grounded),
                                                 (f"within {NEAR_M / 1e3:g} km of ApRES", near))
            for m in pf.MODELS]
    numbers = {}
    for i, (m, mask, where) in enumerate(rows):
        tot = {k: float(part[m][k][mask].sum()) for k in ("bed", "shear", "membrane")}
        total = sum(tot.values())
        left = 0.0
        for key, _, col in PART:
            f = 100 * tot[key] / total
            ax.barh(i, f, left=left, height=0.62, color=col, edgecolor="white", lw=1.0)
            if f >= 6:
                ax.text(left + f / 2, i, f"{f:.0f}%", ha="center", va="center",
                        fontsize=fs.FS_TICK, color="white" if key != "shear" else BLACK)
            elif f > 0:      # too narrow to hold its label: print it just above the bar
                ax.text(left + f / 2, i - 0.36, f"{f:.1f}%", ha="center", va="bottom",
                        fontsize=fs.FS_TICK - 1, color=BLACK)
            left += f
        ax.text(101.5, i, f"{total / 1e9:.1f} GW", ha="left", va="center", fontsize=fs.FS_TICK)
        numbers.setdefault(where, {})[m] = dict(total_GW=total / 1e9,
                                                **{k: 100 * v / total for k, v in tot.items()})
    ax.set_yticks(range(len(rows)),
                  [f"{pf.SHORT[m]}, {where}" for m, _, where in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of dissipated power (%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fs.panel_letter(ax, "d", x=-0.33, y=1.0)

    from matplotlib.patches import Patch
    fs.top_legend(fig, [pf.GL_KEY, pf.STATION_KEY]
                  + [Patch(color=col, label=lab) for _, lab, col in PART], ncol=5, y=1.0)
    pf.save(fig, "F_dissipation")
    json.dump(numbers, open(pf.OUT / "dissipation_numbers.json", "w"), indent=2)
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
