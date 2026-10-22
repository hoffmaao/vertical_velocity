r"""Where SSA and the hybrid model dissipate their energy: ice or bed.

Plots results/paper/dissipation_<model>.npz (dissipation_partition.py) in the
paper's figure grammar (figstyle.py):

  (a)  share of the hybrid model's dissipation on grounded ice that happens at
       the bed, Phi_bed / (Phi_bed + Phi_ice);
  (b)  share of the hybrid model's ice dissipation carried by vertical shear
       rather than membrane stretching (zero by construction in SSA);
  (c)  the partition of the power dissipated over the whole model domain, for
       SSA and the hybrid model: bed, vertical shear and stretching, as a
       percentage of the total, which is printed at the end of each bar, with
       the colour key beside the bars.

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
from dissipation_partition import GROUNDED_PHI
from figstyle import BLACK

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
    part, _ = load_partition(ex)
    grounded = part["ssa"]["phi"] > GROUNDED_PHI

    # Nodal power densities (W/m^2) of the hybrid model for the maps; floating ice masked.
    a_node = to_nodes(part["ssa"]["area"], cells, nn)
    q = {k: to_nodes(part["hybrid"][k], cells, nn) / a_node for k in ("ice", "bed", "shear")}
    tri_g = Triangulation(tri.x, tri.y, tri.triangles, mask=~grounded)

    box = pf.full_box(d)
    fig, axs, caxs = fs.map_grid(1, [1, 1], pf.aspect(box), bottom=1.45)
    share = Normalize(0, 1)
    # Floating cells carry bed power at roundoff (slightly negative); clip so the masked
    # shelf does not add a spurious extend to the colorbar.
    bed_share = np.clip(q["bed"] / np.maximum(q["bed"] + q["ice"], 1e-12), 0, 1)
    mp = fs.tri_field(axs[0][0], tri_g, bed_share, "share", share)
    fs.colorbar(fig, mp, caxs[0][0], r"Bed share $\Phi_b/(\Phi_b+\Phi_i)$")
    mp = fs.tri_field(axs[0][1], tri_g, q["shear"] / np.maximum(q["ice"], 1e-12), "share", share)
    fs.colorbar(fig, mp, caxs[0][1], "Vertical-shear share\nof ice dissipation")
    pf.finish_maps(axs, box, d, tri, stations=True)

    # (c) partition of the power dissipated over the whole domain.
    W = fig.get_figwidth()
    H = fig.get_figheight()
    ax = fig.add_axes([0.30 / W + 0.085, 0.40 / H, 0.50, 0.80 / H])
    numbers = {}
    for i, m in enumerate(pf.MODELS):
        tot = {k: float(part[m][k].sum()) for k in ("bed", "shear", "membrane")}
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
        numbers[m] = dict(total_GW=total / 1e9, **{k: 100 * v / total for k, v in tot.items()})
    ax.set_yticks(range(len(pf.MODELS)), [pf.SHORT[m] for m in pf.MODELS])
    ax.invert_yaxis()
    ax.set_ylim(len(pf.MODELS) - 0.45, -0.75)       # headroom for the narrow-segment labels
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of power dissipated over the model domain (%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fs.panel_letter(ax, "c", x=-0.12, y=0.95)
    # The partition key sits beside the bars, past the printed totals; the map overlays
    # (grounding line, stations) are described in the caption, so there is no top legend.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col, label=lab) for _, lab, col in PART],
              loc="center left", bbox_to_anchor=(1.24, 0.5), frameon=False,
              handlelength=1.4, borderaxespad=0.0)
    pf.save(fig, "F_dissipation")
    json.dump(numbers, open(pf.OUT / "dissipation_numbers.json", "w"), indent=2)
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
