r"""Ridgeline (joyplot) of the Thwaites retreat distribution over time.

Uses ALL posterior forward realizations (sample_vaf_*.npz, each storing a full VAF(t)
trajectory from a distinct θ-initialization). At a set of horizon years it builds the
across-ensemble distribution of the Thwaites sea-level contribution
    SLE(t) = (VAF(0) − VAF(t)) / 361.8   [mm]
and stacks them as a ridgeline (earliest at top), à la IPCC SLR ridge plots. The growing
spread and the heavy right tail (collapse trajectories) are the story.

Writes figures/uq_ridgeline_retreat.png.
"""
import glob
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
VAF0 = 46836.1          # Gt, common initial VAF
GT_PER_MM = 361.8       # Gt of grounded-ice loss per mm sea-level equivalent
ROW_YEARS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def main():
    trajs = []
    for f in sorted(glob.glob(str(DATA / "results" / "sample_vaf_*.npz"))):
        d = np.load(f)
        if bool(d["failed"]):
            continue
        trajs.append((np.asarray(d["time"]), np.asarray(d["vaf_Gt"])))
    N = len(trajs)
    print(f"loaded {N} realizations")

    # SLE(year) across the ensemble at each horizon
    rows = {}
    for yr in ROW_YEARS:
        vaf_t = np.array([np.interp(yr, t, v) for (t, v) in trajs])
        rows[yr] = (VAF0 - vaf_t) / GT_PER_MM      # mm SLE
    allsle = np.concatenate(list(rows.values()))
    xmax = np.percentile(allsle, 99.7) * 1.15

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(11, 8.5))
    xs = np.linspace(-0.3, xmax, 500)
    spacing, scale = 1.0, 1.9
    nrow = len(ROW_YEARS)
    for i, yr in enumerate(ROW_YEARS):
        sle = rows[yr]
        base = (nrow - 1 - i) * spacing            # earliest (i=0) at top
        kde = gaussian_kde(sle)
        dens = kde(xs); dens = dens / dens.max() * scale
        color = cm.viridis(i / (nrow - 1))
        z = i
        ax.fill_between(xs, base, base + dens, color=color, alpha=0.78, zorder=z)
        ax.plot(xs, base + dens, color="k", lw=0.8, zorder=z)
        ax.hlines(base, xs[0], xmax, color="0.5", lw=0.6, zorder=z)
        # markers: median + 95th percentile
        ax.plot(np.median(sle), base, "|", color="k", ms=10, mew=1.5, zorder=z + 0.5)
        ax.text(xs[0] - 0.15, base + 0.05, f"{yr} yr", ha="right", va="bottom",
                fontweight="bold", fontsize=12)

    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(xs[0] - 1.4, xmax)
    ax.set_xlabel("Thwaites sea-level contribution  [mm SLE]", fontsize=13)
    ax.set_title(f"Thwaites 50-yr retreat: posterior distribution vs time\n"
                 f"(N={N} realizations, each a distinct fluidity/friction initialization)",
                 fontsize=13)
    # secondary axis: VAF loss in Gt
    secax = ax.secondary_xaxis("top", functions=(lambda x: x * GT_PER_MM, lambda g: g / GT_PER_MM))
    secax.set_xlabel("VAF loss  [Gt]", fontsize=11, color="0.4")
    fig.tight_layout()
    fp = DATA / "figures" / "uq_ridgeline_retreat.png"
    fig.savefig(str(fp), dpi=200, bbox_inches="tight")
    print(f"Saved {fp}")
    # quick stats table
    print(f"\n{'year':>5} {'median':>8} {'5-95% mm':>16} {'max(mm)':>8}")
    for yr in ROW_YEARS:
        s = rows[yr]
        print(f"{yr:>5} {np.median(s):>8.2f} [{np.percentile(s,5):>5.2f},{np.percentile(s,95):>5.2f}] {s.max():>10.2f}")


if __name__ == "__main__":
    main()
