r"""Retreat distribution vs time on a SINGLE axis, distributions colored by horizon year.

Same ensemble as uq_ridgeline.py, but instead of stacking the per-year distributions as a
ridgeline they are overlaid on one baseline; a colorbar encodes the year. The x-axis is
shared between Thwaites sea-level contribution [mm SLE] (bottom) and VAF loss [Gt] (top).

Writes figures/uq_density_timeline.png.
"""
import glob
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
VAF0 = 46836.1
GT_PER_MM = 361.8
ROW_YEARS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def main():
    trajs = []
    for f in sorted(glob.glob(str(DATA / "results" / "sample_vaf_[0-9b-i]*.npz"))):  # hybrid only; SSA runs are sample_vaf_ssa*/ssaw*
        d = np.load(f)
        if bool(d["failed"]):
            continue
        trajs.append((np.asarray(d["time"]), np.asarray(d["vaf_Gt"])))
    N = len(trajs)
    print(f"loaded {N} realizations")

    rows = {yr: (VAF0 - np.array([np.interp(yr, t, v) for (t, v) in trajs])) / GT_PER_MM
            for yr in ROW_YEARS}
    xmax = np.percentile(np.concatenate(list(rows.values())), 99.7) * 1.12

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.linspace(-0.3, xmax, 600)
    cmap = cm.viridis
    norm = Normalize(vmin=ROW_YEARS[0], vmax=ROW_YEARS[-1])

    for yr in ROW_YEARS:
        sle = rows[yr]
        dens = gaussian_kde(sle)(xs)
        dens = dens / dens.max()                 # equal peak height → readable across years
        c = cmap(norm(yr))
        ax.fill_between(xs, 0, dens, color=c, alpha=0.35, lw=0, zorder=yr)
        ax.plot(xs, dens, color=c, lw=2.0, zorder=yr + 100)

    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.set_ylabel("probability density (peak-normalized)", fontsize=11)
    ax.set_xlim(xs[0], xmax)
    ax.set_xlabel("Thwaites sea-level contribution  [mm SLE]", fontsize=13)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)

    # shared x-axis, second scale = VAF loss [Gt]
    secax = ax.secondary_xaxis("top", functions=(lambda x: x * GT_PER_MM, lambda g: g / GT_PER_MM))
    secax.set_xlabel("VAF loss  [Gt]", fontsize=13)

    # colorbar = time
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cb.set_label("forward time  [yr]", fontsize=12)
    cb.set_ticks(ROW_YEARS)

    ax.set_title(f"Thwaites retreat distribution vs time  (N={N} realizations, "
                 f"each a distinct fluidity/friction initialization)", fontsize=12.5, pad=28)
    fig.tight_layout()
    fp = DATA / "figures" / "uq_density_timeline.png"
    fig.savefig(str(fp), dpi=200, bbox_inches="tight")
    print(f"Saved {fp}")


if __name__ == "__main__":
    main()
