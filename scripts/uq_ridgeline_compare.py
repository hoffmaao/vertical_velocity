r"""SSA vs Hybrid retreat-distribution ridgeline (IPCC multi-method style).

Two model ensembles (Hybrid: sample_vaf_{'',b..i}*.npz; SSA: sample_vaf_ssa[0-9]*.npz),
each a 50-yr VAF trajectory from a posterior θ-draw. Stacks the across-ensemble
Thwaites sea-level-contribution distribution by horizon year, with Hybrid and SSA
overlaid per row (à la the IPCC SLR ridge with multiple methods). Shared x-axis:
mm SLE (bottom) / VAF loss Gt (top).

Writes figures/uq_ridgeline_compare.png.
"""
import glob
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
VAF0 = 46836.1
GT_PER_MM = 361.8
ROW_YEARS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def load(patterns):
    trajs = []
    for pat in patterns:
        for f in glob.glob(str(DATA / "results" / pat)):
            d = np.load(f)
            if not bool(d["failed"]):
                trajs.append((np.asarray(d["time"]), np.asarray(d["vaf_Gt"])))
    return trajs


def sle_rows(trajs):
    return {yr: (VAF0 - np.array([np.interp(yr, t, v) for (t, v) in trajs])) / GT_PER_MM
            for yr in ROW_YEARS}


def main():
    hyb = load(["sample_vaf_[0-9b-i]*.npz"])
    ssa = load(["sample_vaf_ssa[0-9]*.npz"])  # Coulomb only; excludes sample_vaf_ssaw*
    print(f"Hybrid: {len(hyb)} realizations | SSA: {len(ssa)}")
    rows_h, rows_s = sle_rows(hyb), sle_rows(ssa)
    allv = np.concatenate([np.concatenate(list(rows_h.values())), np.concatenate(list(rows_s.values()))])
    xmax = np.percentile(allv, 99.5) * 1.1

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 8.5))
    xs = np.linspace(-0.3, xmax, 600)
    spacing, scale = 1.0, 1.9
    nrow = len(ROW_YEARS)
    for i, yr in enumerate(ROW_YEARS):
        base = (nrow - 1 - i) * spacing
        for rows, color, lbl in [(rows_h, "C0", "Hybrid"), (rows_s, "C3", "SSA")]:
            v = rows[yr]
            dens = gaussian_kde(v)(xs); dens = dens / dens.max() * scale
            ax.fill_between(xs, base, base + dens, color=color, alpha=0.45,
                            zorder=i, label=lbl if i == 0 else None)
            ax.plot(xs, base + dens, color=color, lw=1.0, zorder=i)
            ax.plot(np.median(v), base, "|", color=color, ms=9, mew=1.5, zorder=i + 0.5)
        ax.hlines(base, xs[0], xmax, color="0.6", lw=0.5, zorder=i)
        ax.text(xs[0] - 0.2, base + 0.05, f"{yr} yr", ha="right", va="bottom", fontweight="bold", fontsize=11)

    ax.set_yticks([]); ax.set_xlim(xs[0] - 1.6, xmax)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Thwaites sea-level contribution  [mm SLE]", fontsize=13)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    secax = ax.secondary_xaxis("top", functions=(lambda x: x * GT_PER_MM, lambda g: g / GT_PER_MM))
    secax.set_xlabel("VAF loss  [Gt]", fontsize=11, color="0.4")
    ax.set_title(f"Thwaites 50-yr retreat: SSA vs Hybrid posterior distributions\n"
                 f"(Hybrid N={len(hyb)}, SSA N={len(ssa)}; spread = fluidity/friction uncertainty)",
                 fontsize=12.5)
    fig.tight_layout()
    fp = DATA / "figures" / "uq_ridgeline_compare.png"
    fig.savefig(str(fp), dpi=200, bbox_inches="tight")
    print(f"Saved {fp}")
    print(f"\n{'year':>5} {'Hybrid med (mm)':>16} {'SSA med (mm)':>14} {'Δ (mm)':>8}")
    for yr in ROW_YEARS:
        mh, ms = np.median(rows_h[yr]), np.median(rows_s[yr])
        print(f"{yr:>5} {mh:>16.2f} {ms:>14.2f} {ms-mh:>8.2f}")


if __name__ == "__main__":
    main()
