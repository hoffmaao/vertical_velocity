"""Figure 3 - near-surface strain rate against what the surface velocity implies.

Incompressibility gives eps_zz = -(du/dx + dv/dy), so a surface velocity field
predicts a vertical strain rate at every site.  That prediction is a statement
about the surface, and it should hold best just below it, so the radar side of
the comparison is averaged over 100-400 m depth: below the firn, and shallow
enough that the surface strain rate ought still to describe the column.  If the
ice deformed as plug flow with the surface strain rate carried through the
depth being sampled, the points would fall on the 1:1 line.

Each axis is a strain rate in the same units, so the panel is square and the
two axes carry identical limits.  Anything else would make the departure from
1:1 a property of the aspect ratio.

Velocities are the 2022 and 2023 Amundsen Sea Embayment annual mosaics,
smoothed over 4 km before differencing, with the one 15 km hole through the
middle of the transect patched from the MEaSUREs 2019-2020 mosaic.

Both axes carry uncertainties, and they come from different places.  The
vertical bars are the spread of the measured strain rate through the
100-400 m interval, not the standard error of its mean: neighbouring radar
samples are heavily correlated, so a standard error would be optimistic.  The
horizontal bars are dominated not by the velocity measurement error, which
propagates to about 4e-5 yr-1, but by the choice of smoothing length before
differencing, which moves the answer by about 1.7e-3 yr-1 - roughly forty
times more, and comparable to the signal itself.  That is the honest position:
a surface-divergence strain rate on this transect is barely resolved, and the
scatter here is not attributable to the radar alone.

Usage
-----
    python scripts/figures/fig03_strain_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import data  # noqa: E402
import paths  # noqa: E402
import style  # noqa: E402

DEPTH_MIN_M = 100.0
DEPTH_MAX_M = 400.0
MIN_SAMPLES = 5


def near_surface_strain() -> pd.DataFrame:
    """Mean measured strain rate over the near-surface depth interval."""
    import xarray as xr

    ds = xr.open_dataset(paths.VERTICAL_VELOCITY / "strain_profiles.nc")
    rows = []
    for i, site in enumerate(str(s) for s in ds["site"].values):
        z = ds["depth"].values[i]
        e = ds["strain_rate"].values[i]
        keep = (
            np.isfinite(z) & np.isfinite(e)
            & (z >= DEPTH_MIN_M) & (z <= DEPTH_MAX_M)
        )
        if keep.sum() < MIN_SAMPLES:
            continue
        rows.append(
            {
                "site": site,
                "eps_apres": float(np.mean(e[keep])),
                # Spread of the strain rate through the interval, not the
                # standard error of its mean: neighbouring samples are heavily
                # correlated, so the spread is the honest error bar.
                "eps_apres_sd": float(np.std(e[keep])),
                "n_samples": int(keep.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    style.use_paper_style()

    merged = near_surface_strain().merge(
        pd.read_csv(paths.SITES / "surface_strain.csv"), on="site", how="inner"
    )
    keep = set(data.load_summary()["site"])
    merged = merged[merged["site"].isin(keep)]

    x = merged["eps_zz_surface"].values
    y = merged["eps_apres"].values
    yerr = merged["eps_apres_sd"].values
    xerr = merged.get("eps_zz_surface_err", pd.Series(np.nan, index=merged.index)).values
    good = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(1.18 * style.ONE_COL, 1.18 * style.ONE_COL))

    lim = float(np.nanmax(np.abs(np.concatenate([x[good], y[good]])))) * 1.12
    ax.plot([-lim, lim], [-lim, lim], color=style.INK_MUTED, lw=0.8,
            ls=(0, (4, 3)), zorder=1, label="1:1")
    ax.axhline(0.0, color=style.GRID, lw=0.6, zorder=0)
    ax.axvline(0.0, color=style.GRID, lw=0.6, zorder=0)

    ax.errorbar(
        x[good], y[good], yerr=yerr[good], xerr=xerr[good], fmt="none",
        ecolor=style.INK_MUTED, elinewidth=0.5, alpha=0.5, zorder=2,
    )
    ax.scatter(x[good], y[good], s=24, color=style.CATEGORICAL[0],
               edgecolor="white", linewidth=0.4, zorder=4,
               label=f"ApRES site ($n={int(good.sum())}$)")

    # Least squares on every site, and on the survey-period velocities alone.
    # The fit statistics go in the legend rather than in a text block, so the
    # only words on the panel are keyed to something drawn on it.
    # Ordinary least squares.  The fit statistics go in the legend rather than
    # in a text block, so the only words on the panel are keyed to something
    # drawn on it.
    if good.sum() >= 3:
        slope, intercept = np.polyfit(x[good], y[good], 1)
        r = np.corrcoef(x[good], y[good])[0, 1]
        xs = np.array([-lim, lim])
        ax.plot(xs, slope * xs + intercept, color=style.INK, lw=1.1, zorder=5,
                label=f"fit: slope {slope:.2f}, $R^2$ {r ** 2:.2f}")
        print(f"  slope={slope:+.3f} intercept={intercept:+.2e} "
              f"R2={r ** 2:.3f} n={int(good.sum())}")
        print(f"  median sigma_x={np.nanmedian(xerr[good]):.2e} "
              f"sigma_y={np.nanmedian(yerr[good]):.2e} /yr")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"surface $\dot\varepsilon_{zz}$ from $-\nabla\!\cdot\mathbf{u}$ (yr$^{-1}$)")
    ax.set_ylabel(rf"ApRES $\dot\varepsilon_{{zz}}$, {DEPTH_MIN_M:.0f}-{DEPTH_MAX_M:.0f} m (yr$^{{-1}}$)")
    # Lower right: the upper left is where the 1:1 line runs and where the
    # extensional sites sit.
    ax.legend(loc="lower right", fontsize=6.6, borderaxespad=0.4,
              frameon=True, framealpha=1.0, facecolor="white",
              edgecolor=style.GRID)

    style.save(fig, paths.FIGURES / "fig03_strain_regression.png")


if __name__ == "__main__":
    main()
