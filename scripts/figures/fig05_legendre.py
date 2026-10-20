"""Figure 5 - the shifted-Legendre decomposition, and what it can resolve.

Four panels, left to right:

  * the orthonormal basis on the common fitting window.  Degree 0 is a uniform
    offset, degree 1 a uniform vertical strain rate - plug flow - and every
    higher degree is curvature in w(z), which under incompressibility requires
    a depth-varying horizontal velocity, that is, vertical shear.
  * one measured profile with the degree-1 and the full degree-4 fit.  Which
    site it is, and its shear and plug amplitudes, are printed when the script
    runs and belong in the caption rather than on the panel.  At this
    scale the two are indistinguishable, which is the point of the next panel.
  * the same profile's departure from plug flow, with the fitted shear terms.
    The shear signal is of order 0.04 m/yr on a profile that spans several
    m/yr, so it is only visible once the plug-flow part is removed.  This is
    also why the significance test has to be done carefully: the quantity
    being tested is two orders of magnitude below the measurement it is drawn
    from.
  * the coefficient magnitudes at every site.  Power falls by roughly a factor
    of twenty between degrees 1 and 2 and by a similar factor again at each
    higher degree, so degree 4 is where the expansion stops resolving ice and
    starts resolving radar noise.

Usage
-----
    python scripts/figures/fig05_legendre.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
from apres.legendre import design_matrix, fit_profile  # noqa: E402

DEGREE = 4
MIN_SAMPLES_FOR_EXAMPLE = 300


def choose_example(df):
    """A well-sampled site with a typical shear share."""
    ok = df[(df["shear_resolved"] == 1) & (df["n_fitted"] >= MIN_SAMPLES_FOR_EXAMPLE)]
    if ok.empty:
        ok = df
    target = ok["shear_share"].median()
    return ok.iloc[(ok["shear_share"] - target).abs().argsort().iloc[0]]


def main() -> None:
    style.use_paper_style()

    df = data.load_summary()
    ds = data.load_profiles()
    if df.empty:
        raise SystemExit("no sites survive the quality screen")
    lo, hi = data.fitting_window(df)

    # Each panel is filled by its curves from edge to edge, so a key placed
    # inside any of them would sit on the data.  All four legends go above
    # their axes instead, and the figure is made taller to carry them.
    fig, axs = plt.subplots(
        1, 4, figsize=(style.TWO_COL, 0.36 * style.TWO_COL),
        gridspec_kw={"wspace": 0.34, "width_ratios": [1.0, 1.0, 1.0, 1.15],
                     "top": 0.80, "bottom": 0.17, "left": 0.075, "right": 0.985},
    )

    def legend_above(ax, *, ncol, **kw):
        """Key centred above the axes, clear of the data."""
        return ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=ncol,
            fontsize=5.8, frameon=False, borderaxespad=0.0,
            columnspacing=0.9, handletextpad=0.5, labelspacing=0.35, **kw,
        )

    # ------------------------------------------------ the orthonormal basis
    ax = axs[0]
    zeta = np.linspace(lo, hi, 400)
    basis = design_matrix(zeta, DEGREE, (lo, hi))
    names = ["$L_0$", "$L_1$ plug", "$L_2$", "$L_3$", "$L_4$"]
    for n in range(DEGREE + 1):
        colour = style.INK_MUTED if n == 0 else style.CATEGORICAL[(n - 1) % len(style.CATEGORICAL)]
        ax.plot(basis[:, n], zeta, color=colour, lw=1.1, label=names[n])
    ax.axvline(0.0, color=style.GRID, lw=0.6, zorder=0)
    ax.set_xlabel("basis amplitude")
    ax.set_ylabel(r"normalised depth $\zeta = z/H$")
    ax.set_ylim(hi, lo)
    legend_above(ax, ncol=3, handlelength=1.1)

    # ------------------------------------------------------- example profile
    row = choose_example(df)
    site = str(row["site"])
    i = list(ds["site"].values).index(site)
    z = ds["depth"].values[i]
    w = ds["w"].values[i]
    sw = ds["sigma_w"].values[i]
    good = np.isfinite(z) & np.isfinite(w)
    H = float(row["thickness_m"])

    full = fit_profile(z[good], w[good], sigma_w=sw[good], thickness_m=H,
                       degree=DEGREE, window=(lo, hi))
    plug = fit_profile(z[good], w[good], sigma_w=sw[good], thickness_m=H,
                       degree=1, window=(lo, hi))
    if full is None or plug is None:
        raise SystemExit(f"could not fit the example site {site}")
    order = np.argsort(full.zeta)
    zf = full.zeta[order]

    ax = axs[1]
    ax.plot(full.w_observed[order], zf, color=style.CATEGORICAL[0], lw=0.9,
            label="measured")
    ax.plot(plug.w_model[np.argsort(plug.zeta)], np.sort(plug.zeta),
            color=style.INK, lw=0.9, ls=(0, (4, 2)), label="degree 1")
    ax.plot(full.w_model[order], zf, color=style.CATEGORICAL[1], lw=0.9,
            ls=(0, (1, 1.4)), label=f"degree {DEGREE}")
    ax.set_xlabel("$w$ (m yr$^{-1}$)")
    ax.set_ylim(hi, lo)
    ax.tick_params(labelleft=False)
    legend_above(ax, ncol=2, handlelength=1.3)

    # ------------------------------------------- departure from plug flow
    ax = axs[2]
    plug_model = np.interp(zf, np.sort(plug.zeta),
                           plug.w_model[np.argsort(plug.zeta)])
    departure = full.w_observed[order] - plug_model
    shear_model = full.w_model[order] - plug_model
    ax.plot(departure, zf, color=style.CATEGORICAL[0], lw=0.7, alpha=0.85,
            label="measured $-$ plug")
    ax.plot(shear_model, zf, color=style.CATEGORICAL[1], lw=1.2,
            label=r"fitted $n\geq2$")
    ax.axvline(0.0, color=style.GRID, lw=0.6, zorder=0)
    ax.set_xlabel("departure (m yr$^{-1}$)")
    ax.set_ylim(hi, lo)
    ax.tick_params(labelleft=False)
    legend_above(ax, ncol=2, handlelength=1.3)
    # -------------------------------------------------- coefficient spectrum
    ax = axs[3]
    degrees = np.arange(DEGREE + 1)
    cols = [f"c{n}" for n in degrees]
    coeffs = np.abs(df[cols].values)
    for series in coeffs:
        ax.plot(degrees, series, color=style.CATEGORICAL[0], lw=0.4, alpha=0.16,
                zorder=1)
    median = np.nanmedian(coeffs, axis=0)
    ax.plot(degrees, median, color=style.INK, lw=1.4, marker="o", ms=3.4,
            mec="white", mew=0.5, zorder=3, label="median")
    ax.fill_between(
        degrees, np.nanpercentile(coeffs, 25, axis=0),
        np.nanpercentile(coeffs, 75, axis=0),
        color=style.INK, alpha=0.13, lw=0, zorder=2, label="IQR",
    )
    ax.set_yscale("log")
    ax.set_xticks(degrees)
    # Degree is a non-negative integer, so the axis stops at zero rather than
    # opening a margin into values the quantity cannot take.  The markers at
    # the ends are left unclipped so they still draw in full.
    ax.set_xlim(0, DEGREE)
    for artist in ax.lines:
        artist.set_clip_on(False)
    ax.set_xlabel("polynomial degree $n$")
    ax.set_ylabel("$|c_n|$ (m yr$^{-1}$)")
    legend_above(ax, ncol=2, handlelength=1.2)

    for a in axs:
        a.margins(y=0.02)

    print(f"example site: {site} at {row['distance_km']:.1f} km, "
          f"A_s={row['shear_amplitude']:.3f}, A_p={row['plug_amplitude']:.3f} m/yr")
    style.save(fig, paths.FIGURES / "fig05_legendre.png")


if __name__ == "__main__":
    main()
