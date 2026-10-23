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
    """A well-sampled site with a typical shear share.

    With an even number of candidates the median sits exactly halfway between
    two sites, so ties are real; they go to the cleaner fit (smaller residual).
    """
    ok = df[(df["shear_resolved"] == 1) & (df["n_fitted"] >= MIN_SAMPLES_FOR_EXAMPLE)]
    if ok.empty:
        ok = df
    target = ok["shear_share"].median()
    ranked = ok.assign(_gap=(ok["shear_share"] - target).abs())
    return ranked.sort_values(["_gap", "residual_rms_m_yr"], kind="stable").iloc[0]


def main() -> None:
    style.use_paper_style()

    df = data.load_summary()
    ds = data.load_profiles()
    if df.empty:
        raise SystemExit("no sites survive the quality screen")
    lo, hi = data.fitting_window(df)

    fig, axs = plt.subplots(
        1, 4, figsize=(style.TWO_COL, 0.40 * style.TWO_COL),
        gridspec_kw={"wspace": 0.42, "top": 0.96, "bottom": 0.34,
                     "left": 0.07, "right": 0.99},
    )
    for ax in axs:
        style.grid(ax)
    every = 40          # one marker per 40 samples: series stay identifiable

    def series(n):
        """Colour and marker of basis degree n: L0 black diamonds, then the
        fixed order (red circles, blue squares, green, purple)."""
        if n == 0:
            return style.INK, "D"
        return style.CATEGORICAL[n - 1], style.MARKERS[n - 1]

    # ------------------------------------------------ (a) the orthonormal basis
    ax = axs[0]
    zeta = np.linspace(lo, hi, 400)
    basis = design_matrix(zeta, DEGREE, (lo, hi))
    basis_handles = []
    for n in range(DEGREE + 1):
        colour, marker = series(n)
        (h,) = ax.plot(basis[:, n], zeta, color=colour, marker=marker,
                       markevery=(n * 8, every), label=rf"$\tilde{{L}}_{n}$")
        basis_handles.append(h)
    ax.axvline(0.0, color=style.INK_MUTED, lw=0.6, zorder=1)
    ax.set_xlabel("Basis amplitude")
    ax.set_ylabel(r"$\zeta$ = depth / $H$")
    ax.set_ylim(hi, lo)

    # ------------------------------------------------------- (b) example profile
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
    po = np.argsort(plug.zeta)

    ax = axs[1]
    (h_meas,) = ax.plot(full.w_observed[order], zf, color=style.INK, lw=0.8,
                        label="ApRES measured")
    (h_plug,) = ax.plot(plug.w_model[po], plug.zeta[po], color=style.CATEGORICAL[0],
                        marker=style.MARKERS[0], markevery=(0, every),
                        label="degree 1 (plug flow)")
    (h_full,) = ax.plot(full.w_model[order], zf, color=style.CATEGORICAL[1],
                        marker=style.MARKERS[1], markevery=(every // 2, every),
                        label=f"degree {DEGREE}")
    ax.set_xlabel(r"$w$ (m yr$^{-1}$)")
    ax.set_ylabel(r"$\zeta$")
    ax.set_ylim(hi, lo)

    # ------------------------------------------- (c) departure from plug flow
    # In mm/yr: the shear signal is two orders of magnitude below w itself.
    ax = axs[2]
    plug_model = np.interp(zf, plug.zeta[po], plug.w_model[po])
    departure = 1e3 * (full.w_observed[order] - plug_model)
    shear_model = 1e3 * (full.w_model[order] - plug_model)
    ax.plot(departure, zf, color=style.INK, lw=0.8)
    ax.plot(shear_model, zf, color=style.CATEGORICAL[1], marker=style.MARKERS[1],
            markevery=(every // 2, every))
    ax.axvline(0.0, color=style.INK_MUTED, lw=0.6, zorder=1)
    ax.set_xlabel(r"$w - w_\mathrm{plug}$ (mm yr$^{-1}$)")
    ax.set_ylabel(r"$\zeta$")
    ax.set_ylim(hi, lo)

    # -------------------------------------------------- (d) coefficient spectrum
    ax = axs[3]
    degrees = np.arange(DEGREE + 1)
    cols = [f"c{n}" for n in degrees]
    coeffs = np.abs(df[cols].values)
    for c in coeffs:
        ax.plot(degrees, c, color="0.78", lw=0.4, zorder=1)
    h_iqr = ax.fill_between(
        degrees, np.nanpercentile(coeffs, 25, axis=0),
        np.nanpercentile(coeffs, 75, axis=0),
        color="0.45", alpha=0.35, lw=0, zorder=2, label="interquartile range",
    )
    (h_med,) = ax.plot(degrees, np.nanmedian(coeffs, axis=0), color=style.INK,
                       marker="D", zorder=3, label="median, all stations")
    ax.set_yscale("log")
    ax.set_xticks(degrees)
    ax.set_xlim(-0.2, DEGREE + 0.2)
    ax.set_xlabel("Polynomial degree $n$")
    ax.set_ylabel(r"$|c_n|$ (m yr$^{-1}$)")

    for a, letter in zip(axs, "abcd"):
        a.margins(y=0.02)
        style.panel_letter(a, letter, x=0.13, y=0.92)

    # Below the panels, clear of their x labels.
    style.bottom_legend(
        fig, [h_meas, h_plug, h_full, h_med, h_iqr, *basis_handles], ncol=5, y=0.15,
    )

    print(f"example site: {site} at {row['distance_km']:.1f} km, "
          f"A_s={row['shear_amplitude']:.3f}, A_p={row['plug_amplitude']:.3f} m/yr")
    style.save(fig, paths.FIGURES / "fig05_legendre.png")


if __name__ == "__main__":
    main()
