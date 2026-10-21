"""Decompose the measured vertical-velocity profiles on a common depth window.

This is deliberately a separate stage from ``compute_vertical_velocities.py``.
The cross-correlation that produces w(z) costs minutes per site; the
decomposition costs milliseconds.  Keeping them apart means the fitting window,
the polynomial degree and the screening thresholds can be revised without
re-running the radar processing.

The window is chosen as the deepest normalised depth that *every* retained site
resolves, so that a coefficient means the same thing at every site.  Sites that
do not reach it are reported, not silently dropped.

Depth is normalised by the BedMachine thickness rather than by the ApRES bed
pick.  The basal return on this transect stands only 10-30 dB above the noise
floor, the pick disagrees with BedMachine by a median of 47 m and repeats
between seasons at only about half the sites, and because the thickness sets
both the normalised coordinate and the conversion from the degree-1
coefficient to a strain rate, an unstable pick injects that instability into
every derived quantity.  A consistent external thickness costs a little
accuracy at any one site and buys comparability along the whole transect.

Also reports, for each site, the depth-averaged vertical strain rate over the
same window.  That is the mean of the measured dw/dz profile, not the slope of
the fitted polynomial, so it does not inherit the polynomial's behaviour and
can be compared directly against a depth-averaged model strain rate.

Outputs
-------
results/vertical_velocity/legendre_fits.csv

Usage
-----
    python scripts/fit_legendre.py [--degree 4] [--min-segments 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths  # noqa: E402
from apres.legendre import fit_profile  # noqa: E402

OUT = paths.VERTICAL_VELOCITY / "legendre_fits.csv"

# Fine-alignment windows are 6 m wide stepped every 2 m, so adjacent samples
# share two thirds of their data.  The F-test degrees of freedom are deflated
# by this factor towards an effective sample size.
OVERSAMPLING = 3.0


def choose_window(zeta_max: np.ndarray, *, quantile: float = 0.10,
                  floor: float = 0.30) -> tuple[float, float]:
    """Common normalised-depth window for the survey.

    Taking the strict minimum of ``zeta_max`` would let one shallow profile set
    the window for all of them, so a low quantile is used and the sites that
    fall short are reported instead.
    """
    top = float(np.nanquantile(zeta_max, quantile))
    return (0.06, max(top, floor))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--min-segments", type=int, default=100)
    ap.add_argument("--window", type=float, nargs=2, default=None,
                    help="override the normalised depth window, e.g. 0.06 0.42")
    args = ap.parse_args()

    import xarray as xr
    from scipy import stats

    ds = xr.open_dataset(paths.VV_PROFILES)
    summary = pd.read_csv(paths.VV_SUMMARY).set_index("site")
    # The pair catalog is the authority on what kind of site this is and how
    # far off the survey line it sits; the velocity summary may predate those
    # columns if it was written by an older run.
    catalog = pd.read_csv(paths.PAIR_CATALOG).set_index("site")
    for col in ("site_type", "offline_km"):
        if col in catalog.columns:
            summary[col] = catalog[col]
    if "site_type" not in summary.columns:
        summary["site_type"] = "point"
    sites = [str(s) for s in ds["site"].values]

    # Measured strain-rate profiles, used for the depth average.
    strain_path = paths.VERTICAL_VELOCITY / "strain_profiles.nc"
    strain = xr.open_dataset(strain_path) if strain_path.exists() else None
    strain_sites = [str(s) for s in strain["site"].values] if strain is not None else []

    depth = ds["depth"].values
    thickness = np.array([
        float(summary["bm_thickness_m"].get(s, np.nan)) for s in sites
    ])
    # Fall back to the ApRES pick only where BedMachine has no value.
    fallback = ds["thickness_m"].values
    thickness = np.where(np.isfinite(thickness), thickness, fallback)
    zeta_max = np.array(
        [np.nanmax(depth[i]) / thickness[i] if np.isfinite(thickness[i]) else np.nan
         for i in range(len(sites))]
    )
    n_seg = np.array([int(np.isfinite(depth[i]).sum()) for i in range(len(sites))])

    eligible = n_seg >= args.min_segments
    window = tuple(args.window) if args.window else choose_window(zeta_max[eligible])
    print(f"fitting window: zeta in [{window[0]:.3f}, {window[1]:.3f}]")
    print(f"  = the top {100 * window[1]:.0f}% of the ice column")

    rows = []
    for i, site in enumerate(sites):
        z = depth[i]
        w = ds["w"].values[i]
        sw = ds["sigma_w"].values[i]
        good = np.isfinite(z) & np.isfinite(w)
        H = float(thickness[i])
        rec = {
            "site": site,
            "n_segments": int(good.sum()),
            "thickness_m": H,
            "thickness_source": "BedMachine",
            "zeta_max": float(zeta_max[i]),
            "reaches_window": bool(zeta_max[i] >= window[1]),
            "passes_screen": bool(n_seg[i] >= args.min_segments),
        }
        fit = (
            fit_profile(z[good], w[good], sigma_w=sw[good], thickness_m=H,
                        degree=args.degree, window=window)
            if good.sum() >= 5 and np.isfinite(H)
            else None
        )
        if fit is None:
            rec["status"] = "no-fit"
            rows.append(rec)
            continue

        higher = fit.coefficients[2:]
        plug_amp = float(abs(fit.coefficients[1]))
        shear_amp = float(np.sqrt(np.sum(higher ** 2)))
        total = float(np.hypot(plug_amp, shear_amp))

        # Depth-averaged measured strain rate over the same window.
        eps_mean = np.nan
        eps_mean_sd = np.nan
        if site in strain_sites:
            j = strain_sites.index(site)
            zs = strain["depth"].values[j]
            es = strain["strain_rate"].values[j]
            m = np.isfinite(zs) & np.isfinite(es)
            m &= (zs >= window[0] * H) & (zs <= window[1] * H)
            if m.sum() >= 5:
                eps_mean = float(np.mean(es[m]))
                eps_mean_sd = float(np.std(es[m]))

        # F-test against an effective sample size rather than the raw count.
        n_eff = max(fit.n_samples / OVERSAMPLING, 6.0)
        dof_eff = max(n_eff - (args.degree + 1), 1.0)
        p_eff = float(stats.f.sf(fit.f_statistic, args.degree - 1, dof_eff))

        # Share of the resolved deformation carried by shear.  A plain
        # shear/plug quotient is unbounded and diverges wherever the plug term
        # passes through zero, which it does several times along this
        # transect; the share is bounded on [0, 1], is monotonic in the
        # quotient, and stays finite there.
        shear_share = (shear_amp / (shear_amp + plug_amp)) if (shear_amp + plug_amp) > 0 else np.nan

        rec.update(
            status="ok",
            shear_share=shear_share,
            n_fitted=fit.n_samples,
            strain_rate_yr=fit.strain_rate_linear,
            strain_rate_err_yr=fit.strain_rate_linear_err,
            plug_amplitude=plug_amp,
            shear_amplitude=shear_amp,
            shear_ratio=(shear_amp / total) if total > 0 else np.nan,
            shear_fraction=fit.shear_fraction,
            residual_rms_m_yr=fit.residual_rms,
            strain_rate_depth_mean_yr=eps_mean,
            strain_rate_depth_sd_yr=eps_mean_sd,
            f_statistic=fit.f_statistic,
            p_value=fit.p_value,
            p_value_eff=p_eff,
            significant_shear=int(fit.significant),
            significant_shear_eff=int(p_eff < 0.05),
            residual_autocorr_time=fit.residual_autocorr_time,
            shear_chi2=fit.shear_chi2,
            shear_p_value=fit.shear_p_value,
            shear_resolved=int(fit.shear_significant),
            shear_amplitude_err=fit.shear_amplitude_err,
            **{f"c{k}": float(c) for k, c in enumerate(fit.coefficients)},
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    meta = ["distance_km", "lon", "lat", "x_ps_m", "y_ps_m",
            "bm_thickness_m", "bm_surface_m", "bm_bed_m", "speed_m_yr",
            "bed_depth_m", "bed_ok", "mean_coherence", "z_max_m",
            "site_type", "offline_km"]
    have = [c for c in meta if c in summary.columns]
    out = out.join(summary[have], on="site").sort_values("distance_km")
    out.attrs["window"] = window
    out["window_lo"] = window[0]
    out["window_hi"] = window[1]
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")

    ok = out[(out["status"] == "ok") & out["passes_screen"]]
    print()
    print(f"  sites fitted:                {len(ok)} of {len(out)}")
    print(f"  reach the window:            {int(ok['reaches_window'].sum())}")
    print(f"  strain rate (1/yr):          {ok['strain_rate_yr'].min():.2e} to {ok['strain_rate_yr'].max():.2e}")
    print(f"  plug amplitude (m/yr):       {ok['plug_amplitude'].min():.2f} to {ok['plug_amplitude'].max():.2f}")
    print(f"  shear amplitude (m/yr):      {ok['shear_amplitude'].min():.2f} to {ok['shear_amplitude'].max():.2f}")
    print(f"  shear share:                 {ok['shear_share'].min():.2f} to {ok['shear_share'].max():.2f}")
    print(f"  F-test significant (raw n):  {int(ok['significant_shear'].sum())} of {len(ok)}")
    print(f"  shear resolved (autocorr):   {int(ok['shear_resolved'].sum())} of {len(ok)}")
    print(f"  residual autocorr time:      {ok['residual_autocorr_time'].median():.1f} samples")
    if ok["strain_rate_depth_mean_yr"].notna().any():
        v = ok["strain_rate_depth_mean_yr"]
        print(f"  depth-mean strain (1/yr):    {v.min():.2e} to {v.max():.2e}")
    for col in ("shear_ratio", "plug_amplitude", "strain_rate_yr"):
        v = ok[col].values
        rough = np.median(np.abs(np.diff(v)))
        spread = np.subtract(*np.percentile(v, [75, 25]))
        print(f"  along-flow roughness {col:16s} {rough:.4f}  (IQR {spread:.4f})")


if __name__ == "__main__":
    main()
