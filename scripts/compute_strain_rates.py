"""Vertical strain rate with depth at every repeat ApRES site.

Reads the vertical-velocity profiles produced by
``compute_vertical_velocities.py`` and differentiates each one with respect to
depth, using the local-slope estimator in :mod:`apres.strainrate`.

The strain rate, not the velocity, is the comparable quantity between sites:
w(z) is referenced to the shallowest tracked segment at each site, so its
absolute value carries that arbitrary choice, while its gradient does not.

Outputs
-------
results/vertical_velocity/strain_profiles.nc

Usage
-----
    python scripts/compute_strain_rates.py [--window-m 60]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths  # noqa: E402
from apres.strainrate import strain_rate_profile  # noqa: E402

OUT = paths.VERTICAL_VELOCITY / "strain_profiles.nc"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-m", type=float, default=60.0,
                    help="full width of the local-fit depth window")
    args = ap.parse_args()

    import xarray as xr

    ds = xr.open_dataset(paths.VV_PROFILES)
    sites = [str(s) for s in ds["site"].values]
    n_max = ds.sizes["sample"]

    depth = np.full((len(sites), n_max), np.nan)
    eps = np.full((len(sites), n_max), np.nan)
    err = np.full((len(sites), n_max), np.nan)

    for i, _ in enumerate(sites):
        z = ds["depth"].values[i]
        w = ds["w"].values[i]
        sw = ds["sigma_w"].values[i]
        good = np.isfinite(z) & np.isfinite(w)
        if good.sum() < 10:
            continue
        p = strain_rate_profile(z[good], w[good], sigma_w=sw[good],
                                window_m=args.window_m)
        k = p.depth_m.size
        depth[i, :k] = p.depth_m
        eps[i, :k] = p.strain_rate_yr
        err[i, :k] = p.strain_rate_err_yr

    out = xr.Dataset(
        {
            "depth": (("site", "sample"), depth),
            "strain_rate": (("site", "sample"), eps),
            "strain_rate_err": (("site", "sample"), err),
            "distance_km": ("site", ds["distance_km"].values),
            "surface_m": ("site", ds["surface_m"].values),
            "thickness_m": ("site", ds["thickness_m"].values),
            "x_ps_m": ("site", ds["x_ps_m"].values),
            "y_ps_m": ("site", ds["y_ps_m"].values),
        },
        coords={"site": sites, "sample": np.arange(n_max)},
    )
    out["strain_rate"].attrs = {
        "units": "yr-1",
        "long_name": "vertical strain rate dw/dz",
    }
    out["strain_rate_err"].attrs = {
        "units": "yr-1",
        "long_name": "1-sigma uncertainty from the residual scatter of the local fit",
    }
    out.attrs = {
        "title": "Thwaites Glacier repeat ApRES vertical strain rate profiles",
        "window_m": args.window_m,
        "source": "scripts/compute_strain_rates.py",
    }
    out.to_netcdf(OUT)
    print(f"wrote {OUT}")

    finite = np.isfinite(eps)
    print(f"  sites:                 {len(sites)}")
    print(f"  samples with a slope:  {int(finite.sum())}")
    print(f"  strain rate (1/yr):    {np.nanpercentile(eps, 1):.2e} to {np.nanpercentile(eps, 99):.2e}")
    print(f"  median uncertainty:    {np.nanmedian(err):.2e}")
    frac = np.nanmean(np.abs(eps[finite]) > 2 * err[finite])
    print(f"  |eps| > 2 sigma at:    {100 * frac:.0f}% of samples")


if __name__ == "__main__":
    main()
