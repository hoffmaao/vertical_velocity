"""Vertical strain rate implied by the divergence of surface velocity.

Incompressibility ties the vertical strain rate to the horizontal one:

    eps_zz = -(du/dx + dv/dy),

so a surface velocity field predicts, at every ApRES site, the vertical strain
rate the column must have if the ice is incompressible and the horizontal
strain rate measured at the surface holds through the depth being sampled.
Comparing that prediction against the strain rate the radar actually measures
is the test of whether the column deforms as the surface implies - which is
exactly the plug-flow assumption the paper is examining.

Velocities are the 2022 and 2023 Amundsen Sea Embayment annual mosaics
(250 m), averaged over the years that bracket the repeat survey.

Smoothing matters and is not incidental.  Differentiating a feature-tracked
velocity field amplifies its noise, and the divergence of a raw 250 m mosaic
is dominated by it.  The field is therefore smoothed before differencing, over
a length comparable to the ice thickness (~2.2 km here), which is the scale
below which surface strain rates are not transmitted coherently through the
column anyway.  The sensitivity to that choice is reported when this runs.

Uncertainty
-----------
Two contributions are reported and combined.

The first is propagated from the mosaics' own ``ex``/``ey`` error fields.
Smoothing and differencing is one linear operator, so its impulse response is
computed numerically and the error is propagated through the kernel weights:
sigma^2 = sum_ij w_ij^2 sigma_u,ij^2, taking the error field as locally
uniform.  This assumes the per-pixel velocity errors are independent, which
they are not - feature-tracking errors are correlated over the template size -
so this term is a lower bound.

The second is the sensitivity to the smoothing length itself, taken as half
the spread of the strain rate across the 2, 4 and 8 km smoothings.  Choosing a
smoothing length is a real and unavoidable choice, and on this transect it
moves the answer more than the measurement error does.

Outputs
-------
results/sites/surface_strain.csv

Usage
-----
    python scripts/compute_surface_strain.py [--smooth-m 4000]
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

OUT = paths.SITES / "surface_strain.csv"
SMOOTH_SCALES_M = (2000.0, 4000.0, 8000.0)


def load_velocity(region_m, pad_m: float = 20_000.0, fill: bool = True):
    """Mean vx, vy over the survey years, clipped to the region.

    The ASE mosaics leave one contiguous 15 km hole through the middle of the
    transect, which is exactly where the measured strain rate is most strongly
    negative.  With ``fill`` the hole is patched from the full-coverage
    MEaSUREs 2019-2020 mosaic before differencing.  That mixes epochs, so the
    patched sites are flagged in the output rather than being passed off as
    survey-period data.
    """
    import rioxarray as rxr
    import xarray as xr

    x0, x1, y0, y1 = region_m
    out = {}
    for comp in ("vx", "vy"):
        stack = []
        for year in paths.ASE_VELOCITY_YEARS:
            da = rxr.open_rasterio(paths.ase_velocity(year, comp), masked=True)
            # sortby, not isel(y=slice(None, None, -1)): on the lazy rasterio
            # backend a negative-step slice returns an empty array (xarray
            # 2025.12, rioxarray 0.18) while the y coordinate keeps its length.
            da = da.squeeze("band", drop=True).sortby("y")
            da = da.sel(x=slice(x0 - pad_m, x1 + pad_m),
                        y=slice(y0 - pad_m, y1 + pad_m))
            stack.append(da)
        out[comp] = xr.concat(stack, dim="year").mean(dim="year", skipna=True)

    filled = np.zeros(out["vx"].shape, dtype=bool)
    if fill and paths.MEASURES_VELOCITY.exists():
        ref = xr.open_dataset(paths.MEASURES_VELOCITY)
        for comp, name in (("vx", "VX"), ("vy", "VY")):
            da = ref[name].sortby("y")
            on_grid = da.interp(x=out["vx"]["x"], y=out["vx"]["y"], method="linear")
            gap = ~np.isfinite(out[comp].values)
            filled |= gap
            out[comp] = out[comp].where(np.isfinite(out[comp]), on_grid)
    return out["vx"], out["vy"], filled


def operator_weight_sq(vx, smooth_m: float) -> float:
    """Sum of squared kernel weights for (Gaussian smooth then d/dx).

    The two steps compose into a single linear filter, so its impulse response
    gives the weights that multiply each pixel's error.  Computed numerically
    rather than analytically so it stays correct if the smoothing or the
    difference stencil changes.
    """
    from scipy import ndimage

    dx = float(np.abs(np.median(np.diff(vx["x"].values))))
    sigma = smooth_m / dx
    half = int(np.ceil(4 * sigma)) + 2
    n = 2 * half + 1
    delta = np.zeros((n, n))
    delta[half, half] = 1.0
    impulse = ndimage.gaussian_filter(delta, sigma, mode="constant")
    kernel = np.gradient(impulse, dx, axis=1)
    return float(np.sum(kernel ** 2))


def load_errors(region_m, template, pad_m: float = 20_000.0):
    """Mean ex, ey over the survey years, on the velocity grid."""
    import rioxarray as rxr
    import xarray as xr

    x0, x1, y0, y1 = region_m
    out = {}
    for comp in ("ex", "ey"):
        stack = []
        for year in paths.ASE_VELOCITY_YEARS:
            try:
                da = rxr.open_rasterio(paths.ase_velocity(year, comp), masked=True)
            except FileNotFoundError:
                continue
            # sortby, not isel(y=slice(None, None, -1)): on the lazy rasterio
            # backend a negative-step slice returns an empty array (xarray
            # 2025.12, rioxarray 0.18) while the y coordinate keeps its length.
            da = da.squeeze("band", drop=True).sortby("y")
            stack.append(da.sel(x=slice(x0 - pad_m, x1 + pad_m),
                                y=slice(y0 - pad_m, y1 + pad_m)))
        if not stack:
            return None, None
        merged = xr.concat(stack, dim="year").mean(dim="year", skipna=True)
        out[comp] = merged.interp(x=template["x"], y=template["y"], method="nearest")
    return out.get("ex"), out.get("ey")


def divergence(vx, vy, smooth_m: float):
    """Horizontal divergence of a smoothed velocity field, in 1/yr."""
    from scipy import ndimage

    dx = float(np.abs(np.median(np.diff(vx["x"].values))))
    dy = float(np.abs(np.median(np.diff(vx["y"].values))))
    sigma = smooth_m / np.mean([dx, dy])

    def smooth(a):
        arr = np.asarray(a.values, dtype=float)
        finite = np.isfinite(arr)
        filled = np.where(finite, arr, 0.0)
        num = ndimage.gaussian_filter(filled, sigma, mode="nearest")
        den = ndimage.gaussian_filter(finite.astype(float), sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0.05, num / den, np.nan)
        return np.where(finite | (den > 0.5), out, np.nan)

    u = smooth(vx)
    v = smooth(vy)
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    return du_dx + dv_dy


def sample(field, vx, x_m, y_m):
    """Nearest-cell sample of a 2D array laid out on vx's grid."""
    xs = vx["x"].values
    ys = vx["y"].values
    ix = np.clip(np.searchsorted(xs, x_m), 0, xs.size - 1)
    iy = np.clip(np.searchsorted(ys, y_m), 0, ys.size - 1)
    return field[iy, ix]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smooth-m", type=float, default=4000.0,
                    help="Gaussian smoothing length before differencing")
    args = ap.parse_args()

    pairs = pd.read_csv(paths.PAIR_CATALOG)
    x = pairs["x_ps_m"].values
    y = pairs["y_ps_m"].values
    region = (x.min(), x.max(), y.min(), y.max())
    vx, vy, filled = load_velocity(region)

    rows = {"site": pairs["site"].values, "distance_km": pairs["distance_km"].values}
    rows["velocity_filled"] = sample(filled.astype(float), vx, x, y) > 0.5
    for scale in sorted({*SMOOTH_SCALES_M, args.smooth_m}):
        div = divergence(vx, vy, scale)
        rows[f"eps_zz_surface_{int(scale)}m"] = -sample(div, vx, x, y)

    out = pd.DataFrame(rows)
    out["eps_zz_surface"] = out[f"eps_zz_surface_{int(args.smooth_m)}m"]

    # --- uncertainty -------------------------------------------------------
    scales = sorted({*SMOOTH_SCALES_M, args.smooth_m})
    stack = np.column_stack([out[f"eps_zz_surface_{int(s)}m"].values for s in scales])
    out["eps_zz_surface_err_scale"] = 0.5 * (np.nanmax(stack, axis=1)
                                             - np.nanmin(stack, axis=1))

    ex, ey = load_errors(region, vx)
    if ex is not None:
        w2 = operator_weight_sq(vx, args.smooth_m)
        sx = sample(np.asarray(ex.values, float), vx, x, y)
        sy = sample(np.asarray(ey.values, float), vx, x, y)
        out["eps_zz_surface_err_vel"] = np.sqrt(w2 * (sx ** 2 + sy ** 2))
    else:
        out["eps_zz_surface_err_vel"] = np.nan

    out["eps_zz_surface_err"] = np.sqrt(
        np.nan_to_num(out["eps_zz_surface_err_vel"].values) ** 2
        + np.nan_to_num(out["eps_zz_surface_err_scale"].values) ** 2
    )
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")

    print(f"\n  sites:                 {len(out)}")
    print(f"  smoothing used:        {args.smooth_m:.0f} m")
    print(f"  gap-filled from MEaSUREs: {int(out['velocity_filled'].sum())} sites")
    for col, tag in (("eps_zz_surface_err_vel", "from ex/ey"),
                     ("eps_zz_surface_err_scale", "from smoothing"),
                     ("eps_zz_surface_err", "combined")):
        v = out[col].dropna()
        if len(v):
            print(f"  sigma {tag:16s} median {v.median():.2e}  "
                  f"p90 {v.quantile(0.9):.2e} /yr")
    for scale in sorted({*SMOOTH_SCALES_M, args.smooth_m}):
        v = out[f"eps_zz_surface_{int(scale)}m"]
        print(f"  {int(scale):5d} m smoothing: "
              f"{np.nanpercentile(v, 5):+.2e} to {np.nanpercentile(v, 95):+.2e} /yr "
              f"(median {np.nanmedian(v):+.2e}, {int(v.notna().sum())} finite)")


if __name__ == "__main__":
    main()
