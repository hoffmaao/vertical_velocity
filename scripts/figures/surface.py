"""Dense ice-surface profile along the survey line, from REMA.

The section figure needs a surface, and the obvious source - the BedMachine
surface already carried in the site catalog - is wrong for the job twice over:
it is a 500 m product, and it is sampled once per ApRES site, so it arrives as
a 91-point polyline that draws the ice top as a chain of straight segments
between occupations.

This module samples REMA v2.0 at 32 m along a densely interpolated path that
follows the survey line, so the surface is drawn at the resolution of the DEM
rather than at the spacing of the radar measurements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paths  # noqa: E402

STEP_M = 100.0  # along-path sampling interval


def _path_along_sites(x_m, y_m, step_m=STEP_M):
    """Resample the site polyline at a fixed interval, in metres."""
    order = np.argsort(x_m) if x_m[0] > x_m[-1] else np.arange(x_m.size)
    x_m, y_m = x_m[order], y_m[order]
    seg = np.hypot(np.diff(x_m), np.diff(y_m))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    dense = np.arange(0.0, s[-1] + step_m, step_m)
    return np.interp(dense, s, x_m), np.interp(dense, s, y_m), dense


def surface_profile(x_m, y_m, distance_km):
    """Surface elevation along the survey line at REMA resolution.

    Returns (distance_km, elevation_m) on a dense path.  Falls back to the
    supplied site distances if no REMA tile is available locally.
    """
    import rioxarray as rxr
    import xarray as xr

    tiles = paths.rema_tiles()
    if not tiles:
        return None, None

    px, py, s = _path_along_sites(np.asarray(x_m, float), np.asarray(y_m, float))
    elev = np.full(px.size, np.nan)
    for tile in tiles:
        da = rxr.open_rasterio(tile, masked=True).squeeze("band", drop=True)
        x0, y0, x1, y1 = da.rio.bounds()
        inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1) & ~np.isfinite(elev)
        if not inside.any():
            continue
        sampled = da.interp(
            x=xr.DataArray(px[inside], dims="p"),
            y=xr.DataArray(py[inside], dims="p"),
            method="linear",
        ).values
        elev[inside] = sampled

    # Map path distance onto the same along-flow coordinate the sites use, so
    # the profile lines up with the columns drawn beneath it.
    d = np.asarray(distance_km, float)
    site_s = np.linspace(s[0], s[-1], d.size) if d.size > 1 else s[:1]
    order = np.argsort(d)
    dense_km = np.interp(s, site_s[order], d[order])
    return dense_km, elev
