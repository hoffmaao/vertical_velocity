"""Figure 1 - setting of the repeat ApRES survey on Thwaites Glacier.

A regional panel covering the whole model domain, showing surface speed over
the Thwaites trunk during the survey period, the grounding line, the ice-shelf
front, the model domain outline, and the 88 sites occupied in both field
seasons, with an Antarctic locator inset.  The three autonomous stations, which
logged through the year between the two visits, are drawn the same way as every
other site: they are the same instrument making the same measurement.

Data
----
Surface speed is the mean of the 2022 and 2023 Amundsen Sea Embayment annual
mosaics (250 m, EPSG:3031), which together bracket the December 2022 to
December 2023 interval spanned by the repeat measurements.  Those two years
cover about three quarters of this map; the remaining quarter, where feature
tracking fails over the low-contrast interior, is filled from the MEaSUREs
2019-2020 mosaic so that the field is continuous.  The background is
the MOA 125 m high-pass composite, which suppresses the broad brightness field
and leaves surface slope, so it reads as shaded relief beneath the colour.

Both boundaries are epoch-matched to the survey.  Ice-shelf fronts are the
IceLines monthly Sentinel-1 product (DLR EOC) for Thwaites1 and Thwaites2,
which together give fifteen front positions between October 2022 and February
2024; all of them are drawn as hairlines, so the width of the bundle is the front's
migration over the survey rather than a single snapshot.  The grounding-line
traces are drawn the same way for the same reason: their spread is the
grounding zone, not plotting error.  The grounding line
is MEaSUREs InSAR v2.1 (NSIDC-0498), whose features carry their own
acquisition dates, restricted to the same window - 71 of them fall in it over
this map.  Greene's coastline, which ends in 2021, is kept only for
continental context in the locator inset.

Design notes
------------
The map extends over the whole model domain plus a margin.  The domain outline
is the boundary of the model mesh read from paths.MODEL_EXTRACT; without that
file the map falls back to a fixed region and draws no outline.

Speed is drawn on the magma ramp in 10 discrete log-spaced bands, as on the
paper's other speed maps, with no contours.  The range is extended to 4000
m/yr because the Thwaites trunk runs an order of magnitude faster than the
fabric_anisotropy study's 400 m/yr ceiling and would otherwise saturate to a
single colour across the whole grounding zone.

The frame carries a graticule, parallels and meridians labelled where they
leave the frame, and a scale bar.  Every symbol is keyed in one unframed
legend in a single row under the map, and the locator inset sits top left
without a backing box.  The map is drawn smaller than the text width so it
does not fill a page, with the site markers and lines sized up to match.

Usage
-----
    python scripts/figures/fig01_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import data  # noqa: E402
import mapfurniture  # noqa: E402
import paths  # noqa: E402
import style  # noqa: E402

# Map region in EPSG:3031 kilometres: the whole model domain plus a margin, so
# the reader sees the area the model assimilates the survey into.  Used as is
# when the model mesh extract is not available.
REGION_KM = [-1633, -1263, -569, -276]
DOMAIN_MARGIN_KM = 8.0
# Map width at print size: about two thirds of the 17.8 cm text width, so the
# figure does not fill a page; the paper includes it at that physical size.
WIDTH_CM = 11.5
DOMAIN_PEN = f"1.3p,{style.CATEGORICAL[1]}"
SITE_STYLE = "c0.17c"
DETAIL_PEN = "1.3p,black"
# Graticule: parallels every degree, meridians every four degrees of
# longitude (about 110 km at this latitude).
GRATICULE_LATS = (-75, -76, -77, -78)
# Legend: one row under map and colour bar, at a size that keeps it to one line.
LEGEND_PT = 6.5
COLORBAR_BLOCK_CM = 2.6          # parallel labels, colour bar and its annotation
GRATICULE_LONS = (-100, -104, -108, -112)

# Black with a white rim.  The sites sit on the mid-range of the magma ramp,
# where a black dot carries furthest, and the white rim separates it from the
# darker ice upstream without the marker losing its shape.
SITE_COLOR = "black"
SITE_PEN = "0.2p,white"

# Boundary traces are shaded by acquisition date, light for the start of the
# survey and dark for the end, so the bundle reads as a time series rather
# than as a single smeared line.
DATE_MIN, DATE_MAX = 2022.75, 2024.2
GL_RAMP = ("#b8b8b8", "#8a8a8a", "#5c5c5c", "#2e2e2e", "#000000")
FRONT_RAMP = ("#ffc4b8", "#f7907e", "#e35745", "#c22015", "#7f0d08")

# Magma on a logarithmic scale in N_LEVELS discrete bands, as on the paper's
# other speed maps.  The ceiling is 4000 m/yr so the trunk does not saturate.
SPEED_MIN, SPEED_MAX = 10.0, 4000.0


def _to_km(da):
    return da.assign_coords(x=da["x"] / 1000.0, y=da["y"] / 1000.0)


def _clip(da, region_km, step: int = 1):
    """Clip to the map region and force ascending y, which GMT requires."""
    x0, x1, y0, y1 = region_km
    da = _to_km(da)
    if da["y"].values[0] > da["y"].values[-1]:
        da = da.isel(y=slice(None, None, -1))
    da = da.sel(x=slice(x0, x1), y=slice(y0, y1))
    if step > 1:
        da = da.isel(x=slice(None, None, step), y=slice(None, None, step))
    return da


def load_speed(region_km):
    """Mean surface speed over the survey years, as log10(m/yr)."""
    import rioxarray as rxr

    import xarray as xr

    stack = []
    for year in paths.ASE_VELOCITY_YEARS:
        da = rxr.open_rasterio(paths.ase_velocity(year, "vv"), masked=True)
        da = da.squeeze("band", drop=True)
        stack.append(_clip(da, region_km))
    # Averaged with skipna, so a gap in one year's mosaic is filled by the
    # other rather than propagating into the composite.  A plain mean would
    # punch the union of both years' gaps through the field.
    speed = xr.concat(stack, dim="year").mean(dim="year", skipna=True)
    speed = speed.where(np.isfinite(speed) & (speed > 0))

    # Feature tracking fails over the low-contrast interior, so the two annual
    # mosaics together still leave about a quarter of this map without a
    # solution, in streaks that read as holes in the ice rather than as
    # missing data.  Those gaps are filled from the full-coverage MEaSUREs
    # 2019-2020 mosaic.  The survey-period data are used wherever they exist;
    # the fill is a different epoch and the caption says so.
    speed = speed.where(np.isfinite(speed), _measures_speed_on(speed))
    speed = speed.where(np.isfinite(speed) & (speed > 0))
    out = np.log10(speed.clip(SPEED_MIN, SPEED_MAX))
    out.name = "log_speed"
    return out, speed


def _measures_speed_on(template):
    """MEaSUREs surface speed interpolated onto another grid, in km coords."""
    import xarray as xr

    ds = xr.open_dataset(paths.require(paths.MEASURES_VELOCITY, "MEaSUREs velocity"))
    vx = _to_km(ds["VX"])
    vy = _to_km(ds["VY"])
    if vx["y"].values[0] > vx["y"].values[-1]:
        vx = vx.isel(y=slice(None, None, -1))
        vy = vy.isel(y=slice(None, None, -1))
    speed = np.hypot(vx, vy)
    speed = speed.where(np.isfinite(speed) & (speed > 0))
    return speed.interp(x=template["x"], y=template["y"], method="linear")


def load_moa(region_km):
    """MOA 125 m high-pass composite, subsampled for the map region."""
    import rioxarray as rxr

    da = rxr.open_rasterio(paths.require(paths.MOA, "MOA 125 m mosaic"), masked=True)
    da = da.squeeze("band", drop=True)
    da = _clip(da, region_km, step=4)
    da.name = "moa"
    return da


def load_mask_full(region_km, step: int = 30):
    """BedMachine mask over a wide region, coarsened for the locator inset."""
    import xarray as xr

    ds = xr.open_dataset(paths.require(paths.BEDMACHINE, "BedMachine Antarctica"))
    mask = _clip(ds["mask"], region_km, step=step).astype(float)
    mask.name = "mask"
    return mask


def load_mask(region_km):
    import xarray as xr

    ds = xr.open_dataset(paths.require(paths.BEDMACHINE, "BedMachine Antarctica"))
    mask = _clip(ds["mask"], region_km, step=2).astype(float)
    mask.name = "mask"
    return mask


def load_icelines():
    """IceLines calving fronts in the survey window, with their dates.

    Returns (x_km, y_km, decimal_year) per polyline.
    """
    import geopandas as gpd
    import pandas as pd

    out = []
    for path in sorted(paths.ICELINES_DIR.glob("*.gpkg")):
        try:
            gdf = gpd.read_file(path).to_crs(paths.EPSG_POLAR)
        except Exception:
            continue
        stamps = gdf["DATE_"].astype(str) if "DATE_" in gdf.columns else None
        for i, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty:
                continue
            year = np.nan
            if stamps is not None:
                token = stamps.iloc[i]
                if len(token) >= 6 and token[:6].isdigit():
                    year = int(token[:4]) + (int(token[4:6]) - 0.5) / 12.0
            parts = geom.geoms if hasattr(geom, "geoms") else [geom]
            for part in parts:
                xy = np.asarray(part.coords, dtype=float)
                if xy.size:
                    out.append((xy[:, 0] / 1000.0, xy[:, 1] / 1000.0, year))
    _ = pd
    return out


def load_grounding_lines():
    """MEaSUREs InSAR grounding lines in the survey window, with their dates.

    Returns (x_km, y_km, decimal_year) per polyline.
    """
    import geopandas as gpd
    import pandas as pd

    if not paths.MEASURES_GL.exists():
        return []
    gdf = gpd.read_file(paths.MEASURES_GL)
    when = pd.to_datetime(gdf["Date_1"], errors="coerce")
    keep = (when >= pd.Timestamp(paths.SURVEY_START)) & (
        when <= pd.Timestamp(paths.SURVEY_END)
    )
    gdf = gdf[keep]
    years = when[keep]
    out = []
    for geom, stamp in zip(gdf.geometry, years):
        if geom is None or geom.is_empty:
            continue
        year = stamp.year + (stamp.dayofyear - 0.5) / 365.25
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for part in parts:
            xy = np.asarray(part.coords, dtype=float)
            if xy.size:
                out.append((xy[:, 0] / 1000.0, xy[:, 1] / 1000.0, year))
    return out


def load_front():
    """Greene coastline, in kilometres, as separate polylines.

    The file is comma-delimited EPSG:3031 metres with NaN,NaN rows separating
    ice regions.  Its header mixes '#' comments with one quoted line, so the
    header is stripped by hand rather than with a comment character.
    """
    path = paths.greene_coastline()
    if not path.exists():
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith('"'):
                continue
            a, _, b = line.partition(",")
            try:
                rows.append((float(a), float(b)))
            except ValueError:
                continue
    if not rows:
        return []
    xy = np.asarray(rows, dtype=float)
    return list(mapfurniture.nan_split_polylines(xy[:, 0] / 1000.0, xy[:, 1] / 1000.0))


def load_model_domain():
    """Outline of the model mesh as closed polylines in km, or [] without it.

    The boundary is the set of triangle edges used by exactly one cell,
    chained into rings."""
    f = paths.MODEL_EXTRACT
    if not f.exists():
        print(f"  model mesh extract not found ({f}); drawing no domain outline")
        return []
    with np.load(f, allow_pickle=False) as z:
        xy, cells = z["xy"] / 1000.0, z["cells"]
    e = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    nbr = {}
    for a, b in u[cnt == 1]:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    rings, seen = [], set()
    for start in nbr:
        if start in seen:
            continue
        ring, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in nbr[cur] if n != prev and (n not in seen or n == start)]
            if not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]
            ring.append(cur)
            seen.add(cur)
        ring.append(start)
        rings.append((xy[ring, 0], xy[ring, 1]))
    return rings


def main() -> None:
    import pygmt

    catalog = pd.read_csv(paths.PAIR_CATALOG).sort_values("distance_km").reset_index(drop=True)
    is_station = catalog.get("site_type", "point").eq("continuous")
    pairs = catalog[~is_station].reset_index(drop=True)
    stations = catalog[is_station].reset_index(drop=True)
    sx = pairs["x_ps_m"].values / 1000.0
    sy = pairs["y_ps_m"].values / 1000.0

    domain = load_model_domain()
    region = list(REGION_KM)
    if domain:
        dx = np.concatenate([r[0] for r in domain])
        dy = np.concatenate([r[1] for r in domain])
        region = [float(np.floor(dx.min() - DOMAIN_MARGIN_KM)), float(np.ceil(dx.max() + DOMAIN_MARGIN_KM)),
                  float(np.floor(dy.min() - DOMAIN_MARGIN_KM)), float(np.ceil(dy.max() + DOMAIN_MARGIN_KM))]
    x0, x1, y0, y1 = region
    height_cm = WIDTH_CM * (y1 - y0) / (x1 - x0)
    proj = f"X{WIDTH_CM}c/{height_cm:.3f}c"
    tmp = Path(tempfile.mkdtemp(prefix="fig01_"))

    pygmt.config(**style.GMT_CONFIG)
    fig = pygmt.Figure()

    # --------------------------------------------- MOA shaded-relief backdrop
    moa = load_moa(region)
    lo, hi = np.nanpercentile(moa.values, [2, 98])
    pygmt.makecpt(cmap="gray", series=[float(lo), float(hi)])
    fig.grdimage(grid=moa, region=region, projection=proj, cmap=True,
                 nan_transparent=True, frame="+n")

    # ------------------------------------------------------------ speed field
    log_speed, _ = load_speed(region)
    edges = np.linspace(np.log10(SPEED_MIN), np.log10(SPEED_MAX), style.N_LEVELS + 1)
    speed_cpt = style.gmt_cpt_bands(style.SEQ_SPEED, edges, str(tmp / "speed.cpt"))
    # Semi-transparent so the MOA surface texture reads through the bands.
    fig.grdimage(grid=log_speed, region=region, projection=proj, cmap=speed_cpt,
                 nan_transparent=True, transparency=18)

    # ------------------------------------------ dated grounding line and fronts
    def _in_view(px, py):
        return not (px.max() < x0 or px.min() > x1 or py.max() < y0 or py.min() > y1)

    def _shade(year, ramp):
        """Light for the earliest trace, dark for the latest."""
        if not np.isfinite(year):
            return ramp[len(ramp) // 2]
        f = (year - DATE_MIN) / (DATE_MAX - DATE_MIN)
        i = int(np.clip(round(f * (len(ramp) - 1)), 0, len(ramp) - 1))
        return ramp[i]

    # Drawn oldest first so the most recent trace sits on top of the bundle.
    for gx, gy, year in sorted(load_grounding_lines(), key=lambda t: (np.isnan(t[2]), t[2])):
        if _in_view(gx, gy):
            fig.plot(x=gx, y=gy, pen=f"0.4p,{_shade(year, GL_RAMP)}",
                     region=region, projection=proj)
    for fx, fy, year in sorted(load_icelines(), key=lambda t: (np.isnan(t[2]), t[2])):
        if _in_view(fx, fy):
            fig.plot(x=fx, y=fy, pen=f"0.45p,{_shade(year, FRONT_RAMP)}",
                     region=region, projection=proj)

    # ----------------------------------------------------------- model domain
    for rx, ry in domain:
        fig.plot(x=rx, y=ry, pen=DOMAIN_PEN, region=region, projection=proj)

    # ------------------------------------------------------------------ sites
    # Every ApRES measurement is drawn the same way, attended or autonomous:
    # they are the same instrument making the same measurement.
    all_x = np.concatenate([sx, stations["x_ps_m"].values / 1000.0]) if len(stations) else sx
    all_y = np.concatenate([sy, stations["y_ps_m"].values / 1000.0]) if len(stations) else sy
    fig.plot(x=all_x, y=all_y, style=SITE_STYLE, fill=SITE_COLOR, pen=SITE_PEN,
             region=region, projection=proj)

    # ------------------------------------------------- detail-figure extent
    dx0, dx1, dy0, dy1 = data.detail_region_km()
    fig.plot(x=[dx0, dx1, dx1, dx0, dx0], y=[dy0, dy0, dy1, dy1, dy0],
             pen=DETAIL_PEN, region=region, projection=proj)

    # ------------------------------------ graticule, frame and scale bar
    mapfurniture.graticule(fig, region=region, projection=proj,
                           lats=GRATICULE_LATS, lons=GRATICULE_LONS)
    fig.plot(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], pen="0.6p,black",
             region=region, projection=proj, no_clip=True)
    mapfurniture.scalebar(fig, region=region, projection=proj, loc=(0.06, 0.05), halo=False)

    # ------------------------------------------------------------- colorbar
    # Log-spaced bands, annotated at the decades.
    tickfile = tmp / "speed_ticks.txt"
    with open(tickfile, "w") as fh:
        for t in (1, 2, 3):
            fh.write(f"{t}\ta\t10@+{t}@+\n")
        for e in edges[1:-1]:
            fh.write(f"{e:.6f}\tf\n")
    # Offset past the parallel labels on the right edge of the frame.
    style.gmt_colorbar(fig, cmap=speed_cpt, length_cm=0.88 * height_cm, offset="0.95c/0c",
                       frame=[f"xc{tickfile}+lSurface speed (m yr@+{style.GMT_MINUS}1@+)"])

    # -------------------------------------------------------- locator inset
    ant_km = [-2800, 2800, -2500, 2300]
    inset_w = 2.9
    inset_h = inset_w * (ant_km[3] - ant_km[2]) / (ant_km[1] - ant_km[0])
    # No backing box and no border: the continent sits straight on the map,
    # with the ocean left transparent.
    with fig.inset(position=f"jTL+w{inset_w}c/{inset_h:.3f}c+o0.15c"):
        # Drawn in projected kilometres rather than with `coast`, so the
        # BedMachine mask can be imaged directly and the ice shelves are a
        # distinct tone rather than being merged into the coastline.
        inset_proj = f"X{inset_w}c/{inset_h:.3f}c"
        mask = load_mask_full(ant_km)
        mask = mask.where(mask != 0)            # ocean transparent
        # One flat colour per BedMachine mask class.
        classes = [
            (1, "gray72"),     # ice-free land
            (2, "gray88"),     # grounded ice
            (3, "#9ecae1"),    # floating ice shelf
            (4, "gray88"),     # Lake Vostok
        ]
        maskcpt = tmp / "mask.cpt"
        with open(maskcpt, "w") as fh:
            for value, colour in classes:
                fh.write(f"{value - 0.5} {colour} {value + 0.5} {colour}\n")
        fig.grdimage(grid=mask, region=ant_km, projection=inset_proj,
                     cmap=str(maskcpt), nan_transparent=True, frame=["+n"])
        fig.plot(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
                 pen="1.0p,black", region=ant_km, projection=inset_proj)

    # ------------------------------------------------ legend under the map
    # One row across the whole figure (map and colour bar), small, below the
    # meridian labels; the survey dates are in the caption.  The domain
    # outline is keyed only when the mesh extract supplied one.
    entries = [
        ("ramp", (GL_RAMP, "1.4p"), "grounding zone"),
        ("ramp", (FRONT_RAMP, "1.4p"), "ice front"),
        *([("line", DOMAIN_PEN, "model domain")] if domain else []),
        ("symbol", (SITE_STYLE, SITE_COLOR, SITE_PEN), "ApRES stations"),
        # No figure number: the paper orders the figures differently from these
        # scripts, so the caption names the figure the box outlines.
        ("symbol", ("s0.28c", None, "1.0p,black"), "extent of station maps"),
    ]
    legend_w = WIDTH_CM + COLORBAR_BLOCK_CM
    legend_h = style.gmt_legend_height(entries, width_cm=legend_w, font_pt=LEGEND_PT)
    fig.shift_origin(yshift=f"-{legend_h + 0.55:.3f}c")
    style.gmt_top_legend(fig, entries, width_cm=legend_w, frame=False, font_pt=LEGEND_PT)

    style.gmt_save(fig, paths.FIGURES / "fig01_overview.png")


if __name__ == "__main__":
    main()
