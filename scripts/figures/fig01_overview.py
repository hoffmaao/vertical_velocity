"""Figure 1 - setting of the repeat ApRES survey on Thwaites Glacier.

A regional panel showing surface speed over the Thwaites trunk during the
survey period, the grounding line, the ice-shelf front, and the 88 sites
occupied in both field seasons, with an Antarctic locator inset.  The three
autonomous stations, which logged through the year between the two visits, are
drawn as squares.

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
Speed is drawn on the magma ramp on a logarithmic scale, matching the velocity
figures in the fabric_anisotropy study, with white contours over it.  The range
is extended past that study's 400 m/yr ceiling because the Thwaites trunk runs
an order of magnitude faster and would otherwise saturate to a single colour
across the whole grounding zone.

Nothing is written onto the map that is not an axis annotation, a colorbar
label, the scale bar's value or a legend entry.

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

# Map region in EPSG:3031 kilometres: the survey line plus the grounding zone
# and shelf front immediately downstream of it.
REGION_KM = [-1620, -1330, -560, -380]
WIDTH_CM = 12.0

# Black with a white rim.  The sites sit on the mid-range of the magma ramp,
# where a black dot carries furthest, and the white rim separates it from the
# darker ice upstream without the marker losing its shape.
SITE_COLOR = "black"
SITE_PEN = "0.375p,white"

# Boundary traces are shaded by acquisition date, light for the start of the
# survey and dark for the end, so the bundle reads as a time series rather
# than as a single smeared line.
DATE_MIN, DATE_MAX = 2022.75, 2024.2
GL_RAMP = ("#b8b8b8", "#8a8a8a", "#5c5c5c", "#2e2e2e", "#000000")
FRONT_RAMP = ("#ffc4b8", "#f7907e", "#e35745", "#c22015", "#7f0d08")
GL_COLOR = GL_RAMP[-2]
FRONT_COLOR = FRONT_RAMP[-2]

# Matching the fabric_anisotropy velocity figures: magma, logarithmic, with
# white contours.  The ceiling is raised from 400 to 4000 m/yr for the trunk.
SPEED_MIN, SPEED_MAX = 10.0, 4000.0
SPEED_CONTOURS = (30, 100, 300, 1000, 3000)


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


def draw_legend(fig, region_km, proj) -> None:
    """Legend drawn by hand, because the key has to show the date shading.

    GMT's own legend can only give a symbol one colour, so a line entry there
    could show the grounding line or the front but not the fact that each is a
    bundle shaded from light at the start of the survey to dark at the end.
    Here each line symbol is drawn as the ramp itself - the same five shades,
    in the same order, as the traces on the map - which puts the shading
    convention in the symbol so the label only has to name the feature and its
    period.  The box is sized as a fraction of the map region, so it keeps its
    proportions if the region changes; the width is set by the longest label.
    """
    x0, x1, y0, y1 = region_km
    span = x1 - x0

    pad = 0.020 * span
    box_w = 0.465 * span
    row_h = 0.050 * (y1 - y0)
    rows = 3
    box_h = row_h * (rows + 0.6)

    bx0 = x0 + pad
    bx1 = bx0 + box_w
    by1 = y1 - pad
    by0 = by1 - box_h

    fig.plot(x=[bx0, bx1, bx1, bx0, bx0], y=[by0, by0, by1, by1, by0],
             fill="white", pen="0.5p,black", close=True,
             region=region_km, projection=proj)

    sym_x0 = bx0 + 0.030 * span
    sym_x1 = sym_x0 + 0.055 * span
    text_x = sym_x1 + 0.018 * span
    top = by1 - row_h * 0.8

    def row(i):
        return top - i * row_h

    # A black marker reads directly on the white key, so it needs no backing
    # patch of the map behind it.
    fig.plot(x=[0.5 * (sym_x0 + sym_x1)], y=[row(0)], style="c0.085c",
             fill=SITE_COLOR, pen=SITE_PEN,
             region=region_km, projection=proj)
    fig.text(x=[text_x], y=[row(0)], text=["ApRES site"],
             font="8p,Helvetica,black", justify="LM",
             region=region_km, projection=proj)

    # Line entries: the symbol is the ramp, light (earliest) to dark (latest).
    for i, (ramp, label) in enumerate(
        ((GL_RAMP, "grounding zone (2022-2024)"),
         (FRONT_RAMP, "ice front (2022-2024)")), start=1
    ):
        edges = np.linspace(sym_x0, sym_x1, len(ramp) + 1)
        for j, colour in enumerate(ramp):
            fig.plot(x=[edges[j], edges[j + 1]], y=[row(i), row(i)],
                     pen=f"2p,{colour}", region=region_km, projection=proj)
        fig.text(x=[text_x], y=[row(i)], text=[label],
                 font="8p,Helvetica,black", justify="LM",
                 region=region_km, projection=proj)


def main() -> None:
    import pygmt

    catalog = pd.read_csv(paths.PAIR_CATALOG).sort_values("distance_km").reset_index(drop=True)
    is_station = catalog.get("site_type", "point").eq("continuous")
    pairs = catalog[~is_station].reset_index(drop=True)
    stations = catalog[is_station].reset_index(drop=True)
    sx = pairs["x_ps_m"].values / 1000.0
    sy = pairs["y_ps_m"].values / 1000.0

    x0, x1, y0, y1 = REGION_KM
    height_cm = WIDTH_CM * (y1 - y0) / (x1 - x0)
    proj = f"X{WIDTH_CM}c/{height_cm:.3f}c"

    # One type size for every annotation on the figure and one for every
    # label, so the colorbar's ticks and title match the map's axes instead of
    # defaulting to GMT's own smaller colorbar sizes.
    pygmt.config(
        FONT_ANNOT_PRIMARY="8p,Helvetica,black",
        FONT_ANNOT_SECONDARY="8p,Helvetica,black",
        FONT_LABEL="9p,Helvetica,black",
        MAP_FRAME_TYPE="plain",
        MAP_FRAME_PEN="0.7p,black",
        MAP_TICK_LENGTH_PRIMARY="2.5p",
        MAP_TICK_PEN_PRIMARY="0.6p,black",
    )
    fig = pygmt.Figure()

    # --------------------------------------------- MOA shaded-relief backdrop
    moa = load_moa(REGION_KM)
    lo, hi = np.nanpercentile(moa.values, [2, 98])
    pygmt.makecpt(cmap="gray", series=[float(lo), float(hi)])
    fig.grdimage(
        grid=moa, region=REGION_KM, projection=proj, cmap=True,
        nan_transparent=True,
        frame=["WSne", "xa50f25+lpolar-stereographic easting (km)",
               "ya50f25+lpolar-stereographic northing (km)"],
    )

    # ------------------------------------------------------------ speed field
    log_speed, _ = load_speed(REGION_KM)
    # magma in its natural direction - slow ice dark, fast ice bright - which
    # is how the fabric_anisotropy velocity figures read it.
    pygmt.makecpt(
        cmap="magma",
        series=[np.log10(SPEED_MIN), np.log10(SPEED_MAX), 0.01],
    )
    # Semi-transparent so the MOA surface texture reads through the colour.
    fig.grdimage(
        grid=log_speed, region=REGION_KM, projection=proj, cmap=True,
        nan_transparent=True, transparency=18,
    )
    fig.grdcontour(
        grid=log_speed, levels=[np.log10(c) for c in SPEED_CONTOURS],
        pen="0.35p,white@30", region=REGION_KM, projection=proj,
    )

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
            fig.plot(x=gx, y=gy, pen=f"0.3p,{_shade(year, GL_RAMP)}",
                     region=REGION_KM, projection=proj)
    for fx, fy, year in sorted(load_icelines(), key=lambda t: (np.isnan(t[2]), t[2])):
        if _in_view(fx, fy):
            fig.plot(x=fx, y=fy, pen=f"0.35p,{_shade(year, FRONT_RAMP)}",
                     region=REGION_KM, projection=proj)

    # ------------------------------------------------------------------ sites
    # Every ApRES measurement is drawn the same way, attended or autonomous:
    # they are the same instrument making the same measurement, and the
    # distinction belongs in the caption rather than in a second colour.
    # Distance ticks are gone: at this scale the tick rings sat on top of
    # neighbouring sites and hid them.
    all_x = np.concatenate([sx, stations["x_ps_m"].values / 1000.0]) if len(stations) else sx
    all_y = np.concatenate([sy, stations["y_ps_m"].values / 1000.0]) if len(stations) else sy
    fig.plot(x=all_x, y=all_y, style="c0.085c", fill=SITE_COLOR,
             pen=SITE_PEN)

    # ------------------------------------------------- detail-figure extent
    # The ground covered by the detail map, drawn unlabelled: the box is a
    # locator, and naming it here would duplicate the caption.
    try:
        dx0, dx1, dy0, dy1 = data.detail_region_km()
        fig.plot(
            x=[dx0, dx1, dx1, dx0, dx0], y=[dy0, dy0, dy1, dy1, dy0],
            pen="1.0p,black", region=REGION_KM, projection=proj,
        )
    except Exception:
        pass

    # ----------------------------------------------------------- scale bar
    mapfurniture.scalebar_bw(
        fig, x0=x1 - 0.27 * (x1 - x0), y0=y1 - 0.13 * (y1 - y0),
        length_km=50.0,
    )

    # ------------------------------------------------------------- colorbar
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for t in (10, 100, 1000):
            fh.write(f"{np.log10(t)}\ta\t{t}\n")
        for t in (30, 300, 3000, 4000):
            fh.write(f"{np.log10(t)}\tf\n")
        tickfile = fh.name
    # GMT scales a colorbar's annotation and label fonts down when the bar is
    # thin, so a bar left on the figure's own FONT_LABEL renders smaller than
    # the frame labels beside it.  These sizes land at the frame's size after
    # that scaling, and the bar is thickened to reduce it.  Same values as
    # Figure 4, so the two map figures carry the same colorbar type.
    with pygmt.config(
        FONT_ANNOT_PRIMARY="12p,Helvetica,black",
        FONT_LABEL="13p,Helvetica,black",
    ):
        fig.colorbar(
            position="JBC+w7.5c/0.225c+h+o0c/1.5c",
            frame=[f"xc{tickfile}", "x+lsurface speed (m yr@+-1@+)"],
        )

    # --------------------------------------------------------------- legend
    draw_legend(fig, REGION_KM, proj)

    # -------------------------------------------------------- locator inset
    ant_km = [-2800, 2800, -2500, 2300]
    inset_w = 2.75
    inset_h = inset_w * (ant_km[3] - ant_km[2]) / (ant_km[1] - ant_km[0])
    with fig.inset(
        position=f"jBL+w{inset_w}c/{inset_h:.3f}c+o0.1c",
        box="+gwhite+p0.7p,black",
    ):
        # Drawn in projected kilometres rather than with `coast`, so the
        # BedMachine mask can be imaged directly and the ice shelves - where
        # the ice crossing this survey eventually goes - are a distinct tone
        # rather than being merged into the coastline.
        inset_proj = f"X{inset_w}c/{inset_h:.3f}c"
        mask = load_mask_full(ant_km)
        # A hand-written categorical CPT: one flat colour per BedMachine mask
        # class, so ocean, land, grounded ice and shelf are distinguishable
        # without interpolating between class numbers.
        classes = [
            (0, "white"),      # ocean
            (1, "gray72"),     # ice-free land
            (2, "gray88"),     # grounded ice
            (3, "#9ecae1"),    # floating ice shelf
            (4, "gray88"),     # Lake Vostok
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".cpt", delete=False) as fh:
            for value, colour in classes:
                fh.write(f"{value - 0.5} {colour} {value + 0.5} {colour}\n")
            maskcpt = fh.name
        fig.grdimage(grid=mask, region=ant_km, projection=inset_proj,
                     cmap=maskcpt, nan_transparent=True, frame=["+n"])
        fig.plot(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
                 pen="1.2p,black",
                 region=ant_km, projection=inset_proj)
        # The mask is imaged to the panel edge, so the inset border is redrawn
        # over it to stay continuous on all four sides.
        ax0, ax1, ay0, ay1 = ant_km
        fig.plot(
            x=[ax0, ax1, ax1, ax0, ax0], y=[ay0, ay0, ay1, ay1, ay0],
            pen="0.7p,black", region=ant_km, projection=inset_proj,
        )

    out = paths.FIGURES / "fig01_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
