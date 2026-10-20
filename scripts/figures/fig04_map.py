"""Figure 4 - map view of the deformation measured along the transect.

Two stacked map panels over the same ground, with bed topography beneath:
the depth-averaged vertical strain rate above, and the share of the
deformation carried by shear rather than plug flow below.  The bed is shared
between the panels and carries one colorbar, at the foot of the figure, on a
terrestrial green-yellow-brown-grey ramp so that it reads apart from the blues
of the strain-rate scale above it.

Every ApRES measurement is drawn the same way, whether it came from an
attended transect occupation or from one of the three autonomous stations,
which are paired from the first and last acquisition of their year-long
records.  All of them are decomposed on the same degree-4 shifted-Legendre
basis over the same normalised-depth window.

Whether shear is resolved at a site is carried by the marker outline - white
where the degree->=2 coefficients are jointly distinguishable from zero, black
where they are not.  The convention belongs in the caption; there is no key on
the panel.

Why a strip and not a square map
--------------------------------
The transect is 68 km long and 11 km wide, and the two off-line stations sit
about 20 km north of it.  Drawn on a square map the whole survey occupies a
thin band across an otherwise empty page.  The panels are cropped to the data
and given the aspect ratio it actually has.  Figure 1 carries the regional
context this crop removes.

Colour
------
The depth-averaged strain rate changes sign along flow, so it is drawn on a
diverging ramp with a neutral midpoint pinned to zero.  The shear share is
bounded and one-signed, so it gets a single-hue sequential ramp, compressed
because three quarters of the sites sit below 0.15.

Usage
-----
    python scripts/figures/fig04_map.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import data  # noqa: E402
import mapfurniture  # noqa: E402
import paths  # noqa: E402

BUFFER_KM = 4.0
BED_CONTOUR_M = 200.0
WIDTH_CM = 15.0
PANEL_GAP_CM = 1.5
SCALEBAR_KM = 20.0

SHEAR_GAMMA = 0.65


def load_bed(region_km):
    """BedMachine bed elevation clipped to the map strip, in km coordinates."""
    import xarray as xr

    x0, x1, y0, y1 = region_km
    ds = xr.open_dataset(paths.require(paths.BEDMACHINE, "BedMachine Antarctica"))
    bed = ds["bed"].assign_coords(x=ds["bed"]["x"] / 1000.0, y=ds["bed"]["y"] / 1000.0)
    if bed["y"].values[0] > bed["y"].values[-1]:
        bed = bed.isel(y=slice(None, None, -1))
    pad = 6.0
    bed = bed.sel(x=slice(x0 - pad, x1 + pad), y=slice(y0 - pad, y1 + pad))
    bed.name = "bed"
    return bed


def write_cpt(entries, *, clamp: bool = True) -> str:
    """Write a CPT file from (z_low, colour, z_high, colour) rows.

    With ``clamp`` the background and foreground slots are set to the end
    colours of the ramp.  Without it GMT paints everything outside the ramp
    black and white, which on a bed clipped to its 2nd-98th percentile turns
    the deepest troughs into black blotches and the highest ridges into holes.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cpt", delete=False) as fh:
        for lo, c_lo, hi, c_hi in entries:
            fh.write(f"{lo} {c_lo} {hi} {c_hi}\n")
        if clamp and entries:
            fh.write(f"B {entries[0][1]}\n")
            fh.write(f"F {entries[-1][3]}\n")
            fh.write("N white\n")
        return fh.name


def clamp_cpt(path: str) -> str:
    """Rewrite a CPT's background and foreground to its own end colours.

    ``makecpt``'s ``background`` argument is ignored for a discrete CPT such as
    ibcso, which leaves B and F at black and white; grdimage then paints every
    cell outside the chosen range as a black or white blob.  Rewriting the
    slots directly is the only reliable fix.
    """
    rows = []
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines:
        if line.startswith(("#", "B", "F", "N")) or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            rows.append(parts)
    if not rows:
        return path
    first, last = rows[0][1], rows[-1][3]
    with open(path, "w") as fh:
        for line in lines:
            if line.startswith(("B", "F", "N")):
                continue
            fh.write(line)
        fh.write(f"B {first}\n")
        fh.write(f"F {last}\n")
        fh.write("N white\n")
    return path


def _cpt_rows(path: str):
    """(z_low, colour, z_high, colour) rows of a CPT file, ignoring B/F/N."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("#", "B", "F", "N")) or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                rows.append((float(parts[0]), parts[1], float(parts[2]), parts[3]))
    return rows


# Anchors of the bed ramp, as (fraction along the ramp, R, G, B).  Built by
# hand rather than borrowed from a named CPT because the ones GMT ships either
# skip the yellow band between green and brown (bukavu, oleron) or run to cream
# rather than grey at the top (relief).  The top stays grey instead of white so
# the highest ridges still read as terrain rather than as blank page.
TERRAIN_ANCHORS = (
    (0.00, (26, 87, 30)),     # dark green, deepest
    (0.22, (120, 150, 50)),   # green
    (0.40, (232, 220, 110)),  # yellow
    (0.60, (150, 112, 62)),   # brown
    (0.80, (175, 170, 165)),  # grey-brown
    (1.00, (208, 208, 208)),  # grey, highest
)


def terrain_cpt(z0: float, z1: float, n: int = 160) -> str:
    """A terrestrial green-yellow-brown-grey ramp over an arbitrary range.

    The Thwaites bed lies entirely below sea level, so any topographic CPT
    handed this range returns only its marine half - blues, which collide with
    the diverging strain-rate ramp and the dark end of the shear ramp in the
    panels above.  This builds the terrestrial sequence directly and lays it
    over the bed's own range, so the colours encode relief rather than height
    above sea level; the colorbar is labelled in metres either way.
    """
    fracs = np.array([f for f, _ in TERRAIN_ANCHORS], dtype=float)
    cols = np.array([c for _, c in TERRAIN_ANCHORS], dtype=float)
    edges = np.linspace(0.0, 1.0, n + 1)

    def colour_at(f):
        rgb = [np.interp(f, fracs, cols[:, k]) for k in range(3)]
        return "/".join(f"{int(round(v))}" for v in rgb)

    rows = []
    for i in range(n):
        f0, f1 = edges[i], edges[i + 1]
        rows.append((z0 + (z1 - z0) * f0, colour_at(f0),
                     z0 + (z1 - z0) * f1, colour_at(f1)))
    return write_cpt(rows)


def power_cpt(vmax: float, gamma: float, cmap: str, n: int = 64) -> str:
    """A CPT whose steps are spaced as a power law, mimicking PowerNorm.

    GMT has no equivalent of matplotlib's PowerNorm, so the compression is
    baked into the CPT's z values instead: the colour sequence is sampled
    uniformly and the breakpoints are placed at v = vmax * f**(1/gamma).
    """
    import pygmt

    with tempfile.NamedTemporaryFile("w", suffix=".cpt", delete=False) as fh:
        tmp = fh.name
    pygmt.makecpt(cmap=cmap, series=[0, 1, 1.0 / n], output=tmp)
    rows = []
    with open(tmp) as fh:
        for line in fh:
            if line.startswith(("#", "B", "F", "N")) or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                rows.append((float(parts[0]), parts[1], float(parts[2]), parts[3]))
    out = []
    for lo, c_lo, hi, c_hi in rows:
        out.append((vmax * lo ** (1.0 / gamma), c_lo, vmax * hi ** (1.0 / gamma), c_hi))
    return write_cpt(out)


def draw_panel(fig, *, bed, bed_cpt, region, proj, frame, values, cpt,
               x, y, resolved, stations_xy, stations_v):
    """One map panel: bed underneath, every ApRES site on top."""
    import pygmt

    fig.grdimage(grid=bed, region=region, projection=proj, cmap=bed_cpt,
                 nan_transparent=True, frame=frame)
    # Every other contour is annotated, with labels placed at a fixed spacing
    # along each line so a reader can put a number on the relief without
    # tracing it back to a key.
    # Levels are given as an interval, not as an explicit list: GMT only
    # annotates contours it generated from an interval, so a list silently
    # produces unlabelled lines.  Every contour is annotated because the bed
    # spans only ~700 m here, and the labels are white because the deep end of
    # an IBCSO ramp is nearly black.
    fig.grdcontour(
        grid=bed, levels=BED_CONTOUR_M, pen="0.4p,white@25",
        annotation=f"{BED_CONTOUR_M:.0f}+f7p,Helvetica,white+u m",
        label_placement="d7c", region=region, projection=proj,
    )

    # Sites where shear is resolved carry a white outline; the rest are ringed
    # in black and drawn last so they stay findable.
    res = np.asarray(resolved, dtype=bool)
    if res.any():
        fig.plot(x=x[res], y=y[res], style="c0.285c", fill=values[res],
                 cmap=cpt, pen="0.5p,white", region=region, projection=proj)
    if (~res).any():
        fig.plot(x=x[~res], y=y[~res], style="c0.285c", fill=values[~res],
                 cmap=cpt, pen="0.9p,black", region=region, projection=proj)
    if len(stations_xy[0]):
        fig.plot(x=stations_xy[0], y=stations_xy[1], style="c0.285c",
                 fill=stations_v, cmap=cpt, pen="0.5p,white",
                 region=region, projection=proj)

    mapfurniture.scalebar_bw(
        fig,
        x0=region[1] - 0.30 * (region[1] - region[0]),
        y0=region[2] + 0.08 * (region[3] - region[2]),
        length_km=SCALEBAR_KM, height_km=0.9,
    )


def main() -> None:
    import pygmt

    df = data.load_summary()
    if df.empty:
        raise SystemExit("no sites survive the quality screen")
    point, stations = data.split_by_type(df)
    lo_w, hi_w = data.fitting_window(df)
    print(data.describe(point))
    print(f"fitting window: zeta in [{lo_w:.2f}, {hi_w:.2f}]")
    print(f"stations: {', '.join(stations['site'])}")

    x = point["x_ps_m"].values / 1000.0
    y = point["y_ps_m"].values / 1000.0
    sx = stations["x_ps_m"].values / 1000.0
    sy = stations["y_ps_m"].values / 1000.0
    region = data.detail_region_km(BUFFER_KM)
    height_cm = WIDTH_CM * (region[3] - region[2]) / (region[1] - region[0])
    proj = f"X{WIDTH_CM}c/{height_cm:.3f}c"
    bed = load_bed(region)
    # The bed lies entirely below sea level, so a topographic CPT given this
    # range returns only its marine half - blues, which collide with the
    # diverging strain-rate ramp and the dark end of the shear ramp above.
    # land_cpt samples the terrestrial side instead and rescales it onto the
    # bed range.
    bed_lo, bed_hi = np.nanpercentile(bed.values, [1, 99])
    bed_cpt = terrain_cpt(float(bed_lo), float(bed_hi))

    col_eps = "strain_rate_depth_mean_yr"
    if not np.isfinite(df[col_eps].values).any():
        col_eps = "strain_rate_yr"
    eps = point[col_eps].values
    eps_all = np.concatenate([eps, stations[col_eps].values])
    lim = float(np.nanmax(np.abs(eps_all[np.isfinite(eps_all)])))

    ratio = point["shear_share"].values
    ratio_all = np.concatenate([ratio, stations["shear_share"].values])
    # The ramp spans the full range of the data, so nothing falls outside it
    # and the bar needs no out-of-range arrow.  The power-law spacing, not a
    # clipped ceiling, is what keeps the low values apart.
    cap = float(np.nanmax(ratio_all[np.isfinite(ratio_all)]))
    resolved = point["shear_resolved"].astype(bool).values

    pygmt.config(
        FONT_ANNOT_PRIMARY="8p,Helvetica,black",
        FONT_LABEL="9p,Helvetica,black",
        MAP_FRAME_TYPE="plain",
        MAP_FRAME_PEN="0.7p,black",
        MAP_TICK_LENGTH_PRIMARY="2.5p",
        MAP_TICK_PEN_PRIMARY="0.6p,black",
    )
    fig = pygmt.Figure()

    # GMT scales a colorbar's annotation and label fonts down when the bar is
    # thin, so a bar set to the map's own FONT_LABEL still renders smaller
    # than the axis labels beside it.  These sizes are chosen so the bars
    # match the frame after that scaling, and the bar is thickened a little to
    # reduce it.
    CBAR_FONTS = dict(
        FONT_ANNOT_PRIMARY="12p,Helvetica,black",
        FONT_LABEL="13p,Helvetica,black",
    )
    CBAR_THICK = "0.225c"

    # ------------------------------ upper: depth-averaged vertical strain rate
    eps_cpt = tempfile.NamedTemporaryFile(suffix=".cpt", delete=False).name
    # GMT does not ship ColorBrewer's RdBu, so the diverging ramp is Crameri's
    # vik: same blue-negative / red-positive reading, neutral at the midpoint,
    # and perceptually uniform.
    pygmt.makecpt(cmap="vik", series=[-lim, lim, 2 * lim / 128], output=eps_cpt)
    draw_panel(
        fig, bed=bed, bed_cpt=bed_cpt, region=region, proj=proj,
        frame=["WSne", "xa10f5", "ya5f5+lpolar-stereographic northing (km)"],
        values=eps, cpt=eps_cpt, x=x, y=y, resolved=resolved,
        stations_xy=(sx, sy), stations_v=stations[col_eps].values,
    )
    with pygmt.config(**CBAR_FONTS):
        fig.colorbar(
            cmap=eps_cpt,
            position="JMR+w{:.1f}c/{}+o0.75c/0c".format(0.8 * height_cm, CBAR_THICK),
            frame=["xa0.005f0.001+ldepth-averaged @~e@~@-zz@- (yr@+-1@+)"],
        )

    # ----------------------------------------- lower: shear share of deformation
    fig.shift_origin(yshift=f"-{height_cm + PANEL_GAP_CM:.3f}c")
    shear_cpt = power_cpt(cap, SHEAR_GAMMA, "magma")
    draw_panel(
        fig, bed=bed, bed_cpt=bed_cpt, region=region, proj=proj,
        frame=["WSne", "xa10f5+lpolar-stereographic easting (km)",
               "ya5f5+lpolar-stereographic northing (km)"],
        values=ratio, cpt=shear_cpt, x=x, y=y, resolved=resolved,
        stations_xy=(sx, sy), stations_v=stations["shear_share"].values,
    )
    with pygmt.config(**CBAR_FONTS):
        fig.colorbar(
            cmap=shear_cpt,
            position="JMR+w{:.1f}c/{}+o0.75c/0c".format(0.8 * height_cm, CBAR_THICK),
            frame=["xa0.2f0.05+lshear share A@-s@-/(A@-s@-+A@-p@-)"],
        )

    # One bed colorbar for both panels, under the lower one.
    with pygmt.config(**CBAR_FONTS):
        fig.colorbar(
            cmap=bed_cpt,
            position="JBC+w{:.1f}c/{}+h+o0c/1.6c".format(0.55 * WIDTH_CM, CBAR_THICK),
            frame=["xa200f100+lbed elevation (m)"],
        )

    out = paths.FIGURES / "fig04_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
