"""Figure 4 - map view of the deformation measured along the transect.

Two stacked map panels over the same ground, with bed topography beneath:
the depth-averaged vertical strain rate above, and the share of the
deformation carried by shear rather than plug flow below.  The bed is shared
between the panels and carries one colour bar, at the foot of the figure; it
is drawn as a conventional bed map, a continuous bathymetric ramp with a
faint hillshade laid over it.  The panels follow the paper's map grammar (figstyle): bare frames
with a scale bar, discrete colour bands for the measured quantities, panel
letters and a row label on the left.

Every ApRES measurement is drawn the same way, whether it came from an
attended transect occupation or from one of the three autonomous stations,
which are paired from the first and last acquisition of their year-long
records.  All of them are decomposed on the same degree-4 shifted-Legendre
basis over the same normalised-depth window.

Whether shear is resolved at a site is carried by the marker outline - thin
where the degree->=2 coefficients are jointly distinguishable from zero, heavy
where they are not.  The convention is given in the caption; the figure
carries no legend.

Why a strip and not a square map
--------------------------------
The transect is 68 km long and 11 km wide, and the two off-line stations sit
about 20 km north of it.  Drawn on a square map the whole survey occupies a
thin band across an otherwise empty page.  The panels are cropped to the data
and given the aspect ratio it actually has.  Figure 1 carries the regional
context this crop removes.

Colour
------
The bed lies entirely below sea level, so the bed map is blue, and neither
quantity drawn over it uses blue.  The depth-averaged strain rate changes sign
along flow, so it is drawn on a purple-orange diverging ramp with a neutral
midpoint pinned to zero.  The shear share is bounded and one-signed, so it
gets a yellow-orange-red ramp whose band edges are spaced as a power law,
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
import style  # noqa: E402

BUFFER_KM = 4.0
# Drawn at print width: map + row label + colour bar fill the 17.8 cm text width.
WIDTH_CM = 15.85
PANEL_GAP_CM = 0.25
SITE_STYLE = "c0.20c"
PEN_RESOLVED = "0.4p,black"
PEN_UNRESOLVED = "1.3p,black"
BED_SPACING_KM = 0.1
BED_SMOOTH_KM = 1.0
# The bed ramp: abyss without its near-black deep end, brightened by a gamma
# stretch so the shallow half of the bed reaches the light blues.
BED_RAMP_START = 0.125
BED_GAMMA = 0.6
# The hillshade is a grey layer over the bed at SHADE_OPACITY: sun from the
# north-west at 45 degrees, relief exaggerated SHADE_EXAGGERATION times so the
# gentle, smoothed bed still reads.
SHADE_AZIMUTH_DEG = 315.0
SHADE_ALTITUDE_DEG = 45.0
SHADE_EXAGGERATION = 20.0
SHADE_OPACITY = 0.35

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


def smooth_bed(bed):
    """BedMachine bed ready for hillshading: Gaussian-smoothed over
    BED_SMOOTH_KM, then resampled (cubic) to BED_SPACING_KM.

    Shaded at its native 500 m, or after a bicubic resample, the hillshade is
    dominated by the seams of BedMachine's interpolation between radar lines
    (straight streaks and crosses) rather than by the relief; a 1 km smoothing
    removes them and leaves the troughs and ridges.  The resample after it is
    cubic because a linear one leaves flat facets that catch the light as
    white glints."""
    from scipy import ndimage

    dx = float(abs(bed["x"].values[1] - bed["x"].values[0]))
    smoothed = bed.copy(data=ndimage.gaussian_filter(bed.values, BED_SMOOTH_KM / dx,
                                                     mode="nearest"))
    x = np.arange(float(bed["x"].min()), float(bed["x"].max()), BED_SPACING_KM)
    y = np.arange(float(bed["y"].min()), float(bed["y"].max()), BED_SPACING_KM)
    return smoothed.interp(x=x, y=y, method="cubic")


def abyss_table(tmp: Path):
    """The bed ramp (style.BED_CPT, GMT's abyss) sampled at 64 positions on
    [0, 1]: (positions, RGB 0-255)."""
    import pygmt

    path = tmp / "abyss.cpt"
    pygmt.makecpt(cmap=style.BED_CPT, series=[0.0, 1.0, 1.0 / 64], output=str(path))
    pos, rgb = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        # Slices are "z0 rgb z1 rgb [annotation]"; take each at its centre.
        if len(parts) >= 4 and line[0] not in "#BFN":
            pos.append(0.5 * (float(parts[0]) + float(parts[2])))
            rgb.append([float(v) for v in parts[1].split("/")])
    return np.asarray(pos), np.asarray(rgb)


def bed_rgb(z, lo: float, hi: float, table):
    """Bed colour (RGB 0-255) of elevations ``z`` on the stretched abyss ramp.

    Stretched linearly over the bed's range, abyss puts most of this bed in its
    darkest navy and leaves the light blues to a few ridges.  Elevation z maps
    to ramp position BED_RAMP_START + (1 - BED_RAMP_START) * f**BED_GAMMA,
    f = (z - lo) / (hi - lo), which skips the near-black end and gives the
    shallower bed more of the light half."""
    pos, rgb = table
    f = np.clip((np.asarray(z, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    t = BED_RAMP_START + (1.0 - BED_RAMP_START) * f ** BED_GAMMA
    return np.stack([np.interp(t, pos, rgb[:, k]) for k in range(3)], axis=-1)


def bed_cpt(lo: float, hi: float, table, path: str) -> str:
    """The bed ramp as a continuous CPT, for the colour bar."""
    z = np.linspace(lo, hi, 65)
    c = ["/".join(str(round(v)) for v in row) for row in bed_rgb(z, lo, hi, table)]
    with open(path, "w") as fh:
        for i in range(z.size - 1):
            fh.write(f"{z[i]:.6g} {c[i]} {z[i + 1]:.6g} {c[i + 1]}\n")
        fh.write(f"B {c[0]}\nF {c[-1]}\nN white\n")
    return path


def bed_image(bed, lo: float, hi: float, table):
    """The bed map as one RGB image: the bed ramp with a grey hillshade laid
    over it at SHADE_OPACITY.

    Blended here rather than drawn as a second, transparent GMT layer, which
    rasterises with thin seams where the bed shows through unshaded.  A flat
    bed (shade = sin(altitude)) gets mid grey, so the layer darkens slopes
    facing away from the sun and lightens those facing it by the same amount,
    and nothing is pushed towards white."""
    import xarray as xr

    colour = bed_rgb(bed.values, lo, hi, table)
    flat = float(np.sin(np.deg2rad(SHADE_ALTITUDE_DEG)))
    grey = 255.0 * np.clip(hillshade(bed).values - flat + 0.5, 0.0, 1.0)
    out = (1.0 - SHADE_OPACITY) * colour + SHADE_OPACITY * grey[..., None]
    # GMT reads images north-up.
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)[::-1]
    return xr.DataArray(
        np.moveaxis(out, -1, 0), dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": bed["y"].values[::-1], "x": bed["x"].values},
    )


def hillshade(bed):
    """Lambertian hillshade of the bed, 0 (facing away) to 1 (facing the sun).

    Computed directly rather than with grdgradient's normalisation, which on
    this smooth bed pins a quarter of the grid at its darkest value and leaves
    the layer blotchy."""
    dx_m = 1000.0 * float(abs(bed["x"].values[1] - bed["x"].values[0]))
    dy_m = 1000.0 * float(abs(bed["y"].values[1] - bed["y"].values[0]))
    z = SHADE_EXAGGERATION * np.asarray(bed.values, dtype=float)
    dz_dy, dz_dx = np.gradient(z, dy_m, dx_m)
    az = np.deg2rad(SHADE_AZIMUTH_DEG)
    alt = np.deg2rad(SHADE_ALTITUDE_DEG)
    # Unit vector towards the sun (x east, y north, z up) dotted with the normal.
    sun = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    norm = np.sqrt(1.0 + dz_dx ** 2 + dz_dy ** 2)
    shade = (-dz_dx * sun[0] - dz_dy * sun[1] + sun[2]) / norm
    return bed.copy(data=np.clip(shade, 0.0, 1.0))


def shear_edges(cap: float) -> np.ndarray:
    """Band edges for the shear share, spaced as a power law up to ``cap``.

    Three quarters of the sites sit below 0.15, so uniform bands would put
    nearly all of them in the bottom two; spacing the edges as
    cap * f**(1/SHEAR_GAMMA) keeps the low values apart while the bands stay
    discrete, as on every other colour scale in the paper."""
    f = np.linspace(0.0, 1.0, style.N_LEVELS + 1)
    return cap * f ** (1.0 / SHEAR_GAMMA)


def draw_panel(fig, *, bed, region, proj, values, cpt, x, y,
               resolved, stations_xy, stations_v, stations_resolved, letter,
               row_label, scale=False):
    """One bare map panel: the hillshaded bed image, every ApRES site on top."""
    fig.grdimage(grid=bed, region=region, projection=proj, frame="+n")

    # Sites where shear is resolved carry a thin outline; the rest a heavy one,
    # drawn last so they stay findable.  The convention is given in the caption.
    res = np.asarray(resolved, dtype=bool)
    st_res = np.asarray(stations_resolved, dtype=bool)
    sx, sy = stations_xy
    for keep, pen in ((st_res, PEN_RESOLVED), (~st_res, PEN_UNRESOLVED)):
        if keep.any():
            fig.plot(x=sx[keep], y=sy[keep], style=SITE_STYLE, fill=stations_v[keep],
                     cmap=cpt, pen=pen, region=region, projection=proj)
    if res.any():
        fig.plot(x=x[res], y=y[res], style=SITE_STYLE, fill=values[res],
                 cmap=cpt, pen=PEN_RESOLVED, region=region, projection=proj)
    if (~res).any():
        fig.plot(x=x[~res], y=y[~res], style=SITE_STYLE, fill=values[~res],
                 cmap=cpt, pen=PEN_UNRESOLVED, region=region, projection=proj)

    x0, x1, y0, y1 = region
    fig.plot(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], pen="0.6p,black",
             region=region, projection=proj, no_clip=True)
    style.gmt_panel_letter(fig, letter, region=region, projection=proj,
                           at=(0.025, 0.86))
    fig.text(x=[x0 - 0.012 * (x1 - x0)], y=[0.5 * (y0 + y1)], text=[row_label],
             font=style.GMT_TEXT, angle=90, justify="CB", no_clip=True,
             region=region, projection=proj)
    if scale:
        # Upper left, under the letter: the transect runs through the lower left.
        mapfurniture.scalebar(fig, region=region, projection=proj, loc=(0.03, 0.70))


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
    tmp = Path(tempfile.mkdtemp(prefix="fig04_"))

    # Bed: a continuous bathymetric ramp with a hillshade laid over it, as on a
    # conventional bed map.  The bed lies 800-1600 m below sea level, so a
    # land/sea ramp hinged at zero would give only its darkest slice; a
    # bathymetry ramp is stretched over the bed's own range instead, rounded
    # outward so no cell falls off it.  The hillshade is a faint grey layer
    # over the colours (bed_image) rather than an intensity that lightens and
    # darkens them: that read as glossy and washed the shallow bed to white.
    bed = smooth_bed(load_bed(region))
    b_lo = 100.0 * np.floor(float(np.nanmin(bed.values)) / 100.0)
    b_hi = 100.0 * np.ceil(float(np.nanmax(bed.values)) / 100.0)
    table = abyss_table(tmp)
    image = bed_image(bed, b_lo, b_hi, table)
    bed_cpt_path = bed_cpt(b_lo, b_hi, table, str(tmp / "bed.cpt"))

    # Strain rate in 10^-3 /yr, symmetric about zero at the 99th percentile of
    # |value| to the nearest 5, as in Fig. 2.
    col_eps = "strain_rate_depth_mean_yr"
    if not np.isfinite(df[col_eps].values).any():
        col_eps = "strain_rate_yr"
    k = style.STRAIN_SCALE
    eps = k * point[col_eps].values
    eps_st = k * stations[col_eps].values
    eps_all = np.abs(np.concatenate([eps, eps_st]))
    lim = 5.0 * max(1.0, np.round(float(np.nanpercentile(eps_all[np.isfinite(eps_all)], 99)) / 5.0))
    eps_cpt = style.gmt_cpt_bands(style.DIV_VELOCITY, np.linspace(-lim, lim, style.N_LEVELS + 1),
                                  str(tmp / "eps.cpt"))

    ratio = point["shear_share"].values
    ratio_all = np.concatenate([ratio, stations["shear_share"].values])
    cap = np.ceil(10.0 * float(np.nanmax(ratio_all[np.isfinite(ratio_all)]))) / 10.0
    shear_cpt = style.gmt_cpt_bands(style.SEQ_SHARE, shear_edges(cap), str(tmp / "shear.cpt"))
    resolved = point["shear_resolved"].astype(bool).values
    st_resolved = stations["shear_resolved"].astype(bool).values

    pygmt.config(**style.GMT_CONFIG)
    fig = pygmt.Figure()
    cbar_len = 0.88 * height_cm

    # ------------------------------ (a) depth-averaged vertical strain rate
    draw_panel(
        fig, bed=image, region=region, proj=proj, values=eps,
        cpt=eps_cpt, x=x, y=y, resolved=resolved, stations_xy=(sx, sy),
        stations_v=eps_st, stations_resolved=st_resolved, letter="a",
        row_label=f"Depth-averaged {style.GMT_EPS_ZZ}", scale=True,
    )
    style.gmt_colorbar(fig, cmap=eps_cpt, length_cm=cbar_len,
                 frame=[f"xa5f{2 * lim / style.N_LEVELS:g}",
                        f"x+l{style.GMT_EPS_ZZ} ({style.GMT_STRAIN_UNIT})"])

    # ----------------------------------------- (b) shear share of deformation
    fig.shift_origin(yshift=f"-{height_cm + PANEL_GAP_CM:.3f}c")
    draw_panel(
        fig, bed=image, region=region, proj=proj, values=ratio,
        cpt=shear_cpt, x=x, y=y, resolved=resolved, stations_xy=(sx, sy),
        stations_v=stations["shear_share"].values, stations_resolved=st_resolved,
        letter="b",
        row_label="Shear share",
    )
    # The bands span 0 to the rounded-up maximum, so nothing overflows them.
    style.gmt_colorbar(fig, cmap=shear_cpt, length_cm=cbar_len, extend=False,
                 frame=["xa0.2f0.1+l@%Times-Italic%A@%%@-s@-/(@%Times-Italic%A@%%@-s@-"
                        " + @%Times-Italic%A@%%@-p@-)"])

    # One bed colour bar for both panels, under the lower one.
    # The ramp spans the bed's full range, so the bar needs no overflow arrows.
    style.gmt_colorbar(fig, cmap=bed_cpt_path, length_cm=0.45 * WIDTH_CM, anchor="JBC",
                       offset="0c/0.45c", horizontal=True, extend=False,
                       frame=["xa200f100+lBed elevation, BedMachine v4 (m)"])

    style.gmt_save(fig, paths.FIGURES / "fig04_map.png")


if __name__ == "__main__":
    main()
