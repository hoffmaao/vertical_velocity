"""Figure 4 - map view of the deformation measured along the transect.

Two stacked map panels over the same ground, with bed topography beneath:
the depth-averaged vertical strain rate above, and the share of the
deformation carried by shear rather than plug flow below.  The bed is shared
between the panels and carries one colour bar, at the foot of the figure, in
grey relief so the coloured symbols above it carry all the hue.  The panels
follow the paper's map grammar (figstyle): bare frames with a scale bar,
discrete colour bands, panel letters and a row label on the left.

Every ApRES measurement is drawn the same way, whether it came from an
attended transect occupation or from one of the three autonomous stations,
which are paired from the first and last acquisition of their year-long
records.  All of them are decomposed on the same degree-4 shifted-Legendre
basis over the same normalised-depth window.

Whether shear is resolved at a site is carried by the marker outline - thin
where the degree->=2 coefficients are jointly distinguishable from zero, heavy
where they are not - and keyed in the legend above the panels.

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
bounded and one-signed, so it gets a sequential ramp whose band edges are
spaced as a power law, because three quarters of the sites sit below 0.15.

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


def shear_edges(cap: float) -> np.ndarray:
    """Band edges for the shear share, spaced as a power law up to ``cap``.

    Three quarters of the sites sit below 0.15, so uniform bands would put
    nearly all of them in the bottom two; spacing the edges as
    cap * f**(1/SHEAR_GAMMA) keeps the low values apart while the bands stay
    discrete, as on every other colour scale in the paper."""
    f = np.linspace(0.0, 1.0, style.N_LEVELS + 1)
    return cap * f ** (1.0 / SHEAR_GAMMA)


def draw_panel(fig, *, bed, bed_cpt, region, proj, values, cpt, x, y, resolved,
               stations_xy, stations_v, letter, row_label, scale=False):
    """One bare map panel: banded bed underneath, every ApRES site on top."""
    fig.grdimage(grid=bed, region=region, projection=proj, cmap=bed_cpt,
                 nan_transparent=True, frame="+n")

    # Sites where shear is resolved carry a thin outline; the rest a heavy one,
    # drawn last so they stay findable.  Both are keyed in the legend.
    res = np.asarray(resolved, dtype=bool)
    if len(stations_xy[0]):
        fig.plot(x=stations_xy[0], y=stations_xy[1], style=SITE_STYLE,
                 fill=stations_v, cmap=cpt, pen=PEN_RESOLVED,
                 region=region, projection=proj)
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

    # Bed: grey relief in N_LEVELS bands on round 100 m edges, under the
    # coloured symbols (figstyle "relief").
    bed = load_bed(region)
    b_lo, b_hi = np.nanpercentile(bed.values, [1, 99])
    b_lo, b_hi = 100.0 * np.floor(b_lo / 100.0), 100.0 * np.ceil(b_hi / 100.0)
    bed_cpt = style.gmt_cpt_bands(style.SEQ_RELIEF, np.linspace(b_lo, b_hi, style.N_LEVELS + 1),
                                  str(tmp / "bed.cpt"))

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

    pygmt.config(**style.GMT_CONFIG)
    fig = pygmt.Figure()
    cbar_len = 0.88 * height_cm

    # Legend across the top, then the panels below it.
    legend_entries = [
        ("symbol", (SITE_STYLE, "white", PEN_RESOLVED), "ApRES station, shear resolved"),
        ("symbol", (SITE_STYLE, "white", PEN_UNRESOLVED), "shear not resolved"),
    ]
    style.gmt_top_legend(fig, legend_entries, width_cm=WIDTH_CM)
    fig.shift_origin(yshift=f"-{height_cm + 0.25:.3f}c")

    # ------------------------------ (a) depth-averaged vertical strain rate
    draw_panel(
        fig, bed=bed, bed_cpt=bed_cpt, region=region, proj=proj, values=eps,
        cpt=eps_cpt, x=x, y=y, resolved=resolved, stations_xy=(sx, sy),
        stations_v=eps_st, letter="a",
        row_label=f"Depth-averaged {style.GMT_EPS_ZZ}", scale=True,
    )
    style.gmt_colorbar(fig, cmap=eps_cpt, length_cm=cbar_len,
                 frame=[f"xa5f{2 * lim / style.N_LEVELS:g}",
                        f"x+l{style.GMT_EPS_ZZ} ({style.GMT_STRAIN_UNIT})"])

    # ----------------------------------------- (b) shear share of deformation
    fig.shift_origin(yshift=f"-{height_cm + PANEL_GAP_CM:.3f}c")
    draw_panel(
        fig, bed=bed, bed_cpt=bed_cpt, region=region, proj=proj, values=ratio,
        cpt=shear_cpt, x=x, y=y, resolved=resolved, stations_xy=(sx, sy),
        stations_v=stations["shear_share"].values, letter="b",
        row_label="Shear share",
    )
    # The bands span 0 to the rounded-up maximum, so nothing overflows them.
    style.gmt_colorbar(fig, cmap=shear_cpt, length_cm=cbar_len, extend=False,
                 frame=["xa0.2f0.1+l@%Times-Italic%A@%%@-s@-/(@%Times-Italic%A@%%@-s@-"
                        " + @%Times-Italic%A@%%@-p@-)"])

    # One bed colour bar for both panels, under the lower one.
    style.gmt_colorbar(fig, cmap=bed_cpt, length_cm=0.45 * WIDTH_CM, anchor="JBC",
                       offset="0c/0.45c", horizontal=True,
                       frame=["xa200f100+lBed elevation, BedMachine v4 (m)"])

    style.gmt_save(fig, paths.FIGURES / "fig04_map.png")


if __name__ == "__main__":
    main()
