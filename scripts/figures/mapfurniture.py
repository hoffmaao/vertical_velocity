"""Shared map furniture for the PyGMT map figures.

Keeps the scale bar identical between figures, and identical to the one on
the paper's matplotlib maps (figstyle.scalebar), so a reader who has learned
to read one map can read the others without relearning its furniture.
The graticule labels each kind of line along one chosen frame edge.
"""

from __future__ import annotations

import numpy as np


def scalebar(fig, *, region, projection, loc=(0.06, 0.07), halo: bool = True) -> None:
    """Kilometre scale bar sized to a round fifth of the panel width, as
    figstyle.scalebar: a black bar, with a thin white halo unless ``halo`` is
    False, its length above it.

    ``region`` is in projected kilometres; ``loc`` is the bar's left end as a
    fraction of the panel."""
    import style

    x0, x1, y0, y1 = region
    raw = (x1 - x0) / 5.0
    mag = 10 ** np.floor(np.log10(raw))
    length = float(min([1, 2, 5, 10], key=lambda m: abs(m * mag - raw)) * mag)
    bx = x0 + loc[0] * (x1 - x0)
    by = y0 + loc[1] * (y1 - y0)
    for pen in (("3.6p,white", "2p,black") if halo else ("2p,black",)):
        fig.plot(x=[bx, bx + length], y=[by, by], pen=pen, region=region,
                 projection=projection)
    fig.text(x=[bx + 0.5 * length], y=[by], text=[f"{length:g} km"],
             font=style.GMT_ANNOT, justify="CB", offset="0c/0.12c",
             region=region, projection=projection)


def nan_split_polylines(x, y):
    """Yield contiguous finite runs of a NaN-separated polyline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if not good.any():
        return
    edges = np.flatnonzero(np.diff(good.astype(int)))
    starts = [0] if good[0] else []
    starts += list(edges[good[edges + 1]] + 1)
    ends = list(edges[good[edges]] + 1)
    if good[-1]:
        ends.append(good.size)
    for a, b in zip(starts, ends):
        if b - a > 1:
            yield x[a:b], y[a:b]


def _edge_crossings(x, y, region, edge):
    """Points where the polyline (x, y) crosses one frame edge ('left',
    'right', 'bottom' or 'top'), as (x, y) pairs in region units."""
    x0, x1, y0, y1 = region
    out = []
    if edge in ("left", "right"):
        xe = x0 if edge == "left" else x1
        for i in range(len(x) - 1):
            if (x[i] - xe) * (x[i + 1] - xe) <= 0 and x[i] != x[i + 1]:
                t = (xe - x[i]) / (x[i + 1] - x[i])
                yc = y[i] + t * (y[i + 1] - y[i])
                if y0 <= yc <= y1:
                    out.append((xe, yc))
    else:
        ye = y0 if edge == "bottom" else y1
        for i in range(len(y) - 1):
            if (y[i] - ye) * (y[i + 1] - ye) <= 0 and y[i] != y[i + 1]:
                t = (ye - y[i]) / (y[i + 1] - y[i])
                xc = x[i] + t * (x[i + 1] - x[i])
                if x0 <= xc <= x1:
                    out.append((xc, ye))
    return out


def graticule(fig, *, region, projection, lats, lons, lat_edge="bottom", lon_edge="left",
              pen="0.3p,gray45,2_2:0p", epsg="EPSG:3031") -> None:
    """Parallels and meridians over a map drawn in projected kilometres.

    Parallels are labelled where they cross ``lat_edge`` and meridians where
    they cross ``lon_edge``, just outside the frame, so each kind of label
    reads along a single edge as on a conventional map; a line that never
    reaches its edge is drawn but left unlabelled.  Which edges suit depends
    on where the map sits on the polar stereographic plane, so the caller
    chooses them and leaves room for the labels beside those edges.

    ``region`` is [x0, x1, y0, y1] in km of ``epsg``; ``lats`` and ``lons`` are
    in degrees, negative south and west."""
    import style
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    x0, x1, y0, y1 = region

    def draw(lon_deg, lat_deg):
        X, Y = tr.transform(lon_deg, lat_deg)
        X, Y = X / 1000.0, Y / 1000.0
        inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        for xs, ys in nan_split_polylines(np.where(inside, X, np.nan), np.where(inside, Y, np.nan)):
            fig.plot(x=xs, y=ys, pen=pen, region=region, projection=projection)
        return X, Y

    justify = {"left": "RM", "right": "LM", "bottom": "TC", "top": "BC"}
    offset = {"left": "-0.1c/0c", "right": "0.1c/0c", "bottom": "0c/-0.1c", "top": "0c/0.1c"}

    def label(X, Y, text, edge):
        hits = _edge_crossings(X, Y, region, edge)
        if hits:
            px, py = hits[0]
            fig.text(x=[px], y=[py], text=[text], font=style.GMT_ANNOT,
                     justify=justify[edge], offset=offset[edge], no_clip=True,
                     region=region, projection=projection)

    lon_dense = np.linspace(-180.0, 180.0, 3601)
    for lat in lats:
        X, Y = draw(lon_dense, np.full_like(lon_dense, float(lat)))
        hemi = "S" if lat < 0 else "N"
        label(X, Y, f"{abs(lat):g}\\260{hemi}", lat_edge)
    lat_dense = np.linspace(-90.0, -60.0, 1201)
    for lon in lons:
        X, Y = draw(np.full_like(lat_dense, float(lon)), lat_dense)
        hemi = "W" if lon < 0 else "E"
        label(X, Y, f"{abs(lon):g}\\260{hemi}", lon_edge)
