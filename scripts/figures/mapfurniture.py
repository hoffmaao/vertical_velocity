"""Shared map furniture for the PyGMT map figures.

Keeps the scale bar identical between figures, so a reader who has learned to
read one map can read the other without relearning its furniture.  Both map
figures are drawn with PyGMT, so only the GMT bar is needed.
"""

from __future__ import annotations

import numpy as np


def scalebar_bw(fig, *, x0: float, y0: float, length_km: float,
                n_segments: int = 4, height_km: float = 2.2,
                font: str = "7p,Helvetica,black") -> None:
    """Alternating black-and-white bar scale, drawn in projected kilometres.

    GMT's own fancy scale bar is unavailable on a Cartesian projection, and a
    plain line carries no intermediate divisions.  This draws the conventional
    chequered bar: equal segments alternating black and white with a hairline
    frame, labelled at its ends and annotated with its unit.

    Parameters
    ----------
    x0, y0
        Left end of the bar, in the same projected kilometres as the map.
    length_km
        Total length of the bar.
    n_segments
        Number of alternating cells; 4 gives quarter divisions.
    height_km
        Bar thickness, in map units.
    """
    seg = float(length_km) / int(n_segments)
    for i in range(int(n_segments)):
        xa = x0 + i * seg
        xb = xa + seg
        fill = "black" if i % 2 == 0 else "white"
        fig.plot(
            x=[xa, xb, xb, xa, xa],
            y=[y0, y0, y0 + height_km, y0 + height_km, y0],
            fill=fill, pen="0.5p,black", close=True,
        )
    # End and midpoint labels, plus the unit on the right-hand end.
    ticks = [0.0, length_km / 2.0, length_km]
    for t in ticks:
        label = f"{t:g}" if t != length_km else f"{t:g} km"
        fig.text(
            x=[x0 + t], y=[y0 + height_km + 1.0], text=[label],
            font=font, justify="CB",
        )


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
