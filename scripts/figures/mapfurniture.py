"""Shared map furniture for the PyGMT map figures.

Keeps the scale bar identical between figures, and identical to the one on
the paper's matplotlib maps (figstyle.scalebar), so a reader who has learned
to read one map can read the others without relearning its furniture.
"""

from __future__ import annotations

import numpy as np


def scalebar(fig, *, region, projection, loc=(0.06, 0.07)) -> None:
    """Kilometre scale bar sized to a round fifth of the panel width, as
    figstyle.scalebar: a black bar with a thin white halo, its length above it.

    ``region`` is in projected kilometres; ``loc`` is the bar's left end as a
    fraction of the panel."""
    import style

    x0, x1, y0, y1 = region
    raw = (x1 - x0) / 5.0
    mag = 10 ** np.floor(np.log10(raw))
    length = float(min([1, 2, 5, 10], key=lambda m: abs(m * mag - raw)) * mag)
    bx = x0 + loc[0] * (x1 - x0)
    by = y0 + loc[1] * (y1 - y0)
    for pen in ("3.6p,white", "2p,black"):
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
