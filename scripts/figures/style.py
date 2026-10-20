"""Shared figure style for the Thwaites ApRES manuscript.

One place to set type sizes, colours and output paths so that every panel in
the paper reads as part of the same system.

Colour is assigned by the job the quantity does, not by taste:

  * magnitude (return power, uncertainty, speed, shear fraction)
        -> a single-hue sequential ramp, light to dark;
  * polarity (vertical velocity and vertical strain rate, which change sign
        along the transect)
        -> a diverging ramp with a neutral grey midpoint, centred on zero, so
           that the sign change is the thing the eye lands on;
  * identity (a handful of highlighted profiles)
        -> a fixed categorical order, never cycled.

Type sizes match the PyGMT map figures - 8 pt annotations, 9 pt labels - so
that a reader moving between a section and a map sees one typographic system.

AGU single-column width is 95 mm and full width is 190 mm.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

# ------------------------------------------------------------------ geometry
MM = 1.0 / 25.4
ONE_COL = 95 * MM
TWO_COL = 190 * MM

# --------------------------------------------------------------------- colour
# Sequential: single hue, light -> dark.
SEQ_POWER = "cividis"        # return power; perceptually uniform and CVD-safe
SEQ_MAGNITUDE = "viridis"    # uncertainty, speed
SEQ_SHEAR = "magma"          # shear fraction

# Diverging: two hues, neutral midpoint, always centred on zero.
DIV_VELOCITY = "RdBu_r"

# Identity: fixed order, never cycled.
CATEGORICAL = ("#3B6FB6", "#D1622B", "#4C9A6A", "#8C5BA8", "#B5893B")

# Ink and furniture
INK = "#1a1a1a"
INK_SECONDARY = "#4d4d4d"
INK_MUTED = "#808080"
GRID = "#d9d9d9"
SURFACE = "#ffffff"

# Reference geometry drawn on sections
COLOR_SURFACE_LINE = "#1a1a1a"
COLOR_BED_APRES = "#1a1a1a"
COLOR_BED_BEDMACHINE = "#B5893B"


def use_paper_style() -> None:
    """Apply the manuscript rcParams.  Call once at the top of a figure script."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_SECONDARY,
            "axes.linewidth": 0.6,
            "axes.facecolor": SURFACE,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "lines.linewidth": 1.0,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "text.color": INK,
        }
    )


def symmetric_limits(values, quantile: float = 99.0) -> tuple[float, float]:
    """Symmetric colour limits about zero for a diverging quantity.

    Centring on zero is what makes a diverging ramp legible: without it the
    neutral midpoint no longer marks the sign change.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    m = float(np.nanpercentile(np.abs(v), quantile))
    m = m if m > 0 else float(np.nanmax(np.abs(v)))
    return (-m, m)


def thin_colorbar(fig, mappable, ax, label: str, *, orientation: str = "vertical",
                  pad: float = 0.02, fraction: float = 0.025, **kw):
    """A recessive colorbar: thin, hairline-framed, single-line label."""
    cb = fig.colorbar(
        mappable, ax=ax, orientation=orientation, pad=pad, fraction=fraction,
        **kw,
    )
    cb.set_label(label, fontsize=9, color=INK)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(INK_SECONDARY)
    cb.ax.tick_params(labelsize=8, width=0.5, length=2, color=INK_SECONDARY)
    return cb


def save(fig, path: Path, *, also: Path | None = None) -> None:
    """Write a figure and echo where it went."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")
    if also is not None:
        also = Path(also)
        also.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(also)
        print(f"wrote {also}")
