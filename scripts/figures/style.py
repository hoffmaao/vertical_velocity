"""Figure style for the ApRES pipeline figures (GRL manuscript Figs. 1-5).

The paper's figures all follow one grammar, after Morrow, Ghelichkhan &
Kramer (2026), defined once in ``scripts/figstyle.py``.  This module adapts it
to these figure scripts and to PyGMT, so a value changed there changes here:

  * type: Times with STIX math, 8.5 pt text, 9 pt axis labels, 7.5 pt ticks;
  * every figure drawn at the width it is printed (``FULL`` = 17.8 cm, or
    ``ONE_COL`` for the single-panel figure placed at 0.55 of the text width),
    so those sizes are the sizes on the page;
  * series in a fixed colour and marker order, never cycled;
  * fields in ``N_LEVELS`` discrete colour bands, with the colormap chosen by
    the quantity (``figstyle.STYLES``, except the strain-rate and shear-share
    ramps below, which avoid the blue of Fig. 4's bed); Fig. 4's bed itself is
    a continuous hillshaded ramp under the banded symbols;
  * panel letters in grey circles, one legend across the top (framed, except
    in Fig. 1; Fig. 4 has none and keys its markers in the caption);
  * PNG and PDF written without timestamps, so an unchanged figure gives an
    empty git diff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import figstyle as fs  # noqa: E402

# ------------------------------------------------------------------ geometry
CM = 1.0 / 2.54
TWO_COL = fs.FULL                  # 17.8 cm, included at \textwidth
ONE_COL = 0.55 * fs.FULL           # included at 0.55\textwidth
TWO_COL_CM = TWO_COL / CM
FS, FS_LABEL, FS_TICK = fs.FS, fs.FS_LABEL, fs.FS_TICK
N_LEVELS = fs.N_LEVELS

# --------------------------------------------------------------------- colour
SEQ_POWER = fs.STYLES["power"][0]        # radar return
SEQ_SPEED = fs.STYLES["speed"][0]        # surface speed
# The bed map (Fig. 4) is a hillshaded bathymetric ramp, which is all blue
# because the bed lies below sea level, so the quantities drawn over it avoid
# blue: a purple-orange diverging ramp for strain rate (orange extending,
# purple thinning, light grey at zero) and yellow-orange-red for shear share.
# Fig. 2 uses the same strain ramp so the quantity reads the same everywhere.
BED_CPT = "abyss"                        # GMT bathymetry ramp, deep navy to light blue
SEQ_SHARE = "YlOrRd"                     # shear share
DIV_VELOCITY = "PuOr_r"                  # strain rate, centred on zero

# Series order (Morrow et al. reported.py): colour and marker travel together.
CATEGORICAL = (fs.RED, fs.BLUE, fs.GREEN, fs.PURPLE, fs.BLACK)
MARKERS = ("o", "s", "^", "v", "D")
INK = fs.BLACK
INK_MUTED = fs.GREY
GRID = "0.85"

COLOR_SURFACE_LINE = fs.BLACK
COLOR_BED_BEDMACHINE = fs.BLACK          # drawn dashed, as in the model figures

# Strain rates are quoted in 10^-3 yr^-1 throughout the paper.
STRAIN_SCALE = 1e3
STRAIN_UNIT = r"$10^{-3}$ yr$^{-1}$"
EPS_ZZ = r"$\dot{\varepsilon}_{zz}$"

# GMT spellings of the same notation (ISOLatin1+ encoding; @!\227 overstrikes
# the dot accent, @~ is the Symbol font, \055 in it a true minus).
GMT_MINUS = "@~\\055@~"
GMT_EPS_ZZ = "@!\\227@~e@~@-@%Times-Italic%zz@%%@-"
GMT_STRAIN_UNIT = f"10@+{GMT_MINUS}3@+ yr@+{GMT_MINUS}1@+"

# ----------------------------------------------------------------- GMT type
# Times-Roman at the same point sizes as the matplotlib figures, so a map and a
# section side by side read as one typographic system.
GMT_FONT = "Times-Roman"
GMT_TEXT = f"{FS:g}p,{GMT_FONT},black"
GMT_ANNOT = f"{FS_TICK:g}p,{GMT_FONT},black"
GMT_LABEL = f"{FS_LABEL:g}p,{GMT_FONT},black"
GMT_CONFIG = dict(
    FONT_ANNOT_PRIMARY=GMT_ANNOT,
    FONT_ANNOT_SECONDARY=GMT_ANNOT,
    FONT_LABEL=GMT_LABEL,
    FONT_TAG=GMT_TEXT,
    MAP_FRAME_TYPE="plain",
    MAP_FRAME_PEN="0.6p,black",
    MAP_TICK_LENGTH_PRIMARY="3p",
    MAP_TICK_PEN_PRIMARY="0.6p,black",
    MAP_ANNOT_OFFSET_PRIMARY="2.5p",
    MAP_LABEL_OFFSET="4p",
    FORMAT_FLOAT_MAP="%.12lg",
    PS_CHAR_ENCODING="ISOLatin1+",
)


def use_paper_style() -> None:
    """Apply the paper's rcParams.  Call once at the top of a figure script."""
    fs.apply_rc()


grid = fs.grid
panel_letter = fs.panel_letter
top_legend = fs.top_legend
num = fs.num
nice = fs.nice
bands = fs.bands
discrete = fs.discrete


def discrete_cmap(name: str, norm):
    """(cmap, BoundaryNorm) of colormap ``name`` in N_LEVELS bands, as
    figstyle.discrete for a colormap outside figstyle.STYLES."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    cmap = plt.get_cmap(name)
    return cmap, BoundaryNorm(bands(norm), cmap.N, extend="both")


def symmetric_limits(values, percentile: float = 1.0) -> tuple[float, float]:
    """Round symmetric colour limits about zero (figstyle.limits, kind 'strain')."""
    return fs.limits(values, "strain", percentile)


def colorbar(fig, mappable, cax, label: str):
    """Discrete-band colour bar with its label rotated alongside."""
    return fs.colorbar(fig, mappable, cax, label)


def save(fig, path: Path) -> None:
    """PNG at 300 dpi and a vector PDF beside it, with no embedded timestamp."""
    fs.save(fig, Path(path))


# --------------------------------------------------------------- GMT helpers
def gmt_panel_letter(fig, letter: str, *, region, projection,
                     at=(0.045, 0.92)) -> None:
    """Grey-circle panel letter at the top left, as figstyle.panel_letter."""
    x0, x1, y0, y1 = region
    x = x0 + at[0] * (x1 - x0)
    y = y0 + at[1] * (y1 - y0)
    fig.plot(x=[x], y=[y], style="c0.42c", fill="211/211/211", pen="0.6p,black",
             region=region, projection=projection)
    fig.text(x=[x], y=[y], text=[letter], font=GMT_TEXT, justify="CM",
             region=region, projection=projection)


def gmt_cpt_bands(cmap: str, edges, path: str) -> str:
    """A GMT CPT of flat bands between ``edges``, coloured from a matplotlib
    colormap sampled at the band centres, so the maps use the same colours and
    banding as the matplotlib panels.  Out-of-range values take the end colours."""
    import matplotlib.pyplot as plt

    cm = plt.get_cmap(cmap)
    edges = np.asarray(edges, dtype=float)
    n = edges.size - 1

    def rgb(f):
        r, g, b, _ = cm(f)
        return f"{round(255 * r)}/{round(255 * g)}/{round(255 * b)}"

    with open(path, "w") as fh:
        for i in range(n):
            c = rgb((i + 0.5) / n)
            fh.write(f"{edges[i]:.10g} {c} {edges[i + 1]:.10g} {c}\n")
        fh.write(f"B {rgb(0.5 / n)}\nF {rgb(1 - 0.5 / n)}\nN white\n")
    return path


def gmt_top_legend(fig, entries, *, width_cm: float, frame: bool = True) -> float:
    """One legend centred across the figure, as figstyle.top_legend; framed
    unless ``frame`` is False.

    ``entries`` are (kind, spec, label): kind "symbol" with spec (style, fill,
    pen) (fill None for an open symbol), "line" with spec = pen, or "ramp"
    with spec = (colours, width): a line in equal steps of the given colours,
    for a bundle of traces shaded by date.  Entries run in as many aligned
    columns as fit in ``width_cm``, row by row.

    Drawn in its own centimetre frame with its lower edge at the current
    origin; returns its height in cm.  Draw it first and shift the origin down
    to the panels: PyGMT loses content when the origin is shifted up and back
    down again."""
    char_cm = 0.47 * FS / 72 * 2.54          # mean Times advance at FS
    sym_cm, gap_cm, pad_cm, col_cm, row_cm = 0.7, 0.2, 0.3, 0.55, 0.45
    widths = [sym_cm + gap_cm + char_cm * len(_plain(lab)) for _, _, lab in entries]

    def layout(ncol):
        cols = [max(widths[c::ncol]) for c in range(ncol)]
        return cols, sum(cols) + col_cm * (ncol - 1) + 2 * pad_cm

    for ncol in range(len(entries), 0, -1):
        cols, box_w = layout(ncol)
        if box_w <= width_cm:
            break
    nrow = -(-len(entries) // ncol)
    box_h = nrow * row_cm + 0.12
    x0 = 0.5 * (width_cm - box_w)
    region = [0, width_cm, 0, box_h]
    proj = f"X{width_cm}c/{box_h}c"
    # Without a frame the box is still drawn, in white: PyGMT crops the page
    # to the drawn content, and bare text alone gets its ascenders cut off.
    fig.plot(x=[x0, x0 + box_w, x0 + box_w, x0, x0], y=[0, 0, box_h, box_h, 0],
             fill="white", pen="0.6p,black" if frame else "0.6p,white",
             region=region, projection=proj)
    for i, (kind, spec, label) in enumerate(entries):
        r, c = divmod(i, ncol)
        x = x0 + pad_cm + sum(cols[:c]) + col_cm * c
        yc = box_h - 0.06 - (r + 0.5) * row_cm
        if kind == "symbol":
            sty, fill, pen = spec
            kw = {} if fill is None else {"fill": fill}
            fig.plot(x=[x + 0.5 * sym_cm], y=[yc], style=sty, pen=pen,
                     region=region, projection=proj, **kw)
        elif kind == "ramp":
            colours, width = spec
            edges = np.linspace(x, x + sym_cm, len(colours) + 1)
            for a, b, col in zip(edges[:-1], edges[1:], colours):
                fig.plot(x=[a, b], y=[yc, yc], pen=f"{width},{col}", region=region,
                         projection=proj)
        else:
            fig.plot(x=[x, x + sym_cm], y=[yc, yc], pen=spec, region=region,
                     projection=proj)
        fig.text(x=[x + sym_cm + gap_cm], y=[yc], text=[label], font=GMT_TEXT,
                 justify="LM", region=region, projection=proj)
    return box_h


def _plain(text: str) -> str:
    """A GMT label with its escape codes removed, for width estimates."""
    import re

    return re.sub(r"@%[^%]*%|@[-+~!%]|@\\\d{3}|\\\d{3}", "", text)


def gmt_colorbar(fig, *, cmap: str, length_cm: float, anchor: str = "JMR",
                 offset: str = "0.25c/0c", horizontal: bool = False,
                 extend: bool = True, frame) -> None:
    """A colour bar whose text prints at the paper's tick and label sizes.

    GMT shrinks a colour bar's fonts by sqrt(length / 15 cm) for bars shorter
    than 15 cm (measured: 0.515 at 4 cm, 0.731 at 8 cm, 0.964 at 14 cm), so the
    sizes handed to it are divided by that factor.  ``extend`` draws the
    out-of-range arrows; leave it off when no value falls outside the bands."""
    import pygmt

    f = min(1.0, float(np.sqrt(length_cm / 15.0)))
    fonts = dict(FONT_ANNOT_PRIMARY=f"{FS_TICK / f:.2f}p,{GMT_FONT},black",
                 FONT_LABEL=f"{FS_LABEL / f:.2f}p,{GMT_FONT},black",
                 MAP_ANNOT_OFFSET_PRIMARY=f"{2.5 / f:.2f}p",
                 MAP_LABEL_OFFSET=f"{4.0 / f:.2f}p",
                 MAP_TICK_LENGTH_PRIMARY=f"{2.5 / f:.2f}p")
    pos = (f"{anchor}+w{length_cm:.2f}c/0.25c+o{offset}" + ("+e" if extend else "")
           + ("+h" if horizontal else ""))
    with pygmt.config(**fonts):
        fig.colorbar(cmap=cmap, position=pos, frame=frame)


def gmt_save(fig, path: Path) -> None:
    """PNG at 300 dpi and a PDF beside it, as save() does for matplotlib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=fs.DPI)
    fig.savefig(path.with_suffix(".pdf"))
    print(f"Saved {path.with_suffix('')}.{{png,pdf}}", flush=True)
