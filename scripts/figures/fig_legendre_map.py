"""Paper Fig. 3 - the Legendre decomposition and its results along the line, as one figure.

The decomposition (fig05_legendre.py) and the map of what it gives at every
station (fig04_map.py) answer one question: how far the measured profiles
depart from plug flow, and where.  The paper shows them as one figure, panels
a-d above e-f, which also keeps the main text within the journal's figure
budget.

The two halves are drawn by their own scripts, matplotlib for the profiles and
PyGMT for the maps, with the panel letters passed in; this script then stacks
the two vector PDFs at their native size (both are set to the 17.8 cm text
width, so neither is rescaled and the type stays at the paper's sizes) with
plain pdfTeX, and renders a PNG from the result.  Dates and the
PDF trailer ID are pinned so an unchanged figure re-renders byte-identical.

Usage
-----
    python scripts/figures/fig_legendre_map.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import fig04_map  # noqa: E402
import fig05_legendre  # noqa: E402
import paths  # noqa: E402

GAP_MM = 2.0          # between the legend under the profiles and the top map
PNG_DPI = 300

# Plain pdfTeX, so the composition needs no LaTeX packages: the page is sized to the
# two parts, each placed at its natural size and centred, profiles above maps.
STACK = r"""\pdfpagewidth=%(w).3fbp \pdfpageheight=%(h).3fbp
\pdfhorigin=0bp \pdfvorigin=0bp \hsize=%(w).3fbp \vsize=%(h).3fbp
\parindent=0pt \nopagenumbers \offinterlineskip
\pdfinfoomitdate=1 \pdfsuppressptexinfo=-1 \pdftrailerid{}
\pdfximage{legendre.pdf}\edef\profilesbox{\the\pdflastximage}
\pdfximage{map.pdf}\edef\mapsbox{\the\pdflastximage}
\vbox to \vsize{\hbox to \hsize{\hss\pdfrefximage\profilesbox\hss}\vskip %(gap).3fbp
\hbox to \hsize{\hss\pdfrefximage\mapsbox\hss}\vss}
\bye
"""


def page_size(pdf: Path) -> tuple[float, float]:
    """Width and height of a one-page PDF, in PostScript points."""
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True)
    line = next(l for l in info.stdout.splitlines() if l.startswith("Page size:"))
    w, _, h = line.split(":", 1)[1].split()[:3]
    return float(w), float(h)


def compose(legendre_pdf: Path, map_pdf: Path, out_pdf: Path) -> None:
    """Stack the two PDFs, profiles above maps, centred, at their native size."""
    (w1, h1), (w2, h2) = page_size(legendre_pdf), page_size(map_pdf)
    gap = GAP_MM / 25.4 * 72.0
    with tempfile.TemporaryDirectory(prefix="fig_legendre_map_") as tmp:
        tmp = Path(tmp)
        shutil.copy(legendre_pdf, tmp / "legendre.pdf")
        shutil.copy(map_pdf, tmp / "map.pdf")
        (tmp / "stack.tex").write_text(STACK % {"w": max(w1, w2), "h": h1 + gap + h2, "gap": gap})
        env = dict(os.environ, SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1")
        run = subprocess.run(["pdftex", "-interaction=nonstopmode", "-halt-on-error",
                              "stack.tex"], cwd=tmp, env=env, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError("pdftex failed composing the figure:\n" + run.stdout[-2000:])
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp / "stack.pdf", out_pdf)


def main() -> None:
    out = paths.FIGURES / "fig_legendre_map.pdf"
    with tempfile.TemporaryDirectory(prefix="fig_legendre_map_parts_") as parts:
        parts = Path(parts)
        fig05_legendre.main(letters="abcd", out=parts / "legendre.png")
        fig04_map.main(letters="ef", out=parts / "map.png")
        compose(parts / "legendre.pdf", parts / "map.pdf", out)
    subprocess.run(["pdftoppm", "-r", str(PNG_DPI), "-png", "-singlefile", str(out),
                    str(out.with_suffix(""))], check=True)
    print(f"Saved {out.with_suffix('')}.{{png,pdf}}", flush=True)


if __name__ == "__main__":
    main()
