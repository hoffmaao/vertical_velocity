"""Shared loading, screening and derived quantities for the manuscript figures.

Every figure applies the same quality screen and computes the same derived
measures here, so that a number quoted in one panel cannot disagree with the
same number in another.

Two decisions are made in this module and are worth stating plainly.

Source
------
Figures read ``legendre_fits.csv``, written by ``scripts/fit_legendre.py``,
which decomposes every profile on one common normalised-depth window.  Fitting
each site over whatever depth range it happened to reach made the coefficients
incomparable between sites and left the along-flow signal buried in noise: the
median site-to-site change in strain rate was 47% of the survey interquartile
range before the common window and 9% after it.

Quality screen
--------------
A profile is retained only if enough coherent segments survived the fine
alignment.  Profiles built from a few dozen segments span too little of the
column to constrain curvature, and they are where the implausible strain rates
sit: the single site with a strain rate of 0.13 yr-1, two orders of magnitude
above the survey median, is built from 52 segments.

Shear measured as an amplitude, not as a ratio
----------------------------------------------
The natural summary is the share of the Legendre power carried by degrees >= 2.
That ratio is ill-conditioned: its denominator contains the degree-1 (plug
flow) coefficient, which passes through zero where the column stops thinning,
and the ratio then swings to 1 for numerical rather than physical reasons.
Adjacent sites, 0.5-2 km apart, differ in it by 0.39 on a 0-1 scale - far more
than the ice can plausibly change over that distance.

The amplitudes themselves are well determined, so both are reported in m/yr:

    plug amplitude  = |c1|
    shear amplitude = sqrt(sum_{n>=2} c_n^2)

Both are in the same units on the same axis, so they can be compared directly
without a ratio and without a second y-scale.  The ratio is still provided for
continuity, but it should only be read where the plug amplitude is resolved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paths  # noqa: E402

MIN_SEGMENTS = 100
COEFF_COLUMNS = ("c0", "c1", "c2", "c3", "c4")

# Fine-alignment windows are 6 m wide and stepped every 2 m, so neighbouring
# samples share two thirds of their data and the profile is oversampled by
# about this factor.  It is used to deflate the degrees of freedom of the
# F-test towards an effective sample size.
OVERSAMPLING = 3.0


FITS = paths.VERTICAL_VELOCITY / "legendre_fits.csv"

# Autonomous stations farther than this from the survey line are off it.  LTG
# sits 1.6 km from the line; Lake1 and Lake2 sit 23 and 18 km north of it.
ONLINE_KM = 5.0


def load_summary(*, screen: bool = True, require_window: bool = True) -> pd.DataFrame:
    """Per-site fits with the quality screen applied.

    ``require_window`` keeps only the sites whose returns actually reach the
    bottom of the common fitting window, so that no coefficient is reported
    from a profile that was extrapolated into a depth range it never sampled.
    """
    if not FITS.exists():
        raise SystemExit("run scripts/fit_legendre.py first")
    df = pd.read_csv(FITS)
    df = df[df["status"] == "ok"].copy()
    if screen:
        df = df[df["passes_screen"]]
        if require_window:
            df = df[df["reaches_window"]]
    return df.sort_values("distance_km").reset_index(drop=True)


def load_stations() -> pd.DataFrame:
    """The autonomous stations, screened on sample count only.

    One of the three, Lake2, returns coherently to 0.45 of the ice thickness
    and so falls just short of the common fitting window that the transect
    sites define.  Excluding it on that basis would drop a year-long record
    for a few percent of column depth, so stations are exempted from the
    window screen; their depth-averaged strain rate is still computed over
    whatever part of the window they do reach, and the shortfall is reported
    by ``reaches_window``.
    """
    df = pd.read_csv(FITS)
    df = df[(df["status"] == "ok") & df["passes_screen"]]
    if "site_type" not in df.columns:
        return df.iloc[:0]
    return df[df["site_type"] == "continuous"].sort_values("distance_km")


def split_by_type(df: pd.DataFrame):
    """Separate the attended transect sites from the autonomous stations.

    Only LTG of the three autonomous stations lies on the survey line; Lake1
    and Lake2 sit 23 and 18 km off it.  Their along-flow distance is a
    projection onto an axis they never occupied, so they belong on the map
    figures, where their true position is drawn, and not on a section.
    """
    if "site_type" not in df.columns:
        return df, df.iloc[:0]
    point = df[df["site_type"] != "continuous"]
    return point, load_stations()


def on_line(stations: pd.DataFrame) -> pd.DataFrame:
    """The autonomous stations that sit on the survey line.

    The along-flow map is about the strain measured along the transect, so it
    draws only these; Lake1 and Lake2 are kept on the overview, where their
    position relative to the line is shown.
    """
    return stations[stations["offline_km"] <= ONLINE_KM]


def detail_region_km(buffer_km: float = 4.0) -> list[float]:
    """Map extent of the detail figure, in EPSG:3031 kilometres.

    The extent covers the transect sites and the stations on the line only
    (``on_line``); the off-line lake stations lie outside it.

    Defined here rather than inside the detail figure so the overview can draw
    the same box: a locator that disagreed with the panel it points at would
    be worse than no locator.
    """
    df = load_summary()
    point, stations = split_by_type(df)
    stations = on_line(stations)
    x = np.concatenate([point["x_ps_m"].values, stations["x_ps_m"].values]) / 1000.0
    y = np.concatenate([point["y_ps_m"].values, stations["y_ps_m"].values]) / 1000.0
    return [
        float(x.min() - buffer_km), float(x.max() + buffer_km),
        float(y.min() - buffer_km), float(y.max() + buffer_km),
    ]


def fitting_window(df: pd.DataFrame) -> tuple[float, float]:
    return float(df["window_lo"].iloc[0]), float(df["window_hi"].iloc[0])


def load_profiles():
    import xarray as xr

    return xr.load_dataset(paths.VV_PROFILES)


def describe(df: pd.DataFrame) -> str:
    """One-line provenance string for logging."""
    return (
        f"{len(df)} sites, {df['distance_km'].min():.1f}-{df['distance_km'].max():.1f} km, "
        f"strain rate {df['strain_rate_yr'].min():.2e} to {df['strain_rate_yr'].max():.2e} /yr"
    )
