"""Central path and external-dataset registry for the Thwaites ApRES analysis.

Every script in this directory resolves its inputs through this module so that
a single edit re-points the whole pipeline at a different machine.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- repository
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RESULTS = ROOT / "results"
# Figures 1-5 of the GRL manuscript as it currently stands (the versions in
# paper-thwaites-nonlinear-deformation/figures).  Kept apart from figures/manuscript,
# which manuscript_figures.py writes.
FIGURES = ROOT / "figures" / "strain_pipeline"

PREPROCESSED = RESULTS / "preprocessed"
SITES = RESULTS / "sites"
VERTICAL_VELOCITY = RESULTS / "vertical_velocity"

SEASON_1 = "2022-2023"
SEASON_2 = "2023-2024"

SITE_CATALOG = SITES / "site_catalog.csv"
PAIR_CATALOG = SITES / "repeat_pairs.csv"
VV_PROFILES = VERTICAL_VELOCITY / "profiles.nc"
VV_SUMMARY = VERTICAL_VELOCITY / "site_summary.csv"

# ------------------------------------------------------- external reference data
# Overridable with environment variables so the pipeline is portable.
def _env_path(var: str, default: str | Path) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


BEDMACHINE = _env_path(
    "BEDMACHINE_ANTARCTICA",
    "~/projects/hydrology_models/data/bedmachine/"
    "NSIDC-0756_4-20260219_211506/"
    "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc",
)

MEASURES_VELOCITY = _env_path(
    "MEASURES_VELOCITY",
    "~/Downloads/Antarctica_ice_velocity_2019_2020_1km_v01.1.nc",
)

# Amundsen Sea Embayment annual velocity mosaics, 250 m, EPSG:3031, extracted
# from the tar releases on the Hoffman drive at
# projects/ase_velocities/annual/Releases/annualThrough2023NoTSX.
# The repeat ApRES survey spans Dec 2022 to Dec 2023, so the two annual
# mosaics that bracket it are averaged rather than either used alone.
ASE_VELOCITY_DIR = _env_path("ASE_VELOCITY_DIR", "~/data/ase_velocities")
ASE_VELOCITY_YEARS = (2022, 2023)

# REMA v2.0 32 m mosaic tiles (PGC).  Two tiles, 26_16 and 26_17, cover the
# transect either side of x = -1400 km.  Used for the ice surface in the
# section figure: BedMachine's surface is a 500 m product and, sampled once
# per ApRES site, gives a 91-point polyline where the section wants a
# continuous profile.
REMA_DIR = _env_path("REMA_DIR", "~/data/rema")

# MOA 125 m high-pass composite.  The hp1 product is the slope-sensitive one:
# it suppresses the broad brightness field and leaves surface topography, so
# it works as a shaded background under a colour field.
MOA = _env_path("MOA_MOSAIC", "~/Downloads/moa125_2014_hp1_v01.tif")

# Greene Antarctic coastlines, EPSG:3031 eastings/northings with NaN-separated
# polygons (doi:10.5281/zenodo.5903643).  The series ends at 2021, so it is
# kept only for continental context in the locator inset.
GREENE_COASTLINE_DIR = _env_path("GREENE_COASTLINE_DIR", "~/data/greene_coastlines")
GREENE_COASTLINE_YEAR = "2021.2"

# IceLines (DLR EOC): monthly Sentinel-1 calving-front positions per ice shelf,
# https://download.geoservice.dlr.de/icelines/files/<shelf>/monthly/fronts/.
# Thwaites1 and Thwaites2 together cover Oct 2022 to Feb 2024, which spans the
# repeat survey; this is the record that continues past Greene's 2021 endpoint.
ICELINES_DIR = _env_path("ICELINES_DIR", "~/data/icelines")

# MEaSUREs InSAR grounding lines, v2.1 (NSIDC-0498).  One GeoPackage whose
# features each carry their own acquisition dates, running to 2025, so the
# grounding line can be restricted to the survey period rather than taken from
# a single undated mask.
MEASURES_GL = _env_path(
    "MEASURES_GL", "~/data/measures_gl/InSAR_GL_Antarctica_v02.1.gpkg"
)

# Seismic attributes of the ice base from the 2022/23 along-flow vibroseismic
# section 20230551 (Zeising et al., 2026, doi:10.1038/s43247-026-03502-2): per
# 200 m CMP group, the basal reflection coefficient R1 and acoustic impedance Zb
# with their uncertainties.  PANGAEA doi:10.1594/PANGAEA.987691 (CC-BY-4.0),
# file DS4_20230551_SeismicAttributes_Bed_v02.csv.  The section follows the
# ApRES flowline, within about 2 km of the stations over the survey span.
VIBROSEIS_ATTRIBUTES = _env_path(
    "VIBROSEIS_ATTRIBUTES",
    DATA / "vibroseis" / "DS4_20230551_SeismicAttributes_Bed_v02.csv",
)

# Model mesh (nodes and triangles) exported by paper_extract.py, read for the
# model-domain outline on the overview map.  Any of the extracts will do: the
# SSA and hybrid runs share one mesh.
MODEL_EXTRACT = _env_path(
    "MODEL_EXTRACT", str(ROOT / "results" / "paper" / "extract_ssa.npz")
)

# Period the repeat ApRES survey spans, used to select dated boundary products.
SURVEY_START = "2022-10-01"
SURVEY_END = "2024-03-01"

# ------------------------------------------------------------------ constants
EPSG_POLAR = "EPSG:3031"
EPSG_WGS84 = "EPSG:4326"


def require(path: Path, what: str) -> Path:
    """Fail loudly and early when a reference dataset is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n"
            f"Set the corresponding environment variable in scripts/paths.py."
        )
    return path


def ase_velocity(year: int, component: str) -> Path:
    """Path to one ASE annual mosaic component ('vv', 'vx', 'vy', 'ex', 'ey')."""
    folder = ASE_VELOCITY_DIR / f"Vel-{year}-01-01.{year}-12-31"
    matches = sorted(folder.glob(f"*_{component}_*.tif"))
    if not matches:
        raise FileNotFoundError(
            f"no {component} mosaic for {year} under {folder}.\n"
            f"Extract it from the Hoffman drive: "
            f"projects/ase_velocities/annual/Releases/annualThrough2023NoTSX"
        )
    return matches[0]


def rema_tiles() -> list[Path]:
    """REMA mosaic tiles available locally."""
    return sorted(REMA_DIR.glob("*_dem.tif"))


def greene_coastline(year: str | None = None) -> Path:
    """Path to one Greene coastline file."""
    year = year or GREENE_COASTLINE_YEAR
    return GREENE_COASTLINE_DIR / f"antarctic_coastline_{year}.txt"


def ensure_dirs() -> None:
    for d in (SITES, VERTICAL_VELOCITY, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
