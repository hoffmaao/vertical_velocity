"""Basal acoustic impedance along the survey line from the vibroseismic section.

The 2022/23 vibroseismic profile (section 20230551; Zeising et al., 2026) was
driven along the same flowline as the repeat ApRES survey, so its basal
attributes can be placed on the figures' distance axis.  The attributes come
per 200 m CMP group: the basal reflection coefficient R1 and the acoustic
impedance of the bed material Zb derived from it, each with an uncertainty.

Zeising et al. classify the bed by Zb: soft (dilatant till) below 2.72, hard
(consolidated and lithified sediments) above 3.24, and intermediate between,
in 10^6 kg m^-2 s^-1.  Those bounds are kept here so a figure can shade the
same classes.

The CMP groups are projected onto the ApRES line with the same principal-axis
projector the site catalogue uses (build_site_catalog.flow_axis), so distance
means the same thing for both data sets; groups farther off the line than
MAX_OFFLINE_KM are dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paths  # noqa: E402
from build_site_catalog import flow_axis  # noqa: E402

ZB_SCALE = 1e-6                      # kg m^-2 s^-1 -> 10^6 kg m^-2 s^-1
ZB_SOFT_MAX = 2.72                   # 10^6 kg m^-2 s^-1, Zeising et al. (2026)
ZB_HARD_MIN = 3.24
MAX_OFFLINE_KM = 3.0

COLUMNS = {
    "lon": "longitude(°)",
    "lat": "latitude(°)",
    "x": "x(m)(EPSG:3031)",
    "y": "y(m)(EPSG:3031)",
    "R1": "R1",
    "R1_err": "R1 error",
    "Zb": "Zb (km m^(-2) s^(-1))",        # header typo in the file: values are kg m^-2 s^-1
    "Zb_err": "Zb error (km m^(-2) s^(-1))",
}


def bed_attributes(pairs: pd.DataFrame, max_offline_km: float = MAX_OFFLINE_KM) -> pd.DataFrame:
    """Basal R1 and Zb (10^6 kg m^-2 s^-1) on the ApRES distance axis.

    ``pairs`` is the repeat-pair catalogue whose attended sites define the
    line.  Returns one row per CMP group within ``max_offline_km`` of it, with
    ``distance_km`` and ``offline_km``, sorted along the line.
    """
    path = paths.require(paths.VIBROSEIS_ATTRIBUTES, "vibroseismic bed attributes")
    raw = pd.read_csv(path)
    missing = [c for c in COLUMNS.values() if c not in raw.columns]
    if missing:
        raise KeyError(f"{path.name} lacks columns {missing}")
    project = flow_axis(pairs["x_ps_m"].values, pairs["y_ps_m"].values)
    s, off = project(raw[COLUMNS["x"]].values, raw[COLUMNS["y"]].values)
    out = pd.DataFrame({
        "distance_km": s,
        "offline_km": off,
        "R1": raw[COLUMNS["R1"]].values,
        "R1_err": raw[COLUMNS["R1_err"]].values,
        "Zb": ZB_SCALE * raw[COLUMNS["Zb"]].values,
        "Zb_err": ZB_SCALE * raw[COLUMNS["Zb_err"]].values,
    })
    out = out[out["offline_km"] <= max_offline_km]
    return out.sort_values("distance_km").reset_index(drop=True)
