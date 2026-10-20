"""Build the master catalog of Thwaites ApRES sites.

Joins, for every attended ApRES site in both field seasons:

  * the raw and preprocessed file paths and acquisition timestamps,
  * GPS position (Garmin waypoint export for 2022-2023, GPX for 2023-2024),
  * BedMachine Antarctica v4.1 surface, bed, thickness and mask,
  * MEaSUREs 2019-2020 surface velocity and its reported uncertainty,
  * along-flow distance, measured from the most upstream repeat site.

Two kinds of site are catalogued.  The attended point sites were occupied once
in each field season, and pair across the two.  The three autonomous stations -
Lake1, Lake2 and LTG - logged continuously through the year between, and are
paired here from the first and last acquisition of their own record, which
spans the same interval.  They are flagged ``site_type='continuous'`` because
only LTG lies on the survey line: Lake1 and Lake2 sit 23 and 18 km off it, so
they belong on the map figures but not on an along-flow section.

Outputs
-------
results/sites/site_catalog.csv   one row per (season, site)
results/sites/repeat_pairs.csv   one row per repeat pair, both kinds

Usage
-----
    python scripts/build_site_catalog.py
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths  # noqa: E402

WAYPOINT_XLSX = paths.DATA / paths.SEASON_1 / "Waypoints_GHOST2022-2023.xlsx"
GARMIN_KML = paths.DATA / paths.SEASON_1 / "ApRES_Locations_22-23_fromGarmin.kml"
GPX_DIR = paths.DATA / paths.SEASON_2 / "GPX"

# ApRES timestamps are encoded in the raw filename; the two seasons used
# slightly different conventions.
_TS_PATTERNS = (
    re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})[_-](?P<time>\d{6})"),
    re.compile(r"DATA(?P<date>\d{4}-\d{2}-\d{2})-(?P<time>\d{4})"),
)


# --------------------------------------------------------------------- readers
def read_waypoint_table() -> pd.DataFrame:
    """Season-1 site positions from the field waypoint spreadsheet."""
    df = pd.read_excel(WAYPOINT_XLSX, sheet_name="waypoint-gps")
    df = df.rename(
        columns={"Waypoint": "site", "xcoord": "lon", "ycoord": "lat", "zcoord": "elev_gps_m"}
    )
    df = df[["site", "lon", "lat", "elev_gps_m"]].dropna(subset=["site"])
    df["site"] = df["site"].astype(str).str.strip()
    # A handful of positions were interpolated in the field notebook and carry
    # an elevation but no coordinate; fill them by interpolating along the line.
    return df.drop_duplicates(subset="site", keep="first")


def read_kml_waypoints(path: Path) -> pd.DataFrame:
    """Garmin KML export: <name> plus lon,lat,elev triples."""
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(path)
    rows = []
    for pm in tree.iterfind(".//k:Placemark", ns):
        name = pm.findtext("k:name", default="", namespaces=ns).strip()
        coord = pm.findtext(".//k:coordinates", default="", namespaces=ns).strip()
        if not name or not coord:
            continue
        parts = coord.split(",")
        if len(parts) < 2:
            continue
        lon, lat = float(parts[0]), float(parts[1])
        elev = float(parts[2]) if len(parts) > 2 else np.nan
        rows.append({"gps_id": name, "lon": lon, "lat": lat, "elev_gps_m": elev})
    return pd.DataFrame(rows)


def read_gpx_waypoints(directory: Path) -> pd.DataFrame:
    """Season-2 site positions from the Garmin GPX exports."""
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    rows = []
    for path in sorted(directory.glob("*.gpx")):
        tree = ET.parse(path)
        for wpt in tree.iterfind(".//g:wpt", ns):
            name = wpt.findtext("g:name", default="", namespaces=ns).strip()
            if not name:
                continue
            rows.append(
                {
                    "site": name,
                    "lat": float(wpt.attrib["lat"]),
                    "lon": float(wpt.attrib["lon"]),
                    "elev_gps_m": float(wpt.findtext("g:ele", default="nan", namespaces=ns)),
                    "gpx_file": path.name,
                }
            )
    return pd.DataFrame(rows)


def parse_timestamp(name: str) -> pd.Timestamp:
    for pat in _TS_PATTERNS:
        m = pat.search(name)
        if m:
            time = m.group("time")
            if len(time) == 4:
                time = time + "00"
            return pd.Timestamp(f"{m.group('date')} {time[:2]}:{time[2:4]}:{time[4:6]}", tz="UTC")
    return pd.NaT


def scan_season(season: str) -> pd.DataFrame:
    """Every attended point-mode acquisition in one season."""
    raw_root = paths.DATA / season / "Point"
    pre_root = paths.PREPROCESSED / season / "Point"
    rows = []
    if not raw_root.exists():
        return pd.DataFrame(rows)
    for site_dir in sorted(raw_root.iterdir()):
        if not site_dir.is_dir():
            continue
        for raw in sorted(list(site_dir.glob("*.dat")) + list(site_dir.glob("*.DAT"))):
            pre = pre_root / site_dir.name / (raw.stem + ".npz")
            rows.append(
                {
                    "season": season,
                    "site": site_dir.name,
                    "raw_file": str(raw.relative_to(paths.ROOT)),
                    "preprocessed_file": (
                        str(pre.relative_to(paths.ROOT)) if pre.exists() else ""
                    ),
                    "timestamp": parse_timestamp(raw.name),
                    "polarization": _polarization(raw.stem),
                }
            )
    return pd.DataFrame(rows)


def _polarization(stem: str) -> str:
    m = re.search(r"-(HH|HV|VH|VV)$", stem, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


# ------------------------------------------------------------------- sampling
def sample_bedmachine(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    import xarray as xr

    ds = xr.open_dataset(paths.require(paths.BEDMACHINE, "BedMachine Antarctica"))
    pts = {"x": xr.DataArray(x, dims="site"), "y": xr.DataArray(y, dims="site")}
    out = {}
    for var, col in [
        ("surface", "bm_surface_m"),
        ("bed", "bm_bed_m"),
        ("thickness", "bm_thickness_m"),
        ("firn", "bm_firn_m"),
        ("mask", "bm_mask"),
    ]:
        out[col] = ds[var].sel(**pts, method="nearest").values.astype(float)
    ds.close()
    return pd.DataFrame(out)


def sample_velocity(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    import xarray as xr

    ds = xr.open_dataset(paths.require(paths.MEASURES_VELOCITY, "MEaSUREs velocity mosaic"))
    pts = {"x": xr.DataArray(x, dims="site"), "y": xr.DataArray(y, dims="site")}
    vx = ds["VX"].sel(**pts, method="nearest").values.astype(float)
    vy = ds["VY"].sel(**pts, method="nearest").values.astype(float)
    ex = ds["ERRX"].sel(**pts, method="nearest").values.astype(float)
    ey = ds["ERRY"].sel(**pts, method="nearest").values.astype(float)
    ds.close()
    speed = np.hypot(vx, vy)
    return pd.DataFrame(
        {
            "vx_m_yr": vx,
            "vy_m_yr": vy,
            "speed_m_yr": speed,
            "vel_err_m_yr": np.hypot(ex, ey),
            "flow_azimuth_deg": np.degrees(np.arctan2(vx, vy)),
        }
    )


CONTINUOUS_SITES = ("Lake1", "Lake2", "LTG")


def scan_continuous() -> pd.DataFrame:
    """First and last valid acquisition of each autonomous station."""
    rows = []
    for site in CONTINUOUS_SITES:
        root = paths.PREPROCESSED / "continuous" / site
        if not root.exists():
            continue
        stamped = []
        for npz in sorted(root.rglob("*.npz")):
            try:
                with np.load(npz, allow_pickle=True) as z:
                    stamp = str(z["timestamp"])
            except Exception:
                continue
            # A few bursts carry an unset clock (year 2000); they would
            # otherwise be chosen as the start of the record.
            if not stamp[:4].isdigit() or int(stamp[:4]) < 2022:
                continue
            stamped.append((pd.Timestamp(stamp), npz))
        if len(stamped) < 2:
            continue
        stamped.sort(key=lambda t: t[0])
        (t1, f1), (t2, f2) = stamped[0], stamped[-1]

        coords = paths.DATA / "Continuous" / site / f"ApRES_{site}.csv"
        lon = lat = np.nan
        if coords.exists():
            c = pd.read_csv(coords)
            lat = float(c["Latitude"].iloc[0])
            lon = float(c["Longitude"].iloc[0])
        rows.append(
            {
                "site": site,
                "site_type": "continuous",
                "file_1": str(f1.relative_to(paths.ROOT)),
                "file_2": str(f2.relative_to(paths.ROOT)),
                "raw_1": "", "raw_2": "",
                "t1": t1, "t2": t2,
                "lon": lon, "lat": lat,
                "n_acquisitions": len(stamped),
            }
        )
    return pd.DataFrame(rows)


def flow_axis(x: np.ndarray, y: np.ndarray):
    """Principal axis of the attended survey line.

    Returned as a projector so that off-line stations can be placed on the same
    distance coordinate without being allowed to rotate the axis itself.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    pts = np.column_stack([x[ok], y[ok]])
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis, perp = vt[0], vt[1]
    origin = float(((pts - centre) @ axis).min())

    def project(px, py):
        d = np.column_stack([np.asarray(px, float), np.asarray(py, float)]) - centre
        return (d @ axis - origin) / 1000.0, np.abs(d @ perp) / 1000.0

    return project


def along_flow_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance along the survey line, measured from its upstream end."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full(x.shape, np.nan)
    project = flow_axis(x, y)
    s = np.full(x.shape, np.nan)
    s[ok], _ = project(x[ok], y[ok])
    return s


# ----------------------------------------------------------------------- main
def main() -> None:
    paths.ensure_dirs()
    from pyproj import Transformer

    frames = [scan_season(paths.SEASON_1), scan_season(paths.SEASON_2)]
    cat = pd.concat(frames, ignore_index=True)
    if cat.empty:
        raise SystemExit("No point-mode acquisitions found under data/<season>/Point.")

    # --- positions ---------------------------------------------------------
    wp1 = read_waypoint_table()
    wp2 = read_gpx_waypoints(GPX_DIR) if GPX_DIR.exists() else pd.DataFrame()

    pos = wp1.set_index("site")[["lon", "lat", "elev_gps_m"]]
    if not wp2.empty:
        # Season 2 reoccupied the season-1 stakes, so season-1 coordinates are
        # authoritative; GPX only fills sites that season 1 never occupied.
        extra = wp2.drop_duplicates(subset="site").set_index("site")[
            ["lon", "lat", "elev_gps_m"]
        ]
        pos = pos.combine_first(extra)

    cat = cat.join(pos, on="site")

    # Sites whose waypoint was recorded as "interp" have no coordinate; fill by
    # linear interpolation in the site ordering along each survey box.
    cat = cat.sort_values(["site"]).reset_index(drop=True)
    for col in ("lon", "lat"):
        cat[col] = cat.groupby(cat["site"].str.slice(0, 2))[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )

    # --- projection and reference data ------------------------------------
    tr = Transformer.from_crs(paths.EPSG_WGS84, paths.EPSG_POLAR, always_xy=True)
    x, y = tr.transform(cat["lon"].values, cat["lat"].values)
    cat["x_ps_m"], cat["y_ps_m"] = x, y

    cat = pd.concat(
        [cat, sample_bedmachine(x, y), sample_velocity(x, y)], axis=1
    )
    cat["distance_km"] = along_flow_distance(x, y)

    cat = cat.sort_values(["season", "distance_km", "site"]).reset_index(drop=True)
    cat.to_csv(paths.SITE_CATALOG, index=False)
    print(f"wrote {paths.SITE_CATALOG}  ({len(cat)} acquisitions)")

    # --- repeat pairs ------------------------------------------------------
    # One acquisition per (season, site): prefer HH, else the first in time.
    def pick(g: pd.DataFrame) -> pd.Series:
        hh = g[g["polarization"] == "HH"]
        return (hh if len(hh) else g).sort_values("timestamp").iloc[0]

    best = cat.groupby(["season", "site"], as_index=False).apply(pick, include_groups=False)
    s1 = best[best["season"] == paths.SEASON_1].set_index("site")
    s2 = best[best["season"] == paths.SEASON_2].set_index("site")
    common = sorted(set(s1.index) & set(s2.index))

    pairs = pd.DataFrame(
        {
            "site": common,
            "file_1": s1.loc[common, "preprocessed_file"].values,
            "file_2": s2.loc[common, "preprocessed_file"].values,
            "raw_1": s1.loc[common, "raw_file"].values,
            "raw_2": s2.loc[common, "raw_file"].values,
            "t1": s1.loc[common, "timestamp"].values,
            "t2": s2.loc[common, "timestamp"].values,
            "lon": s1.loc[common, "lon"].values,
            "lat": s1.loc[common, "lat"].values,
            "x_ps_m": s1.loc[common, "x_ps_m"].values,
            "y_ps_m": s1.loc[common, "y_ps_m"].values,
            "distance_km": s1.loc[common, "distance_km"].values,
            "bm_surface_m": s1.loc[common, "bm_surface_m"].values,
            "bm_bed_m": s1.loc[common, "bm_bed_m"].values,
            "bm_thickness_m": s1.loc[common, "bm_thickness_m"].values,
            "speed_m_yr": s1.loc[common, "speed_m_yr"].values,
            "flow_azimuth_deg": s1.loc[common, "flow_azimuth_deg"].values,
        }
    )
    pairs["site_type"] = "point"

    # --- autonomous stations ------------------------------------------------
    cont = scan_continuous()
    if not cont.empty:
        cx, cy = tr.transform(cont["lon"].values, cont["lat"].values)
        cont["x_ps_m"], cont["y_ps_m"] = cx, cy
        # Placed on the distance coordinate defined by the attended line, not
        # allowed to redefine it: two of the three sit tens of km off the line.
        project = flow_axis(pairs["x_ps_m"].values, pairs["y_ps_m"].values)
        cont["distance_km"], cont["offline_km"] = project(cx, cy)
        cont = pd.concat(
            [cont.reset_index(drop=True),
             sample_bedmachine(cx, cy).reset_index(drop=True),
             sample_velocity(cx, cy).reset_index(drop=True)],
            axis=1,
        )
        pairs = pd.concat([pairs, cont], ignore_index=True)

    pairs["offline_km"] = pairs.get("offline_km", pd.Series(0.0, index=pairs.index)).fillna(0.0)
    # Both kinds of timestamp are tz-aware UTC but arrive from different
    # readers, so they are normalised before differencing.
    pairs["dt_days"] = (
        pd.to_datetime(pairs["t2"], utc=True) - pd.to_datetime(pairs["t1"], utc=True)
    ).dt.total_seconds() / 86400.0
    pairs = pairs.sort_values("distance_km").reset_index(drop=True)
    pairs.to_csv(paths.PAIR_CATALOG, index=False)
    print(f"wrote {paths.PAIR_CATALOG}  ({len(pairs)} repeat pairs)")

    # --- provenance summary -------------------------------------------------
    print()
    print(f"  sites season 1 ({paths.SEASON_1}): {s1.shape[0]}")
    print(f"  sites season 2 ({paths.SEASON_2}): {s2.shape[0]}")
    npoint = int((pairs["site_type"] == "point").sum())
    ncont = int((pairs["site_type"] == "continuous").sum())
    print(f"  repeat pairs:                     {len(pairs)} ({npoint} point, {ncont} continuous)")
    if ncont:
        for _, r in pairs[pairs["site_type"] == "continuous"].iterrows():
            print(f"    {r['site']:6s} {r['distance_km']:5.1f} km along, "
                  f"{r['offline_km']:4.1f} km off line, {r['dt_days']:.0f} days")
    print(f"  missing coordinates:              {int(pairs['lon'].isna().sum())}")
    print(f"  transect length:                  {np.nanmax(pairs['distance_km']):.1f} km")
    print(
        f"  dt (days):                        "
        f"{np.nanmin(pairs['dt_days']):.1f} to {np.nanmax(pairs['dt_days']):.1f}"
    )
    print(
        f"  BedMachine thickness (m):         "
        f"{np.nanmin(pairs['bm_thickness_m']):.0f} to {np.nanmax(pairs['bm_thickness_m']):.0f}"
    )
    print(
        f"  surface speed (m/yr):             "
        f"{np.nanmin(pairs['speed_m_yr']):.0f} to {np.nanmax(pairs['speed_m_yr']):.0f}"
    )


if __name__ == "__main__":
    main()
