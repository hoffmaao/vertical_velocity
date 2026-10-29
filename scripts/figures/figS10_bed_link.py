"""Figure S10 - how the englacial deformation relates to the bed beneath it.

For each fitted station on the survey line (the 73 transect stations and LTG)
this tests three relations the paper's discussion leans on:

  (a)  depth-averaged vertical strain rate against the slope of the bed along
       flow (BedMachine v4.1 along the station path, smoothed over 2 km);
  (b)  shear share S against the basal shear stress of the velocity-only (SSA)
       inversion, which never saw the ApRES data, so the relation is not built
       into the inversion that assimilates them;
  (c)  shear share S by bed class from the vibroseismic acoustic impedance
       (Zeising et al., 2026), the mean of the 200 m groups within 0.5 km.

Significance
------------
Stations 0.5 km apart are not independent samples: the bed, the drag and the
deformation all vary smoothly along the line.  Each Spearman correlation is
therefore tested against the same statistic with one series shifted
circularly along the line by 5 to n - 5 stations, which keeps the spatial
autocorrelation of both series; p is the fraction of shifted correlations at
least as strong, counting the observed one, so with 74 stations the smallest
possible p is 1/66.  The partial correlation of S with the basal shear stress
removes the ranks' linear dependence on bed elevation, which also varies with
the drag.

Writes figures/strain_pipeline/figS10_bed_link.{png,pdf} and
results/vertical_velocity/bed_link.json with every number the paper quotes.

Usage
-----
    python scripts/figures/figS10_bed_link.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

import data  # noqa: E402
import paths  # noqa: E402
import seismic  # noqa: E402
import style  # noqa: E402

PATH_STEP_KM = 0.1          # sampling of the bed along the station path
BED_SMOOTH_KM = 2.0         # Gaussian sigma for the bed before differencing, about H
IMPEDANCE_RADIUS_KM = 0.5   # vibroseismic groups averaged around each station
MIN_SHIFT = 5               # smallest circular shift, in stations
SECONDS_PER_YEAR = 3.15576e7
CLASS_NAMES = ("soft", "intermediate", "hard")


def stations() -> pd.DataFrame:
    """The fitted stations on the survey line, in order along it."""
    fit = data.load_summary()
    point, cont = data.split_by_type(fit)
    return pd.concat([point, data.on_line(cont)]).sort_values("distance_km").reset_index(drop=True)


def bed_impedance(df: pd.DataFrame) -> np.ndarray:
    """Mean vibroseismic impedance (10^6 kg m^-2 s^-1) within IMPEDANCE_RADIUS_KM, else NaN."""
    att = seismic.bed_attributes(pd.read_csv(paths.PAIR_CATALOG))
    s_att, zb_att = att["distance_km"].values, att["Zb"].values
    out = np.full(len(df), np.nan)
    for i, s in enumerate(df["distance_km"].values):
        near = np.abs(s_att - s) <= IMPEDANCE_RADIUS_KM
        if near.any():
            out[i] = zb_att[near].mean()
    return out


def bed_along_line(df: pd.DataFrame):
    """BedMachine bed elevation (m) and its slope along flow (dimensionless) at each station.

    The bed is sampled every PATH_STEP_KM along the path through the stations and
    smoothed with a Gaussian of BED_SMOOTH_KM before differencing.  Distance grows
    inland and the ice flows towards smaller distances, so the slope along flow is
    minus the slope with distance: positive where the bed rises in the direction of
    flow (a stoss slope)."""
    import xarray as xr

    s, x, y = (df[k].values for k in ("distance_km", "x_ps_m", "y_ps_m"))
    sd = np.arange(s.min(), s.max() + 1e-9, PATH_STEP_KM)
    ds = xr.open_dataset(paths.require(paths.BEDMACHINE, "BedMachine Antarctica"))
    bed = ds["bed"].interp(x=xr.DataArray(np.interp(sd, s, x)),
                           y=xr.DataArray(np.interp(sd, s, y))).values.astype(float)
    bed = gaussian_filter1d(bed, BED_SMOOTH_KM / PATH_STEP_KM)
    slope = -np.gradient(bed, sd * 1000.0)
    return np.interp(s, sd, bed), np.interp(s, sd, slope)


def ssa_basal_shear_stress(df: pd.DataFrame) -> np.ndarray:
    """Basal shear stress (kPa) of the velocity-only SSA inversion at each station.

    The basal frictional power per unit area from dissipation_partition.py,
    divided by the sliding speed, which for SSA is the depth-averaged speed."""
    ex = np.load(paths.require(paths.MODEL_EXTRACT, "SSA model extract"))
    dz = np.load(paths.require(paths.MODEL_DISSIPATION, "SSA dissipation partition"))
    cells = ex["cells"]
    centre = ex["xy"][cells].mean(axis=1)
    tree = cKDTree(centre)
    dist, k = tree.query(dz["xy"])
    if dist.max() > 1.0:
        raise ValueError("dissipation cells do not match the extract's mesh")
    power = np.empty(len(cells))
    power[k] = dz["bed"] / dz["area"]                                  # W m^-2
    speed = ex["speed_mod"][cells].mean(axis=1) / SECONDS_PER_YEAR      # m s^-1
    tau = power / np.maximum(speed, 1e-12)
    _, at = tree.query(np.column_stack([df["x_ps_m"].values, df["y_ps_m"].values]))
    return tau[at] / 1000.0


def shifted_spearman(a, b):
    """Spearman rho of a and b, and its circular-shift p (see the module docstring)."""
    def rho(u, v):
        ok = np.isfinite(u) & np.isfinite(v)
        return spearmanr(u[ok], v[ok]).correlation, int(ok.sum())

    r0, n_ok = rho(a, b)
    shifts = range(MIN_SHIFT, len(a) - MIN_SHIFT + 1)
    null = np.array([rho(a, np.roll(b, k))[0] for k in shifts])
    p = (1 + np.sum(np.abs(null) >= abs(r0))) / (1 + len(null))
    return float(r0), float(p), n_ok


def partial_spearman(a, b, control):
    """Spearman correlation of a and b with the ranks' linear dependence on control removed."""
    def resid(u):
        ru, rc = rankdata(u), rankdata(control)
        design = np.column_stack([np.ones_like(rc), rc])
        return ru - design @ np.linalg.lstsq(design, ru, rcond=None)[0]

    return float(np.corrcoef(resid(a), resid(b))[0, 1])


def bed_class(zb):
    """0 soft, 1 intermediate, 2 hard (Zeising et al., 2026); NaN without impedance."""
    out = np.where(zb < seismic.ZB_SOFT_MAX, 0.0, np.where(zb > seismic.ZB_HARD_MIN, 2.0, 1.0))
    return np.where(np.isfinite(zb), out, np.nan)


def shifted_median_difference(values, labels, hi=2.0, lo=0.0):
    """Median of values on label hi minus on label lo, with its circular-shift p."""
    def diff(lab):
        return np.median(values[lab == hi]) - np.median(values[lab == lo])

    d0 = diff(labels)
    null = np.array([diff(np.roll(labels, k))
                     for k in range(MIN_SHIFT, len(values) - MIN_SHIFT + 1)])
    p = (1 + np.sum(np.abs(null) >= abs(d0))) / (1 + len(null))
    return float(d0), float(p)


def main() -> None:
    style.use_paper_style()
    df = stations()
    S = df["shear_share"].values
    eps = df["strain_rate_depth_mean_yr"].values
    resolved = df["shear_resolved"].astype(bool).values
    zb = bed_impedance(df)
    cls = bed_class(zb)
    bed, slope = bed_along_line(df)
    tau = ssa_basal_shear_stress(df)

    out = {"n_stations": len(df), "n_with_impedance": int(np.isfinite(zb).sum()),
           "p_floor": 1.0 / (len(df) - 2 * MIN_SHIFT + 2)}
    for key, a, b in (("eps_vs_slope", eps, slope),
                      ("S_vs_tau_ssa", S, tau),
                      ("S_vs_bed_elevation", S, bed),
                      ("S_vs_impedance", S, zb),
                      ("abs_eps_vs_impedance", np.abs(eps), zb)):
        r, p, m = shifted_spearman(a, b)
        out[key] = dict(rho=r, p=p, n=m)
    out["S_vs_tau_ssa"]["partial_rho_given_bed"] = partial_spearman(S, tau, bed)
    out["tau_ssa_kPa"] = dict(median=float(np.median(tau)), min=float(tau.min()),
                              max=float(tau.max()))
    out["by_class"] = {}
    for c, name in enumerate(CLASS_NAMES):
        sel = cls == c
        out["by_class"][name] = dict(n=int(sel.sum()), median_S=float(np.median(S[sel])),
                                     median_abs_eps_yr=float(np.median(np.abs(eps[sel]))),
                                     resolved=int(resolved[sel].sum()))
    d, p = shifted_median_difference(np.abs(eps), cls)
    out["abs_eps_hard_minus_soft"] = dict(median_difference_yr=d, p=p)
    d, p = shifted_median_difference(S, cls)
    out["S_hard_minus_soft"] = dict(median_difference=d, p=p)

    k = style.STRAIN_SCALE
    fig, axs = plt.subplots(1, 3, figsize=(style.TWO_COL, 0.36 * style.TWO_COL),
                            gridspec_kw={"wspace": 0.36, "left": 0.08, "right": 0.99,
                                         "top": 0.95, "bottom": 0.2})
    for ax in axs:
        style.grid(ax)
    mark = dict(ls="none", marker="D", ms=3.2, color=style.INK)

    def stat(ax, text):
        ax.text(0.96, 0.95, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=style.FS_TICK)

    def p_text(p):
        floor = out["p_floor"]
        return rf"$p \leq {floor:.3f}$" if p <= floor + 1e-12 else rf"$p = {p:.2f}$"

    # (a) vertical strain rate against bed slope along flow
    ax = axs[0]
    ax.axhline(0, color=style.INK_MUTED, lw=0.6, zorder=1)
    ax.axvline(0, color=style.INK_MUTED, lw=0.6, zorder=1)
    ax.plot(100 * slope, k * eps, zorder=3, **mark)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.3 * (hi - lo))      # headroom for the letter and the statistics
    ax.set_xlabel("Bed slope along flow (%)")
    ax.set_ylabel(rf"Depth-averaged {style.EPS_ZZ} ({style.STRAIN_UNIT})")
    r = out["eps_vs_slope"]
    stat(ax, rf"$\rho = {r['rho']:+.2f}$" "\n" + p_text(r["p"]))

    # (b) shear share against the velocity-only basal shear stress
    ax = axs[1]
    ax.plot(tau[resolved], S[resolved], zorder=3, **mark)
    ax.plot(tau[~resolved], S[~resolved], zorder=3, **dict(mark, mfc="white"))
    ax.set_xscale("log")
    ax.set_xlabel("Basal shear stress, SSA inversion (kPa)")
    ax.set_ylabel(r"Shear share $S$")
    r = out["S_vs_tau_ssa"]
    stat(ax, rf"$\rho = {r['rho']:+.2f}$" "\n" + p_text(r["p"]) + "\n"
         + rf"partial $\rho = {r['partial_rho_given_bed']:+.2f}$")

    # (c) shear share by vibroseismic bed class
    ax = axs[2]
    rng = np.random.default_rng(0)       # fixed jitter so the figure is reproducible
    for c in range(3):
        sel = cls == c
        xj = c + rng.uniform(-0.18, 0.18, sel.sum())
        ax.plot(xj[resolved[sel]], S[sel][resolved[sel]], zorder=3, **mark)
        ax.plot(xj[~resolved[sel]], S[sel][~resolved[sel]], zorder=3, **dict(mark, mfc="white"))
        ax.plot([c - 0.3, c + 0.3], [np.median(S[sel])] * 2, color=style.CATEGORICAL[0],
                lw=1.6, zorder=4)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xlabel("Bed class, vibroseismic impedance")
    ax.set_ylabel(r"Shear share $S$")
    r = out["S_vs_impedance"]
    stat(ax, rf"$\rho = {r['rho']:+.2f}$ with impedance" "\n" + p_text(r["p"]))

    # S is at most 1; the band above it holds the statistics clear of the data.
    for ax in axs[1:]:
        ax.set_ylim(-0.03, 1.2)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    for ax, letter in zip(axs, "abc"):
        style.panel_letter(ax, letter, x=0.1, y=0.91)

    style.save(fig, paths.FIGURES / "figS10_bed_link.png")
    target = paths.VERTICAL_VELOCITY / "bed_link.json"
    target.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
