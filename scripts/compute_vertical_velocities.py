"""Vertical velocity profiles from the Thwaites repeat ApRES survey.

For every site occupied in both the 2022-2023 and 2023-2024 field seasons this
script cross-correlates the two range profiles segment by segment, converts the
resulting englacial displacements to vertical velocities, and decomposes each
profile on the shifted-Legendre basis to separate plug flow from vertical
shear.

Processing chain
----------------
1.  Load the two cached range profiles for the site.
2.  Set the deepest usable range from the signal-to-noise ratio of *both*
    acquisitions, and pick the bed against a BedMachine thickness prior.
3.  Coarse (amplitude) alignment to get integer range-bin offsets vs depth.
4.  Fine (phase) alignment to refine those offsets to a fraction of a
    wavelength, with the coherence of each segment retained for screening.
5.  Convert displacement to vertical velocity, w = dh / dt, referenced to the
    shallowest accepted segment so that the profile describes deformation
    rather than absolute motion.
6.  Fit the shifted-Legendre decomposition and the nested F-test.

Outputs
-------
results/vertical_velocity/profiles.nc      w(z) and its uncertainty per site
results/vertical_velocity/site_summary.csv one row per site

Usage
-----
    python scripts/compute_vertical_velocities.py [--sites G1-21-05 ...]
                                                  [--jobs N] [--degree 4]
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paths  # noqa: E402
from apres import ProcessingConfig  # noqa: E402
from apres.bedpick import noise_floor_db, pick_bed  # noqa: E402
from apres.legendre import fit_profile  # noqa: E402
from apres.serialization import load_preprocessed_npz  # noqa: E402
import apres.strain as strain  # noqa: E402

DAYS_PER_YEAR = 365.25

# Depth range used for the decomposition.  The upper limit excludes firn
# compaction, which is not englacial ice deformation; the lower limit is set
# per site from the radar signal-to-noise ratio.
FIRN_DEPTH_M = 100.0

# Coherence below which a fine-alignment segment is discarded.
MIN_COHERENCE = 0.80


def usable_depth(profile, *, margin_db: float = 10.0, smooth_m: float = 10.0,
                 edge_exclude_m: float = 60.0) -> float:
    """Deepest range at which the return is ``margin_db`` above the noise floor.

    This, not the bed pick, is what limits layer tracking: below it the phase of
    the return is noise and the cross-correlation reports a spurious offset.
    """
    r = np.asarray(profile.range_m, dtype=float)
    power_db = 20.0 * np.log10(np.abs(profile.spec_cor) + 1e-30)
    floor = noise_floor_db(r, power_db)
    n = max(int(smooth_m / float(np.nanmedian(np.diff(r)))), 1)
    n += n % 2 == 0
    smooth = np.convolve(power_db, np.ones(n) / n, mode="same")
    above = np.where((smooth > floor + margin_db) & (r < r.max() - edge_exclude_m))[0]
    return float(r[above[-1]]) if above.size else float("nan")


def make_config(max_depth_m: float) -> ProcessingConfig:
    """Correlation settings for thick, cold, grounded ice.

    The defaults in :mod:`apres.config` were tuned for ice-shelf sites with a
    bright bed a few hundred metres down.  Over 2 km of Thwaites ice the
    per-segment correlation is lower, so the acceptance thresholds are relaxed
    and screening is done afterwards on coherence, which is the quantity that
    actually controls the phase error.
    """
    return ProcessingConfig(
        max_range_m=2500.0,
        min_depth_m=FIRN_DEPTH_M,
        # Coarse amplitude alignment
        min_ampcor=0.70,
        min_ampcor_prom=0.02,
        coarse_chunk_width_m=10.0,
        coarse_step_m=2.0,
        # Fine phase alignment
        fine_chunk_width_m=6.0,
        fine_step_m=2.0,
        min_cohere_coarse=0.70,
        min_cohere_fine=0.70,
        # Grounded ice: no basal melt estimate
        do_melt_estimate=False,
        bed_search_max_m=max_depth_m,
    ).validate()


def process_pair(row: dict, degree: int) -> dict:
    """Full chain for one repeat site.  Returns a summary dict plus arrays."""
    site = row["site"]
    out: dict = {"site": site, "status": "ok", "error": ""}
    try:
        p1 = load_preprocessed_npz(paths.ROOT / row["file_1"])
        p2 = load_preprocessed_npz(paths.ROOT / row["file_2"])

        dt_days = (p2.timestamp - p1.timestamp).total_seconds() / 86400.0
        if dt_days <= 0:
            raise ValueError(f"non-positive dt: {dt_days}")
        dt_years = dt_days / DAYS_PER_YEAR

        # --- depth limits -------------------------------------------------
        z_max = float(np.nanmin([usable_depth(p1), usable_depth(p2)]))
        if not np.isfinite(z_max) or z_max < 400.0:
            raise ValueError(f"no usable depth range (z_max={z_max})")

        prior = float(row["bm_thickness_m"])
        bed1 = pick_bed(p1.range_m, p1.spec_cor, prior_m=prior)
        bed2 = pick_bed(p2.range_m, p2.spec_cor, prior_m=prior)

        # The alignment routines use ``bed_depth_m`` / ``noise_depth_m`` as the
        # window limits, so they are set to the SNR-limited depth rather than
        # to the bed itself.
        for p in (p1, p2):
            p.bed_depth_m = z_max
            p.noise_depth_m = z_max

        cfg = make_config(z_max)
        coarse = strain._coarse_alignment(p1, p2, cfg)
        fine = strain._fine_alignment(p1, p2, cfg, coarse)

        # --- vertical velocity -------------------------------------------
        z = np.asarray(fine.range_m, dtype=float)
        coherence = np.abs(np.asarray(fine.coherence))
        good = (
            np.isfinite(fine.dh_m)
            & (coherence >= MIN_COHERENCE)
            & (z >= FIRN_DEPTH_M)
            & (z <= z_max)
        )
        if good.sum() < 20:
            raise ValueError(f"only {int(good.sum())} coherent segments")

        # Reference displacement to the shallowest accepted segment so the
        # profile measures deformation of the column, not bulk motion of the
        # instrument or the surface.
        ref = np.argmax(good)
        dh = np.asarray(fine.dh_m, dtype=float) - float(fine.dh_m[ref])
        w = dh / dt_years
        sigma_w = np.asarray(fine.dhe_m, dtype=float) / dt_years

        # --- decomposition -------------------------------------------------
        thickness = bed1.depth_m if bed1.ok else prior
        fit = fit_profile(
            z[good], w[good], sigma_w=sigma_w[good],
            thickness_m=thickness, degree=degree,
        )

        out.update(
            dt_days=dt_days,
            z_max_m=z_max,
            n_segments=int(good.sum()),
            n_segments_total=int(z.size),
            mean_coherence=float(np.nanmean(coherence[good])),
            bed_depth_m=bed1.depth_m,
            bed_snr_db=bed1.snr_db,
            bed_ok=int(bed1.ok),
            bed_repeat_diff_m=float(bed2.depth_m - bed1.depth_m),
            thickness_used_m=thickness,
            w_shallow_m_yr=float(np.nanmedian(w[good][:10])),
            w_deep_m_yr=float(np.nanmedian(w[good][-10:])),
            w_range_m_yr=float(np.nanmax(w[good]) - np.nanmin(w[good])),
            sigma_w_median_m_yr=float(np.nanmedian(sigma_w[good])),
        )
        if fit is not None:
            out.update(
                strain_rate_yr=fit.strain_rate_linear,
                strain_rate_err_yr=fit.strain_rate_linear_err,
                shear_fraction=fit.shear_fraction,
                shear_to_plug=fit.shear_to_plug,
                f_statistic=fit.f_statistic,
                p_value=fit.p_value,
                significant_shear=int(fit.significant),
                residual_rms_m_yr=fit.residual_rms,
                **{f"c{i}": float(c) for i, c in enumerate(fit.coefficients)},
            )
        else:
            out["status"] = "no-fit"

        out["_arrays"] = {
            "z": z[good],
            "w": w[good],
            "sigma_w": sigma_w[good],
            "coherence": coherence[good],
        }
    except Exception as exc:  # noqa: BLE001 - one bad site must not stop the survey
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["_traceback"] = traceback.format_exc()
    return out


def write_netcdf(records: list[dict], pairs: pd.DataFrame, path: Path) -> None:
    """Store the profiles on a common padded depth axis."""
    import xarray as xr

    usable = [r for r in records if "_arrays" in r]
    if not usable:
        raise SystemExit("no profiles to write")
    n_max = max(len(r["_arrays"]["z"]) for r in usable)
    sites = [r["site"] for r in usable]

    def stack(key: str) -> np.ndarray:
        out = np.full((len(usable), n_max), np.nan)
        for i, r in enumerate(usable):
            a = r["_arrays"][key]
            out[i, : len(a)] = a
        return out

    meta = pairs.set_index("site")
    ds = xr.Dataset(
        {
            "depth": (("site", "sample"), stack("z")),
            "w": (("site", "sample"), stack("w")),
            "sigma_w": (("site", "sample"), stack("sigma_w")),
            "coherence": (("site", "sample"), stack("coherence")),
            "distance_km": ("site", meta.loc[sites, "distance_km"].values),
            "lon": ("site", meta.loc[sites, "lon"].values),
            "lat": ("site", meta.loc[sites, "lat"].values),
            "x_ps_m": ("site", meta.loc[sites, "x_ps_m"].values),
            "y_ps_m": ("site", meta.loc[sites, "y_ps_m"].values),
            "surface_m": ("site", meta.loc[sites, "bm_surface_m"].values),
            "thickness_m": ("site", [r["thickness_used_m"] for r in usable]),
            "speed_m_yr": ("site", meta.loc[sites, "speed_m_yr"].values),
        },
        coords={"site": sites, "sample": np.arange(n_max)},
    )
    ds["depth"].attrs = {"units": "m", "long_name": "depth below surface"}
    ds["w"].attrs = {
        "units": "m yr-1",
        "long_name": "vertical velocity relative to the shallowest tracked segment",
    }
    ds["sigma_w"].attrs = {"units": "m yr-1", "long_name": "1-sigma uncertainty on w"}
    ds.attrs = {
        "title": "Thwaites Glacier repeat ApRES vertical velocity profiles",
        "seasons": f"{paths.SEASON_1} to {paths.SEASON_2}",
        "min_coherence": MIN_COHERENCE,
        "firn_depth_m": FIRN_DEPTH_M,
        "source": "scripts/compute_vertical_velocities.py",
    }
    ds.to_netcdf(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", nargs="*", default=None, help="subset of site names")
    ap.add_argument("--jobs", type=int, default=6, help="parallel worker processes")
    ap.add_argument("--degree", type=int, default=4, help="highest Legendre degree")
    args = ap.parse_args()

    paths.ensure_dirs()
    if not paths.PAIR_CATALOG.exists():
        raise SystemExit("run scripts/build_site_catalog.py first")

    pairs = pd.read_csv(paths.PAIR_CATALOG)
    if args.sites:
        pairs = pairs[pairs["site"].isin(args.sites)]
    pairs = pairs[pairs["file_1"].astype(bool) & pairs["file_2"].astype(bool)]
    print(f"processing {len(pairs)} repeat sites with {args.jobs} workers")

    records: list[dict] = []
    rows = pairs.to_dict("records")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_pair, r, args.degree): r["site"] for r in rows}
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                records.append(rec)
                flag = {"ok": " ", "no-fit": "?", "failed": "!"}.get(rec["status"], "!")
                extra = (
                    f"n={rec.get('n_segments', 0):4d} "
                    f"w_deep={rec.get('w_deep_m_yr', float('nan')):7.2f} "
                    f"shear={rec.get('shear_fraction', float('nan')):5.3f}"
                    if rec["status"] != "failed"
                    else rec["error"][:70]
                )
                print(f"[{i:3d}/{len(rows)}] {flag} {rec['site']:12s} {extra}", flush=True)

    order = {s: i for i, s in enumerate(pairs["site"])}
    records.sort(key=lambda r: order.get(r["site"], 1e9))

    summary = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in records])
    summary = summary.merge(
        pairs[["site", "distance_km", "lon", "lat", "x_ps_m", "y_ps_m",
               "bm_thickness_m", "bm_surface_m", "bm_bed_m", "speed_m_yr",
               "site_type", "offline_km"]],
        on="site", how="left",
    ).sort_values("distance_km")
    summary.to_csv(paths.VV_SUMMARY, index=False)
    print(f"\nwrote {paths.VV_SUMMARY}")

    write_netcdf(records, pairs, paths.VV_PROFILES)
    print(f"wrote {paths.VV_PROFILES}")

    ok = summary[summary["status"] == "ok"]
    print(f"\n  sites processed:      {len(summary)}")
    print(f"  succeeded:            {len(ok)}")
    print(f"  failed:               {int((summary['status'] == 'failed').sum())}")
    if len(ok):
        print(f"  segments per profile: {ok['n_segments'].min():.0f}-{ok['n_segments'].max():.0f}")
        print(f"  significant shear:    {int(ok['significant_shear'].sum())} of {len(ok)}")
        print(f"  shear fraction:       {ok['shear_fraction'].min():.3f}-{ok['shear_fraction'].max():.3f}")
        print(f"  strain rate (1/yr):   {ok['strain_rate_yr'].min():.2e} to {ok['strain_rate_yr'].max():.2e}")


if __name__ == "__main__":
    main()
