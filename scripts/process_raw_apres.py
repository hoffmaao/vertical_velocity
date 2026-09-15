r"""Process raw ApRES .dat file pairs to derive vertical strain rates.

For each GHOST site with paired 2022 and 2023 raw .dat files:
  1. Preprocess both profiles (range conversion, phase calibration)
  2. Coarse alignment (amplitude cross-correlation)
  3. Fine alignment (complex cross-correlation, phase unwrapping)
  4. Menke-weighted linear strain fit
  5. Compare with MATLAB .mat result
  6. Save per-site diagnostic figure

Uses the scripts/apres processing pipeline.

Usage:
    python process_raw_apres.py
"""
import sys
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add scripts directory to path so we can import the apres package
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from apres.config import ProcessingConfig
from apres.preprocess import preprocess_file
from apres.strain import strain_melt_between_profiles

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "raw_strain_profiles"


def find_paired_sites():
    """Find sites with both 2022 and 2023 raw .dat files."""
    files_2022 = {}
    for f in glob.glob(str(DATA_DIR / "2022-2023" / "G*" / "*.dat")):
        site = os.path.basename(os.path.dirname(f))
        files_2022[site] = f

    files_2023 = {}
    for f in glob.glob(str(DATA_DIR / "Repeats" / "G*" / "*.dat")):
        site = os.path.basename(os.path.dirname(f))
        # Take the first .dat file if multiple exist
        if site not in files_2023:
            files_2023[site] = f

    paired = {}
    for site in sorted(set(files_2022.keys()) & set(files_2023.keys())):
        paired[site] = (files_2022[site], files_2023[site])

    return paired


def load_matlab_vsr(site):
    """Load MATLAB strain rate for comparison."""
    mat_path = DATA_DIR / "Strain" / f"{site}_2023_2024_strain.mat"
    if not mat_path.exists():
        return None
    try:
        import mat73
        dat = mat73.loadmat(str(mat_path))
        vdat = dat["vdat_strain"]
        if "fit_ice" in vdat and isinstance(vdat["fit_ice"], dict):
            return float(vdat["fit_ice"]["vsr"])
    except Exception:
        pass
    return None


def main():
    print("=" * 60)
    print("Processing raw ApRES .dat pairs -> strain rates")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    paired = find_paired_sites()
    print(f"Found {len(paired)} paired sites\n")

    # Load known thicknesses from .mat files for bed detection
    import mat73
    thickness_map = {}
    for mat_path in glob.glob(str(DATA_DIR / "Strain" / "*.mat")):
        try:
            dat = mat73.loadmat(str(mat_path))
            name = dat["vdat_strain"]["Name"]
            thickness_map[name] = float(dat["vdat_strain"]["H"])
        except Exception:
            pass
    print(f"Loaded thickness for {len(thickness_map)} sites from .mat files\n")

    # Base config for Thwaites grounded ice
    base_cfg = ProcessingConfig(
        max_range_m=2500.0,      # Thwaites ice is up to ~2500m thick
        firn_depth_m=100.0,      # Fit below firn
        min_depth_m=20.0,        # Start processing from 20m
        do_melt_estimate=False,  # Grounded ice, no basal melt
    )

    results = []
    for site, (f1, f2) in paired.items():
        print(f"  {site}:")
        print(f"    file1: {os.path.basename(f1)}")
        print(f"    file2: {os.path.basename(f2)}")

        try:
            # Per-site config with known thickness
            # Enable melt estimate to get the bed reflector phase/range shift
            H_known = thickness_map.get(site)
            if H_known:
                cfg = ProcessingConfig(
                    station=site,
                    max_range_m=2500.0,
                    firn_depth_m=100.0,
                    min_depth_m=20.0,
                    do_melt_estimate=True,   # needed for bed shift
                    ice_thickness_method="use",
                    ice_thickness_use_m=H_known,
                    bed_search_min_m=max(H_known - 300, 500),
                    bed_search_max_m=H_known + 200,
                )
            else:
                cfg = ProcessingConfig(
                    station=site,
                    max_range_m=2500.0,
                    firn_depth_m=100.0,
                    min_depth_m=20.0,
                    do_melt_estimate=True,
                )

            # Preprocess both profiles independently
            p1 = preprocess_file(Path(f1), cfg)
            p2 = preprocess_file(Path(f2), cfg)

            # Compute strain rate
            result, coarse, fine, fit, bed = strain_melt_between_profiles(
                p1, p2, cfg
            )

            matlab_vsr = load_matlab_vsr(site)
            matlab_str = f"{matlab_vsr:.5f}" if matlab_vsr else "N/A"

            print(f"    VSR: {result.vsr_per_year:+.5f} ± {result.vsr_err_per_year:.5f} 1/yr"
                  f"  (MATLAB: {matlab_str})")
            print(f"    dt: {result.dt_days:.1f} days, "
                  f"n_fit: {result.n_fit_points}, "
                  f"R²: {fit.r2:.4f}")
            print(f"    Bed shift: {bed.dh_m:+.4f} ± {bed.dhe_m:.4f} m, "
                  f"phase: {bed.phase_rad:.3f} rad, "
                  f"|coherence|: {abs(bed.coherence):.3f}")
            print(f"    Melt: {result.melt_rate_m_per_year:.3f} ± "
                  f"{result.melt_rate_err_m_per_year:.3f} m/yr")

            results.append({
                "site": site,
                "vsr": result.vsr_per_year,
                "vsr_err": result.vsr_err_per_year,
                "matlab_vsr": matlab_vsr,
                "dt_days": result.dt_days,
                "n_fit": result.n_fit_points,
                "r2": fit.r2,
                "bed_shift_m": bed.dh_m,
                "bed_shift_err_m": bed.dhe_m,
                "bed_phase_rad": bed.phase_rad,
                "bed_coherence": abs(bed.coherence),
                "melt_rate": result.melt_rate_m_per_year,
                "melt_rate_err": result.melt_rate_err_m_per_year,
            })

            # ── Per-site figure: 5 panels ──
            dt_yr = result.dt_days / 365.25
            fig, axes = plt.subplots(1, 5, figsize=(25, 6))

            # Panel 1: Vertical velocity w(z) = dh/dt
            ax = axes[0]
            w_fine = fine.dh_m / dt_yr
            ax.plot(w_fine, fine.range_m, "k.", ms=1, alpha=0.3, label="data")
            # Menke linear velocity fit
            depth_line = np.linspace(fine.range_m.min(), fine.range_m.max(), 200)
            dh_line = fit.m0 + fit.m1 * (depth_line - fit.mid_range_m)
            ax.plot(dh_line / dt_yr, depth_line, "r-", lw=2, label="Menke fit")
            ax.axhspan(100, min(1100, (H_known or 2500) * 0.98),
                       alpha=0.08, color="blue", label="fit range")
            ax.invert_yaxis()
            ax.set_xlabel("w (m/yr)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Vertical velocity")
            ax.legend(fontsize=7)

            # Panel 2: Displacement (raw fine alignment)
            ax = axes[1]
            ax.plot(fine.dh_m, fine.range_m, "k.", ms=1, alpha=0.3)
            dh_fit_line = fit.m0 + fit.m1 * (depth_line - fit.mid_range_m)
            ax.plot(dh_fit_line, depth_line, "r-", lw=2)
            ax.invert_yaxis()
            ax.set_xlabel("Displacement (m)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Fine alignment displacement")

            # Panel 3: Coherence
            ax = axes[2]
            coh = np.abs(fine.coherence)
            ax.plot(coh, fine.range_m, "b.", ms=1, alpha=0.3)
            ax.axvline(cfg.min_cohere_fine, color="r", ls="--", lw=1,
                       label=f"threshold={cfg.min_cohere_fine}")
            ax.invert_yaxis()
            ax.set_xlabel("Coherence")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Coherence")
            ax.legend(fontsize=7)

            # Panel 4: Vertical strain rate
            ax = axes[3]
            eps_fd = np.gradient(w_fine, fine.range_m)
            ax.plot(eps_fd, fine.range_m, "k.", ms=1, alpha=0.3, label="dw/dz")
            ax.axvline(result.vsr_per_year, color="r", lw=2,
                       label=f"ours: {result.vsr_per_year:.4f}")
            if matlab_vsr:
                ax.axvline(matlab_vsr, color="g", ls="--", lw=1.5,
                           label=f"MATLAB: {matlab_vsr:.4f}")
            ax.axhspan(100, min(1100, (H_known or 2500) * 0.98),
                       alpha=0.08, color="blue")
            ax.invert_yaxis()
            ax.set_xlabel("ε_zz (1/yr)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Vertical strain rate")
            ax.legend(fontsize=7)

            # Panel 5: Bed reflector and deformation with depth
            ax = axes[4]
            # Plot the full displacement profile with the Menke fit extrapolated to bed
            ax.plot(w_fine, fine.range_m, "k.", ms=1, alpha=0.3, label="w(z)")
            # Menke fit extrapolated to full depth
            depth_full = np.linspace(0, H_known or fine.range_m.max(), 300)
            dh_full = fit.m0 + fit.m1 * (depth_full - fit.mid_range_m)
            ax.plot(dh_full / dt_yr, depth_full, "r-", lw=1.5, alpha=0.7,
                    label="Menke extrapolation")
            # Mark bed position and bed shift
            bed_depth = H_known or p1.bed_depth_m
            ax.axhline(bed_depth, color="brown", ls="-", lw=1.5, alpha=0.7,
                       label=f"bed ({bed_depth:.0f}m)")
            # Predicted displacement at bed from strain fit
            dh_pred_bed = fit.m0 + fit.m1 * (bed_depth - fit.mid_range_m)
            w_pred_bed = dh_pred_bed / dt_yr
            # Actual bed shift
            w_bed_obs = bed.dh_m / dt_yr
            ax.plot(w_pred_bed, bed_depth, "rv", ms=10, zorder=5,
                    label=f"strain pred: {w_pred_bed:.2f} m/yr")
            ax.plot(w_bed_obs, bed_depth, "g^", ms=10, zorder=5,
                    label=f"bed obs: {w_bed_obs:.2f} m/yr")
            ax.invert_yaxis()
            ax.set_xlabel("w (m/yr)")
            ax.set_ylabel("Depth (m)")
            ax.set_title(f"Bed reflector\nmelt={result.melt_rate_m_per_year:.2f} m/yr")
            ax.legend(fontsize=6, loc="lower left")

            H_str = f"H={H_known:.0f}m" if H_known else "H=unknown"
            fig.suptitle(f"{site}  ({H_str}, dt={dt_yr:.2f}yr, "
                         f"n={result.n_fit_points}, R²={fit.r2:.3f})",
                         fontsize=12)
            fig.tight_layout()
            fig.savefig(str(FIG_DIR / f"{site}.png"), dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "site": site, "vsr": np.nan, "vsr_err": np.nan,
                "matlab_vsr": load_matlab_vsr(site),
                "dt_days": np.nan, "n_fit": 0, "r2": np.nan,
            })

    # Summary
    print(f"\n{'='*60}")
    print(f"Processed {len(results)} sites")
    n_ok = sum(1 for r in results if np.isfinite(r["vsr"]))
    print(f"Successful: {n_ok}, Failed: {len(results) - n_ok}")

    if n_ok > 0:
        vsr_ours = np.array([r["vsr"] for r in results if np.isfinite(r["vsr"]) and r["matlab_vsr"]])
        vsr_matlab = np.array([r["matlab_vsr"] for r in results if np.isfinite(r["vsr"]) and r["matlab_vsr"]])
        if len(vsr_ours) > 0:
            diff = vsr_ours - vsr_matlab
            print(f"\nRaw vs MATLAB comparison ({len(diff)} sites):")
            print(f"  Mean diff: {diff.mean():.6f} 1/yr")
            print(f"  RMS diff:  {np.sqrt((diff**2).mean()):.6f} 1/yr")

    print(f"\nFigures: {FIG_DIR}/")


if __name__ == "__main__":
    main()
