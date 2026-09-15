r"""Process each GHOST ApRES site: compute vertical strain rate profiles.

For each .mat file in data/Strain/:
  1. Extract displacement profile dh(z) and uncertainty dhe(z)
  2. Convert to velocity: w(z) = dh / dt
  3. Compute vertical strain rate: eps_zz(z) = dw/dz
  4. Fit Legendre polynomials to eps_zz over 100-1100m
  5. Save a per-site diagnostic figure

The strain rate is computed following the ApRES processing approach:
  - Use the Menke-weighted linear fit from fit_ice when available
  - Also compute the depth-varying strain rate via finite differences
  - Compare our processing against the MATLAB fit_ice.vsr result

Usage:
    python process_strain_profiles.py
"""
import numpy as np
import mat73
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STRAIN_DIR = DATA_DIR / "Strain"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "strain_profiles"

# Depth range for fitting
FIT_DEPTH_MIN = 100.0   # m
FIT_DEPTH_MAX = 1100.0  # m


def menke_fit(x, y, sigma=None):
    """Weighted least-squares linear fit (Menke 1984).

    Fits y = m0 + m1 * (x - x_mid).

    Returns m0, m1, m0_err, m1_err, x_mid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mid = np.mean(x)
    G = np.column_stack([np.ones_like(x), x - x_mid])

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.maximum(sigma, 1e-12)
        w = 1.0 / sigma ** 2
        W = np.diag(w)
        GTWG = G.T @ W @ G
        GTWy = G.T @ W @ y
    else:
        GTWG = G.T @ G
        GTWy = G.T @ y

    M = np.linalg.solve(GTWG, GTWy)

    resid = y - G @ M
    n, p = G.shape
    var = np.sum(resid ** 2) / max(n - p, 1)
    cov = var * np.linalg.inv(GTWG)
    Me = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return M[0], M[1], Me[0], Me[1], x_mid


def process_site(mat_path):
    """Process one .mat file and return strain rate profile."""
    dat = mat73.loadmat(str(mat_path))
    vdat = dat["vdat_strain"]

    name = vdat["Name"]
    H = float(vdat["H"])
    dt_days = float(vdat["dt"])
    dt_yr = dt_days / 365.25

    depth = vdat["range"]
    dh = vdat["dh"]
    dhe = vdat["dhe"]

    # Vertical velocity (m/yr)
    w = dh / dt_yr
    w_err = dhe / dt_yr

    # Vertical strain rate via finite differences: eps_zz = dw/dz (1/yr)
    eps_zz = np.gradient(w, depth)
    # Uncertainty propagation (central diff: σ_ε ≈ √2 σ_w / Δz)
    dz = np.abs(np.gradient(depth))
    dz = np.maximum(dz, 0.5)
    eps_err = np.sqrt(2) * w_err / dz

    # Filter to fit range
    depth_max = min(FIT_DEPTH_MAX, H * 0.98)
    mask = (depth >= FIT_DEPTH_MIN) & (depth <= depth_max) & np.isfinite(eps_zz)
    depth_fit = depth[mask]
    w_fit = w[mask]
    w_err_fit = w_err[mask]
    eps_fit = eps_zz[mask]
    eps_err_fit = eps_err[mask]

    # Menke fit on DISPLACEMENT (as MATLAB does) within the fit range
    dh_fit = dh[mask]
    dhe_fit = dhe[mask]
    dhe_fit_clean = np.maximum(dhe_fit, 1e-6)

    m0, m1, m0e, m1e, mid = menke_fit(depth_fit, dh_fit, sigma=dhe_fit_clean)
    vsr_menke = m1 * 365.25 / dt_days  # 1/yr

    # MATLAB result for comparison
    matlab_vsr = None
    if "fit_ice" in vdat and isinstance(vdat["fit_ice"], dict):
        fit_ice = vdat["fit_ice"]
        if "vsr" in fit_ice:
            matlab_vsr = float(fit_ice["vsr"])

    # Terrain-following coordinate
    zeta = depth_fit / H

    return {
        "name": name,
        "H": H,
        "dt_yr": dt_yr,
        "depth": depth,
        "w": w,
        "w_err": w_err,
        "eps_zz": eps_zz,
        "eps_err": eps_err,
        "depth_fit": depth_fit,
        "zeta_fit": zeta,
        "eps_fit": eps_fit,
        "eps_err_fit": eps_err_fit,
        "w_fit": w_fit,
        "vsr_menke": vsr_menke,
        "vsr_menke_err": m1e * 365.25 / dt_days,
        "menke_m0": m0,
        "menke_m1": m1,
        "menke_mid": mid,
        "matlab_vsr": matlab_vsr,
    }


def plot_site(result, fig_path):
    """Plot strain rate profile for one site."""
    r = result
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Panel 1: Vertical velocity w(z)
    ax = axes[0]
    ax.plot(r["w"], r["depth"], "k-", lw=0.5, alpha=0.5)
    ax.fill_betweenx(r["depth"],
                      r["w"] - r["w_err"], r["w"] + r["w_err"],
                      alpha=0.1, color="gray")
    # Highlight fit range
    ax.axhspan(FIT_DEPTH_MIN, min(FIT_DEPTH_MAX, r["H"] * 0.98),
               alpha=0.1, color="blue", label="fit range")
    # Menke linear fit
    depth_line = np.linspace(r["depth_fit"].min(), r["depth_fit"].max(), 100)
    dh_line = r["menke_m0"] + r["menke_m1"] * (depth_line - r["menke_mid"])
    w_line = dh_line / r["dt_yr"]
    ax.plot(w_line, depth_line, "r-", lw=2, label="Menke fit")
    ax.invert_yaxis()
    ax.set_xlabel("w (m/yr)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Vertical velocity")
    ax.legend(fontsize=8)

    # Panel 2: Strain rate eps_zz(z) in 1/yr
    ax = axes[1]
    ax.plot(r["eps_zz"], r["depth"], "k-", lw=0.5, alpha=0.5)
    ax.fill_betweenx(r["depth"],
                      r["eps_zz"] - r["eps_err"], r["eps_zz"] + r["eps_err"],
                      alpha=0.1, color="gray")
    ax.axhspan(FIT_DEPTH_MIN, min(FIT_DEPTH_MAX, r["H"] * 0.98),
               alpha=0.1, color="blue")
    # Menke constant strain rate
    ax.axvline(r["vsr_menke"], color="r", lw=2,
               label=f'Menke: {r["vsr_menke"]:.4f} 1/yr')
    if r["matlab_vsr"] is not None:
        ax.axvline(r["matlab_vsr"], color="g", ls="--", lw=1.5,
                   label=f'MATLAB: {r["matlab_vsr"]:.4f} 1/yr')
    ax.invert_yaxis()
    ax.set_xlabel("ε_zz (1/yr)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Vertical strain rate")
    ax.legend(fontsize=8)

    # Panel 3: Strain rate vs ζ = depth/H
    ax = axes[2]
    zeta_all = r["depth"] / r["H"]
    ax.plot(r["eps_zz"], zeta_all, "k-", lw=0.5, alpha=0.5)
    zeta_min = FIT_DEPTH_MIN / r["H"]
    zeta_max = min(FIT_DEPTH_MAX, r["H"] * 0.98) / r["H"]
    ax.axhspan(zeta_min, zeta_max, alpha=0.1, color="blue")
    ax.axvline(r["vsr_menke"], color="r", lw=2)
    if r["matlab_vsr"] is not None:
        ax.axvline(r["matlab_vsr"], color="g", ls="--", lw=1.5)
    ax.invert_yaxis()
    ax.set_xlabel("ε_zz (1/yr)")
    ax.set_ylabel("ζ = depth / H")
    ax.set_title("Strain rate (terrain-following)")
    ax.set_ylim(1.0, 0.0)

    fig.suptitle(f'{r["name"]}  (H={r["H"]:.0f}m, dt={r["dt_yr"]:.2f}yr)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)


def main():
    print("=" * 60)
    print("Processing GHOST ApRES strain rate profiles")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(glob.glob(str(STRAIN_DIR / "*.mat")))
    print(f"Found {len(mat_files)} .mat files\n")

    results = []
    for mat_path in mat_files:
        r = process_site(mat_path)
        results.append(r)

        fig_path = FIG_DIR / f"{r['name']}.png"
        plot_site(r, fig_path)

        matlab_str = f"{r['matlab_vsr']:.5f}" if r["matlab_vsr"] else "N/A"
        print(f"  {r['name']:>10s}: "
              f"vsr_menke={r['vsr_menke']:+.5f} ± {r['vsr_menke_err']:.5f}, "
              f"matlab={matlab_str}")

    # Summary comparison
    menke_vals = np.array([r["vsr_menke"] for r in results])
    matlab_vals = np.array([r["matlab_vsr"] if r["matlab_vsr"] else np.nan
                            for r in results])
    valid = np.isfinite(matlab_vals)
    if valid.any():
        diff = menke_vals[valid] - matlab_vals[valid]
        print(f"\nMenke vs MATLAB comparison ({valid.sum()} sites):")
        print(f"  Mean diff: {diff.mean():.6f} 1/yr")
        print(f"  Max |diff|: {np.abs(diff).max():.6f} 1/yr")
        print(f"  RMS diff:  {np.sqrt((diff**2).mean()):.6f} 1/yr")

    print(f"\nFigures saved to {FIG_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
