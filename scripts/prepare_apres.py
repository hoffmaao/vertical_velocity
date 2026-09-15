r"""Process ApRES vertical velocity profiles into Legendre polynomial fits.

For each GHOST 2022-2023 ApRES site on Thwaites:
  1. Load raw .mat file (range, dh, dhe)
  2. Convert depth to terrain-following coordinate: zeta = range / H
  3. Compute vertical velocity: w = dh / dt (m/yr) and uncertainty
  4. Fit orthonormal shifted Legendre polynomials P~_k(zeta) on [0,1]
     using weighted linear least squares (weighted by 1/sigma)
  5. Compute full covariance matrix for each fit order
  6. Run incremental F-tests to determine statistically significant order
  7. Convert site coordinates from WGS84 to EPSG:3031

The Legendre basis P~_k(zeta) = sqrt(2k+1) * P_k(2*zeta - 1) is the same
basis used by icepack's HybridModel GL vertical discretization, enabling
direct comparison of model and observed polynomial coefficients.

Output: HDF5 file with per-site coefficients, covariances, F-test results,
and raw profiles for direct comparison.
"""
import numpy as np
import pandas as pd
import mat73
import h5py
import glob
from numpy.polynomial.legendre import legval
from scipy.stats import f as f_dist
from pyproj import Transformer
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STRAIN_DIR = DATA_DIR / "Strain"
WAYPOINTS_FILE = Path(
    "/media/andrew/wd1/projects/thwaitesfirn/data/strainrates/"
    "Waypoints_GHOST2022-2023.xlsx"
)
OUTPUT_FILE = DATA_DIR / "apres_legendre_fits.h5"

# Maximum polynomial order to fit (quartic = 4 -> 5 coefficients)
MAX_ORDER = 4

# Depth range for fitting (m below surface)
# Below firn (~100m) and within reliable radar range (~1100m)
FIT_DEPTH_MIN = 100.0   # m, exclude firn compaction
FIT_DEPTH_MAX = 1100.0  # m, stay within reliable radar range

# Minimum uncertainty floor (m/yr) to avoid zero-weight issues
SIGMA_FLOOR_FRACTION = 0.01  # 1% of the data range as floor


# ═══════════════════════════════════════════════════════════════════
# Legendre basis
# ═══════════════════════════════════════════════════════════════════

def shifted_legendre(n, zeta):
    """Evaluate orthonormal shifted Legendre polynomial P~_n on [0,1].

    P~_n(zeta) = sqrt(2n+1) * P_n(2*zeta - 1)

    where P_n is the standard Legendre polynomial. This basis satisfies
    <P~_n, P~_m> = integral_0^1 P~_n P~_m dzeta = delta_{nm}.
    """
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return np.sqrt(2 * n + 1) * legval(2 * zeta - 1, coeffs)


def build_design_matrix(zeta, n_modes):
    """Build the N_data x n_modes design matrix Phi[i,k] = P~_k(zeta_i)."""
    return np.column_stack([shifted_legendre(k, zeta) for k in range(n_modes)])


# ═══════════════════════════════════════════════════════════════════
# Weighted linear least squares with proper covariance
# ═══════════════════════════════════════════════════════════════════

def weighted_legendre_fit(zeta, w, sigma, n_modes):
    """Fit n_modes Legendre coefficients to w(zeta) using weighted LSQ.

    Parameters
    ----------
    zeta : array (N,)
        Terrain-following coordinate in [0, 1].
    w : array (N,)
        Observed vertical velocity (m/yr).
    sigma : array (N,)
        Observation uncertainty (m/yr), must be > 0.
    n_modes : int
        Number of Legendre modes to fit (2=linear, 3=quadratic, etc.).

    Returns
    -------
    coeffs : array (n_modes,)
        Fitted Legendre coefficients.
    cov : array (n_modes, n_modes)
        Covariance matrix of coefficients (scaled by chi2/dof if chi2/dof > 1).
    cov_formal : array (n_modes, n_modes)
        Formal covariance (not scaled, from inverse Fisher information).
    chi2 : float
        Weighted sum of squared residuals.
    dof : int
        Degrees of freedom (N - n_modes).
    rss : float
        Unweighted residual sum of squares.
    """
    N = len(zeta)
    dof = N - n_modes

    # Design matrix
    Phi = build_design_matrix(zeta, n_modes)

    # Weight matrix (diagonal, 1/sigma)
    W_inv = 1.0 / sigma
    Phi_w = Phi * W_inv[:, None]  # W^{1/2} @ Phi
    w_weighted = w * W_inv        # W^{1/2} @ w

    # Solve normal equations: (Phi^T W Phi) x = Phi^T W w
    # Equivalently: lstsq(Phi_w, w_weighted)
    gram = Phi_w.T @ Phi_w
    coeffs = np.linalg.solve(gram, Phi_w.T @ w_weighted)

    # Residuals
    residuals = w - Phi @ coeffs
    chi2 = np.sum((residuals / sigma) ** 2)
    rss = np.sum(residuals ** 2)

    # Formal covariance (from inverse Fisher information)
    cov_formal = np.linalg.inv(gram)

    # Scaled covariance: if chi2/dof > 1, uncertainties are underestimated
    # Scale the covariance to account for model inadequacy / noise mismatch
    scale = max(chi2 / dof, 1.0) if dof > 0 else 1.0
    cov = cov_formal * scale

    return coeffs, cov, cov_formal, chi2, dof, rss


def incremental_f_test(rss_simple, rss_complex, dof_simple, dof_complex, n_data):
    """F-test for whether adding modes significantly improves the fit.

    Tests H0: the simpler model is adequate.

    Parameters
    ----------
    rss_simple : float
        Weighted RSS of the simpler (fewer mode) model.
    rss_complex : float
        Weighted RSS of the more complex model.
    dof_simple, dof_complex : int
        Degrees of freedom for each model.
    n_data : int
        Number of data points.

    Returns
    -------
    f_stat : float
        F-statistic.
    p_value : float
        p-value (probability of observing this F under H0).
        Small p-value -> reject H0 -> complex model is justified.
    """
    delta_dof = dof_simple - dof_complex
    if delta_dof <= 0 or dof_complex <= 0 or rss_complex <= 0:
        return np.nan, np.nan

    f_stat = ((rss_simple - rss_complex) / delta_dof) / (rss_complex / dof_complex)
    p_value = 1 - f_dist.cdf(f_stat, delta_dof, dof_complex)
    return f_stat, p_value


# ═══════════════════════════════════════════════════════════════════
# Process a single site
# ═══════════════════════════════════════════════════════════════════

def process_site(mat_path, thickness_override=None):
    """Process a single ApRES .mat file into Legendre fits.

    Parameters
    ----------
    mat_path : str or Path
        Path to the .mat file.
    thickness_override : float, optional
        If provided, use this thickness instead of the .mat file's H value.
        This should come from the mesh (BedMachine) for consistent zeta.

    Returns
    -------
    result : dict
        Site name, raw data, and fits for each polynomial order.
    """
    dat = mat73.loadmat(str(mat_path))
    vdat = dat["vdat_strain"]

    name = vdat["Name"]
    H_mat = float(vdat["H"])
    H = thickness_override if thickness_override is not None else H_mat
    dt_days = float(vdat["dt"])
    dt_yr = dt_days / 365.25

    depth = vdat["range"]       # depth from surface (m)
    dh = vdat["dh"]             # vertical displacement (m)
    dhe = vdat["dhe"]           # displacement uncertainty (m)

    # Convert to velocity (m/yr)
    w = dh / dt_yr
    w_sigma = dhe / dt_yr

    # Compute vertical strain rate: ε_zz = dw/dz (1/yr)
    # Central differences for interior points, one-sided at boundaries
    dz = np.gradient(depth)
    eps_zz = np.gradient(w, depth)
    # Propagate uncertainty: σ_ε ≈ sqrt(2) * σ_w / Δz (central diff)
    dz_avg = np.abs(dz)
    dz_avg[dz_avg < 0.5] = 0.5  # floor to avoid division by zero
    eps_sigma = np.sqrt(2) * w_sigma / dz_avg

    # Filter to fit depth range: below firn, within reliable radar range
    depth_max = min(FIT_DEPTH_MAX, H * 0.98)  # don't exceed 98% of thickness
    valid = (
        (depth >= FIT_DEPTH_MIN)
        & (depth <= depth_max)
        & np.isfinite(eps_zz)
        & np.isfinite(eps_sigma)
    )
    depth_fit = depth[valid]
    eps_zz = eps_zz[valid]
    eps_sigma = eps_sigma[valid]

    # Also keep the velocity data for reference
    w_fit = w[valid]
    w_sigma_fit = w_sigma[valid]

    # Normalize depth to terrain-following coordinate using ApRES thickness
    # ζ = depth / H_apres (0 at surface, 1 at bed)
    zeta = depth_fit / H

    if len(zeta) < MAX_ORDER + 2:
        return None

    # Apply uncertainty floor: fraction of the strain rate data range
    data_range = np.ptp(eps_zz)
    sigma_floor = max(SIGMA_FLOOR_FRACTION * data_range, 1e-6)
    eps_sigma = np.maximum(eps_sigma, sigma_floor)

    # Fit progressively higher polynomial orders to vertical strain rate
    fits = {}
    for order in range(1, MAX_ORDER + 1):
        n_modes = order + 1  # linear=2, quadratic=3, cubic=4, quartic=5
        coeffs, cov, cov_formal, chi2, dof, rss = weighted_legendre_fit(
            zeta, eps_zz, eps_sigma, n_modes
        )
        fits[order] = {
            "n_modes": n_modes,
            "coeffs": coeffs,
            "cov": cov,
            "cov_formal": cov_formal,
            "sigma": np.sqrt(np.diag(cov)),
            "chi2": chi2,
            "dof": dof,
            "rss": rss,
        }

    # F-tests: does adding each higher mode significantly improve the fit?
    f_tests = {}
    for order in range(2, MAX_ORDER + 1):
        f_stat, p_value = incremental_f_test(
            fits[order - 1]["chi2"],
            fits[order]["chi2"],
            fits[order - 1]["dof"],
            fits[order]["dof"],
            len(zeta),
        )
        f_tests[order] = {"f_stat": f_stat, "p_value": p_value}

    # Determine best order (highest order where p < 0.05)
    best_order = 1
    for order in range(2, MAX_ORDER + 1):
        if f_tests[order]["p_value"] < 0.05:
            best_order = order

    return {
        "name": name,
        "H_mat": H_mat,
        "H_used": H,
        "dt_days": dt_days,
        "n_points": len(zeta),
        "fit_depth_min": FIT_DEPTH_MIN,
        "fit_depth_max": float(depth_max),
        "depth": depth_fit,
        "zeta": zeta,
        "eps_zz": eps_zz,          # vertical strain rate (1/yr)
        "eps_sigma": eps_sigma,     # strain rate uncertainty (1/yr)
        "w": w_fit,                 # vertical velocity (m/yr), for reference
        "w_sigma": w_sigma_fit,
        "fits": fits,
        "f_tests": f_tests,
        "best_order": best_order,
    }


# ═══════════════════════════════════════════════════════════════════
# Load waypoints and convert coordinates
# ═══════════════════════════════════════════════════════════════════

def load_waypoints():
    """Load GHOST waypoints and convert WGS84 -> EPSG:3031."""
    wp = pd.read_excel(str(WAYPOINTS_FILE), "waypoint-gps")
    wp = wp.rename(columns={"xcoord": "lon", "ycoord": "lat", "zcoord": "elev_m"})

    # Convert WGS84 (lat, lon) -> EPSG:3031 (Antarctic Polar Stereographic)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    valid_mask = wp["lon"].notna() & wp["lat"].notna()
    x_stereo, y_stereo = transformer.transform(
        wp.loc[valid_mask, "lon"].values,
        wp.loc[valid_mask, "lat"].values,
    )
    wp["x"] = np.nan
    wp["y"] = np.nan
    wp.loc[valid_mask, "x"] = x_stereo
    wp.loc[valid_mask, "y"] = y_stereo

    # Interpolate missing coordinates from neighbors within same transect
    for idx in wp.index[wp["x"].isna()]:
        name = str(wp.loc[idx, "Waypoint"])
        parts = name.split("-")
        transect = parts[0]
        neighbors = wp[
            wp["Waypoint"].str.startswith(transect + "-") & wp["x"].notna()
        ]
        if len(neighbors) >= 2:
            # Linear interpolation by waypoint number
            num = int(parts[1])
            nums = neighbors["Waypoint"].apply(lambda s: int(s.split("-")[1]))
            sort_idx = nums.argsort()
            x_interp = np.interp(num, nums.iloc[sort_idx], neighbors["x"].iloc[sort_idx])
            y_interp = np.interp(num, nums.iloc[sort_idx], neighbors["y"].iloc[sort_idx])
            wp.loc[idx, "x"] = x_interp
            wp.loc[idx, "y"] = y_interp

    return wp


# ═══════════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════════

def save_results(results, waypoints, output_path, thickness_map=None):
    """Save all results to HDF5."""
    if thickness_map is None:
        thickness_map = {}
    with h5py.File(str(output_path), "w") as f:
        # Site summary table
        names = [r["name"] for r in results]
        wp_lookup = waypoints.set_index("Waypoint")

        xs = np.array([wp_lookup.loc[n, "x"] if n in wp_lookup.index else np.nan
                        for n in names])
        ys = np.array([wp_lookup.loc[n, "y"] if n in wp_lookup.index else np.nan
                        for n in names])

        summary = f.create_group("summary")
        summary.create_dataset("names", data=[n.encode() for n in names])
        summary.create_dataset("x", data=xs)
        summary.create_dataset("y", data=ys)
        summary.create_dataset("H_apres", data=[r["H_mat"] for r in results])
        # Store mesh (BedMachine) thickness for reference
        H_mesh_vals = np.array([
            thickness_map.get(n, np.nan) for n in names
        ])
        summary.create_dataset("H_mesh", data=H_mesh_vals)
        summary.create_dataset("n_points", data=[r["n_points"] for r in results])
        summary.create_dataset("best_order", data=[r["best_order"] for r in results])

        # Per-site groups
        for r in results:
            g = f.create_group(f"sites/{r['name']}")

            # Raw profile data (for direct comparison)
            g.create_dataset("depth", data=r["depth"])
            g.create_dataset("zeta", data=r["zeta"])
            g.create_dataset("eps_zz", data=r["eps_zz"])
            g.create_dataset("eps_sigma", data=r["eps_sigma"])
            g.create_dataset("w", data=r["w"])
            g.create_dataset("w_sigma", data=r["w_sigma"])
            g.attrs["fit_depth_min"] = r["fit_depth_min"]
            g.attrs["fit_depth_max"] = r["fit_depth_max"]

            # Fits for each order
            for order, fit in r["fits"].items():
                fg = g.create_group(f"order_{order}")
                fg.create_dataset("coeffs", data=fit["coeffs"])
                fg.create_dataset("cov", data=fit["cov"])
                fg.create_dataset("cov_formal", data=fit["cov_formal"])
                fg.create_dataset("sigma", data=fit["sigma"])
                fg.attrs["chi2"] = fit["chi2"]
                fg.attrs["dof"] = fit["dof"]
                fg.attrs["rss"] = fit["rss"]
                fg.attrs["n_modes"] = fit["n_modes"]

            # F-test results
            for order, ft in r["f_tests"].items():
                g.attrs[f"f_stat_{order}"] = ft["f_stat"]
                g.attrs[f"f_prob_{order}"] = ft["p_value"]

            g.attrs["best_order"] = r["best_order"]
            g.attrs["H_apres"] = r["H_mat"]
            g.attrs["H_used"] = r["H_used"]
            if r["name"] in thickness_map:
                g.attrs["H_mesh"] = thickness_map[r["name"]]

    print(f"Saved {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def get_mesh_thickness_at_sites(waypoints):
    """Query the mesh's BedMachine thickness at each ApRES site location.

    Returns a dict mapping waypoint name -> thickness (m), or None if
    the mesh file doesn't exist or the site is outside the mesh.
    """
    import firedrake

    mesh_path = Path(__file__).resolve().parent.parent / "mesh" / "thwaites.h5"
    if not mesh_path.exists():
        print("   WARNING: mesh/thwaites.h5 not found, using .mat thicknesses")
        return {}

    with firedrake.CheckpointFile(str(mesh_path), "r") as chk:
        mesh = chk.load_mesh()
        H = chk.load_function(mesh, "thickness")

    thickness_map = {}
    for _, row in waypoints.iterrows():
        name = str(row["Waypoint"])
        x, y = row.get("x"), row.get("y")
        if pd.isna(x) or pd.isna(y):
            continue
        try:
            h_val = float(H.at((x, y), tolerance=1e-3))
            if h_val > 0:
                thickness_map[name] = h_val
        except Exception:
            pass  # site outside mesh domain

    return thickness_map


def main():
    print("=" * 60)
    print("Processing ApRES vertical velocity profiles")
    print("=" * 60)

    # Load waypoints
    print("\n1. Loading waypoints...")
    waypoints = load_waypoints()
    n_valid = waypoints["x"].notna().sum()
    print(f"   {len(waypoints)} waypoints, {n_valid} with valid coordinates")

    # Query mesh thickness at sites for reference (not used for zeta conversion)
    print("\n2. Loading mesh thickness at ApRES sites (for reference)...")
    thickness_map = get_mesh_thickness_at_sites(waypoints)
    print(f"   Mesh thickness available for {len(thickness_map)} sites")

    # Find .mat files
    mat_files = sorted(glob.glob(str(STRAIN_DIR / "*.mat")))
    print(f"\n3. Found {len(mat_files)} .mat files")

    # Process each site using ApRES-measured thickness for zeta conversion.
    # The ApRES H is the radar-measured thickness at the site, which defines
    # the coordinate system the observations live in. BedMachine thickness
    # on the coarse mesh may differ due to (a) mesh smoothing of bed
    # topography and (b) genuine BedMachine vs radar disagreement.
    print(f"\n4. Fitting Legendre polynomials (orders 1-{MAX_ORDER})...")
    results = []
    for mat_path in mat_files:
        result = process_site(mat_path, thickness_override=None)
        if result is None:
            print(f"   SKIP {Path(mat_path).stem}: too few data points")
            continue
        results.append(result)

        # Print summary
        best = result["best_order"]
        fit = result["fits"][best]
        coeffs_str = ", ".join([f"{c:.3f}+/-{s:.3f}"
                                for c, s in zip(fit["coeffs"], fit["sigma"])])
        f_probs = [result["f_tests"].get(o, {}).get("p_value", np.nan)
                   for o in range(2, MAX_ORDER + 1)]
        f_str = ", ".join([f"{p:.1e}" if np.isfinite(p) else "---" for p in f_probs])
        print(f"   {result['name']:>10s}: best_order={best}, "
              f"chi2/dof={fit['chi2']/max(fit['dof'],1):.1f}, "
              f"coeffs=[{coeffs_str}], f_probs=[{f_str}]")

    print(f"\n5. Processed {len(results)} sites")

    # Summary statistics
    orders = [r["best_order"] for r in results]
    for o in range(1, MAX_ORDER + 1):
        n = sum(1 for x in orders if x == o)
        print(f"   Order {o}: {n} sites")

    # Report thickness comparison (ApRES vs mesh)
    n_compared = sum(1 for r in results if r["name"] in thickness_map)
    if n_compared:
        diffs = [thickness_map[r["name"]] - r["H_mat"] for r in results
                 if r["name"] in thickness_map]
        print(f"   ApRES vs mesh thickness ({n_compared} sites): "
              f"dH=[{min(diffs):.0f}, {max(diffs):.0f}] m, "
              f"mean={np.mean(diffs):.0f} m")

    # Save
    print(f"\n6. Saving results...")
    save_results(results, waypoints, OUTPUT_FILE, thickness_map)

    # Quick diagnostic plot
    print(f"\n7. Plotting diagnostics...")
    plot_diagnostics(results, waypoints)

    print("\nDone!")


def plot_diagnostics(results, waypoints):
    """Plot example fits and site map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot example profiles (first 6 sites)
    for i, ax in enumerate(axes.flat):
        if i >= len(results):
            break
        r = results[i]
        zeta_fine = np.linspace(0.01, 0.99, 200)

        # Raw strain rate data
        ax.errorbar(r["eps_zz"], r["zeta"], xerr=r["eps_sigma"],
                     fmt="k.", ms=1, elinewidth=0.3, alpha=0.3, label="data")

        # Fits
        colors = ["coral", "mediumpurple", "dodgerblue", "forestgreen"]
        labels = ["linear", "quadratic", "cubic", "quartic"]
        for order in range(1, MAX_ORDER + 1):
            fit = r["fits"][order]
            Phi = build_design_matrix(zeta_fine, fit["n_modes"])
            eps_fit = Phi @ fit["coeffs"]
            lw = 2 if order == r["best_order"] else 0.8
            ax.plot(eps_fit, zeta_fine, color=colors[order - 1],
                    lw=lw, label=labels[order - 1])

        ax.invert_yaxis()
        ax.set_xlabel("ε_zz (1/yr)")
        ax.set_ylabel("ζ (normalized depth)")
        ax.set_title(f"{r['name']} (best: {labels[r['best_order']-1]})")
        if i == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig_path = str(Path(__file__).resolve().parent.parent / "figures" / "apres_fits.png")
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"   Saved {fig_path}")


if __name__ == "__main__":
    main()
