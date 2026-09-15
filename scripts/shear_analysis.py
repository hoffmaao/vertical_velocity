r"""Legendre shear analysis of ApRES vertical velocity profiles.

Fits orthonormal shifted Legendre polynomials to the vertical velocity
w(ζ) at each GHOST ApRES site and computes the degree of vertical shear
from the ratio of Legendre coefficients.

Following the icepack hybrid tutorial (06-hybrid-ice-stream-xyz):
  - a₀ = depth-averaged velocity (plug flow component)
  - a₁ = linear shear component (proportional to basal shear)
  - shear_ratio = |a₁| / |a₀| (0 = plug flow, >0 = shearing)

The F-test determines whether each higher-order coefficient is
statistically significant, identifying where the ice column has
measurable vertical shear vs pure plug flow.

Uses the same orthonormal Legendre basis as icepack's GL vertical
discretization:
  P̃_k(ζ) = sqrt(2k+1) * P_k(2ζ - 1)   on [0, 1]

Usage:
    python shear_analysis.py
"""
import numpy as np
import mat73
import h5py
import glob
import pandas as pd
from numpy.polynomial.legendre import legval
from scipy.stats import f as f_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STRAIN_DIR = DATA_DIR / "Strain"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Fit range (meters below surface)
FIT_DEPTH_MIN = 100.0
FIT_DEPTH_MAX = 1100.0

# Maximum polynomial order
MAX_ORDER = 4

# Minimum uncertainty floor
SIGMA_FLOOR_FRAC = 0.01


def shifted_legendre(n, zeta):
    """Orthonormal shifted Legendre polynomial on [0,1]."""
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return np.sqrt(2 * n + 1) * legval(2 * zeta - 1, coeffs)


def weighted_legendre_fit(zeta, y, sigma, n_modes):
    """Weighted least squares fit to Legendre basis."""
    N = len(zeta)
    dof = N - n_modes
    Phi = np.column_stack([shifted_legendre(k, zeta) for k in range(n_modes)])
    W_inv = 1.0 / sigma
    Phi_w = Phi * W_inv[:, None]
    y_w = y * W_inv
    gram = Phi_w.T @ Phi_w
    coeffs = np.linalg.solve(gram, Phi_w.T @ y_w)
    residuals = y - Phi @ coeffs
    chi2 = np.sum((residuals / sigma) ** 2)
    cov_formal = np.linalg.inv(gram)
    scale = max(chi2 / dof, 1.0) if dof > 0 else 1.0
    cov = cov_formal * scale
    return coeffs, cov, chi2, dof


def f_test(chi2_simple, chi2_complex, dof_simple, dof_complex):
    """F-test: is the complex model significantly better?"""
    delta_dof = dof_simple - dof_complex
    if delta_dof <= 0 or dof_complex <= 0 or chi2_complex <= 0:
        return np.nan, np.nan
    f_stat = ((chi2_simple - chi2_complex) / delta_dof) / (chi2_complex / dof_complex)
    p_value = 1 - f_dist.cdf(f_stat, delta_dof, dof_complex)
    return f_stat, p_value


def process_site(mat_path):
    """Process one site: fit Legendre polynomials to velocity."""
    dat = mat73.loadmat(str(mat_path))
    vdat = dat["vdat_strain"]

    name = vdat["Name"]
    H = float(vdat["H"])
    dt_yr = float(vdat["dt"]) / 365.25
    depth = vdat["range"]
    dh = vdat["dh"]
    dhe = vdat["dhe"]

    w = dh / dt_yr
    w_sigma = dhe / dt_yr

    # Filter to fit range
    depth_max = min(FIT_DEPTH_MAX, H * 0.98)
    valid = (depth >= FIT_DEPTH_MIN) & (depth <= depth_max) & np.isfinite(w) & np.isfinite(w_sigma)
    depth_fit = depth[valid]
    w_fit = w[valid]
    w_sigma_fit = w_sigma[valid]

    # Terrain-following coordinate (normalized by ApRES thickness)
    zeta = depth_fit / H

    if len(zeta) < MAX_ORDER + 2:
        return None

    # Uncertainty floor
    data_range = np.ptp(w_fit)
    sigma_floor = max(SIGMA_FLOOR_FRAC * data_range, 1e-4)
    w_sigma_fit = np.maximum(w_sigma_fit, sigma_floor)

    # Fit orders 1 through MAX_ORDER
    fits = {}
    for order in range(1, MAX_ORDER + 1):
        n_modes = order + 1
        coeffs, cov, chi2, dof = weighted_legendre_fit(zeta, w_fit, w_sigma_fit, n_modes)
        sigma = np.sqrt(np.diag(cov))
        fits[order] = {
            "coeffs": coeffs,
            "sigma": sigma,
            "cov": cov,
            "chi2": chi2,
            "dof": dof,
        }

    # F-tests
    f_tests = {}
    for order in range(2, MAX_ORDER + 1):
        f_stat, p_val = f_test(
            fits[order - 1]["chi2"], fits[order]["chi2"],
            fits[order - 1]["dof"], fits[order]["dof"],
        )
        f_tests[order] = {"f_stat": f_stat, "p_value": p_val}

    # Best order
    best_order = 1
    for order in range(2, MAX_ORDER + 1):
        if f_tests[order]["p_value"] < 0.05:
            best_order = order

    # Shear analysis using best-order fit
    best = fits[best_order]
    a0 = best["coeffs"][0]   # depth-averaged (plug flow)
    a0_err = best["sigma"][0]

    # Shear coefficients (all orders >= 1)
    a1 = best["coeffs"][1] if best_order >= 1 else 0.0
    a1_err = best["sigma"][1] if best_order >= 1 else np.nan
    a2 = best["coeffs"][2] if best_order >= 2 else 0.0
    a3 = best["coeffs"][3] if best_order >= 3 else 0.0
    a4 = best["coeffs"][4] if best_order >= 4 else 0.0

    # Shear ratio: sqrt(a1² + a2² + a3² + a4²) / |a0|
    # This captures ALL depth-varying deformation relative to the
    # depth-uniform component, not just the linear shear
    nonlinear_energy = np.sqrt(a1**2 + a2**2 + a3**2 + a4**2)
    shear_ratio = nonlinear_energy / abs(a0) if abs(a0) > 1e-10 else np.nan

    # Is shear significant? (F-test: does adding ANY higher-order term matter?)
    # Use the highest significant order's test
    shear_significant = any(
        f_tests.get(o, {}).get("p_value", 1.0) < 0.05
        for o in range(2, MAX_ORDER + 1)
    )

    # Depth-averaged vertical strain rate from Menke linear fit of displacement
    # dh = m0 + m1 * (z - mid), so dw/dz = m1 / dt_yr = vsr
    dt_yr = float(vdat["dt"]) / 365.25
    depth_max_fit = min(FIT_DEPTH_MAX, H * 0.98)
    mask_fit = (depth >= FIT_DEPTH_MIN) & (depth <= depth_max_fit) & np.isfinite(dh) & np.isfinite(dhe)
    depth_menke = depth[mask_fit]
    dh_menke = dh[mask_fit]
    dhe_menke = dhe[mask_fit]
    if len(depth_menke) > 2:
        from numpy.linalg import solve as lsolve
        mid_m = np.mean(depth_menke)
        G_m = np.column_stack([np.ones(len(depth_menke)), depth_menke - mid_m])
        sigma_m = np.maximum(dhe_menke, 1e-6)
        W_m = np.diag(1.0 / sigma_m ** 2)
        M_m = lsolve(G_m.T @ W_m @ G_m, G_m.T @ W_m @ dh_menke)
        vsr = M_m[1] * 365.25 / float(vdat["dt"])
        # Uncertainty
        resid_m = dh_menke - G_m @ M_m
        var_m = np.sum(resid_m ** 2) / max(len(depth_menke) - 2, 1)
        cov_m = var_m * np.linalg.inv(G_m.T @ W_m @ G_m)
        vsr_err = np.sqrt(max(cov_m[1, 1], 0)) * 365.25 / float(vdat["dt"])
    else:
        vsr = np.nan
        vsr_err = np.nan

    return {
        "name": name,
        "H": H,
        "n_points": len(zeta),
        "best_order": best_order,
        "a0": a0,
        "a0_err": a0_err,
        "a1": a1,
        "a1_err": a1_err,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "nonlinear_energy": nonlinear_energy,
        "vsr": vsr,
        "vsr_err": vsr_err,
        "shear_ratio": shear_ratio,
        "shear_significant": shear_significant,
        "fits": fits,
        "f_tests": f_tests,
        "zeta": zeta,
        "w_fit": w_fit,
        "w_sigma_fit": w_sigma_fit,
        "depth_fit": depth_fit,
    }


def main():
    print("=" * 60, flush=True)
    print("Legendre shear analysis of ApRES vertical velocity profiles", flush=True)
    print("=" * 60, flush=True)

    # Load waypoints for coordinates
    wp = pd.read_excel(
        "/media/andrew/wd1/projects/thwaitesfirn/data/strainrates/Waypoints_GHOST2022-2023.xlsx",
        "waypoint-gps",
    )
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    valid_mask = wp["xcoord"].notna() & wp["ycoord"].notna()
    x_s, y_s = transformer.transform(
        wp.loc[valid_mask, "xcoord"].values,
        wp.loc[valid_mask, "ycoord"].values,
    )
    wp["x"] = np.nan
    wp["y"] = np.nan
    wp.loc[valid_mask, "x"] = x_s
    wp.loc[valid_mask, "y"] = y_s
    wp_lookup = wp.set_index("Waypoint")

    mat_files = sorted(glob.glob(str(STRAIN_DIR / "*.mat")))
    print(f"Found {len(mat_files)} .mat files\n", flush=True)

    results = []
    for mat_path in mat_files:
        r = process_site(mat_path)
        if r is None:
            continue
        results.append(r)

        sig_str = "***" if r["shear_significant"] else "   "
        print(f"  {r['name']:>10s}: a0={r['a0']:+8.3f}±{r['a0_err']:.3f}, "
              f"a1={r['a1']:+8.3f}±{r['a1_err']:.3f}, "
              f"ratio={r['shear_ratio']:.3f} {sig_str} "
              f"order={r['best_order']}", flush=True)

    # Summary
    n_sig = sum(1 for r in results if r["shear_significant"])
    print(f"\n{'='*60}", flush=True)
    print(f"Significant linear shear: {n_sig}/{len(results)} sites", flush=True)
    ratios = [r["shear_ratio"] for r in results if np.isfinite(r["shear_ratio"])]
    print(f"Shear ratio (all nonlinear/a0): mean={np.mean(ratios):.3f}, "
          f"median={np.median(ratios):.3f}, "
          f"max={np.max(ratios):.3f}", flush=True)

    # ── Summary plot ──
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 9,
    })
    fig = plt.figure(figsize=(12, 10))
    gs_top = fig.add_gridspec(1, 3, left=0.07, right=0.95, top=0.98, bottom=0.74, wspace=0.40)
    gs_maps = fig.add_gridspec(2, 1, left=0.07, right=0.88, top=0.70, bottom=0.06,
                                hspace=-0.15)
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    axes_bot = [fig.add_subplot(gs_maps[0, 0]), fig.add_subplot(gs_maps[1, 0])]

    # Panel 1: One example site with significant nonlinear shear — show data + all fits
    # Pick a site with high shear and quartic best order
    example = None
    for r in sorted(results, key=lambda r: r["shear_ratio"], reverse=True):
        if r["best_order"] >= 3 and 0.3 < r["shear_ratio"] < 2.0:
            example = r
            break
    if example is None:
        example = results[0]

    ax = axes_top[0]
    r = example
    # Plot raw data with uncertainty
    ax.errorbar(r["w_fit"], r["zeta"], xerr=r["w_sigma_fit"],
                fmt="k.", ms=2, elinewidth=0.3, alpha=0.3, label="data", zorder=1)

    # Plot each polynomial order fit
    zeta_fine = np.linspace(r["zeta"].min(), r["zeta"].max(), 300)
    fit_colors = {"1": "coral", "2": "mediumpurple", "3": "dodgerblue", "4": "forestgreen"}
    fit_labels = {"1": "linear", "2": "quadratic", "3": "cubic", "4": "quartic"}
    for order in range(1, MAX_ORDER + 1):
        fit = r["fits"][order]
        Phi = np.column_stack([shifted_legendre(k, zeta_fine) for k in range(order + 1)])
        w_model = Phi @ fit["coeffs"]
        lw = 2.5 if order == r["best_order"] else 1.0
        ls = "-" if order == r["best_order"] else "--"
        color = fit_colors[str(order)]
        # F-test p-value for this order
        if order >= 2:
            p = r["f_tests"][order]["p_value"]
            p_str = f" (p={p:.1e})" if np.isfinite(p) else ""
        else:
            p_str = ""
        label = f"{fit_labels[str(order)]}{p_str}"
        if order == r["best_order"]:
            label = f"{label} [best]"
        ax.plot(w_model, zeta_fine, color=color, lw=lw, ls=ls, label=label, zorder=2+order)

    ax.invert_yaxis()
    ax.set_xlabel("$w$ (m yr$^{-1}$)")
    ax.set_ylabel("$\\zeta = z / H$")
    ax.legend(fontsize=7, loc="best")
    ax.set_ylim(1, 0)
    ax.set_box_aspect(1)

    # Panel 2: Best polynomial order histogram
    ax = axes_top[1]
    orders = [r["best_order"] for r in results]
    ax.hist(orders, bins=np.arange(0.5, MAX_ORDER + 1.5, 1), color="steelblue", edgecolor="k")
    ax.set_xlabel("Significant polynomial order (F-test, p < 0.05)")
    ax.set_ylabel("Number of sites")
    ax.set_xticks(range(1, MAX_ORDER + 1))
    ax.set_xticklabels(["linear", "quadratic", "cubic", "quartic"])
    ax.set_box_aspect(1)

    # Panel 3: Horizontal velocity divergence vs ApRES vertical strain rate
    # From incompressibility: ε_zz = -div(u_horizontal)
    # Compare the two independent measurements at sites where the strain
    # rate profile is approximately linear (depth-uniform)
    ax = axes_top[2]

    # Compute div(u) from the surface velocity on the mesh
    import firedrake
    with firedrake.CheckpointFile(str(DATA_DIR.parent / "mesh" / "thwaites.h5"), "r") as chk:
        mesh_vel = chk.load_mesh()
        u_obs_vel = chk.load_function(mesh_vel, "velocity")
    Q_vel = firedrake.FunctionSpace(mesh_vel, "CG", 1)
    div_u = firedrake.Function(Q_vel).project(firedrake.div(u_obs_vel))

    # Evaluate -div(u) at each ApRES site
    vsr_arr = np.array([r["vsr"] for r in results])
    div_u_at_sites = np.full(len(results), np.nan)
    for i, r in enumerate(results):
        name = r["name"]
        if name in wp_lookup.index:
            xi, yi = wp_lookup.loc[name, "x"], wp_lookup.loc[name, "y"]
            if np.isfinite(xi) and np.isfinite(yi):
                try:
                    div_u_at_sites[i] = -float(div_u.at([xi, yi], tolerance=1e-3))
                except Exception:
                    pass

    valid_corr = np.isfinite(vsr_arr) & np.isfinite(div_u_at_sites)

    ax.scatter(div_u_at_sites[valid_corr], vsr_arr[valid_corr],
               c="steelblue", s=20, alpha=0.7, edgecolors="k", linewidths=0.3)

    # 1:1 line
    lims = [min(div_u_at_sites[valid_corr].min(), vsr_arr[valid_corr].min()),
            max(div_u_at_sites[valid_corr].max(), vsr_arr[valid_corr].max())]
    ax.plot(lims, lims, "k--", lw=0.8, label="1:1")
    ax.set_xlabel("$-\\nabla \\cdot \\mathbf{u}_{\\rm surface}$ (yr$^{-1}$)")
    ax.set_ylabel("$\\dot{\\varepsilon}_{zz}$ ApRES (yr$^{-1}$)")
    ax.legend(fontsize=7)
    # Equal axes for correlation
    all_vals = np.concatenate([div_u_at_sites[valid_corr], vsr_arr[valid_corr]])
    lim = max(abs(all_vals.min()), abs(all_vals.max())) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.02))

    # Panel 4: Map of shear ratios
    ax = axes_bot[0]
    xs = np.array([wp_lookup.loc[r["name"], "x"] if r["name"] in wp_lookup.index else np.nan
                    for r in results])
    ys = np.array([wp_lookup.loc[r["name"], "y"] if r["name"] in wp_lookup.index else np.nan
                    for r in results])
    shear_vals = np.array([r["shear_ratio"] for r in results])
    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(shear_vals)

    # Load BedMachine surface elevation raster for background contours
    import netCDF4 as nc
    bm_path = "/media/andrew/wd1/projects/ismip7/data/bedmachine/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
    ds_bm = nc.Dataset(bm_path)
    x_bm = ds_bm.variables["x"][:]
    y_bm = ds_bm.variables["y"][:]
    pad_bm = 30000
    ix0 = max(0, np.searchsorted(x_bm, xs[valid].min() - pad_bm) - 1)
    ix1 = min(len(x_bm), np.searchsorted(x_bm, xs[valid].max() + pad_bm) + 1)
    iy0 = max(0, np.searchsorted(-y_bm, -(ys[valid].max() + pad_bm)) - 1)
    iy1 = min(len(y_bm), np.searchsorted(-y_bm, -(ys[valid].min() - pad_bm)) + 1)
    x_sub = x_bm[ix0:ix1]
    y_sub = y_bm[iy0:iy1]
    surf_bm = np.array(ds_bm.variables["surface"][iy0:iy1, ix0:ix1]).astype(float)
    ds_bm.close()
    X_bm, Y_bm = np.meshgrid(x_sub, y_sub)

    # Background: BedMachine surface elevation contours
    levels = np.arange(0, 3000, 100)
    ax.contour(X_bm, Y_bm, surf_bm, levels=levels, colors="0.6", linewidths=0.3)
    ax.contourf(X_bm, Y_bm, surf_bm, levels=levels, cmap="Greys", alpha=0.12)
    sc = ax.scatter(xs[valid], ys[valid], c=shear_vals[valid], cmap="YlOrRd",
                    s=50, edgecolors="k", linewidths=0.4, vmin=0, vmax=0.5, zorder=5)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(sc, cax=cax, label="shear ratio")
    pad = 5000
    ax.set_xlim(xs[valid].min() - pad, xs[valid].max() + pad)
    ax.set_ylim(ys[valid].min() - pad, ys[valid].max() + pad)
    ax.set_ylabel("y$_{\\rm stereo}$ (km)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}"))
    ax.tick_params(labelbottom=False)  # share x-axis with bottom map
    ax.set_aspect("equal")

    # Panel 5: Depth-averaged vertical strain rate map
    ax = axes_bot[1]
    vsr_vals = np.array([r["vsr"] for r in results])
    # Background: BedMachine surface elevation contours
    ax.contour(X_bm, Y_bm, surf_bm, levels=levels, colors="0.6", linewidths=0.3)
    ax.contourf(X_bm, Y_bm, surf_bm, levels=levels, cmap="Greys", alpha=0.12)
    vsr_max = np.nanpercentile(np.abs(vsr_vals[valid]), 95)
    sc = ax.scatter(xs[valid], ys[valid], c=vsr_vals[valid], cmap="RdBu_r",
                    s=50, edgecolors="k", linewidths=0.4,
                    vmin=-vsr_max, vmax=vsr_max, zorder=5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(sc, cax=cax, label="$\\dot{\\varepsilon}_{zz}$ (yr$^{-1}$)")
    ax.set_xlim(xs[valid].min() - pad, xs[valid].max() + pad)
    ax.set_ylim(ys[valid].min() - pad, ys[valid].max() + pad)
    ax.set_xlabel("x$_{\\rm stereo}$ (km)")
    ax.set_ylabel("y$_{\\rm stereo}$ (km)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}"))
    ax.set_aspect("equal")

    # No overarching title
    fig.savefig(str(FIG_DIR / "shear_analysis.png"), dpi=200)
    print(f"\nSaved {FIG_DIR / 'shear_analysis.png'}", flush=True)

    # Save results to HDF5
    output = DATA_DIR / "apres_shear_analysis.h5"
    with h5py.File(str(output), "w") as f:
        names_arr = [r["name"] for r in results]
        f.create_dataset("names", data=[n.encode() for n in names_arr])
        f.create_dataset("x", data=xs)
        f.create_dataset("y", data=ys)
        f.create_dataset("a0", data=[r["a0"] for r in results])
        f.create_dataset("a0_err", data=[r["a0_err"] for r in results])
        f.create_dataset("a1", data=[r["a1"] for r in results])
        f.create_dataset("a1_err", data=[r["a1_err"] for r in results])
        f.create_dataset("a2", data=[r["a2"] for r in results])
        f.create_dataset("a3", data=[r["a3"] for r in results])
        f.create_dataset("shear_ratio", data=[r["shear_ratio"] for r in results])
        f.create_dataset("shear_significant", data=[r["shear_significant"] for r in results])
        f.create_dataset("best_order", data=[r["best_order"] for r in results])
        f.create_dataset("H", data=[r["H"] for r in results])
        f.create_dataset("vsr", data=[r["vsr"] for r in results])
        f.create_dataset("vsr_err", data=[r["vsr_err"] for r in results])
    print(f"Saved {output}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
