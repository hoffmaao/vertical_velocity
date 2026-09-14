r"""Eigenvalue significance analysis.

Tests whether the eigenspectrum shows modes that are statistically
significant (data-informed) vs noise (prior-dominated).

The null hypothesis H₀: "mode i carries no information beyond the prior"
is rejected when λ_i > 1, i.e., D_i = λ_i/(1+λ_i) > 0.5.

Usage:
    python plot_eigendec_significance.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent
EIGVAL_FILE = DATA_DIR / "results" / "eigenvalues.txt"
PRIOR_FILE = DATA_DIR / "results" / "prior_params.json"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    data = np.loadtxt(str(EIGVAL_FILE))
    idx = data[:, 0].astype(int)
    vals = data[:, 1]
    N = len(vals)

    with open(PRIOR_FILE) as f:
        prior = json.load(f)

    D = vals / (1.0 + vals)
    weights = 1.0 / np.sqrt(1.0 + vals)
    cumD = np.cumsum(D)

    # Power-law fit: log(λ) vs log(i+1)
    log_i = np.log(idx[5:] + 1)  # skip first 5 for stable fit
    log_lam = np.log(vals[5:])
    coeffs = np.polyfit(log_i, log_lam, 1)
    b = -coeffs[0]  # decay exponent
    lam_0 = np.exp(coeffs[1])

    # Extrapolate to find i where λ = 1
    if b > 0:
        i_cross = lam_0**(1.0/b) - 1
    else:
        i_cross = np.inf

    print(f"Eigenvalue spectrum:", flush=True)
    print(f"  {N} computed modes, all with λ >> 1", flush=True)
    print(f"  λ range: [{vals.min():.2e}, {vals.max():.2e}]", flush=True)
    print(f"  Power-law decay: λ ~ {lam_0:.2e} × (i+1)^(-{b:.2f})", flush=True)
    print(f"  Extrapolated λ=1 crossover: i ≈ {i_cross:.0e}", flush=True)
    print(f"  Prior delta = {prior['delta']:.2e} (1/area)", flush=True)
    print(f"  Prior gamma = {prior['gamma_A']:.2e} (L²/area)", flush=True)
    print(f"  area/N_vel ≈ {prior['area']/3037:.0e} (normalization ratio)", flush=True)

    # ── Figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Eigenvalue spectrum (log-log)
    ax = axes[0, 0]
    ax.loglog(idx + 1, vals, "ko", ms=6, label="Computed", zorder=3)
    # Fit line
    i_fit = np.logspace(0, np.log10(N + 1), 100)
    ax.loglog(i_fit, lam_0 * i_fit**(-b), "r--", lw=1, alpha=0.7,
              label=f"Fit: $\\lambda \\sim (i+1)^{{-{b:.1f}}}$")
    ax.axhline(1.0, color="C3", ls="-", lw=2, alpha=0.5,
               label="$\\lambda = 1$ (H₀ rejection threshold)")
    ax.set_xlabel("Mode index $i+1$", fontsize=12)
    ax.set_ylabel("Eigenvalue $\\lambda_i$", fontsize=12)
    ax.set_title("(a) Prior-preconditioned Hessian spectrum", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")

    # (b) Eigenvalue decay rate (successive ratio)
    ax = axes[0, 1]
    ratios = vals[:-1] / vals[1:]
    ax.plot(idx[:-1], ratios, "s-", color="C1", ms=4, lw=1)
    ax.axhline(1.0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("Mode index $i$", fontsize=12)
    ax.set_ylabel("$\\lambda_i / \\lambda_{i+1}$", fontsize=12)
    ax.set_title("(b) Successive eigenvalue ratio", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0.8, 2.5)

    # (c) Posterior weight = perturbation amplitude
    ax = axes[1, 0]
    ax.semilogy(idx, weights, "s-", color="C2", ms=5, lw=1)
    ax.set_xlabel("Mode index $i$", fontsize=12)
    ax.set_ylabel("$1/\\sqrt{1+\\lambda_i}$", fontsize=12)
    ax.set_title("(c) Posterior perturbation weight", fontsize=12)
    ax.grid(True, alpha=0.2, which="both")

    # Interpretation box
    ax.text(0.35, 0.25,
            f"All weights < {weights.max():.1e}\n"
            f"→ posterior ≈ MAP (data\n"
            f"   overwhelms prior in\n"
            f"   every computed direction)\n\n"
            f"Prior uses $\\delta = 1/|\\Omega|$\n"
            f"Misfit uses $1/N$ normalization\n"
            f"Ratio $|\\Omega|/N \\approx {prior['area']/3037:.0e}$\n"
            f"inflates all $\\lambda_i$",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="lightyellow", alpha=0.9),
            verticalalignment="top")

    # (d) Cumulative D_i
    ax = axes[1, 1]
    ax.plot(idx, cumD, "ko-", ms=4, lw=1.5)
    ax.set_xlabel("Mode index $i$", fontsize=12)
    ax.set_ylabel("$\\sum_{j \\leq i} D_j$", fontsize=12)
    ax.set_title(f"(d) Cumulative $n_{{\\mathrm{{eff}}}}$ = {cumD[-1]:.1f}", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, N + 2)

    # Diagonal reference (n_eff = i+1 means every mode is fully constrained)
    ax.plot([0, N-1], [1, N], "r--", lw=1, alpha=0.5, label="$n_{\\mathrm{eff}} = i+1$ (fully constrained)")
    ax.legend(fontsize=9)

    fig.suptitle(f"Eigendecomposition significance: {N} modes, "
                 f"$L = {prior['ELL']/1e3:.0f}$ km, "
                 f"all $\\lambda_i > 1$ (all significant)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "eigendec_significance.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
