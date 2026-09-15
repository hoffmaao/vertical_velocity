r"""Fit eigenvalue spectrum to estimate required number of modes.

Compares power-law and exponential decay models, extrapolates
to find where λ crosses significance thresholds (λ=100, 10, 1).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
EIGVAL_FILE = DATA_DIR / "results" / "eigenvalues.txt"
FIG_DIR = DATA_DIR / "figures"


def main():
    data = np.loadtxt(str(EIGVAL_FILE))
    # Filter to positive eigenvalues only
    pos = data[:, 1] > 0
    data = data[pos]
    idx = np.arange(len(data))
    vals = data[:, 1]
    N = len(vals)
    log_i = np.log(idx + 1)
    log_lam = np.log(vals)

    # Power-law fit on modes 10-40 (stable tail)
    c_pow = np.polyfit(log_i[10:], log_lam[10:], 1)
    b_pow = -c_pow[0]
    lam0_pow = np.exp(c_pow[1])

    # Exponential fit on modes 10-40
    c_exp = np.polyfit(idx[10:], log_lam[10:], 1)
    rate_exp = -c_exp[0]
    lam0_exp = np.exp(c_exp[1])

    # R² in log-space
    pred_pow = np.exp(np.polyval(c_pow, log_i[10:]))
    pred_exp = np.exp(np.polyval(c_exp, idx[10:]))
    ss_tot = np.sum((log_lam[10:] - log_lam[10:].mean())**2)
    r2_pow = 1 - np.sum((log_lam[10:] - np.log(pred_pow))**2) / ss_tot
    r2_exp = 1 - np.sum((log_lam[10:] - np.log(pred_exp))**2) / ss_tot

    # Crossover predictions
    thresholds = {"$\\lambda = 100$ ($D = 0.99$)": 100,
                  "$\\lambda = 10$ ($D = 0.91$)": 10,
                  "$\\lambda = 1$ ($D = 0.5$)": 1}

    print(f"{'Model':>15s}  {'R²':>8s}  ", end="")
    for label in thresholds:
        print(f"{'i('+label[-7:-1]+')':>10s}  ", end="")
    print()

    for model, r2, crossfn in [
        ("Power law", r2_pow,
         lambda t: (lam0_pow / t)**(1.0/b_pow) - 1),
        ("Exponential", r2_exp,
         lambda t: np.log(lam0_exp / t) / rate_exp),
    ]:
        print(f"{model:>15s}  {r2:8.4f}  ", end="")
        for label, t in thresholds.items():
            ic = crossfn(t)
            print(f"{ic:10.0f}  ", end="")
        print()

    # ── Figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Extrapolation range
    i_max = int(np.log(lam0_exp / 0.5) / rate_exp) + 10
    i_ext = np.arange(0, min(i_max, 500))

    # (a) Spectrum + fits
    ax = axes[0]
    ax.semilogy(idx, vals, "ko", ms=6, label="Computed", zorder=3)

    # Power law
    lam_pow = lam0_pow * (i_ext + 1)**(-b_pow)
    ax.semilogy(i_ext, lam_pow, "r--", lw=1.5, alpha=0.7,
                label=f"Power law: $(i+1)^{{-{b_pow:.2f}}}$, $R^2={r2_pow:.3f}$")

    # Exponential
    lam_exp = lam0_exp * np.exp(-rate_exp * i_ext)
    ax.semilogy(i_ext, lam_exp, "b-", lw=1.5, alpha=0.7,
                label=f"Exponential: $e^{{-{rate_exp:.3f}i}}$, $R^2={r2_exp:.3f}$")

    # Thresholds
    for label, t, color in [
        ("$\\lambda=100$", 100, "C1"),
        ("$\\lambda=10$", 10, "C2"),
        ("$\\lambda=1$", 1, "C3"),
    ]:
        ax.axhline(t, color=color, ls=":", lw=1.5, alpha=0.7)
        # Mark exponential crossover
        ic = np.log(lam0_exp / t) / rate_exp
        ax.axvline(ic, color=color, ls=":", lw=0.8, alpha=0.4)
        ax.annotate(f"{label}\ni≈{ic:.0f}",
                    (ic, t), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color=color)

    ax.set_xlabel("Mode index $i$", fontsize=12)
    ax.set_ylabel("Eigenvalue $\\lambda_i$", fontsize=12)
    ax.set_title("(a) Spectrum extrapolation", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlim(-5, min(i_max, 300))
    ax.set_ylim(0.3, vals.max() * 2)

    # (b) Cumulative D with extrapolation
    ax = axes[1]
    D_comp = vals / (1.0 + vals)
    cumD_comp = np.cumsum(D_comp)
    ax.plot(idx, cumD_comp, "ko-", ms=4, lw=1.5, label="Computed")

    D_exp = lam_exp / (1.0 + lam_exp)
    cumD_exp = np.cumsum(D_exp)
    ax.plot(i_ext, cumD_exp, "b-", lw=1.5, alpha=0.7, label="Exponential extrapolation")

    D_pow = lam_pow / (1.0 + lam_pow)
    cumD_pow = np.cumsum(D_pow)
    ax.plot(i_ext, cumD_pow, "r--", lw=1.5, alpha=0.7, label="Power-law extrapolation")

    # Mark where D starts to drop
    ic_99 = np.log(lam0_exp / 100) / rate_exp
    ic_50 = np.log(lam0_exp / 1) / rate_exp
    ax.axvline(ic_99, color="C1", ls=":", lw=1, alpha=0.5)
    ax.axvline(ic_50, color="C3", ls=":", lw=1, alpha=0.5)
    ax.annotate(f"D<0.99\ni≈{ic_99:.0f}", (ic_99, cumD_exp[min(int(ic_99), len(cumD_exp)-1)]),
                textcoords="offset points", xytext=(8, -15), fontsize=9, color="C1")
    ax.annotate(f"D<0.5\ni≈{ic_50:.0f}", (ic_50, cumD_exp[min(int(ic_50), len(cumD_exp)-1)]),
                textcoords="offset points", xytext=(8, -15), fontsize=9, color="C3")

    ax.set_xlabel("Mode index $i$", fontsize=12)
    ax.set_ylabel("Cumulative $n_{\\mathrm{eff}}$", fontsize=12)
    ax.set_title("(b) Effective parameters (extrapolated)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-5, min(i_max, 300))

    fig.suptitle("Eigenspectrum decay: exponential fit suggests "
                 f"~{int(ic_99)} significant modes ($D>0.99$), "
                 f"~{int(ic_50)} total ($D>0.5$)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "eigendec_fit.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}", flush=True)

    print(f"\nRecommendation:")
    print(f"  Compute {int(ic_99)+20} modes to capture all D>0.99 modes")
    print(f"  Compute {min(int(ic_50)+20, 500)} modes for full n_eff (D>0.5)")


if __name__ == "__main__":
    main()
