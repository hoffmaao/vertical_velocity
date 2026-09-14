r"""Post-hoc scan over delta (prior amplitude) at fixed ELL.

The eigendec was run with (delta_0, gamma_0). The eigenvectors are the same
under uniform rescaling of (delta, gamma); eigenvalues scale by delta_0/delta_new
when delta and gamma scale together at fixed ELL = sqrt(gamma/delta).

For each delta_new = factor * delta_0:
  - gamma_new = factor * gamma_0 (preserves ELL)
  - lambda_new = lambda_old / factor
  - D_new = lambda_new / (lambda_new + 1)
  - sigma2_prior_new = sigma2_prior_old / factor**2 (Gamma scales as 1/(delta*gamma))
  - variance_reduction = sum(D_new * a_i**2) where a_i unchanged
  - sigma2_post_new = sigma2_prior_new - variance_reduction

Usage:
    python uq_delta_scan.py
"""
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
EIG_NPZ = DATA_DIR / "results" / "eigenvectors_K1500_gn_recinos.npz"
PRIOR_JSON = DATA_DIR / "results" / "prior_params_K1500_gn_recinos.json"
SUMMARY_JSON = DATA_DIR / "results" / "uq_forward_vaf_50yr_summary.json"

# Factors to scan (relative to current delta = 1e-5)
FACTORS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]


def main():
    eig = np.load(EIG_NPZ)
    vals = eig["eigenvalues"]
    vecs = eig["vecs"]

    with open(PRIOR_JSON) as f:
        prior = json.load(f)
    delta_old = prior["delta"]

    with open(SUMMARY_JSON) as f:
        summary = json.load(f)
    sigma2_prior_old = summary["sigma2_prior_m6"]
    a_squared = None  # need to recompute or load

    # Recompute a_i = vecs.T @ g_full and Sum D a^2 from the summary's a-vector
    # The contribs file has lambda, a, D*a^2 per line
    contribs_path = DATA_DIR / "results" / "uq_forward_vaf_50yr_contribs.txt"
    contribs = np.loadtxt(str(contribs_path), comments="#")
    # cols: i  lambda  a  D*a^2  cum_frac
    a_i = contribs[:, 2]
    Da2_old = contribs[:, 3]
    lambdas_old = contribs[:, 1]

    print(f"Loaded {len(a_i)} eigenpairs from {EIG_NPZ.name}")
    print(f"Original delta = {delta_old:.3e}, sigma_prior(VAF) = "
          f"{np.sqrt(sigma2_prior_old) * 917.0 / 1e12:.1f} Gt\n")

    print(f"{'factor':>8s}  {'delta_new':>10s}  {'sigma_prior_Gt':>14s}  "
          f"{'sigma_post_Gt':>14s}  {'var_reduction%':>15s}  "
          f"{'n_eff (lambda>1)':>17s}")
    print("-" * 90)

    for f in FACTORS:
        delta_new = delta_old * f
        lambdas_new = lambdas_old / f  # eigvals scale as 1/f when both delta and gamma scale by f
        D_new = lambdas_new / (lambdas_new + 1.0)
        var_reduction = float(np.sum(D_new * a_i**2))
        sigma2_prior_new = sigma2_prior_old / f**2  # Gamma scales as 1/(delta*gamma) = 1/(delta**2 * ELL**2)
        sigma2_post_new = max(sigma2_prior_new - var_reduction, 0.0)
        sigma_prior_Gt = np.sqrt(sigma2_prior_new) * 917.0 / 1e12
        sigma_post_Gt = np.sqrt(sigma2_post_new) * 917.0 / 1e12
        if sigma2_prior_new > 0:
            reduction_pct = 100.0 * (1.0 - sigma2_post_new / sigma2_prior_new)
        else:
            reduction_pct = float("nan")
        n_eff = float(np.sum(D_new))
        print(f"{f:>8.1f}  {delta_new:>10.2e}  {sigma_prior_Gt:>14.1f}  "
              f"{sigma_post_Gt:>14.1f}  {reduction_pct:>14.2f}%  "
              f"{n_eff:>17.1f}")


if __name__ == "__main__":
    main()
