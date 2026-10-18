r"""Posterior variance of VAF(T) from a MAP's eigendecomposition and forward gradient.

Isaac et al. (2015, Eq. 20) in the fenics_ice / Recinos et al. (2023) convention:

    sigma^2_post = g . Gamma g  -  sum_i  lambda_i / (lambda_i + 1)  (w_i . g)^2

with g = dVAF(T)/d(theta_A, theta_C) from forward.py --gradient, (lambda_i, w_i)
the Gamma^-1-orthonormal eigenpairs of H w = lambda Gamma^-1 w from eigendec.py,
and Gamma the prior covariance the eigendecomposition was posed with, block
diagonal over the controls: A^-1 M A^-1 for the bilaplacian prior form, A^-1 for
the older h1 form, A = delta M + gamma K (prior_precond.py; the form is read
from the eigendecomposition's json). theta ~ N(0, Gamma), so the first term is
the prior variance sigma^2_prior = g . Gamma g, the sum the reduction the data
buy; with Gamma^-1-orthonormal w_i the
Euclidean dot products are the right ones and sigma^2_post lies in
[0, sigma^2_prior].

    python scripts/uq_vaf.py --map sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km --T 50 --K 1500

reads results/forward_<map>_<T>yr.h5, results/eigendec_<map>_K<K>.{npz,json}
and writes results/uq_<map>_<T>yr_K<K>_summary.json and _contribs.txt (the
per-mode reductions). --forward / --eig override the file stems.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from firedrake import CheckpointFile, Function

from prior_precond import Prior

DATA = Path(__file__).resolve().parent.parent
GT_PER_M3 = 917.0 / 1e12


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True, help="the --name the inversion ran with")
    ap.add_argument("--T", type=float, default=50.0, help="forward run length (years)")
    ap.add_argument("--K", type=int, default=1500, help="eigendecomposition --K")
    ap.add_argument("--forward", default=None,
                    help="forward checkpoint stem under results/ (default forward_<map>_<T>yr)")
    ap.add_argument("--eig", default=None,
                    help="eigendecomposition stem under results/ (default eigendec_<map>_K<K>)")
    ap.add_argument("--nmodes", type=int, default=None,
                    help="use only the leading N modes (default: every positive one)")
    ap.add_argument("--tag", default="",
                    help="appended to the output stem, so a run from a non-default --forward "
                         "does not overwrite the default one")
    args = ap.parse_args()
    args.forward = args.forward or f"forward_{args.map}_{args.T:g}yr"
    args.eig = args.eig or f"eigendec_{args.map}_K{args.K}"
    return args


def prior_covariance(Q, delta, gamma, form="h1"):
    r"""``apply(x) -> Gamma x`` on ``Q`` for the prior of that form (prior_precond.Prior)."""
    pr = Prior(Q, delta, gamma, form)

    def apply(x):
        y = Function(Q)
        y.dat.data[:] = pr.apply_covariance(x.dat.data_ro)
        return y
    return apply


def time_series(args, Q, vecs, D, prior):
    r"""sigma_prior(t), sigma_post(t) of VAF and of the mean loss rate between gradient times.

    Uses every gradient a ``forward.py --gradient-times`` run stored (the final
    time under the plain names). The rate over [t_k-1, t_k] is a linear
    functional too, with gradient (g_k - g_k-1) / (t_k - t_k-1), so its variance
    comes from the same formula and accounts for the correlation between the two
    times. Returns {} for a run without intermediate gradients.
    """
    ts = np.load(str(DATA / "results" / f"{args.forward}_ts.npz"), allow_pickle=True)
    if "gradient_times" not in ts.files or len(ts["gradient_times"]) < 2:
        return {}
    tg = [float(t) for t in ts["gradient_times"]]
    cov_A = prior_covariance(Q, prior["delta_A"], prior["gamma_A"], prior.get("prior_form", "h1"))
    cov_C = prior_covariance(Q, prior["delta_C"], prior["gamma_C"], prior.get("prior_form", "h1"))
    fA, fC = Function(Q), Function(Q)

    def variances(gvec):
        r"""(sigma2_prior, sigma2_post with K modes, its truncation lower bound)."""
        n = Q.dim()
        fA.dat.data[:] = gvec[:n]
        fC.dat.data[:] = gvec[n:]
        s2p = float(gvec[:n] @ cov_A(fA).dat.data_ro + gvec[n:] @ cov_C(fC).dat.data_ro)
        a2 = (vecs.T @ gvec) ** 2
        s2q = s2p - float((D * a2).sum())
        return s2p, s2q, s2q - D[-1] * max(s2p - float(a2.sum()), 0.0)

    grads = []
    with CheckpointFile(str(DATA / "results" / f"{args.forward}.h5"), "r") as chk:
        mesh = chk.load_mesh()
        for k, t in enumerate(tg):
            sfx = "" if k == len(tg) - 1 else f"_t{t:g}yr"
            grads.append(np.concatenate([chk.load_function(mesh, f"dVAF_dtheta{c}{sfx}").dat.data_ro
                                         for c in "AC"]))
    times, vaf = ts["times"], ts["vaf_Gt"]
    vaf_at = np.interp(tg, times, vaf) - vaf[0]
    out = {"times": tg, "dvaf_Gt_t": vaf_at.tolist(), "sigma_prior_Gt_t": [],
           "sigma_post_Gt_t": [], "rate_t_mid": [], "rate_Gt_per_yr": [],
           "sigma_prior_rate": [], "sigma_post_rate": [],
           "sigma_post_lo_Gt_t": [], "sigma_post_lo_rate": []}
    print("\n      t   dVAF(t) Gt   sigma_prior   sigma_post [trunc. bound]   |  interval   "
          "rate Gt/yr  sigma_post [bound]", flush=True)
    prev = (0.0, np.zeros_like(grads[0]), 0.0)
    for t, g, v in zip(tg, grads, vaf_at):
        s2p, s2q, s2l = variances(g)
        out["sigma_prior_Gt_t"].append(np.sqrt(s2p) * GT_PER_M3)
        out["sigma_post_Gt_t"].append(np.sqrt(max(s2q, 0.0)) * GT_PER_M3)
        out["sigma_post_lo_Gt_t"].append(np.sqrt(max(s2l, 0.0)) * GT_PER_M3)
        t0, g0, v0 = prev
        r2p, r2q, r2l = variances((g - g0) / (t - t0))
        out["sigma_post_lo_rate"].append(np.sqrt(max(r2l, 0.0)) * GT_PER_M3)
        out["rate_t_mid"].append(0.5 * (t + t0))
        out["rate_Gt_per_yr"].append((v - v0) / (t - t0))
        out["sigma_prior_rate"].append(np.sqrt(r2p) * GT_PER_M3)
        out["sigma_post_rate"].append(np.sqrt(max(r2q, 0.0)) * GT_PER_M3)
        print(f"  {t:5.1f}  {v:+10.1f}  {out['sigma_prior_Gt_t'][-1]:11.2f}  "
              f"{out['sigma_post_Gt_t'][-1]:11.2f} [{out['sigma_post_lo_Gt_t'][-1]:6.2f}]   |  "
              f"{t0:4.0f}-{t:<4.0f}  {out['rate_Gt_per_yr'][-1]:+9.2f}   "
              f"{out['sigma_post_rate'][-1]:8.3f} [{out['sigma_post_lo_rate'][-1]:6.3f}]", flush=True)
        prev = (t, g, v)
    return out


def main():
    args = parse_args()
    out_stem = f"uq_{args.map}_{args.T:g}yr_K{args.K}" + (f"_{args.tag}" if args.tag else "")
    print(f"UQ of VAF({args.T:g} yr) at {args.map}: {args.forward} x {args.eig}", flush=True)

    eig = np.load(str(DATA / "results" / f"{args.eig}.npz"))
    with open(DATA / "results" / f"{args.eig}.json") as f:
        prior = json.load(f)
    vecs, vals, n = eig["vecs"], eig["eigenvalues"], int(eig["n_dof"])
    n_pos = vals.size
    if args.nmodes is not None:
        vecs, vals = vecs[:, :args.nmodes], vals[:args.nmodes]
    D = vals / (vals + 1.0)
    print(f"  {vals.size} modes of {n_pos} positive: lambda in [{vals[-1]:.3e}, {vals[0]:.3e}], "
          f"D = lambda/(lambda+1) in [{D[-1]:.4f}, {D[0]:.6f}]", flush=True)
    print(f"  prior: theta_A delta={prior['delta_A']:.3e} gamma={prior['gamma_A']:.3e}; "
          f"theta_C delta={prior['delta_C']:.3e} gamma={prior['gamma_C']:.3e}", flush=True)

    with CheckpointFile(str(DATA / "results" / f"{args.forward}.h5"), "r") as chk:
        mesh = chk.load_mesh()
        gA = chk.load_function(mesh, "dVAF_dthetaA")
        gC = chk.load_function(mesh, "dVAF_dthetaC")
    Q = gA.function_space()
    assert Q.dim() == n, f"{Q.dim()} gradient dofs but the eigenvectors have {n} per control"
    g = np.concatenate([gA.dat.data_ro, gC.dat.data_ro])
    print(f"  |dVAF/dtheta_A| = {np.linalg.norm(gA.dat.data_ro):.4e}, "
          f"|dVAF/dtheta_C| = {np.linalg.norm(gC.dat.data_ro):.4e} m^3", flush=True)

    # Prior variance g . A^{-1} g, per control block.
    ΓgA = prior_covariance(Q, prior["delta_A"], prior["gamma_A"], prior.get("prior_form", "h1"))(gA)
    ΓgC = prior_covariance(Q, prior["delta_C"], prior["gamma_C"], prior.get("prior_form", "h1"))(gC)
    σ2_prior_A = float(gA.dat.data_ro @ ΓgA.dat.data_ro)
    σ2_prior_C = float(gC.dat.data_ro @ ΓgC.dat.data_ro)
    σ2_prior = σ2_prior_A + σ2_prior_C

    # Data reduction, mode by mode.
    a = vecs.T @ g
    contrib = D * a ** 2
    reduction = float(contrib.sum())
    σ2_post = σ2_prior - reduction
    σ_prior, σ_post = np.sqrt(σ2_prior), np.sqrt(max(σ2_post, 0.0))
    print(f"\n  sigma_prior(VAF) = {σ_prior * GT_PER_M3:9.3f} Gt  "
          f"(theta_A {np.sqrt(σ2_prior_A) * GT_PER_M3:.3f}, theta_C {np.sqrt(σ2_prior_C) * GT_PER_M3:.3f})",
          flush=True)
    print(f"  sigma_post(VAF)  = {σ_post * GT_PER_M3:9.3f} Gt   "
          f"variance reduction {100 * reduction / σ2_prior:.2f} %, "
          f"sigma ratio {σ_post / σ_prior:.4f}", flush=True)
    # Truncation: the A-orthonormal eigenvectors are complete, so sum_i a_i^2
    # over ALL modes is g . A^{-1} g = sigma2_prior. The modes beyond K hold the
    # prior variance R = sigma2_prior - sum_{i<=K} a_i^2, and each has
    # lambda_i <= lambda_K, so together they can remove at most D_K * R. Hence
    # sigma2_post,K - D_K R <= sigma2_post,true <= sigma2_post,K (= the
    # truncated value, which is therefore conservative).
    R = max(σ2_prior - float((a ** 2).sum()), 0.0)
    σ_post_lo = np.sqrt(max(σ2_post - D[-1] * R, 0.0))
    print(f"  truncation at K={vals.size} (lambda_K={vals[-1]:.3g}, D_K={D[-1]:.3f}): the modes kept "
          f"span {100 * (1 - R / σ2_prior):.2f} % of the prior variance of this QoI;\n"
          f"  true sigma_post lies in [{σ_post_lo * GT_PER_M3:.3f}, {σ_post * GT_PER_M3:.3f}] Gt",
          flush=True)
    cum = np.cumsum(contrib) / σ2_prior
    for frac in (0.5, 0.9, 0.99):
        k = int(np.searchsorted(cum, frac * cum[-1])) + 1
        print(f"  {k:5d} modes give {100 * frac:.0f} % of the reduction", flush=True)

    summary = {
        "map": args.map, "T_years": args.T, "forward": args.forward, "eig": args.eig,
        "n_modes_used": int(vals.size), "n_modes_positive": int(n_pos),
        "lambda_max": float(vals[0]), "lambda_min": float(vals[-1]),
        "delta_A": prior["delta_A"], "gamma_A": prior["gamma_A"],
        "delta_C": prior["delta_C"], "gamma_C": prior["gamma_C"],
        "sigma2_prior_m6": σ2_prior, "sigma2_prior_A_m6": σ2_prior_A,
        "sigma2_prior_C_m6": σ2_prior_C, "sigma2_post_m6": float(σ2_post),
        "sigma_prior_Gt": float(σ_prior * GT_PER_M3), "sigma_post_Gt": float(σ_post * GT_PER_M3),
        "variance_reduction_frac": reduction / σ2_prior,
        "D_K": float(D[-1]), "prior_variance_captured_frac": 1 - R / σ2_prior,
        "sigma_post_lo_Gt": float(σ_post_lo * GT_PER_M3),
        "g_A_norm": float(np.linalg.norm(gA.dat.data_ro)),
        "g_C_norm": float(np.linalg.norm(gC.dat.data_ro)),
    }
    summary.update(time_series(args, Q, vecs, D, prior))
    with open(DATA / "results" / f"{out_stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(DATA / "results" / f"{out_stem}_contribs.txt", "w") as f:
        f.write("# i  lambda_i  a_i=w_i.g  D_i*a_i^2  cumulative_reduction_frac\n")
        for i in range(vals.size):
            f.write(f"{i} {vals[i]:.16e} {a[i]:.16e} {contrib[i]:.16e} {cum[i]:.6f}\n")
    print(f"\nSaved results/{out_stem}_summary.json and _contribs.txt", flush=True)


if __name__ == "__main__":
    main()
