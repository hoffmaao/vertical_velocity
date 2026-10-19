r"""Posterior variance of VAF(t) without eigenmode truncation: preconditioned CG on (H + A).

uq_vaf.py evaluates the Isaac et al. (2015) low-rank formula with the K leading
eigenpairs of H w = lambda A w, which is exact only if the omitted modes carry
no information. For a weak prior the spectrum decays slowly (lambda_K ~ 5 at
K = 3000 for delta = 1e-6, ELL = 1 km) and the truncated value is only an
upper bound. Here the quadratic form itself is solved for,

    sigma^2_post = g . (H + A)^{-1} g,

with g = dVAF(t)/d(theta_A, theta_C) from forward.py, H the Gauss-Newton misfit
Hessian and A the prior precision Gamma^-1 (both from map_hessian.setup, i.e. exactly
what eigendec.py decomposed). The K eigenpairs are used as the preconditioner,

    P = A^{-1} - W diag(lambda / (1 + lambda)) W^T   (= (A + A W Lambda W^T A)^{-1}),

the truncated posterior covariance, so P (H + A) has eigenvalue 1 on the kept
modes and 1 + lambda_i <= 1 + lambda_K on the rest: CG converges in a few tens
of Hessian-vector products regardless of how many modes the data inform.

From x_0 = 0 the CG iterates satisfy g . x_k = x_k . (H + A) x_k, which grows
monotonically to sigma^2_post (Gauss quadrature, Golub & Meurant), so each
iterate is a lower bound, while g . P g, the K-truncated value uq_vaf.py
reports, is an upper bound. Both are printed.

--misfit-scale chi2 inflates the observation errors by sqrt(chi2/N) per data
type (map_hessian.setup), i.e. H_vel / chi2_vel and H_apr / chi2_apr: a
model-discrepancy allowance for data the MAP cannot fit to their stated
errors. The eigenpairs of the unscaled problem still precondition well (the
modes are unchanged when one factor dominates; the eigenvalues are divided by
the velocity factor squared) and the output stem gets a _chi2 suffix.

The rate over [t_k-1, t_k] has gradient (g_k - g_k-1) / dt, whose solve is
(x_k - x_k-1) / dt by linearity, so the rates cost nothing extra.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/uq_vaf_pcg.py \
        --model hybrid --friction rc --map sweep_hybrid_rc_d1e-06_L1km_thwaites_fixed_tau5km \
        --T 50 --K 3000

reads results/forward_<map>_<T>yr.{h5,_ts.npz} and results/eigendec_<map>_K<K>.{npz,json};
writes results/uq_<map>_<T>yr_pcg_summary.json (same time-series keys as
uq_vaf.py) and results/uq_<map>_<T>yr_pcg_solutions.npz, which also serves as
the restart file: solved times are skipped when the script is run again.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from firedrake import CheckpointFile

from map_hessian import setup

DATA = Path(__file__).resolve().parent.parent
GT_PER_M3 = 917.0 / 1e12


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    ap.add_argument("--friction", required=True)
    ap.add_argument("--no-apres", action="store_true",
                    help="hybrid: the MAP came from a velocity-only inversion")
    ap.add_argument("--map", required=True, help="the --name the inversion ran with")
    ap.add_argument("--T", type=float, default=50.0, help="forward run length (years)")
    ap.add_argument("--K", type=int, default=1500, help="eigendecomposition used as preconditioner")
    ap.add_argument("--forward", default=None,
                    help="forward checkpoint stem under results/ (default forward_<map>_<T>yr)")
    ap.add_argument("--eig", default=None,
                    help="eigendecomposition stem under results/ (default eigendec_<map>_K<K>)")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="stop when the preconditioned residual norm falls by this factor")
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--misfit-scale", choices=["chi2"], default=None,
                    help="inflate observation errors by sqrt(chi2/N) per data type")
    ap.add_argument("--tag", default="",
                    help="suffix for the output stem, to keep a run from a non-default "
                         "--forward from overwriting the default one")
    args = ap.parse_args()
    args.apres = args.model == "hybrid" and not args.no_apres
    args.forward = args.forward or f"forward_{args.map}_{args.T:g}yr"
    args.eig = args.eig or f"eigendec_{args.map}_K{args.K}"
    return args


def load_gradients(stem):
    r"""[(t, g)] for every gradient a forward.py run stored; the final time has plain names."""
    ts = np.load(str(DATA / "results" / f"{stem}_ts.npz"), allow_pickle=True)
    tg = ([float(t) for t in ts["gradient_times"]] if "gradient_times" in ts.files
          else [float(ts["T_years"])])
    out = []
    with CheckpointFile(str(DATA / "results" / f"{stem}.h5"), "r") as chk:
        mesh = chk.load_mesh()
        for k, t in enumerate(tg):
            sfx = "" if k == len(tg) - 1 else f"_t{t:g}yr"
            out.append((t, np.concatenate([chk.load_function(mesh, f"dVAF_dtheta{c}{sfx}").dat.data_ro
                                           for c in "AC"])))
    return out, ts


def pcg(op, prec, g, rtol, maxiter, label):
    r"""Solve op x = g from x = 0; returns x and the history of g . x (lower bounds)."""
    x = np.zeros_like(g)
    r = g.copy()
    z = prec(r)
    p = z.copy()
    rz = float(r @ z)
    rz0 = rz
    hist = []
    t0 = time.time()
    for k in range(1, maxiter + 1):
        q = op(p)
        α = rz / float(p @ q)
        x += α * p
        r -= α * q
        z = prec(r)
        rz_new = float(r @ z)
        hist.append(float(g @ x))
        rel = np.sqrt(max(rz_new, 0.0) / rz0)
        if k % 5 == 0 or rel <= rtol:
            print(f"    [{label}] it {k:4d}  g.x = {hist[-1]:.6e}  rel. prec. residual {rel:.2e}  "
                  f"({(time.time() - t0) / k:.1f} s/it)", flush=True)
        if rel <= rtol:
            return x, hist, True
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, hist, False


def main():
    args = parse_args()
    t_start = time.time()
    grads, ts = load_gradients(args.forward)
    stem = (f"uq_{args.map}_{args.T:g}yr_pcg" + ("_chi2" if args.misfit_scale else "")
            + (f"_{args.tag}" if args.tag else ""))
    sol_file = DATA / "results" / f"{stem}_solutions.npz"
    print(f"PCG posterior variance of VAF at {args.map}: {len(grads)} gradient times "
          f"{[t for t, _ in grads]}, preconditioner {args.eig}", flush=True)

    eig = np.load(str(DATA / "results" / f"{args.eig}.npz"))
    with open(DATA / "results" / f"{args.eig}.json") as f:
        eig_meta = json.load(f)
    W, lam = eig["vecs"], eig["eigenvalues"]

    S = setup(args.model, args.friction, args.apres, args.map, misfit_scale=args.misfit_scale)
    n, Hv, Rinv, Rcov = S["n"], S["Hv"], S["Rinv"], S["Rcov"]
    s_vel, s_apr = S["sigma_scale"]
    lam = lam / s_vel ** 2          # the preconditioner's spectrum under the velocity scaling
    prior = dict(zip(("delta_A", "gamma_A", "delta_C", "gamma_C"), S["prior"]))
    for k, v in prior.items():
        if not np.isclose(v, eig_meta[k]):
            raise SystemExit(f"{args.eig} was computed with {k}={eig_meta[k]:g} but the MAP's "
                             f"prior has {v:g}: the preconditioner would not match")
    if eig_meta.get("prior_form", "h1") != S["prior_form"]:
        raise SystemExit(f"{args.eig} was computed with the {eig_meta.get('prior_form', 'h1')} prior "
                         f"form but the MAP was inverted with {S['prior_form']}")
    prior["prior_form"] = S["prior_form"]
    D = lam / (1.0 + lam)
    assert W.shape[0] == 2 * n and grads[0][1].size == 2 * n
    def prec(r):
        return Rcov(r) - W @ (D * (W.T @ r))

    n_hv = [0]

    def op(x):
        n_hv[0] += 1
        return Hv(x) + Rinv(x)

    done = dict(np.load(str(sol_file))) if sol_file.exists() else {}
    xs, info = {}, {}
    for t, g in grads:
        key = f"t{t:g}"
        s2_prior = float(g @ Rcov(g))
        s2_trunc = float(g @ prec(g))
        if key in done:
            xs[t] = done[key]
            info[t] = dict(s2_prior=s2_prior, s2_trunc=s2_trunc, iters=int(done[f"{key}_iters"]),
                           converged=bool(done[f"{key}_conv"]))
            print(f"  t = {t:g} yr: loaded from {sol_file.name}", flush=True)
            continue
        print(f"  t = {t:g} yr: sigma_prior {np.sqrt(s2_prior) * GT_PER_M3:.3f} Gt, "
              f"K-truncated sigma_post (upper bound) {np.sqrt(s2_trunc) * GT_PER_M3:.3f} Gt", flush=True)
        x, hist, conv = pcg(op, prec, g, args.rtol, args.maxiter, f"t={t:g}")
        xs[t] = x
        info[t] = dict(s2_prior=s2_prior, s2_trunc=s2_trunc, iters=len(hist), converged=conv)
        print(f"  t = {t:g} yr: sigma_post {np.sqrt(hist[-1]) * GT_PER_M3:.3f} Gt after {len(hist)} "
              f"iterations ({'converged' if conv else 'NOT converged'})", flush=True)
        done.update({key: x, f"{key}_iters": len(hist), f"{key}_conv": conv,
                     f"{key}_hist": np.array(hist)})
        np.savez(str(sol_file), **done)

    times, vaf = ts["times"], ts["vaf_Gt"]
    tg = [t for t, _ in grads]
    vaf_at = np.interp(tg, times, vaf) - vaf[0]
    G = dict(grads)
    out = {"map": args.map, "T_years": args.T, "forward": args.forward, "eig": args.eig,
           "method": "pcg", "preconditioner_K": int(lam.size), "lambda_K": float(lam[-1]),
           "rtol": args.rtol, "misfit_scale": args.misfit_scale,
           "sigma_scale_vel": s_vel, "sigma_scale_apr": s_apr, **prior,
           "times": tg, "dvaf_Gt_t": vaf_at.tolist(), "sigma_prior_Gt_t": [], "sigma_post_Gt_t": [],
           "sigma_post_trunc_Gt_t": [], "cg_iterations_t": [], "cg_converged_t": [],
           "rate_t_mid": [], "rate_Gt_per_yr": [], "sigma_prior_rate": [], "sigma_post_rate": []}
    print("\n      t   dVAF(t) Gt   sigma_prior   sigma_post  (K-truncated)  |  interval   "
          "rate Gt/yr  sigma_prior  sigma_post", flush=True)
    t0, v0 = 0.0, 0.0
    g0, x0 = np.zeros(2 * n), np.zeros(2 * n)
    for t, v in zip(tg, vaf_at):
        g, x, I = G[t], xs[t], info[t]
        out["sigma_prior_Gt_t"].append(np.sqrt(I["s2_prior"]) * GT_PER_M3)
        out["sigma_post_Gt_t"].append(np.sqrt(float(g @ x)) * GT_PER_M3)
        out["sigma_post_trunc_Gt_t"].append(np.sqrt(I["s2_trunc"]) * GT_PER_M3)
        out["cg_iterations_t"].append(I["iters"])
        out["cg_converged_t"].append(I["converged"])
        dt = t - t0
        dg, dx = (g - g0) / dt, (x - x0) / dt
        out["rate_t_mid"].append(0.5 * (t + t0))
        out["rate_Gt_per_yr"].append((v - v0) / dt)
        out["sigma_prior_rate"].append(np.sqrt(float(dg @ Rcov(dg))) * GT_PER_M3)
        out["sigma_post_rate"].append(np.sqrt(max(float(dg @ dx), 0.0)) * GT_PER_M3)
        print(f"  {t:5.1f}  {v:+10.1f}  {out['sigma_prior_Gt_t'][-1]:11.2f}  "
              f"{out['sigma_post_Gt_t'][-1]:11.2f}  ({out['sigma_post_trunc_Gt_t'][-1]:8.2f})   |  "
              f"{t0:4.0f}-{t:<4.0f}  {out['rate_Gt_per_yr'][-1]:+9.2f}   "
              f"{out['sigma_prior_rate'][-1]:9.3f}  {out['sigma_post_rate'][-1]:9.3f}", flush=True)
        t0, v0, g0, x0 = t, v, g, x
    out.update(sigma_prior_Gt=out["sigma_prior_Gt_t"][-1], sigma_post_Gt=out["sigma_post_Gt_t"][-1],
               sigma_post_trunc_Gt=out["sigma_post_trunc_Gt_t"][-1], n_hessian_products=n_hv[0],
               wall_hours=(time.time() - t_start) / 3600)
    with open(DATA / "results" / f"{stem}_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/{stem}_summary.json ({n_hv[0]} Hessian products, "
          f"{out['wall_hours']:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
