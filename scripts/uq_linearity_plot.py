r"""Aggregate the directional linearity sweep and test the Isaac et al. assumption.

Reads results/uq_linearity_n*.npz (one per amplitude n, from
uq_directional_forward.py) and compares the TRUE forward response
  R_true(n) = ΔVAF(n) − ΔVAF(0)
against the LINEARIZED prediction
  R_lin(n)  = n · σ_post.
A straight line ⇒ the linearization (and hence σ_post) is trustworthy at the
elbow scale; downward curvature on the −n side ⇒ grounding-line nonlinearity
makes σ_post an underestimate.

Writes figures/uq_linearity.png and results/uq_linearity_summary.json.
"""
import json
import glob
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
GT = 917.0 / 1e12


def main():
    rows = []
    sigma_post = None
    for f in glob.glob(str(DATA / "results" / "uq_linearity_n*.npz")):
        d = np.load(f)
        n = float(d["n"])
        sigma_post = float(d["sigma_post_Gt"])
        if bool(d["failed"]):
            rows.append((n, np.nan, np.nan, True))
        else:
            rows.append((n, float(d["dVAF"]), int(d["n_fail"]), False))
    rows.sort(key=lambda r: r[0])
    ns = np.array([r[0] for r in rows])
    dVAF = np.array([r[1] for r in rows])
    nfail = np.array([r[2] for r in rows])

    # baseline at n=0
    i0 = int(np.argmin(np.abs(ns)))
    assert abs(ns[i0]) < 1e-9, "no n=0 baseline run found"
    dVAF0 = dVAF[i0]

    R_true = dVAF - dVAF0            # true VAF response to the perturbation
    R_lin = ns * sigma_post         # linearized prediction
    resid = R_true - R_lin
    resid_pct = 100.0 * resid / np.where(R_lin == 0, np.nan, R_lin)

    print(f"σ_post (linearized, elbow) = {sigma_post:.2f} Gt")
    print(f"ΔVAF(n=0) baseline         = {dVAF0:.2f} Gt\n")
    print(f"{'n':>6} {'ΔVAF':>10} {'R_true':>10} {'R_lin=nσ':>10} "
          f"{'resid':>9} {'resid%':>8} {'fails':>6}")
    print("-" * 66)
    for k in range(len(ns)):
        print(f"{ns[k]:>6.1f} {dVAF[k]:>10.2f} {R_true[k]:>10.2f} {R_lin[k]:>10.2f} "
              f"{resid[k]:>9.2f} {resid_pct[k]:>7.1f}% {nfail[k]:>6d}")

    # Curvature: fit R_true = slope*n + curv*n^2 over finite points
    good = np.isfinite(R_true)
    coef = np.polyfit(ns[good], R_true[good], 2)  # [curv, slope, intercept]
    curv, slope, intercept = coef
    # Asymmetry: |R_true(-n)| vs |R_true(+n)| at matched |n|
    print(f"\nQuadratic fit R_true(n) ≈ {slope:.2f}·n + {curv:+.3f}·n²  (intercept {intercept:+.2f})")
    print(f"  linearized slope σ_post = {sigma_post:.2f} Gt  →  fit slope {slope:.2f} Gt "
          f"({100*(slope/sigma_post-1):+.1f}%)")
    print(f"  curvature coefficient   = {curv:+.3f} Gt per n²  "
          f"(<0 ⇒ accelerating VAF loss ⇒ σ_post underestimates)")

    summary = {
        "sigma_post_Gt_linearized": sigma_post,
        "dVAF0_Gt": float(dVAF0),
        "n": ns.tolist(),
        "dVAF_Gt": dVAF.tolist(),
        "R_true_Gt": R_true.tolist(),
        "R_lin_Gt": R_lin.tolist(),
        "resid_Gt": resid.tolist(),
        "fit_slope_Gt": float(slope),
        "fit_curvature_Gt_per_n2": float(curv),
        "n_failed_runs": int((~good).sum()),
    }
    out = DATA / "results" / "uq_linearity_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved {out}")

    # ── Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    nn = np.linspace(ns.min(), ns.max(), 200)
    ax1.plot(nn, sigma_post * nn, "k--", lw=1.5, label=f"linearized (σ_post={sigma_post:.0f} Gt)")
    ax1.plot(nn, curv * nn**2 + slope * nn + intercept, "C1-", lw=1.2, alpha=0.7, label="quadratic fit")
    ax1.plot(ns[good], R_true[good], "o", color="C0", ms=7, label="true forward")
    ax1.axhline(0, color="gray", lw=0.5); ax1.axvline(0, color="gray", lw=0.5)
    ax1.set_xlabel("amplitude n  (× δθ_1σ)"); ax1.set_ylabel("ΔVAF(n) − ΔVAF(0)  [Gt]")
    ax1.set_title("VAF response along QoI uncertainty direction")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.2)
    # secondary axis: n in units of σ_post (Gt)
    ax2.plot(ns[good], resid[good], "s-", color="C3", ms=6)
    ax2.axhline(0, color="gray", lw=0.5); ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_xlabel("amplitude n  (× δθ_1σ)"); ax2.set_ylabel("nonlinearity residual  R_true − R_lin  [Gt]")
    ax2.set_title("Departure from linearization")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    figpath = DATA / "figures" / "uq_linearity.png"
    fig.savefig(str(figpath), dpi=200, bbox_inches="tight")
    print(f"Saved {figpath}")


if __name__ == "__main__":
    main()
