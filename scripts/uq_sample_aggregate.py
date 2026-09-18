r"""Aggregate the posterior-sample forward ensemble → non-Gaussian VAF uncertainty.

Reads the hybrid results/sample_vaf_[0-9b-i]*.npz (from uq_sample_forward.py) and reports the true
posterior distribution of VAF(T): mean, std, skew, percentiles, and a histogram +
trajectory fan. Compares against the (invalid) linearized σ_post and the MAP run,
so we can see how far the real, threshold-affected distribution departs from the
Gaussian/linear approximation.

Writes figures/uq_sample_vaf.png and results/uq_sample_summary.json.
"""
import glob
import json
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent


def main():
    vafT, dVAF, trajs, times, nfail = [], [], [], None, []
    for f in sorted(glob.glob(str(DATA / "results" / "sample_vaf_[0-9b-i]*.npz"))):  # hybrid only; SSA runs are sample_vaf_ssa*/ssaw*
        d = np.load(f)
        if bool(d["failed"]):
            continue
        vafT.append(float(d["vafT"])); dVAF.append(float(d["dVAF"]))
        nfail.append(int(d["n_fail"]))
        trajs.append(d["vaf_Gt"]); times = d["time"]
    vafT = np.array(vafT); dVAF = np.array(dVAF); nfail = np.array(nfail)
    N = vafT.size
    if N == 0:
        print("no samples yet"); return

    # MAP reference (Recinos n=0 directional run) if available
    map_vafT = None
    p0 = DATA / "results" / "uq_linearity_np0_00.npz"
    if p0.exists():
        dd = np.load(str(p0)); map_vafT = float(dd["vaf0"]) + float(dd["dVAF"])

    def skew(x):
        m = x.mean(); s = x.std()
        return float(((x - m) ** 3).mean() / s ** 3) if s > 0 else 0.0

    mean, std = vafT.mean(), vafT.std(ddof=1)
    pct = {p: float(np.percentile(vafT, p)) for p in (5, 16, 25, 50, 75, 84, 95)}
    sig_lin = 66.5  # linearized σ_post (Gt) — known invalid, for contrast

    print(f"Posterior VAF(T) ensemble: N={N} samples ({(nfail>0).sum()} had ≥1 step fail)")
    print(f"  mean   VAF(T) = {mean:8.1f} Gt")
    print(f"  median VAF(T) = {pct[50]:8.1f} Gt")
    if map_vafT is not None:
        print(f"  MAP    VAF(T) = {map_vafT:8.1f} Gt   (mean−MAP = {mean-map_vafT:+.1f} Gt)")
    print(f"  std           = {std:8.1f} Gt   (linearized σ_post = {sig_lin:.1f} Gt → ratio {std/sig_lin:.2f})")
    print(f"  skew          = {skew(vafT):+.2f}")
    print(f"  5–95%  spread = [{pct[5]:.0f}, {pct[95]:.0f}]  (width {pct[95]-pct[5]:.0f} Gt)")
    print(f"  16–84% (≈±1σ) = [{pct[16]:.0f}, {pct[84]:.0f}]  (half-width {(pct[84]-pct[16])/2:.0f} Gt)")
    print(f"  min / max     = {vafT.min():.0f} / {vafT.max():.0f}")

    summary = {
        "N": int(N), "n_with_fail": int((nfail > 0).sum()),
        "mean_VAF_T_Gt": float(mean), "std_VAF_T_Gt": float(std),
        "skew": skew(vafT), "map_VAF_T_Gt": map_vafT,
        "sigma_linearized_Gt": sig_lin, "percentiles": pct,
        "min": float(vafT.min()), "max": float(vafT.max()),
        "vafT_all": vafT.tolist(),
    }
    out = DATA / "results" / "uq_sample_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"Saved {out}")

    # ── plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for tr in trajs:
        ax1.plot(times, tr, color="C0", alpha=0.25, lw=0.7)
    if map_vafT is not None and times is not None:
        ax1.axhline(map_vafT, color="k", ls="--", lw=1, label=f"MAP VAF(T)={map_vafT:.0f}")
    ax1.set_xlabel("time (yr)"); ax1.set_ylabel("VAF (Gt)")
    ax1.set_title(f"Posterior VAF trajectories (N={N})"); ax1.legend(fontsize=9); ax1.grid(alpha=0.2)

    ax2.hist(vafT, bins=min(20, max(6, N // 3)), color="C0", alpha=0.8, edgecolor="white")
    ax2.axvline(mean, color="C3", lw=2, label=f"mean={mean:.0f}")
    if map_vafT is not None:
        ax2.axvline(map_vafT, color="k", ls="--", lw=1.5, label=f"MAP={map_vafT:.0f}")
    # linearized Gaussian for contrast
    xs = np.linspace(vafT.min() - 50, vafT.max() + 50, 200)
    g = np.exp(-0.5 * ((xs - (map_vafT if map_vafT else mean)) / sig_lin) ** 2)
    g = g / g.max() * np.histogram(vafT, bins=min(20, max(6, N // 3)))[0].max()
    ax2.plot(xs, g, "C1--", lw=1.5, label=f"linearized N(MAP, {sig_lin:.0f}²)")
    ax2.set_xlabel("VAF(T) (Gt)"); ax2.set_ylabel("count")
    ax2.set_title("Posterior VAF(T): sampled vs linearized"); ax2.legend(fontsize=8); ax2.grid(alpha=0.2)
    fig.tight_layout()
    fp = DATA / "figures" / "uq_sample_vaf.png"
    fig.savefig(str(fp), dpi=200, bbox_inches="tight")
    print(f"Saved {fp}")


if __name__ == "__main__":
    main()
