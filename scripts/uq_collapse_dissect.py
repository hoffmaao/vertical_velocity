r"""Dissect the collapse-tail samples: are runaway trajectories linearly predictable?

For every posterior sample we have the true VAF(T) (sample_vaf_*.npz) and its θ-draw
(posterior_samples*.npz). Compute the LINEARIZED ΔVAF prediction a_k = g·(θ_k − θ_MAP)
(g = taped ∂VAF/∂θ) and compare to the TRUE ΔVAF.

  • If collapse samples have extreme-negative a_k (lie on the linear trend) → collapse is
    just "far along the retreat direction", predictable from the gradient.
  • If collapse samples have TYPICAL a_k but extreme-negative true ΔVAF → collapse is a
    NONLINEAR threshold crossing, invisible to the gradient.

Also reports the mean collapse θ-perturbation (signature) per block. Writes
figures/uq_collapse_dissect.png and prints the verdict.
"""
import glob
import numpy as np
import firedrake as fd
from firedrake import CheckpointFile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
FWD = DATA / "results" / "forward_vaf_50yr.h5"
MAP_FILE = DATA / "mesh" / "inversion_hires_apres_vd1.h5"
VAF_MAP = 42813.4          # MAP forward VAF(T), Gt
COLLAPSE = 42400.0         # Gt threshold separating the collapse tail from the bulk
GT = 917.0 / 1e12


def main():
    with CheckpointFile(str(FWD), "r") as c:
        m = c.load_mesh()
        gA = c.load_function(m, "dVAF_dthetaA"); gC = c.load_function(m, "dVAF_dthetaC")
    n = gA.function_space().dim()
    g_full = np.concatenate([gA.dat.data_ro.copy(), gC.dat.data_ro.copy()])
    with CheckpointFile(str(MAP_FILE), "r") as c:
        mm = c.load_mesh()
        mapA = c.load_function(mm, "log_fluidity").dat.data_ro.copy()
        mapC = c.load_function(mm, "log_friction").dat.data_ro.copy()

    a_lin, vafT, labels, dthetas = [], [], [], []
    for L in ["", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]:
        sf = DATA / "results" / (f"posterior_samples_{L}.npz" if L else "posterior_samples.npz")
        if not sf.exists():
            continue
        s = np.load(str(sf)); tA, tC = s["theta_A"], s["theta_C"]
        for i in range(tA.shape[0]):
            vf = DATA / "results" / f"sample_vaf_{L}{i:03d}.npz"
            if not vf.exists():
                continue
            v = float(np.load(str(vf))["vafT"])
            dA = tA[i] - mapA; dC = tC[i] - mapC
            a = float(g_full @ np.concatenate([dA, dC])) * GT
            a_lin.append(a); vafT.append(v); labels.append(f"{L}{i:03d}")
            dthetas.append((dA, dC))
    a_lin = np.array(a_lin); vafT = np.array(vafT)
    dvaf_true = vafT - VAF_MAP
    col = vafT < COLLAPSE
    N = vafT.size
    print(f"N={N} samples paired (θ ↔ VAF); {col.sum()} in collapse tail (<{COLLAPSE:.0f})\n")

    # correlation on the BULK only (collapses would dominate otherwise)
    bulk = ~col
    r_bulk = np.corrcoef(a_lin[bulk], dvaf_true[bulk])[0, 1]
    print(f"BULK: corr(linear a_k, true ΔVAF) = {r_bulk:+.2f}  "
          f"(slope {np.polyfit(a_lin[bulk], dvaf_true[bulk],1)[0]:+.2f})")
    print(f"linear a_k:  bulk range [{a_lin[bulk].min():+.0f}, {a_lin[bulk].max():+.0f}] Gt, "
          f"std {a_lin[bulk].std():.0f}\n")
    print(f"{'sample':>8} {'true ΔVAF':>10} {'linear a_k':>11} {'a_k %ile':>9}  verdict")
    print("-" * 60)
    for j in np.where(col)[0]:
        pctile = 100.0 * (a_lin < a_lin[j]).mean()
        verdict = "PREDICTABLE (extreme a_k)" if pctile < 5 else "NONLINEAR (typical a_k!)"
        print(f"{labels[j]:>8} {dvaf_true[j]:>10.0f} {a_lin[j]:>11.0f} {pctile:>8.1f}%  {verdict}")

    # collapse θ-signature: mean perturbation over collapse samples
    if col.sum():
        cdA = np.mean([dthetas[j][0] for j in np.where(col)[0]], axis=0)
        cdC = np.mean([dthetas[j][1] for j in np.where(col)[0]], axis=0)
        print(f"\nmean collapse perturbation: "
              f"δθ_A max|·|={np.abs(cdA).max():.2f} (rms {np.sqrt((cdA**2).mean()):.3f}), "
              f"δθ_C max|·|={np.abs(cdC).max():.2f} (rms {np.sqrt((cdC**2).mean()):.3f})")
        # save mean collapse perturbation as fields for spatial plotting
        Q = gA.function_space()
        fA = fd.Function(Q, name="collapse_dthetaA"); fA.dat.data[:] = cdA
        fC = fd.Function(Q, name="collapse_dthetaC"); fC.dat.data[:] = cdC
        with CheckpointFile(str(DATA / "results" / "collapse_signature.h5"), "w") as c:
            c.save_mesh(m); c.save_function(fA); c.save_function(fC)
        print(f"Saved mean collapse θ-signature → results/collapse_signature.h5")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(a_lin[bulk], dvaf_true[bulk], s=14, alpha=0.5, color="C0", label="bulk")
    ax.scatter(a_lin[col], dvaf_true[col], s=80, color="C3", marker="*", label="collapse", zorder=5)
    for j in np.where(col)[0]:
        ax.annotate(labels[j], (a_lin[j], dvaf_true[j]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    lo, hi = a_lin.min(), a_lin.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="ΔVAF = a_k (perfect linear)")
    ax.set_xlabel("linearized prediction  a_k = g·δθ  [Gt]")
    ax.set_ylabel("true ΔVAF(50yr)  [Gt]")
    ax.set_title(f"Is collapse linearly predictable?  (N={N})")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fp = DATA / "figures" / "uq_collapse_dissect.png"
    fig.savefig(str(fp), dpi=200, bbox_inches="tight")
    print(f"Saved {fp}")


if __name__ == "__main__":
    main()
