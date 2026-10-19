r"""Summary figure for the linearized VAF uncertainty from uq_vaf.py.

For each (MAP, K) given, reads results/forward_<map>_<T>yr{.h5,_ts.npz} and
results/uq_<map>_<T>yr_K<K>_{summary.json,contribs.txt}, and draws

  (a) the VAF trajectories with the linearized +/- 2 sigma_post bar at T,
      next to the +/- 2 sigma_prior bar, one colour per MAP;
  (b) the cumulative variance reduction against the number of modes, which
      shows whether K reached the data-informed modes (a curve still rising
      at K means sigma_post is an upper bound);
  (c, d) dVAF(T)/dtheta_A and dVAF(T)/dtheta_C for the first MAP, the
      sensitivity maps the posterior variance is projected on.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/plot_uq_vaf.py --T 50 \
        sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km:1500 \
        sweep_hybrid_rc_d1e-06_L1km_thwaites_fixed_tau5km:3000

Each argument is <map>:<K>. Writes figures/uq_vaf_<T>yr<suffix>.png.
"""
import argparse
import json
import re
from pathlib import Path

import firedrake
import matplotlib
import numpy as np
from firedrake import CheckpointFile, Function, TestFunction, assemble, dx

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402

DATA = Path(__file__).resolve().parent.parent
FIG_DIR = DATA / "figures"
GT_PER_M3 = 917.0 / 1e12


def short(name):
    r"""``sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km`` -> ``hybrid_rc d1e-05 L2km``."""
    m = re.match(r"(?:sweep_)?(?P<tag>[a-z]+_[a-z]+(?:_noapres)?)(?:_(?P<pt>d[0-9e.+-]+_L[0-9.]+km))?", name)
    return (f"{m.group('tag')} {m.group('pt').replace('_', ' ')}" if m and m.group("pt")
            else (m.group("tag") if m else name))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="<map>:<K> pairs")
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    runs = [(r.rsplit(":", 1)[0], int(r.rsplit(":", 1)[1])) for r in args.runs]

    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)
    ax_t, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    colours = {}
    for k, (name, K) in enumerate(runs):
        stem = f"{name}_{args.T:g}yr"
        ts = np.load(str(DATA / "results" / f"forward_{stem}_ts.npz"), allow_pickle=True)
        with open(DATA / "results" / f"uq_{stem}_K{K}_summary.json") as f:
            s = json.load(f)
        contrib = np.loadtxt(str(DATA / "results" / f"uq_{stem}_K{K}_contribs.txt"))
        col = colours.setdefault(name, f"C{len(colours)}")
        t, dv = ts["times"], ts["vaf_Gt"] - ts["vaf_Gt"][0]
        if name not in [n for n, _ in runs[:k]]:
            ax_t.plot(t, dv, color=col, lw=1.6, label=short(name))
            ax_t.errorbar(t[-1] + 0.6, dv[-1], yerr=2 * s["sigma_prior_Gt"], color=col,
                          alpha=0.35, capsize=3, lw=1.2)
        x = t[-1] + 1.4 + 0.5 * sum(1 for n, _ in runs[:k] if n == name)
        ax_t.errorbar(x, dv[-1], yerr=2 * s["sigma_post_Gt"], color=col, capsize=3, lw=2)
        ls = "-" if K == max(KK for n, KK in runs if n == name) else ":"
        ax_c.plot(np.arange(1, contrib.shape[0] + 1), 100 * contrib[:, 4], ls, color=col,
                  label=f"{short(name)}, K={K}: $\\sigma_{{post}}$ = {s['sigma_post_Gt']:.1f} Gt "
                        f"(prior {s['sigma_prior_Gt']:.1f})")
        print(f"{short(name):26s} K={K:5d}: dVAF({args.T:g}) = {dv[-1]:+8.1f} Gt, sigma_prior "
              f"{s['sigma_prior_Gt']:6.1f} Gt, sigma_post {s['sigma_post_Gt']:6.1f} Gt, "
              f"reduction {100 * s['variance_reduction_frac']:.1f} %", flush=True)

    ax_t.axhline(0, color="0.6", lw=0.8)
    ax_t.set(xlabel="t (yr)", ylabel="VAF - VAF(0)  (Gt)",
             title="(a) VAF projections; bars at T: $\\pm 2\\sigma$ prior (faint) and posterior")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=9, loc="lower left")
    ax_c.set(xlabel="modes used", ylabel="variance reduction (% of prior variance)",
             title="(b) cumulative variance reduction (dotted: smaller K)", xscale="log")
    ax_c.grid(True, alpha=0.3, which="both")
    ax_c.legend(fontsize=8, loc="upper left")

    name0 = runs[0][0]
    with CheckpointFile(str(DATA / "results" / f"forward_{name0}_{args.T:g}yr.h5"), "r") as chk:
        mesh = chk.load_mesh()
        grads = [chk.load_function(mesh, n) for n in ("dVAF_dthetaA", "dVAF_dthetaC")]
    # The gradients are cotangents (one value per dof, scaling with the dof's
    # area); divide by the lumped mass to plot a density, the VAF change per
    # unit theta per unit area, i.e. metres of ice-above-flotation equivalent.
    area = assemble(TestFunction(grads[0].function_space()) * dx).dat.data_ro
    for j, (g, sym) in enumerate(zip(grads, "AC")):
        ax = fig.add_subplot(gs[1, j])
        dens = Function(g.function_space())
        dens.dat.data[:] = g.dat.data_ro / area
        vmax = np.percentile(np.abs(dens.dat.data_ro), 99)
        tc = firedrake.tripcolor(dens, axes=ax, cmap="RdBu_r",
                                 norm=SymLogNorm(linthresh=vmax / 100, vmin=-vmax, vmax=vmax))
        cb = fig.colorbar(tc, ax=ax, shrink=0.85)
        cb.set_label("m (VAF per m$^2$, per unit $\\theta$)")
        total = g.dat.data_ro.sum() * GT_PER_M3
        ax.set_title(f"({'cd'[j]}) $\\partial$VAF({args.T:g} yr)/$\\partial\\theta_{sym}$, {short(name0)}\n"
                     f"(uniform unit $\\theta_{sym}$ shift: {total:+.0f} Gt)", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Linearized posterior uncertainty of VAF({args.T:g} yr) "
                 "(Isaac et al. 2015; gradient at fixed apparent mass balance)", fontsize=13)
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"uq_vaf_{args.T:g}yr{args.suffix}.png"
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
