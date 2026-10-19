r"""Plot a Hessian eigendecomposition from eigendec.py: spectrum, data influence, leading modes.

For one decomposition, results/eigendec_<name>_K<K>.{npz,json} and the modes in
mesh/eigendec_<name>_K<K>.h5:

  (a) the eigenvalue spectrum lambda_i with a power-law fit to its tail,
      extrapolated to where lambda crosses 100, 10 and 1 (D = 0.99, 0.91, 0.5),
      i.e. how many modes the data constrain beyond the prior and whether K
      reached them;
  (b) the data influence D_i = lambda_i / (1 + lambda_i) and the cumulative
      n_eff = sum D_i, the effective number of constrained parameters;
  (c) the posterior standard-deviation weight 1 / sqrt(1 + lambda_i) per mode;
  (d) the theta_A and theta_C patterns of the leading modes.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/plot_eigendec.py \
        --map sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km --K 1500

    -> figures/eigendec_<name>_K<K>.png

With several --map values (and one --K) the spectra, D_i and cumulative n_eff
are overlaid instead, one line per decomposition:

    -> figures/eigendec_compare_K<K><suffix>.png
"""
import argparse
import json
import re
from pathlib import Path

import firedrake
import matplotlib
import numpy as np
from firedrake import CheckpointFile, Function, FunctionSpace

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

DATA = Path(__file__).resolve().parent.parent
FIG_DIR = DATA / "figures"
THRESHOLDS = ((100.0, "D = 0.99"), (10.0, "D = 0.91"), (1.0, "D = 0.5"))
N_MODES_SHOWN = 4
FIT_FROM = 10          # power law fitted to modes FIT_FROM.. (the leading ones are special)


def short(name):
    r"""``sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km`` -> ``hybrid_rc d1e-05 L2km``."""
    m = re.match(r"(?:sweep_)?(?P<tag>[a-z]+_[a-z]+(?:_noapres)?)(?:_(?P<pt>d[0-9e.+-]+_L[0-9.]+km))?", name)
    return (f"{m.group('tag')} {m.group('pt').replace('_', ' ')}" if m and m.group("pt")
            else (m.group("tag") if m else name))


def load(stem):
    eig = np.load(str(DATA / "results" / f"{stem}.npz"))
    with open(DATA / "results" / f"{stem}.json") as f:
        meta = json.load(f)
    return eig["eigenvalues"], meta


def power_law(vals):
    r"""lambda ~ lam0 (i+1)^-b fitted to the tail; returns (lam0, b, r2, crossings)."""
    i = np.arange(vals.size)
    lo = min(FIT_FROM, max(vals.size - 5, 0))
    x, y = np.log(i[lo:] + 1), np.log(vals[lo:])
    slope, icpt = np.polyfit(x, y, 1)
    b, lam0 = -slope, np.exp(icpt)
    r2 = 1 - np.sum((y - (slope * x + icpt)) ** 2) / np.sum((y - y.mean()) ** 2)
    cross = {t: (lam0 / t) ** (1.0 / b) - 1 if b > 0 else np.inf for t, _ in THRESHOLDS}
    return lam0, b, r2, cross


def spectrum_axes(ax_lam, ax_D, ax_w, vals, label=None, color=None, fit=True):
    i = np.arange(vals.size)
    D = vals / (1 + vals)
    ax_lam.semilogy(i, vals, ".-", ms=3, lw=1, color=color, label=label)
    if fit:
        lam0, b, r2, cross = power_law(vals)
        ii = np.arange(1, int(min(max(cross[1.0], vals.size) * 1.2, 1e6)) + 1)
        ax_lam.semilogy(ii - 1, lam0 * ii ** (-b), "--", lw=1, color="0.4",
                        label=f"tail fit $\\lambda \\approx {lam0:.2g}\\,(i+1)^{{-{b:.2f}}}$ ($R^2$={r2:.3f})")
        for t, dl in THRESHOLDS:
            ax_lam.axhline(t, color="0.75", lw=0.8)
            ax_lam.annotate(f"$\\lambda$={t:g} ({dl}): i $\\approx$ {cross[t]:,.0f}",
                            (0.02, t), xycoords=("axes fraction", "data"), fontsize=8,
                            va="bottom", color="0.3")
    ax_D.plot(i, D, lw=1.2, color=color, label=label)
    ax_D2 = ax_D.twinx() if not hasattr(ax_D, "_neff_axis") else ax_D._neff_axis
    ax_D._neff_axis = ax_D2
    ax_D2.plot(i, np.cumsum(D), ":", lw=1.6, color=color or "r")
    ax_w.semilogy(i, 1 / np.sqrt(1 + vals), lw=1.2, color=color, label=label)
    return D


def finish_spectrum_axes(ax_lam, ax_D, ax_w, n):
    ax_lam.set(xlabel="mode $i$", ylabel="$\\lambda_i$", title="(a) spectrum of $H w = \\lambda A w$")
    ax_lam.grid(True, alpha=0.25, which="both")
    ax_lam.legend(fontsize=8, loc="upper right")
    ax_D.set(xlabel="mode $i$", ylabel="$D_i = \\lambda_i/(1+\\lambda_i)$  (solid)",
             title="(b) data influence and cumulative $n_{\\mathrm{eff}}$", xlim=(-1, n))
    ax_D._neff_axis.set_ylabel("$\\sum_{j \\leq i} D_j$  (dotted)")
    ax_D.grid(True, alpha=0.25)
    ax_w.set(xlabel="mode $i$", ylabel="$1/\\sqrt{1+\\lambda_i}$",
             title="(c) posterior std weight per mode", xlim=(-1, n))
    ax_w.grid(True, alpha=0.25, which="both")


def plot_one(name, K, dpi):
    stem = f"eigendec_{name}_K{K}"
    vals, meta = load(stem)
    n = vals.size
    with CheckpointFile(str(DATA / "mesh" / f"{stem}.h5"), "r") as chk:
        mesh = chk.load_mesh()
        Q = FunctionSpace(mesh, "CG", 1)
        n_show = min(N_MODES_SHOWN, n)
        modes = []
        for i in range(n_show):
            mode = chk.load_function(mesh, name=f"mode_{i:04d}")
            mA, mC = Function(Q), Function(Q)
            mA.dat.data[:] = mode.sub(0).dat.data_ro
            mC.dat.data[:] = mode.sub(1).dat.data_ro
            modes.append((mA, mC))

    fig = plt.figure(figsize=(5.6 * max(n_show, 3), 11.5))
    gs_top = fig.add_gridspec(1, 3, top=0.88, bottom=0.58, wspace=0.45, left=0.05, right=0.95)
    axes = [fig.add_subplot(gs_top[k]) for k in range(3)]
    D = spectrum_axes(*axes, vals)
    finish_spectrum_axes(*axes, n)
    axes[0].set_xlim(-1, n)

    gs_bot = fig.add_gridspec(2, n_show, top=0.48, bottom=0.03, hspace=0.25, wspace=0.3,
                              left=0.05, right=0.95)
    for i, (mA, mC) in enumerate(modes):
        for row, (f, sym) in enumerate(((mA, "A"), (mC, "C"))):
            ax = fig.add_subplot(gs_bot[row, i])
            vmax = max(abs(f.dat.data_ro).max(), 1e-12)
            tc = firedrake.tripcolor(f, axes=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            fig.colorbar(tc, cax=make_axes_locatable(ax).append_axes("right", size="4%", pad=0.03))
            ax.set_title(f"mode {i}: $\\theta_{sym}$" + (f"   $\\lambda$ = {vals[i]:.3g}" if row == 0 else ""),
                         fontsize=10)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)

    prior = (f"$\\delta_A$={meta['delta_A']:.0e}, ELL$_A$={meta['ELL_A'] / 1e3:g} km; "
             f"$\\delta_C$={meta['delta_C']:.0e}, ELL$_C$={meta['ELL_C'] / 1e3:g} km")
    fig.suptitle(f"GN Hessian eigendecomposition at {short(name)}  ({meta['mesh']}, "
                 f"{meta['method']}, K={K}: {n} positive modes, $n_{{\\mathrm{{eff}}}}$ = {D.sum():.1f})\n"
                 f"prior {prior}", fontsize=12, y=0.985)
    out = FIG_DIR / f"{stem}.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    lam0, b, r2, cross = power_law(vals)
    print(f"{stem}: {n} modes, lambda in [{vals[-1]:.3e}, {vals[0]:.3e}], n_eff={D.sum():.1f}; "
          f"tail ~ (i+1)^-{b:.2f} (R2 {r2:.3f}); lambda=1 at i ~ {cross[1.0]:,.0f}", flush=True)
    print(f"Saved {out}", flush=True)


def plot_compare(names, K, suffix, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    n_max = 0
    for k, name in enumerate(names):
        vals, meta = load(f"eigendec_{name}_K{K}")
        n_max = max(n_max, vals.size)
        D = spectrum_axes(*axes, vals, label=f"{short(name)} ($n_{{\\mathrm{{eff}}}}$={vals.size and (vals / (1 + vals)).sum():.0f})",
                          color=f"C{k}", fit=False)
        print(f"{name}: {vals.size} modes, lambda_max {vals[0]:.3e}, lambda_min {vals[-1]:.3e}, "
              f"n_eff {D.sum():.1f}", flush=True)
    for t, dl in THRESHOLDS:
        axes[0].axhline(t, color="0.75", lw=0.8)
    finish_spectrum_axes(*axes, n_max)
    axes[0].set_xlim(-1, n_max)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"GN Hessian spectra, K={K}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / f"eigendec_compare_K{K}{suffix}.png"
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", nargs="+", required=True, help="eigendec.py --map name(s)")
    ap.add_argument("--K", type=int, default=1500, help="eigendec.py --K")
    ap.add_argument("--suffix", default="", help="comparison figure name suffix")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    FIG_DIR.mkdir(exist_ok=True)
    if len(args.map) == 1:
        plot_one(args.map[0], args.K, args.dpi)
    else:
        plot_compare(args.map, args.K, args.suffix, args.dpi)


if __name__ == "__main__":
    main()
