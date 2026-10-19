r"""VAF loss and loss rate over time with linearized posterior uncertainty, SSA vs hybrid.

Reads, for each MAP given, the forward time series results/forward_<map>_<T>yr_ts.npz
and the time-resolved UQ of a forward.py --gradient-times run: by default the
untruncated results/uq_<map>_<T>yr_pcg_summary.json (uq_vaf_pcg.py), with
--uq eig the K-mode results/uq_<map>_<T>yr_K<K>_summary.json (uq_vaf.py). Draws

  (a) VAF(t) - VAF(0) with the +/- 2 sigma_post band, the band interpolated
      linearly between the gradient times (sigma = 0 at t = 0);
  (b) the mean VAF loss rate over each interval between gradient times, with
      its +/- 2 sigma_post bar (the rate's own variance, correlations included);
  (c) sigma_post(t) itself, and the prior sigma faint, which shows how fast
      the parametric uncertainty grows and how much the data take out of it.

Colour = model (SSA / hybrid), line style = friction law. With --tag the files
of a tagged forward and its UQ are read instead,
forward_<map>_<T>yr_<tag>_ts.npz and uq_<map>_<T>yr_<uq>_<tag>_summary.json.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/plot_retreat_uq.py --T 50 \
        sweep_ssa_rc_d1e-05_L2km_thwaites_fixed_tau5km sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km

Writes figures/retreat_uq_<T>yr<suffix>.png.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent
FIG_DIR = DATA / "figures"
MODEL_COLOUR = {"ssa": "C3", "hybrid": "C0"}
MODEL_LABEL = {"ssa": "SSA (velocity only)", "hybrid": "hybrid (velocity + ApRES)"}
FRICTION_STYLE = {"rc": "-", "weertman": "--", "schoof": "-."}
FRICTION_LABEL = {"rc": "regularized Coulomb", "weertman": "Weertman",
                  "schoof": "Schoof (yield stress)"}


def parse(name):
    m = re.match(r"(?:sweep_)?(?P<model>ssa|hybrid)_(?P<friction>[a-z]+)", name)
    return m.group("model"), m.group("friction")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="MAP names (the forward.py --map values)")
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--uq", choices=["pcg", "eig"], default="pcg",
                    help="pcg: uq_vaf_pcg.py (no truncation); eig: uq_vaf.py with --K modes")
    ap.add_argument("--K", type=int, default=1500, help="with --uq eig: the eigendecomposition used")
    ap.add_argument("--tag", default="",
                    help="forward/UQ file tag: read forward_<map>_<T>yr_<tag>_ts.npz and its UQ")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, (ax_v, ax_r, ax_s) = plt.subplots(1, 3, figsize=(19, 5.6))
    n_runs = len(args.runs)
    priors, forcings = set(), set()
    for k, name in enumerate(args.runs):
        model, friction = parse(name)
        col, ls = MODEL_COLOUR[model], FRICTION_STYLE.get(friction, ":")
        label = f"{MODEL_LABEL[model]}, {FRICTION_LABEL.get(friction, friction)}"
        stem = f"{name}_{args.T:g}yr"
        tag = f"_{args.tag}" if args.tag else ""
        ts = np.load(str(DATA / "results" / f"forward_{stem}{tag}_ts.npz"), allow_pickle=True)
        forcings.add(str(ts["forcing"]) if "forcing" in ts.files else "unrecorded")
        uq = "pcg" if args.uq == "pcg" else f"K{args.K}"
        with open(DATA / "results" / f"uq_{stem}_{uq}{tag}_summary.json") as f:
            s = json.load(f)
        priors.add((s["delta_A"], s["gamma_A"], s["delta_C"], s["gamma_C"]))
        if "times" not in s:
            raise SystemExit(f"{stem}: no time-resolved UQ; rerun forward.py with --gradient-times "
                             f"and then uq_vaf_pcg.py / uq_vaf.py")
        t, dv = ts["times"], ts["vaf_Gt"] - ts["vaf_Gt"][0]
        tg = np.r_[0.0, s["times"]]
        sig = np.interp(t, tg, np.r_[0.0, s["sigma_post_Gt_t"]])
        ax_v.plot(t, dv, ls, color=col, lw=1.8, label=label)
        ax_v.fill_between(t, dv - 2 * sig, dv + 2 * sig, color=col, alpha=0.18, lw=0)

        # Rates: one horizontal step per interval, error bar at its midpoint,
        # offset slightly per run so overlapping bars stay readable.
        edges = tg
        rate = np.array(s["rate_Gt_per_yr"])
        srate = np.array(s["sigma_post_rate"])
        ax_r.step(edges, np.r_[rate, rate[-1]], ls, where="post", color=col, lw=1.6, label=label)
        width = np.diff(edges).min()
        off = (k - (n_runs - 1) / 2) * 0.08 * width
        ax_r.errorbar(np.array(s["rate_t_mid"]) + off, rate, yerr=2 * srate, fmt="none",
                      ecolor=col, elinewidth=1.6, capsize=3)

        ax_s.plot(tg, np.r_[0.0, s["sigma_post_Gt_t"]], ls, color=col, lw=1.8, marker="o", ms=3,
                  label=label)
        ax_s.plot(tg, np.r_[0.0, s["sigma_prior_Gt_t"]], ls, color=col, lw=1, alpha=0.35)

        print(f"{label:52s} dVAF({args.T:g}) = {dv[-1]:+8.1f} +/- {2 * s['sigma_post_Gt_t'][-1]:5.1f} Gt "
              f"(2 sigma); last-interval rate {rate[-1]:+7.2f} +/- {2 * srate[-1]:.2f} Gt/yr", flush=True)

    ax_v.axhline(0, color="0.6", lw=0.8)
    ax_v.set(xlabel="t (yr)", ylabel="VAF(t) - VAF(0)  (Gt)",
             title="(a) VAF change, $\\pm 2\\sigma_{\\mathrm{post}}$ band")
    ax_r.axhline(0, color="0.6", lw=0.8)
    ax_r.set(xlabel="t (yr)", ylabel="mean d(VAF)/dt over the interval  (Gt/yr)",
             title="(b) VAF loss rate, $\\pm 2\\sigma_{\\mathrm{post}}$")
    ax_s.set(xlabel="t (yr)", ylabel="$\\sigma$ of VAF(t) - VAF(0)  (Gt)",
             title="(c) posterior $\\sigma$ (bold) and prior $\\sigma$ (faint)")
    for ax in (ax_v, ax_r, ax_s):
        ax.grid(True, alpha=0.3)
    ax_v.legend(fontsize=9, loc="lower left")
    prior = "; ".join(f"prior $\\delta$={dA:.0e}, ELL={np.sqrt(gA / dA) / 1e3:g} km"
                      + ("" if (dA, gA) == (dC, gC) else f" ($\\theta_C$: {dC:.0e}, {np.sqrt(gC / dC) / 1e3:g} km)")
                      for dA, gA, dC, gC in sorted(priors))
    how = "GN Hessian, PCG, no truncation" if args.uq == "pcg" else f"Isaac et al. 2015, K={args.K}"
    fig.suptitle(f"Thwaites VAF projections with linearized posterior uncertainty "
                 f"({how}; {prior}; forcing: {'; '.join(sorted(forcings))})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"retreat_uq_{args.T:g}yr{args.suffix}.png"
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
