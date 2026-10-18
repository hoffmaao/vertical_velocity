r"""Plot the time series of one or more prognostic forward runs (forward.py).

Reads results/forward_<name>_<T>yr_ts.npz for every run given and draws, on
shared axes, the volume above flotation relative to t = 0, the sub-shelf melt
and the peak speed, one line per run. A run is named by its forward.py --map
value; several runs are compared on one figure.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/plot_forward.py --T 1 \
        sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km \
        sweep_hybrid_weertman_d1e-05_L2km_thwaites_fixed_tau5km

Writes figures/forward_<T>yr<suffix>.png, the suffix being --suffix or, for a
single run, that run's name. With --tag the tagged runs
forward_<name>_<T>yr_<tag>_ts.npz are read instead.
"""
import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent
FIG_DIR = DATA / "figures"
# Same convention as plot_retreat_uq.py: colour = model, line style = friction law.
MODEL_COLOUR = {"ssa": "C3", "hybrid": "C0"}
FRICTION_STYLE = {"rc": "-", "weertman": "--", "schoof": "-."}


def style(name):
    r"""Line colour and style for a run, or matplotlib's defaults if the name does not parse."""
    m = re.match(r"(?:sweep_)?(?P<model>ssa|hybrid)_(?P<friction>[a-z]+)", name)
    if not m:
        return {}
    return {"color": MODEL_COLOUR[m.group("model")],
            "ls": FRICTION_STYLE.get(m.group("friction"), ":")}


def short(name):
    r"""``sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km`` -> ``hybrid_rc d1e-05 L2km``."""
    m = re.match(r"(?:sweep_)?(?P<tag>[a-z]+_[a-z]+(?:_noapres)?(?:_bilap)?)(?:_(?P<pt>d[0-9e.+-]+_L[0-9.]+km))?", name)
    return (f"{m.group('tag')} {m.group('pt').replace('_', ' ')}" if m and m.group("pt")
            else (m.group("tag") if m else name))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="forward.py --map names")
    ap.add_argument("--T", type=float, default=50.0, help="run length the files were written with")
    ap.add_argument("--tag", default="",
                    help="forward file tag: read forward_<name>_<T>yr_<tag>_ts.npz")
    ap.add_argument("--suffix", default=None, help="figure name suffix (default: the run name)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    forcings = set()
    for name in args.runs:
        d = np.load(str(DATA / "results" / f"forward_{name}_{args.T:g}yr{'_' + args.tag if args.tag else ''}_ts.npz"), allow_pickle=True)
        t, vaf = d["times"], d["vaf_Gt"]
        label = short(name)
        forcings.add(str(d["forcing"]) if "forcing" in d.files else "unrecorded")
        if int(d["steps_completed"]) < round(float(d["T_years"]) / float(d["dt"])):
            label += f" (stopped at {t[-1]:.2f} yr)"
        kw = style(name)
        axes[0].plot(t, vaf - vaf[0], lw=1.6, label=label, **kw)
        axes[1].plot(t[1:], d["melt_Gt_per_yr"][1:], lw=1.6, **kw)
        axes[2].plot(t, d["u_max_m_per_yr"], lw=1.6, **kw)
        print(f"{name}: VAF {vaf[0]:.1f} -> {vaf[-1]:.1f} Gt ({vaf[-1] - vaf[0]:+.1f}) in {t[-1]:g} yr; "
              f"melt {d['melt_Gt_per_yr'][1]:.1f} -> {d['melt_Gt_per_yr'][-1]:.1f} Gt/yr; "
              f"u_max {d['u_max_m_per_yr'][0]:.0f} -> {d['u_max_m_per_yr'][-1]:.0f} m/yr; "
              f"{int(d['n_rescue'])} rescues, {int(d['n_subcycled_steps'])} subcycled steps",
              flush=True)

    axes[0].axhline(0, color="0.6", lw=0.8)
    axes[0].set_ylabel("VAF - VAF(0)  (Gt)")
    axes[0].set_title("Volume above flotation")
    axes[1].set_ylabel("Gt/yr")
    axes[1].set_title("Sub-shelf melt (Recinos et al. 2023 Eq. 5)")
    axes[2].set_ylabel("m/yr")
    axes[2].set_title("Peak depth-averaged speed")
    for ax in axes:
        ax.set_xlabel("t (yr)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Prognostic forward, {args.T:g} yr, forcing: {'; '.join(sorted(forcings))}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    suffix = args.suffix if args.suffix is not None else (
        args.runs[0] if len(args.runs) == 1 else "compare")
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"forward_{args.T:g}yr_{suffix}.png"
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
