r"""Plot the (delta, ELL) regularization surface produced by sweep_regularization.py.

Reads every results/map_sweep_*.json it can find, so it is safe to run while the
sweep is still going: points that do not exist yet are drawn as empty cells, not
as low values.

The grid axes come from sweep_regularization (DELTAS, ELLS, EXCLUDE), so a cell
is blank only because the point has not finished, or because it was excluded.
Points that stalled on failed forward solves (sweep_regularization.stalled) are
left out of the L-curves, hatched like an excluded cell and still listed in the
text report. The label says what the stall means for that sweep: in the shared
and friction sweeps it happens at the weak-friction end, where theta_C is no
longer constrained and runs away, so the cell is "unphys." (outside the
physical range). In the fluidity sweep it happens at the STRONG-fluidity end
(delta_A >= 1e-3, ELL_A >= 1 km, hybrid Weertman): theta_A is frozen near the
prior mean, the misfit is still enormous, the optimizer drives theta_C up to
+35..41 in a dozen iterations to compensate, the Newton solve fails there and
the line search cannot recover. That is a failed optimization, not a MAP in an
unphysical regime (the RC law converges at the same points with theta_C near
10), so the cell is marked "failed" instead.

For each configuration it draws

  (a) the L-curve along the prior *strength*: misfit against the prior term as
      delta varies at fixed ELL. This is the axis the classic elbow criterion is
      about, so the max-curvature elbow is marked here;
  (b) the same plane traversed along the *correlation length*, one line per
      delta, which shows how much ELL moves the MAP at fixed strength;
  (c, d) chi^2 per observation for velocity and ApRES, on a diverging scale
      centred on chi^2 = 1: blue is fitting inside the observation errors
      (overfitting), red is fitting outside them;
  (e) max |theta| over the domain, which flags where the prior has stopped
      constraining the controls and the MAP is running away;
  (f) the L-BFGS iteration count, with points that did not converge outlined.

and finally a cross-configuration figure overlaying every L-curve family.

    python scripts/plot_lsurface.py
    python scripts/plot_lsurface.py --config hybrid_weertman
    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/plot_lsurface.py

--mode friction / --mode fluidity plot a single-control sweep (see
sweep_regularization.py): the grid is then the swept control's (delta, ELL), the
L-curve's x axis is that control's own prior term R_C or R_A, and the runaway
panel shows that control's max |theta|. --fixed-delta/--fixed-ell pick which
held value of the other control to plot.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/plot_lsurface.py \
        --mode friction --config hybrid_weertman

VV_MESH and VV_TAU_SMOOTH select the sweep exactly as they do for
sweep_regularization.py: only points whose names carry that run suffix are
read, and the figures carry it too, so sweeps on different meshes or anchors
are never mixed on one surface.

Note on reading the elbow (Recinos et al. 2023): the elbow is the best
*data-fitting* compromise, not the right prior for UQ. It tends to be too strong
a prior, which shrinks the posterior variance; expect to want a delta below the
elbow when propagating uncertainty. Panel (e) bounds how far below: past the
point where max |theta| runs away, the MAP is no longer physical.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from sweep_regularization import (
    CONFIG_TAGS, DELTAS, ELLS, EXCLUDE, HELD, MODES, RUN_SUFFIX, SWEPT, mode_infix,
    mode_postfix, stalled,
)

DATA = Path(__file__).resolve().parent.parent
FIG_DIR = DATA / "figures"
def name_re(mode, fixed):
    r"""Result-file stems of one sweep, and nothing else.

    The tag alternation is the closed list of configuration tags, so the
    single-control names (which contain a second _d..._L..km block) can never be
    misread as shared-prior points or vice versa.
    """
    tags = "|".join(sorted(map(re.escape, CONFIG_TAGS), key=len, reverse=True))
    return re.compile(rf"^map_sweep_(?P<tag>{tags}){re.escape(mode_infix(mode))}"
                      r"_d(?P<delta>[0-9.e+-]+)_L(?P<ell>[0-9.]+)km"
                      + re.escape(mode_postfix(mode, fixed) + RUN_SUFFIX) + "$")


def keys(mode):
    r"""(delta key, ELL key, prior-term key, theta-range keys, axis symbol) per mode."""
    if mode == "shared":
        return "delta", "ELL", "R", ("theta_A_range", "theta_C_range"), ""
    X = SWEPT[mode]
    return f"delta_{X}", f"ELL_{X}", f"R_{X}", (f"theta_{X}_range",), f"_{X}"

MISSING_COLOR = "0.93"
FLAG_COLOR = "#d62728"


EXCLUDED_LABEL = "excl."
# How a stalled point is labelled and described, per sweep mode (module docstring).
STALLED_LABEL = {"shared": "unphys.", "friction": "unphys.", "fluidity": "failed"}
STALLED_NOTE = {
    "shared": "stalled on failed forward solves, outside the physical range",
    "friction": "stalled on failed forward solves, outside the physical range",
    "fluidity": "optimizer stalled on failed forward solves at a strong fluidity prior "
                "(theta_C transient), not a MAP",
}


def excluded_cells(tag, mode):
    r"""{(delta, ell_km): label} that sweep_regularization deliberately does not run."""
    out = {}
    if mode != "shared":
        return out
    for model, friction, no_apres, delta, ell in EXCLUDE:
        apres = model == "hybrid" and not no_apres
        etag = f"{model}_{friction}" + ("_noapres" if model == "hybrid" and not apres else "")
        if etag == tag:
            out[(delta, ell / 1000.0)] = EXCLUDED_LABEL
    return out


def load_points(mode, fixed):
    r"""{config tag: {(delta, ell_km): record}} from the sweep result jsons."""
    out = defaultdict(dict)
    rx = name_re(mode, fixed)
    kd, kl, kr, kth, _ = keys(mode)
    for path in sorted((DATA / "results").glob("map_sweep_*.json")):
        m = rx.match(path.stem)
        if not m:
            continue
        try:
            d = json.load(open(path))
        except ValueError:
            print(f"  skipping unreadable {path.name}", flush=True)
            continue
        if d.get(kd) is None or d.get(kr) is None:
            # Shared sweeps need delta/ELL (null when the two priors differ);
            # single-control sweeps need the per-control fields, which runs made
            # before the priors were split do not have.
            print(f"  skipping {path.name}: no {kd}/{kr} in its json", flush=True)
            continue
        rec = {"delta": float(d[kd]), "ell_km": float(d[kl]) / 1000.0,
               "J_vel": d["J_vel"], "J_apr": d["J_apr"], "R": d[kr],
               "misfit": d["J_vel"] + d["J_apr"],
               "chi2_vel": d["chi2_vel_per_obs"], "chi2_apr": d["chi2_apr_per_obs"],
               "theta_max": max(max(abs(v) for v in d[k]) for k in kth),
               "iterations": d["iterations"], "status": d["status"],
               "wall_h": d["wall_hours"], "converged": d["status"].startswith("converged"),
               "stalled": stalled(d["status"])}
        out[m.group("tag")][(rec["delta"], rec["ell_km"])] = rec
    return out


def menger_curvature(x, y):
    r"""Curvature at each interior point of a polyline, by circumscribed circle.

    The standard L-curve elbow criterion, applied in log-log coordinates. Returns
    NaN at the two endpoints, where curvature is undefined.
    """
    k = np.full(len(x), np.nan)
    for i in range(1, len(x) - 1):
        (x1, y1), (x2, y2), (x3, y3) = (x[i - 1], y[i - 1]), (x[i], y[i]), (x[i + 1], y[i + 1])
        a = np.hypot(x2 - x1, y2 - y1)
        b = np.hypot(x3 - x2, y3 - y2)
        c = np.hypot(x3 - x1, y3 - y1)
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if a * b * c > 0:
            k[i] = 4.0 * area / (a * b * c)
    return k


def grid(points, deltas, ells, key):
    r"""(len(deltas), len(ells)) array of ``key``, NaN where the point is missing."""
    g = np.full((len(deltas), len(ells)), np.nan)
    for i, d in enumerate(deltas):
        for j, l in enumerate(ells):
            rec = points.get((d, l))
            if rec is not None and rec[key] is not None:
                g[i, j] = rec[key]
    return g


def physical(points):
    r"""``points`` without the stalled ones (see sweep_regularization.stalled)."""
    return {k: r for k, r in points.items() if not r["stalled"]}


def _text_color(rgba):
    r"""Black or white, whichever reads on this cell (ITU-R BT.601 luminance)."""
    r, g, b, _ = rgba
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else "black"


def heatmap(fig, ax, g, deltas, ells, title, cmap, norm=None, fmt="{:.2f}",
            flag=None, excluded={}, cbar_label=None, sym=""):
    r"""Annotated (delta, ELL) heat map.

    ``flag`` marks cells to outline (points that did not converge); ``excluded``
    maps (delta, ell_km) to the label of a cell drawn hatched (excluded, or
    stalled outside the physical range). Missing points stay empty.
    """
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(MISSING_COLOR)
    masked = np.ma.masked_invalid(g)
    if norm is None:
        finite = g[np.isfinite(g)]
        norm = matplotlib.colors.Normalize(finite.min(), finite.max()) if finite.size else None
    im = ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.set_xticks(range(len(ells)), [f"{l:g}" for l in ells])
    ax.set_yticks(range(len(deltas)), [f"{d:.0e}" for d in deltas])
    ax.set_xlabel(f"ELL{sym} (km)", fontsize=10)
    ax.set_ylabel(rf"$\delta{sym}$", fontsize=10)
    ax.set_title(title, fontsize=11)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if np.isfinite(g[i, j]):
                ax.text(j, i, fmt.format(g[i, j]), ha="center", va="center", fontsize=7.5,
                        color=_text_color(cmap(norm(g[i, j]))))
                if flag is not None and flag[i, j]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec=FLAG_COLOR, lw=1.8, zorder=4))
            elif (deltas[i], ells[j]) in excluded:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                                       hatch="//", ec="0.55", lw=0.6, zorder=3))
                ax.text(j, i, excluded[(deltas[i], ells[j])], ha="center", va="center",
                        fontsize=7, color="0.35")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=9, color="0.55")
    if norm is None:              # no finite values yet: a colorbar would show a fake range
        return im
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9)
    return im


def chi2_heatmap(fig, ax, g, deltas, ells, title, **kw):
    r"""chi^2 per observation, diverging about chi^2 = 1 where the data allow it.

    chi^2 < 1 means the MAP fits inside the observation errors, i.e. it is
    fitting noise; chi^2 > 1 means it cannot reach them.
    """
    finite = g[np.isfinite(g)]
    if finite.size and finite.min() < 1.0 < finite.max():
        norm = TwoSlopeNorm(vcenter=np.log10(1.0),
                            vmin=np.log10(finite.min()), vmax=np.log10(finite.max()))
        im = heatmap(fig, ax, np.log10(g), deltas, ells, title, "RdBu_r", norm=norm,
                     fmt="{:.2f}", cbar_label=r"$\log_{10}\,\chi^2$/obs", **kw)
        # annotate with chi^2 itself, not its log
        for t, v in zip([t for t in ax.texts if t.get_text() not in ("–", EXCLUDED_LABEL, *STALLED_LABEL.values())],
                        [g[i, j] for i in range(g.shape[0]) for j in range(g.shape[1])
                         if np.isfinite(g[i, j])]):
            t.set_text(f"{v:.2f}")
        return im
    if not finite.size:
        return heatmap(fig, ax, g, deltas, ells, title, "magma_r", **kw)
    norm = LogNorm(finite.min(), finite.max()) if finite.min() > 0 else None
    return heatmap(fig, ax, g, deltas, ells, title + "\n(all above 1: never fits the errors)",
                   "magma_r", norm=norm, fmt="{:.1f}", cbar_label=r"$\chi^2$/obs", **kw)


def lcurve_family(ax, points, deltas, ells, by="delta", mark_elbow=True, label_points=True,
                  sym=""):
    r"""Misfit against prior term in log-log.

    ``by="delta"``: one line per ELL, traversed in delta (the prior strength);
    ``by="ell"``: one line per delta, traversed in ELL.
    """
    if by == "delta":
        lines, along, cmap, lab = ells, deltas, plt.cm.plasma, f"ELL{sym} = " + "{:g} km"
    else:
        lines, along, cmap, lab = deltas, ells, plt.cm.viridis, rf"$\delta{sym}$ = " + "{:.0e}"
    colors = cmap(np.linspace(0, 0.85, len(lines)))
    for i, key in enumerate(lines):
        recs = [points.get((v, key) if by == "delta" else (key, v)) for v in along]
        xs = np.array([r["R"] if r else np.nan for r in recs], dtype=float)
        ys = np.array([r["misfit"] if r else np.nan for r in recs], dtype=float)
        m = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
        if m.sum() < 2:
            continue
        ax.plot(xs[m], ys[m], "-o", color=colors[i], ms=4, lw=1.3, label=lab.format(key))
        if label_points:
            fmt = "{:.0e}" if by == "delta" else "{:g}"
            for v, x, y in zip(np.array(along)[m], xs[m], ys[m]):
                ax.annotate(fmt.format(v), (x, y), textcoords="offset points",
                            xytext=(4, 3), fontsize=6.5, color=colors[i],
                            path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
        if mark_elbow and m.sum() >= 3:
            k = menger_curvature(np.log10(xs[m]), np.log10(ys[m]))
            if np.any(np.isfinite(k)):
                e = int(np.nanargmax(k))
                ax.plot(xs[m][e], ys[m][e], "*", color=colors[i], ms=16,
                        markeredgecolor="k", markeredgewidth=0.5, zorder=5)
    if not ax.lines:
        ax.text(0.5, 0.5, "no physical points yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="0.45")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"prior term $R{sym}(\theta{sym})$ at the MAP" if sym else
                  r"prior term $R(\theta)$ at the MAP", fontsize=10)
    ax.set_ylabel(r"misfit $J_{\rm vel} + J_{\rm ApRES}$", fontsize=10)
    ax.grid(alpha=0.3, which="both", lw=0.4)


def report(tag, points, deltas, ells):
    r"""One line per point on stdout, so the numbers are readable without the png."""
    print(f"\n{tag}")
    print(f"  {'delta':>7s} {'ELL':>5s} {'chi2_vel':>9s} {'chi2_apr':>9s} "
          f"{'max|theta|':>10s} {'iters':>6s}  status")
    for d in deltas:
        for l in ells:
            r = points.get((d, l))
            if r is None:
                continue
            apr = f"{r['chi2_apr']:9.1f}" if r["chi2_apr"] is not None else f"{'-':>9s}"
            print(f"  {d:7.0e} {l:5g} {r['chi2_vel']:9.2f} {apr} "
                  f"{r['theta_max']:10.1f} {r['iterations']:6d}  {r['status'][:46]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None, help="only this configuration tag")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--mode", choices=MODES, default="shared",
                    help="which sweep to plot (sweep_regularization.py --mode)")
    ap.add_argument("--fixed-delta", type=float, default=1.0e-5,
                    help="single-control modes: the held prior strength")
    ap.add_argument("--fixed-ell", type=float, default=2000.0,
                    help="single-control modes: the held correlation length (m)")
    args = ap.parse_args()
    FIG_DIR.mkdir(exist_ok=True)
    mode, fixed = args.mode, (args.fixed_delta, args.fixed_ell)
    sym = keys(mode)[4]
    stem = ("" if mode == "shared" else f"_{mode}{mode_postfix(mode, fixed)}") + RUN_SUFFIX
    held = ("" if mode == "shared" else
            rf"   ($\theta_{HELD[mode]}$ prior held at $\delta$={fixed[0]:.0e}, "
            f"ELL={fixed[1] / 1000:g} km)")

    all_points = load_points(mode, fixed)
    if not all_points:
        raise SystemExit("no results/map_sweep_*.json yet - has the sweep produced anything?")
    tags = [args.config] if args.config else sorted(all_points)

    # The full sweep grid, so unfinished points read as gaps rather than as data.
    # A sweep may run extra deltas or ells (--deltas/--ells); include any found.
    found = {k for pts in all_points.values() for k in pts}
    deltas = sorted(set(DELTAS) | {d for d, _ in found}, reverse=True)   # strongest on the bottom row
    ells = sorted({l / 1000.0 for l in ELLS} | {l for _, l in found})

    for tag in tags:
        points = all_points.get(tag)
        if not points:
            print(f"no points for {tag}", flush=True)
            continue
        excl = excluded_cells(tag, mode)
        n_conv = sum(r["converged"] for r in points.values())
        n_cells = len(deltas) * len(ells) - len(excl)
        n_stall = sum(r["stalled"] for r in points.values())
        print(f"{tag}: {len(points)}/{n_cells} points ({n_conv} converged"
              + (f", {n_stall} {STALLED_NOTE[mode]}" if n_stall else "") + ")"
              + (f", {len(excl)} excluded" if excl else ""), flush=True)
        report(tag, points, deltas, ells)
        # Everything below plots only the physical points; stalled ones are hatched.
        excl = {**excl, **{k: STALLED_LABEL[mode] for k, r in points.items() if r["stalled"]}}
        points = physical(points)

        not_conv = np.zeros((len(deltas), len(ells)), dtype=bool)
        for i, d in enumerate(deltas):
            for j, l in enumerate(ells):
                r = points.get((d, l))
                not_conv[i, j] = r is not None and not r["converged"]

        hm = dict(flag=not_conv, excluded=excl, sym=sym)

        def p_lcurve_delta(ax):
            lcurve_family(ax, points, deltas, ells, by="delta", sym=sym)
            ax.set_title("L-curve in the prior strength\n"
                         r"(one line per ELL, annotated with $\delta$; star = elbow)",
                         fontsize=11)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc="best")

        def p_lcurve_ell(ax):
            lcurve_family(ax, points, deltas, ells, by="ell", mark_elbow=False, sym=sym)
            ax.set_title("The same plane along the correlation length\n"
                         r"(one line per $\delta$, annotated with ELL in km)", fontsize=11)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc="best")

        def p_chi2_vel(ax):
            chi2_heatmap(fig, ax, grid(points, deltas, ells, "chi2_vel"), deltas, ells,
                         r"$\chi^2_{\rm vel}$ per observation", **hm)

        def p_chi2_apr(ax):
            chi2_heatmap(fig, ax, grid(points, deltas, ells, "chi2_apr"), deltas, ells,
                         r"$\chi^2_{\rm ApRES}$ per observation", **hm)

        def p_theta(ax):
            heatmap(fig, ax, grid(points, deltas, ells, "theta_max"), deltas, ells,
                    rf"$\max|\theta{sym}|$ at the MAP" "\n(runaway = prior no longer constrains)",
                    "inferno_r", fmt="{:.1f}", cbar_label=rf"$\max|\theta{sym}|$", **hm)

        def p_iters(ax):
            heatmap(fig, ax, grid(points, deltas, ells, "iterations"), deltas, ells,
                    "L-BFGS iterations", "cividis", fmt="{:.0f}",
                    cbar_label="iterations", **hm)

        def p_wall(ax):
            heatmap(fig, ax, grid(points, deltas, ells, "wall_h"), deltas, ells,
                    "wall time (h)", "BuPu", fmt="{:.1f}", cbar_label="hours", **hm)

        panels = [p_lcurve_delta, p_lcurve_ell, p_chi2_vel]
        if np.any(np.isfinite(grid(points, deltas, ells, "chi2_apr"))):
            panels.append(p_chi2_apr)
        panels += [p_theta, p_iters]
        if len(panels) % 3:            # wall time fills the last row rather than leaving a hole
            panels.append(p_wall)
        ncol = 3
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 5.25 * nrow), squeeze=False)
        for ax, panel in zip(axes.ravel(), panels):
            panel(ax)
        for ax in axes.ravel()[len(panels):]:
            ax.set_axis_off()

        fig.suptitle(f"Regularization surface - {tag}{RUN_SUFFIX}, {mode} prior swept{held}\n"
                     f"(red outline: did not converge;  hatched {STALLED_LABEL[mode]}: "
                     f"{STALLED_NOTE[mode]};  – : not run yet)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = FIG_DIR / f"lsurface_{tag}{stem}.png"
        fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}", flush=True)

    if args.config:
        return
    # Cross-configuration comparison of the L-curve families, in the strength axis.
    tags = sorted(all_points)
    ncol = min(3, len(tags))
    nrow = int(np.ceil(len(tags) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 5.0 * nrow), squeeze=False)
    for ax, tag in zip(axes.ravel(), tags):
        lcurve_family(ax, physical(all_points[tag]), deltas, ells, by="delta", label_points=False,
                      sym=sym)
        ax.set_title(tag, fontsize=11)
    for ax in axes.ravel()[len(tags):]:
        ax.set_axis_off()
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(ells)))
    fig.legend(handles=[Line2D([], [], color=c, marker="o", ms=4, label=f"ELL = {l:g} km")
                        for l, c in zip(ells, colors)]
               + [Line2D([], [], color="0.3", marker="*", ms=12, lw=0,
                         label="max-curvature elbow")],
               loc="lower center", ncol=len(ells) + 1, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"L-curve in the prior strength $\delta$, by configuration"
                 + (f"  ({RUN_SUFFIX.lstrip('_')})" if RUN_SUFFIX else ""), fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = FIG_DIR / f"lsurface_all_configs{stem}.png"
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
