r"""Figures of the GRL manuscript "Basal properties drive changes in englacial deformation
on Thwaites Glacier", in the Morrow et al. (2026) figure grammar of ``figstyle.py``.

Pure matplotlib on reduced data: results/manuscript/{apres,maps}.npz from
``manuscript_prepare.py`` and, for the model figures, the archived production runs that
``paper_figures.py`` reads.

    python scripts/manuscript_figures.py             # all figures
    python scripts/manuscript_figures.py --only 2 5

Writes figures/manuscript/fig<NN>.{png,pdf} and numbers.json (every number a caption quotes):
  1  study area     surface speed, grounding line, ice front, model domain, ApRES stations
  2  sections       2022-23 return SNR and the eps_zz section along the ApRES line
  3  legendre       the Legendre basis, one profile's fits and the coefficient spectrum
  4  surface        near-surface ApRES eps_zz against the surface strain rate
  5  maps           depth-averaged eps_zz and the shear share over the bed
  6  profiles       ApRES eps_zz against both models (paper_figures F_apres)
  7  projections    5-yr VAF loss rate and sea-level contribution (paper_figures F_retreat)

ApRES data are black diamonds and ApRES stations black dots in every figure; SSA is red
circles and the hybrid blue squares, as in ``paper_figures.py``.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy.stats import linregress  # noqa: E402

import figstyle as fs  # noqa: E402
import paper_figures as pf  # noqa: E402  (applies the rc)
from figstyle import BLACK, BLUE, GREEN, GREY, PURPLE, RED  # noqa: E402
from manuscript_prepare import basis  # noqa: E402

DATA = Path(__file__).resolve().parent.parent
RES = DATA / "results" / "manuscript"
OUT = DATA / "figures" / "manuscript"
EPS = pf.EPS
STATION_KEY = pf.STATION_KEY
DIAMOND = dict(fmt="d", ms=3.0, color=BLACK, ecolor="0.6", elinewidth=0.5, capsize=0)
COL_WIDTH = 0.7                    # widest station column in the sections (km)


def load():
    a = dict(np.load(str(RES / "apres.npz")))
    order = np.argsort(a["s_km"])
    per_site = [k for k, v in a.items() if np.ndim(v) >= 1 and len(v) == len(a["name"])]
    a.update({k: a[k][order] for k in per_site})
    return a, dict(np.load(str(RES / "maps.npz")))


def save(fig, stem):
    fs.save(fig, OUT / stem)


# ── Fig. 1: study area ─────────────────────────────────────────────────────
def domain_outline(tri):
    r"""Boundary edges of the model mesh as one segment collection."""
    from matplotlib.collections import LineCollection
    e = np.sort(np.vstack([tri.triangles[:, [0, 1]], tri.triangles[:, [1, 2]],
                           tri.triangles[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    b = u[cnt == 1]
    seg = np.stack([np.c_[tri.x[b[:, 0]], tri.y[b[:, 0]]], np.c_[tri.x[b[:, 1]], tri.y[b[:, 1]]]],
                   axis=1)
    return LineCollection(seg, colors=BLUE, linewidths=1.0, zorder=4)


def mask_line(ax, x, y, field, ls="-"):
    cs = ax.contour(x, y, field.astype(float), [0.5], colors=BLACK, linewidths=0.7,
                    linestyles=[ls], zorder=3)
    cs.set(path_effects=fs.HALO_LIGHT)


def fig_study_area(a, mp, model, n):
    from manuscript_prepare import REGION
    box = REGION
    fig, axs, caxs = fs.map_grid(1, [1], pf.aspect(box), left=0.08)
    ax = axs[0][0]
    m = fs.grid_field(ax, mp["speed_x"], mp["speed_y"], mp["speed"], "speed", LogNorm(10, 4000))
    fs.colorbar(fig, m, caxs[0][0], r"Surface speed (m yr$^{-1}$)")
    mx, my, mask = mp["mask_x"], mp["mask_y"], mp["mask"]
    mask_line(ax, mx, my, np.isin(mask, (1, 2, 4)))                 # grounding line
    mask_line(ax, mx, my, mask > 0, ls=(0, (3, 1.5)))               # ice front and coast
    ax.add_collection(domain_outline(model["tri"]))
    xy = a["xy"]
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c=BLACK, linewidths=0, zorder=5)
    b5 = fig5_box(a, mp)
    ax.add_patch(Rectangle((b5[0], b5[2]), b5[1] - b5[0], b5[3] - b5[2], fill=False, ec=RED,
                           lw=1.0, zorder=6))
    ax.annotate("Fig. 5", (b5[1], b5[3]), xytext=(2, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=fs.FS_TICK, color=BLACK)
    fs.frame(ax, box, scale=True)
    pf.locator(fig, ax, np.array([[box[0], box[2]], [box[1], box[3]]]),
               rect=(0.015, 0.60, 0.25, 0.385))
    fs.top_legend(fig, [Line2D([], [], color=BLACK, lw=0.7, label="grounding line"),
                        Line2D([], [], color=BLACK, lw=0.7, ls=(0, (3, 1.5)),
                               label="ice front and coast"),
                        Line2D([], [], color=BLUE, lw=1.0, label="model domain"),
                        STATION_KEY,
                        Patch(fill=False, ec=RED, lw=1.0, label="Fig. 5")], ncol=5)
    save(fig, "fig01")


# ── Fig. 2: sections along the line ────────────────────────────────────────
def columns(s):
    r"""Left and right edges (km) of each station's column: halfway to its neighbours, at
    most COL_WIDTH wide, so gaps in the line stay visible."""
    mid = 0.5 * (s[1:] + s[:-1])
    lo = np.r_[s[0] - COL_WIDTH / 2, mid]
    hi = np.r_[mid, s[-1] + COL_WIDTH / 2]
    return np.maximum(lo, s - COL_WIDTH / 2), np.minimum(hi, s + COL_WIDTH / 2)


def section(ax, a, depth_edges, values, kind, norm, keep):
    r"""One column per station of ``values[station, depth]`` at elevation surface - depth."""
    cmap, bn = fs.discrete(kind, norm)
    lo, hi = columns(a["s_km"])
    for i in range(len(a["s_km"])):
        z = a["surface"][i] - depth_edges
        v = np.where(keep[i], values[i], np.nan)[None, :]
        q = ax.pcolormesh([lo[i], hi[i]], z, v.T, cmap=cmap, norm=bn, shading="flat",
                          rasterized=True, zorder=1)
    return q


def fig_sections(a, mp, model, n):
    fig = plt.figure(figsize=(fs.FULL, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=(1, 0.018), hspace=0.32, wspace=0.025,
                          left=0.08, right=0.92, bottom=0.10, top=0.86)
    ax = [fig.add_subplot(gs[r, 0]) for r in range(2)]
    cax = [fig.add_subplot(gs[r, 1]) for r in range(2)]
    s, H = a["s_km"], a["H"]
    rr = a["snr_range"]
    dr = rr[1] - rr[0]
    e_r = np.r_[rr - dr / 2, rr[-1] + dr / 2]
    # Down to just below the deeper of the radar and BedMachine beds: the radar pick is
    # wrong at a few stations (see the caption), BedMachine shows where the bed echo sits.
    bottom = np.maximum(H, a["surface"] - a["bed"]) + 150
    keep = rr[None, :] <= bottom[:, None]
    q = section(ax[0], a, e_r, a["snr"], "power", Normalize(0, 40), keep)
    fs.colorbar(fig, q, cax[0], "Return SNR (dB)")
    zc = a["sec_depth"]
    dz = zc[1] - zc[0]
    e_z = np.r_[zc - dz / 2, zc[-1] + dz / 2]
    q = section(ax[1], a, e_z, a["eps_sec"] * 1e3, "strain", Normalize(-10, 10),
                np.isfinite(a["eps_sec"]))
    fs.colorbar(fig, q, cax[1], rf"${EPS}$ ($10^{{-3}}$ yr$^{{-1}}$)")
    for k, x in enumerate(ax):
        x.plot(s, a["surface"], color=BLACK, lw=0.8, zorder=3)
        x.plot(s, a["bed"], color=BLACK, lw=0.8, ls=(0, (3, 1.5)), zorder=3)
        x.set(xlim=(-1.0, s.max() + 1.0), ylim=(-1700, 1500), ylabel="Elevation (m)")
        fs.panel_letter(x, "ab"[k], x=0.03, y=0.88)
    ax[1].set_xlabel("Distance along the ApRES line (km)")
    fs.top_legend(fig, [Line2D([], [], color=BLACK, lw=0.8, label="surface (BedMachine v4)"),
                        Line2D([], [], color=BLACK, lw=0.8, ls=(0, (3, 1.5)),
                               label="bed (BedMachine v4)")], y=0.90)
    n["sections"] = dict(n_stations=int(len(s)), line_km=float(s.max()),
                         sigma_cut_mm_per_yr=1.5)
    save(fig, "fig02")


# ── Fig. 3: Legendre decomposition ─────────────────────────────────────────
BASIS_STYLE = [(BLACK, "d"), (RED, "o"), (BLUE, "s"), (GREEN, "^"), (PURPLE, "v")]


def example_site(a):
    r"""The order-4 profile with the largest shear share among the long ones."""
    long = a["n_fit"] >= np.percentile(a["n_fit"], 50)
    cand = np.flatnonzero((a["best"] == 4) & long)
    return int(cand[np.argmax(a["share"][cand])])


def fig_legendre(a, mp, model, n):
    fig, ax = plt.subplots(1, 4, figsize=(fs.FULL, 2.75), layout="constrained")
    i = example_site(a)
    H = a["H"][i]
    k = np.isfinite(a["w"][i]) & (a["depth"][i] >= a["fit"][0]) & \
        (a["depth"][i] <= min(a["fit"][1], 0.98 * H))
    z, w = a["depth"][i][k] / H, a["w"][i][k]
    zz = np.linspace(0, 1, 201)
    keys = []
    for nn, (c, mk) in enumerate(BASIS_STYLE):
        ax[0].plot(basis(nn, zz), zz, color=c, marker=mk, ms=3.0, markevery=25)
        keys.append(Line2D([], [], color=c, marker=mk, ms=3.0, label=rf"$\tilde L_{nn}$"))
    ax[0].axhspan(z.min(), z.max(), color=GREY, alpha=0.15, lw=0, zorder=0)
    ax[0].set(xlabel="Basis amplitude", ylabel=r"$\zeta$ = depth / $H$", ylim=(1, 0),
              xlim=(-3.3, 3.3))

    c1, c4 = a["coef"][i, 0, :2], a["coef"][i, 3, :5]
    G = np.column_stack([basis(m, z) for m in range(5)])
    w0 = w[0]
    ax[1].plot(w - w0, z, color=BLACK, lw=0.6, zorder=4)
    ax[1].plot(G[:, :2] @ c1 - w0, z, color=RED, marker="o", ms=3.0, markevery=60, zorder=3)
    ax[1].plot(G @ c4 - w0, z, color=BLUE, marker="s", ms=3.0, markevery=60, zorder=3)
    ax[1].set(xlabel=r"$w - w_\mathrm{100\,m}$ (m yr$^{-1}$)")
    # Departure from the degree-1 (plug-flow) fit. On the observed part of the column the
    # basis is not orthogonal, so the n >= 2 terms of the degree-4 fit alone are not it.
    lin = G[:, :2] @ c1
    ax[2].plot((w - lin) * 1e3, z, color=BLACK, lw=0.6, zorder=4)
    ax[2].plot((G @ c4 - lin) * 1e3, z, color=BLUE, marker="s", ms=3.0, markevery=60, zorder=3)
    ax[2].axvline(0, color=GREY, lw=0.8, zorder=1)
    ax[2].set(xlabel=r"$w - w_\mathrm{linear}$ (mm yr$^{-1}$)")
    for x in ax[1:3]:
        x.set(ylim=(z.max() + 0.02, z.min() - 0.02), ylabel=r"$\zeta$")

    cn = np.abs(a["coef"][:, 3, :5])
    nn = np.arange(5)
    for row in cn:
        ax[3].plot(nn, row, color="0.75", lw=0.4, zorder=1)
    q1, med, q3 = np.nanpercentile(cn, [25, 50, 75], axis=0)
    ax[3].fill_between(nn, q1, q3, color=GREY, alpha=0.3, lw=0, zorder=2)
    ax[3].plot(nn, med, color=BLACK, marker="d", ms=3.5, zorder=3)
    ax[3].plot(nn, cn[i], color=BLUE, marker="s", ms=3.0, zorder=4)
    ax[3].set(yscale="log", xlabel="Polynomial degree $n$", ylabel=r"$|c_n|$ (m yr$^{-1}$)",
              xticks=nn)
    for x, letter in zip(ax, "abcd"):
        fs.grid(x)
        fs.panel_letter(x, letter, x=0.12, y=0.93)
    ax[3].grid(False, which="minor")
    fs.top_legend(fig, [Line2D([], [], color=BLACK, lw=0.6, label="ApRES measured"),
                        Line2D([], [], color=RED, marker="o", ms=3.0, label="degree 1 (linear)"),
                        Line2D([], [], color=BLUE, marker="s", ms=3.0, label="degree 4"),
                        Line2D([], [], color=BLACK, marker="d", ms=3.5,
                               label=r"median, all stations"),
                        Patch(color=GREY, alpha=0.3, label="interquartile range")] + keys,
                  ncol=5)
    n["legendre"] = dict(example=str(a["name"][i]), example_share=float(a["share"][i]),
                         example_s_km=float(a["s_km"][i]),
                         median_abs_c=[float(v) for v in med])
    save(fig, "fig03")


# ── Fig. 4: near-surface against surface strain rate ───────────────────────
def fig_surface(a, mp, model, n):
    fig, ax = plt.subplots(figsize=(0.52 * fs.FULL, 0.52 * fs.FULL), layout="constrained")
    x, y, sy = a["eps_surf"] * 1e3, a["eps_near"] * 1e3, a["eps_near_sigma"] * 1e3
    ok = np.isfinite(x) & np.isfinite(y)
    r = linregress(x[ok], y[ok])
    lim = 5 * np.ceil(1.05 * max(np.abs(x[ok]).max(), np.abs(y[ok] + sy[ok]).max(),
                                 np.abs(y[ok] - sy[ok]).max()) / 5)
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, ls="--", zorder=1)
    xx = np.array([-lim, lim])
    ax.plot(xx, r.intercept + r.slope * xx, color=RED, zorder=2)
    ax.errorbar(x[ok], y[ok], yerr=sy[ok], zorder=3, **DIAMOND)
    ax.axhline(0, color=GREY, lw=0.5, zorder=0)
    ax.axvline(0, color=GREY, lw=0.5, zorder=0)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
           xlabel=rf"Surface ${EPS} = -\nabla\cdot\mathbf{{u}}$ ($10^{{-3}}$ yr$^{{-1}}$)",
           ylabel=rf"ApRES ${EPS}$, 100$-$400 m ($10^{{-3}}$ yr$^{{-1}}$)")
    fs.grid(ax)
    fs.top_legend(fig, [Line2D([], [], ls="", marker="d", ms=3.0, color=BLACK,
                               label=f"ApRES station (n = {ok.sum()})"),
                        Line2D([], [], color=RED,
                               label=rf"fit: slope {r.slope:.2f}, $R^2$ = {r.rvalue ** 2:.2f}"),
                        Line2D([], [], color=GREY, lw=0.8, ls="--", label="1:1")], ncol=1)
    n["surface"] = dict(n=int(ok.sum()), slope=float(r.slope), intercept=float(r.intercept),
                        r2=float(r.rvalue ** 2), p=float(r.pvalue), smooth_m=float(a["smooth"]))
    save(fig, "fig04")


# ── Fig. 5: maps over the bed ──────────────────────────────────────────────
def fig5_box(a, mp):
    return (float(mp["bed_x"].min()), float(mp["bed_x"].max()),
            float(mp["bed_y"].min()), float(mp["bed_y"].max()))


def fig_maps(a, mp, model, n):
    box = fig5_box(a, mp)
    fig, axs, caxs = fs.map_grid(2, [1], pf.aspect(box), left=0.42, bottom=0.62)
    bed_norm = Normalize(-1600, -900)
    xy = a["xy"]
    rows = ((a["eps_avg"] * 1e3, "strain", Normalize(-10, 10),
             rf"${EPS}$ ($10^{{-3}}$ yr$^{{-1}}$)", rf"Depth-averaged ${EPS}$"),
            (a["share"], "share", Normalize(0, 1), r"$A_s / (A_s + A_p)$", "Shear share"))
    picks = model["picks"]
    for r, (v, kind, norm, clabel, head) in enumerate(rows):
        ax = axs[r][0]
        bm = fs.grid_field(ax, mp["bed_x"], mp["bed_y"], mp["bed"], "relief", bed_norm)
        cmap, bn = fs.discrete(kind, norm)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=cmap, norm=bn, s=16, edgecolors=BLACK,
                        linewidths=0.4, zorder=5)
        fs.colorbar(fig, sc, caxs[r][0], clabel)
        for j, nm in enumerate(picks):
            p = xy[list(a["name"]).index(nm)]
            ax.scatter(*p, s=60, facecolors="none", edgecolors=BLACK, linewidths=0.9, zorder=6)
            ax.annotate("bcde"[j], p, xytext=(0, 7), textcoords="offset points", ha="center",
                        va="bottom", fontsize=fs.FS_TICK, zorder=7)
        fs.frame(ax, box, scale=r == 0)
        fs.row_label(ax, head, x=-0.02)
        fs.panel_letter(ax, "ab"[r], x=0.035, y=0.88)
    # One bed colour bar under both maps.
    W, Hh = fig.get_size_inches()
    p0 = axs[1][0].get_position()
    cb_ax = fig.add_axes([p0.x0 + 0.25 * p0.width, 0.22 / Hh, 0.5 * p0.width, 0.09 / Hh])
    cb = fig.colorbar(bm, cax=cb_ax, orientation="horizontal")
    cb.set_label("Bed elevation, BedMachine v4 (m)", labelpad=2)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.5, length=2.5, labelsize=fs.FS_TICK)
    cb.set_ticks([-1600, -1400, -1200, -1000])
    cb.ax.xaxis.set_major_formatter(lambda t, _: fs.num(float(f"{t:.3g}"), "g"))
    fs.top_legend(fig, [STATION_KEY.__class__([], [], ls="", marker="o", ms=4, mfc="white",
                                              mec=BLACK, mew=0.4, label="ApRES station"),
                        Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=BLACK, mew=0.9,
                               label="profile in Fig. 6 (b-e)")])
    n["maps"] = dict(share_median=float(np.median(a["share"])),
                     share_q=[float(q) for q in np.percentile(a["share"], [25, 75])],
                     eps_avg_range=[float(np.nanmin(a["eps_avg"])), float(np.nanmax(a["eps_avg"]))],
                     best_order_counts={int(o): int((a["best"] == o).sum()) for o in range(1, 5)})
    save(fig, "fig05")


# ── Figs. 6, 7: the model figures of paper_figures.py ──────────────────────
def copy_model_figure(src, stem):
    for ext in ("png", "pdf"):
        shutil.copyfile(pf.OUT / f"{src}.{ext}", OUT / f"{stem}.{ext}")
    print(f"Copied {src} -> {OUT / stem}.{{png,pdf}}", flush=True)


def fig_profiles(a, mp, model, n):
    pf.fig_apres(model["d"], model["tri"], model["area"], model["n"])
    copy_model_figure("F_apres", "fig06")


def fig_projections(a, mp, model, n):
    pf.fig_retreat(model["d"], model["tri"], model["area"], model["n"])
    copy_model_figure("F_retreat_rates", "fig07")
    m = model["n"]
    sle = {k: -m[k]["dvaf"] / pf.GT_PER_MM_SLE for k in pf.MODELS}
    n["projections"] = dict(years=pf.T, start=2015, sle_mm=sle,
                            sle_2sigma_mm={k: 2 * m[k]["sig_post"] / pf.GT_PER_MM_SLE
                                           for k in pf.MODELS},
                            hybrid_less_than_ssa=float(1 - sle["hybrid"] / sle["ssa"]))


FIGS = {"1": fig_study_area, "2": fig_sections, "3": fig_legendre, "4": fig_surface,
        "5": fig_maps, "6": fig_profiles, "7": fig_projections}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=list(FIGS), default=list(FIGS))
    args = ap.parse_args()
    a, mp = load()
    d, tri, area = pf.load()
    nm = pf.numbers(d, area)
    sites = pf.apres_sites()
    model = dict(d=d, tri=tri, area=area, n=nm,
                 picks=[sites[k]["name"] for k in pf.profile_picks(sites)])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "numbers.json"
    n = json.load(open(path)) if path.exists() else {}
    for k in args.only:
        FIGS[k](a, mp, model, n)
    json.dump(n, open(path, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
