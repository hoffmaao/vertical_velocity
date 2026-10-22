r"""Main-text and supplement figures for the Thwaites vertical-velocity paper.

Pure matplotlib on archived results (no simulation): the per-run extracts from
``paper_extract.py`` plus the forward time series, UQ summaries and Hessian
spectra of the production runs (rc friction, bilaplacian prior delta 4e-4,
ELL 3 km, ISMIP7 CESM2-WACCM ssp585 from 2015, 50 yr).

    python scripts/paper_figures.py              # all figures
    python scripts/paper_figures.py --only S1 S4

Writes figures/paper/<id>_<slug>.{png,pdf}:
  F_setting  setting        model domain, surface speed, bed and the ApRES line (main text)
  F_apres    apres          ApRES eps_zz profiles against both models (main text)
  F_retreat  retreat_rates  5-yr VAF loss rate and sea-level contribution (main text)
  S1  summary        what the SSA / hybrid simulations and their UQ show
  S2  controls       MAP theta_A, theta_C and the hybrid - SSA difference
  S3  data_fit       surface speed residuals and the ApRES eps_zz fit
  S4  posterior_std  where the data reduce the pointwise uncertainty of theta
  S5  sensitivity    dVAF(50 yr)/dtheta: where the projection is sensitive
  S6  trajectories   loss rate, melt and peak speed over 50 yr
  S7  grounding      speed, grounding line and thickness at 50 yr
  S8  spectrum       GN Hessian spectrum and how many modes the UQ needs
and figures/paper/retreat_rates_rows.tex, the body of the main-text rate table.

Every visual choice comes from ``figstyle.py``, the Morrow et al. (2026)
figure grammar the group's papers share; this file only decides what to show.
SSA is red circles and the hybrid blue squares throughout; the 2015 grounding
line is black and a 2065 one wears its model's colour. Panels carry a letter
and no heading; what each panel shows is stated in the caption.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

import figstyle as fs  # noqa: E402
from figstyle import BLACK, GREY  # noqa: E402

DATA = Path(__file__).resolve().parent.parent
RES = DATA / "results"
OUT = DATA / "figures" / "paper"
RUN = "sweep_{model}_rc_bilap_d4e-04_L3km_thwaites_fixed_tau5km"
MODELS = ("ssa", "hybrid")
T = 50

COL = {"ssa": fs.RED, "hybrid": fs.BLUE}
MARK = {"ssa": "o", "hybrid": "s"}
LABEL = {"ssa": "SSA (velocity only)", "hybrid": "hybrid (velocity + ApRES)"}
SHORT = {"ssa": "SSA", "hybrid": "hybrid"}
RHO_RATIO = 1024.0 / 917.0          # icepack water / ice density
GT_PER_MM_SLE = 361.8               # Gt of VAF per mm global mean sea level
POST, PRIOR = r"\sigma_\mathrm{post}", r"\sigma_\mathrm{prior}"
EPS = r"\dot\varepsilon_{zz}"
fs.apply_rc()


# ── data ──────────────────────────────────────────────────────────────────────
def load():
    d = {}
    for m in MODELS:
        name = RUN.format(model=m)
        ex = dict(np.load(str(RES / "paper" / f"extract_{m}.npz")))
        ts = dict(np.load(str(RES / f"forward_{name}_{T}yr_ts.npz")))
        pcg = json.load(open(RES / f"uq_{name}_{T}yr_pcg_summary.json"))
        chi = json.load(open(RES / f"uq_{name}_{T}yr_pcg_chi2_summary.json"))
        eig = json.load(open(RES / f"uq_{name}_{T}yr_K1500_summary.json"))
        mp = json.load(open(RES / f"map_{name}.json"))
        diag = json.load(open(RES / f"map_{name}_diagnostics.json"))
        lam = np.load(str(RES / f"eigendec_{name}_K1500.npz"))["eigenvalues"]
        contrib = np.loadtxt(str(RES / f"uq_{name}_{T}yr_K1500_contribs.txt"))
        d[m] = dict(ex=ex, ts=ts, pcg=pcg, chi=chi, eig=eig, map=mp, diag=diag,
                    lam=lam, contrib=contrib)
    ex = d["ssa"]["ex"]
    tri = Triangulation(ex["xy"][:, 0], ex["xy"][:, 1], ex["cells"])
    # Nodal areas (one third of each adjacent triangle) for area integrals.
    x, y = ex["xy"][:, 0], ex["xy"][:, 1]
    c = ex["cells"]
    a = 0.5 * np.abs((x[c[:, 1]] - x[c[:, 0]]) * (y[c[:, 2]] - y[c[:, 0]])
                     - (x[c[:, 2]] - x[c[:, 0]]) * (y[c[:, 1]] - y[c[:, 0]]))
    area = np.zeros(len(x))
    for k in range(3):
        np.add.at(area, c[:, k], a / 3)
    return d, tri, area


def haf(h, bed):
    return h - RHO_RATIO * np.maximum(0.0, -bed)


def stations(ex):
    r"""ApRES station positions, their index per sample, and distance along the line (km)."""
    st, inv = np.unique(np.round(ex["xy_a"], 1), axis=0, return_inverse=True)
    c = st.mean(axis=0)
    _, _, vt = np.linalg.svd(st - c)
    s = (st - c) @ vt[0]
    if np.corrcoef(s, st[:, 0])[0, 1] < 0:      # distance increases inland (east)
        s = -s
    return st, inv.ravel(), (s - s.min()) / 1e3


def gl_dist(d):
    r"""Distance of every node from the 2015 grounding line [m]."""
    from scipy.spatial import cKDTree
    ex = d["ssa"]["ex"]
    on = np.abs(haf(ex["h0"], ex["bed"])) < 20
    return cKDTree(ex["xy"][on]).query(ex["xy"])[0]


def line_dist(d):
    r"""Distance of every node from the nearest ApRES station [m]."""
    from scipy.spatial import cKDTree
    st, _, _ = stations(d["hybrid"]["ex"])
    return cKDTree(st).query(d["ssa"]["ex"]["xy"])[0]


def gz_box(d, pad=30e3):
    ex = d["ssa"]["ex"]
    near = np.abs(haf(ex["h0"], ex["bed"])) < 50
    for m in MODELS:
        near |= np.abs(haf(d[m]["ex"]["h_T"], ex["bed"])) < 50
    xy = ex["xy"][near]
    return xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad




def full_box(d, pad=3e3):
    xy = d["ssa"]["ex"]["xy"]
    return xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad


def aspect(box):
    return (box[3] - box[2]) / (box[1] - box[0])


# ── shared overlays and keys ────────────────────────────────────────────────
def gl2015(ax, tri, d):
    ex = d["ssa"]["ex"]
    fs.grounding_line(ax, tri, haf(ex["h0"], ex["bed"]), color=BLACK, lw=0.7)


def gl2065(ax, tri, d, m):
    fs.grounding_line(ax, tri, haf(d[m]["ex"]["h_T"], d["ssa"]["ex"]["bed"]), color=COL[m], lw=1.0)


def apres(ax, d):
    st, _, _ = stations(d["hybrid"]["ex"])
    ax.scatter(st[:, 0], st[:, 1], s=3, c=BLACK, linewidths=0, zorder=5)


def model_key(m, label=None):
    return Line2D([], [], color=COL[m], marker=MARK[m], label=label or LABEL[m])


GL_KEY = Line2D([], [], color=BLACK, lw=0.7, label="grounding line 2015")
STATION_KEY = Line2D([], [], ls="", marker="o", ms=2.5, mfc=BLACK, mew=0, label="ApRES stations")


def gl2065_key(m):
    return Line2D([], [], color=COL[m], lw=1.0, path_effects=fs.HALO_DARK,
                  label=f"{SHORT[m]} grounding line 2065")


def every(t, step=5):
    """Marker positions every ``step`` years on an annual series (a marker per data point
    would bury a 51-point line)."""
    t = np.asarray(t)
    return [i for i, v in enumerate(t) if abs(v / step - round(v / step)) < 1e-6]


def save(fig, stem):
    fs.save(fig, OUT / stem)


def numbers(d, area):
    r"""Headline numbers quoted in the captions and text (also written to JSON)."""
    ex = d["ssa"]["ex"]
    g0 = haf(ex["h0"], ex["bed"]) > 0
    n = {}
    for m in MODELS:
        r = d[m]
        gT = haf(r["ex"]["h_T"], ex["bed"]) > 0
        n[m] = dict(
            dvaf=r["pcg"]["dvaf_Gt_t"][-1], sig_prior=r["pcg"]["sigma_prior_Gt"],
            sig_post=r["pcg"]["sigma_post_Gt"], sig_post_chi2=r["chi"]["sigma_post_Gt"],
            sig_trunc=r["pcg"]["sigma_post_trunc_Gt"], sig_lo=r["eig"]["sigma_post_lo_Gt"],
            rate=r["pcg"]["rate_Gt_per_yr"][-1], sig_rate=r["pcg"]["sigma_post_rate"][-1],
            chi2_vel=r["map"]["chi2_vel_per_obs"], chi2_apr=r["diag"]["chi2_apr_per_obs"],
            eps_r=r["diag"]["eps_corr"], speed_rmse=r["diag"]["speed_rmse"],
            n_eff=r["eig"].get("n_eff", float(np.sum(r["lam"] / (1 + r["lam"])))),
            lam_min=float(r["lam"].min()), lam_max=float(r["lam"].max()),
            n_pos=int(np.sum(r["lam"] > 0)), cg_its=max(r["pcg"]["cg_iterations_t"]),
            cg_ok=all(r["pcg"]["cg_converged_t"]), apr_scale=r["chi"]["sigma_scale_apr"],
            vel_scale=r["chi"]["sigma_scale_vel"],
            ungrounded_km2=float(area[g0 & ~gT].sum() / 1e6),
            grad_A=[float(np.abs(r["ex"]["dVAF_dthetaA"]).max())])
    n["gap"] = n["hybrid"]["dvaf"] - n["ssa"]["dvaf"]
    return n




# ── S1: summary ─────────────────────────────────────────────────────────────
def fig_summary(d, tri, area, n):
    fig, ax = plt.subplots(2, 3, figsize=(fs.FULL, 5.5), layout="constrained")
    ex0 = d["ssa"]["ex"]

    # (a) VAF change with the 2 sigma posterior band
    a = ax[0, 0]
    for m in MODELS:
        ts, p = d[m]["ts"], d[m]["pcg"]
        dv = ts["vaf_Gt"] - ts["vaf_Gt"][0]
        a.plot(ts["times"], dv, color=COL[m], marker=MARK[m], markevery=every(ts["times"], 10))
        t = np.r_[0.0, p["times"]]
        s = np.r_[0.0, p["sigma_post_Gt_t"]]
        mu = np.interp(t, ts["times"], dv)
        a.fill_between(t, mu - 2 * s, mu + 2 * s, color=COL[m], alpha=0.25, lw=0)
    a.axhline(0, color=GREY, lw=0.8, zorder=1)
    a.set(xlabel="Years after 2015", ylabel="VAF change (Gt)", xlim=(0, T), ylim=(-1700, 400))
    a.text(0.97, 0.93, rf"bands: $\pm 2{POST}$", transform=a.transAxes, ha="right", va="top")

    # (b) prior vs posterior sigma
    b = ax[0, 1]
    for m in MODELS:
        p, e = d[m]["pcg"], d[m]["eig"]
        t = p["times"]
        b.plot(t, p["sigma_prior_Gt_t"], color=COL[m], ls=":", marker=MARK[m], mfc="none")
        b.plot(t, p["sigma_post_Gt_t"], color=COL[m], marker=MARK[m])
        b.fill_between(t, e["sigma_post_lo_Gt_t"], p["sigma_post_trunc_Gt_t"],
                       color=COL[m], alpha=0.15, lw=0)
    b.set_yscale("log")
    b.set(ylim=(1, 1000), xlim=(0, T), xlabel="Years after 2015",
          ylabel=r"$\sigma$ of the VAF change (Gt)")
    b.text(32, 330, "prior", ha="center", va="bottom")
    b.text(37, 6.0, "posterior (PCG)", ha="center", va="top")

    # (c) structural vs parametric
    c = ax[0, 2]
    for i, m in enumerate(MODELS):
        c.errorbar(i, n[m]["dvaf"], yerr=2 * n[m]["sig_prior"], fmt="none", ecolor=COL[m],
                   alpha=0.35, capsize=3, lw=0.9)
        c.errorbar(i, n[m]["dvaf"], yerr=2 * n[m]["sig_post"], fmt=MARK[m], color=COL[m],
                   capsize=3, lw=1.3)
        c.annotate(f"{fs.num(n[m]['dvaf'], '.0f')} $\\pm$ {2 * n[m]['sig_post']:.0f}",
                   (i, n[m]["dvaf"]), xytext=(0, -11), textcoords="offset points", va="top",
                   ha="center")
    c.annotate("", xy=(0.5, n["hybrid"]["dvaf"]), xytext=(0.5, n["ssa"]["dvaf"]),
               arrowprops=dict(arrowstyle="<->", color=BLACK, lw=0.8))
    c.text(0.56, 0.5 * (n["hybrid"]["dvaf"] + n["ssa"]["dvaf"]), f"{n['gap']:.0f} Gt", va="center")
    c.set_xticks([0, 1], [SHORT[m] for m in MODELS])
    c.set_xlim(-0.6, 1.6)
    c.set_ylabel("VAF change by 2065 (Gt)")
    c.text(0.97, 0.04, rf"dark: $\pm 2{POST}$" "\n" rf"faint: $\pm 2{PRIOR}$",
           transform=c.transAxes, ha="right", va="bottom")

    # (d) ApRES eps_zz along the line: observed vs both models
    dd = ax[1, 0]
    st, inv, s_km = stations(d["hybrid"]["ex"])
    for m in MODELS:
        ex = d[m]["ex"]
        st_m, inv_m, s_m = stations(ex)
        mod = np.array([ex["eps_mod"][inv_m == i].mean() for i in range(len(st_m))]) * 1e3
        o = np.argsort(s_m)
        dd.plot(s_m[o], mod[o], color=COL[m], marker=MARK[m], ms=2.5, markevery=6)
    ex = d["hybrid"]["ex"]
    obs = np.array([ex["eps_obs"][inv == i].mean() for i in range(len(st))]) * 1e3
    sig = np.array([np.sqrt(np.mean(ex["eps_sig"][inv == i] ** 2)) for i in range(len(st))]) * 1e3
    o = np.argsort(s_km)
    dd.errorbar(s_km[o], obs[o], yerr=sig[o], fmt="d", ms=2.0, color=BLACK, ecolor="0.6",
                elinewidth=0.5, capsize=0, zorder=3)
    dd.axhline(0, color=GREY, lw=0.8, zorder=1)
    dd.set(xlabel="Distance along the ApRES line (km)",
           ylabel=rf"Upper-ice ${EPS}$ ($10^{{-3}}$ yr$^{{-1}}$)", ylim=(-17, 10))

    # (e) grounding lines at 2065
    e = ax[1, 1]
    e.tricontourf(tri, (haf(ex0["h0"], ex0["bed"]) > 0).astype(float), [-0.5, 0.5, 1.5],
                  colors=["white", "0.88"], zorder=0)
    e.set_rasterization_zorder(0.5)
    gl2015(e, tri, d)
    for m in MODELS:
        gl2065(e, tri, d, m)
    box = gz_box(d)
    e.set_aspect("equal")
    fs.frame(e, (box[0], box[1], box[2], box[3]), scale=True)

    # (f) verification
    h, s = n["hybrid"], n["ssa"]
    fs.verification(ax[1, 2], [
        ("Hessian vs finite differences", "top eigenvalues agree to ~4 digits"),
        ("Hessian spectrum, K = 1500", rf"all $\lambda$ > 0 ({h['lam_min']:.1f} to {fs.sci(s['lam_max'])})"),
        ("exact posterior by PCG", f"converged in {max(s['cg_its'], h['cg_its'])} iterations"),
        ("taped adjoint through 50 yr", "VAF gradient grows smoothly"),
        ("rerun after a machine crash", "hybrid VAF same to 0.01 Gt"),
        (r"$\chi^2$-scaled observation errors",
         rf"${POST}$ +{100 * (h['sig_post_chi2'] / h['sig_post'] - 1):.0f}% "
         rf"(ApRES $\sigma$ $\times${h['apr_scale']:.1f})"),
    ])
    for axi in (a, b, c, dd):
        fs.grid(axi)
    for axi, k in zip(ax.flat, "abcdef"):
        fs.panel_letter(axi, k, x=0.03 if axi is ax[1, 2] else 0.075)
    fs.top_legend(fig, [model_key(m) for m in MODELS]
                  + [Line2D([], [], ls="", marker="d", ms=3, color=BLACK, label=r"ApRES $\pm\sigma$"),
                     GL_KEY])
    save(fig, "S1_summary")


# ── S2: controls ────────────────────────────────────────────────────────────
def fig_controls(d, tri, area, n):
    box = full_box(d)
    fig, axs, caxs = fs.map_grid(2, [2, 1], aspect(box))
    near = line_dist(d) < 5e3
    rows = (("theta_A", r"\theta_A", r"$\theta_A = \ln(A/A_\mathrm{prior})$"),
            ("theta_C", r"\theta_C", r"$\theta_C = \ln(C/C_\mathrm{w0})$"))
    for r, (key, sym, lab) in enumerate(rows):
        for c, m in enumerate(MODELS):
            mp = fs.tri_field(axs[r][c], tri, d[m]["ex"][key], "theta", Normalize(-2.5, 2.5))
        fs.colorbar(fig, mp, caxs[r][0], lab)
        dv = d["hybrid"]["ex"][key] - d["ssa"]["ex"][key]
        mp = fs.tri_field(axs[r][2], tri, dv, "difference", Normalize(-1.5, 1.5))
        fs.colorbar(fig, mp, caxs[r][1], rf"hybrid $-$ SSA ${sym}$")
        n.setdefault("controls", {})[key] = dict(
            rms=float(np.sqrt(np.sum(area * dv ** 2) / area.sum())),
            rms_near=float(np.sqrt(np.sum((area * dv ** 2)[near]) / area[near].sum())))
        fs.row_label(axs[r][0], f"${sym}$")
    finish_maps(axs, box, d, tri, stations=True)
    fs.top_legend(fig, [GL_KEY, STATION_KEY], y=1.0)
    save(fig, "S2_controls")


def finish_maps(axs, box, d, tri, stations=False, gl=True):
    """Frame, overlay and letter a grid of map panels, in Morrow et al. order."""
    k = 0
    for r, row in enumerate(axs):
        for c, ax in enumerate(row):
            if gl:
                gl2015(ax, tri, d)
            if stations:
                apres(ax, d)
            fs.frame(ax, box, scale=r == 0 and c == 0)
            fs.panel_letter(ax, "abcdefgh"[k])
            k += 1


# ── S3: data fit ────────────────────────────────────────────────────────────
def fig_data_fit(d, tri, area, n):
    box = full_box(d)
    fig, axs, caxs = fs.map_grid(1, [1, 2], aspect(box), bottom=2.55)
    ex = d["hybrid"]["ex"]
    mp = fs.tri_field(axs[0][0], tri, ex["speed_obs"], "speed", LogNorm(10, 4000))
    fs.colorbar(fig, mp, caxs[0][0], r"Observed speed (m yr$^{-1}$)")
    for c, m in enumerate(MODELS):
        e = d[m]["ex"]
        mp = fs.tri_field(axs[0][1 + c], tri, e["speed_mod"] - e["speed_obs"], "residual",
                          Normalize(-100, 100))
    fs.colorbar(fig, mp, caxs[0][1], r"Model $-$ observed (m yr$^{-1}$)")
    finish_maps(axs, box, d, tri)
    apres(axs[0][0], d)

    # Bottom row: plain axes under the maps, in the same width.
    W, H = fig.get_size_inches()
    y0, h = 0.42, 1.75
    lefts, w = (0.55, 2.80, 5.22), 1.65
    pa = [fig.add_axes([x / W, y0 / H, w / W, h / H]) for x in lefts]
    lim = max(np.abs(np.r_[d[m]["ex"]["eps_obs"], d[m]["ex"]["eps_mod"]]).max() * 1e3 * 1.05
              for m in MODELS)
    for c, m in enumerate(MODELS):
        e = d[m]["ex"]
        a = pa[c]
        a.errorbar(e["eps_obs"] * 1e3, e["eps_mod"] * 1e3, xerr=e["eps_sig"] * 1e3, fmt="none",
                   ecolor="0.8", elinewidth=0.4, zorder=1)
        a.scatter(e["eps_obs"] * 1e3, e["eps_mod"] * 1e3, s=4, color=COL[m], marker=MARK[m],
                  linewidths=0, zorder=2)
        a.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, ls="--", zorder=1)
        a.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=rf"Observed ${EPS}$ ($10^{{-3}}$ yr$^{{-1}}$)")
        if c == 0:
            a.set_ylabel(rf"Modelled ${EPS}$ ($10^{{-3}}$ yr$^{{-1}}$)")
    a = pa[2]
    bins = np.linspace(-15, 15, 31)
    for m in MODELS:
        e = d[m]["ex"]
        r = (e["eps_mod"] - e["eps_obs"]) / e["eps_sig"]
        cnt, edges = np.histogram(r, bins=bins)
        mid = 0.5 * (edges[1:] + edges[:-1])
        a.plot(mid, cnt, color=COL[m], marker=MARK[m], ms=3)
    a.axvspan(-1, 1, color=GREY, alpha=0.18, lw=0)
    a.set(xlabel=r"(Model $-$ observed) / $\sigma$", ylabel="Samples", ylim=(0, None))
    for a, k in zip(pa, "def"):
        fs.grid(a)
        fs.panel_letter(a, k)
    fs.top_legend(fig, [model_key(m) for m in MODELS] + [GL_KEY, STATION_KEY])
    save(fig, "S3_data_fit")


# ── S4: posterior std ───────────────────────────────────────────────────────
def fig_posterior_std(d, tri, area, n):
    box = full_box(d)
    fig, axs, caxs = fs.map_grid(2, [2, 1], aspect(box))
    for r, k in enumerate("AC"):
        ratio = {m: d[m]["ex"][f"std_post_{k}"] / d[m]["ex"][f"std_prior_{k}"] for m in MODELS}
        for c, m in enumerate(MODELS):
            mp = fs.tri_field(axs[r][c], tri, ratio[m], "ratio", Normalize(0, 1))
        fs.colorbar(fig, mp, caxs[r][0], rf"${POST}/{PRIOR}$")
        dr = ratio["hybrid"] - ratio["ssa"]
        mp = fs.tri_field(axs[r][2], tri, dr, "difference", Normalize(-0.3, 0.3))
        fs.colorbar(fig, mp, caxs[r][1], r"hybrid $-$ SSA")
        fs.row_label(axs[r][0], rf"$\theta_{k}$")
    finish_maps(axs, box, d, tri, stations=True)
    fs.top_legend(fig, [GL_KEY, STATION_KEY])
    save(fig, "S4_posterior_std")


# ── S5: sensitivity ─────────────────────────────────────────────────────────
def fig_sensitivity(d, tri, area, n):
    box = gz_box(d, pad=20e3)
    fig, axs, caxs = fs.map_grid(1, [2, 2], aspect(box), left=0.08, group_gap=0.72)
    for j, k in enumerate("AC"):
        lim = max(fs.limits(d[m]["ex"][f"dVAF_dtheta{k}"], "gradient", percentile=2.0)[1]
                  for m in MODELS)
        for c, m in enumerate(MODELS):
            mp = fs.tri_field(axs[0][2 * j + c], tri, d[m]["ex"][f"dVAF_dtheta{k}"], "gradient",
                              Normalize(-lim, lim))
        fs.colorbar(fig, mp, caxs[0][j], rf"$\partial\,$VAF$_{{2065}}/\partial\theta_{k}$ (m)")
    finish_maps(axs, box, d, tri)
    fs.top_legend(fig, [GL_KEY])
    save(fig, "S5_sensitivity")


# ── S6: trajectories ────────────────────────────────────────────────────────
def fig_trajectories(d, tri, area, n):
    fig, ax = plt.subplots(1, 3, figsize=(fs.FULL, 2.45), layout="constrained")
    for m in MODELS:
        ts, p = d[m]["ts"], d[m]["pcg"]
        ax[0].stairs(p["rate_Gt_per_yr"], np.r_[0.0, p["times"]], color=COL[m], lw=fs.LW,
                     baseline=None)
        ax[0].errorbar(p["rate_t_mid"], p["rate_Gt_per_yr"], yerr=2 * np.array(p["sigma_post_rate"]),
                       fmt=MARK[m], color=COL[m], capsize=2, lw=0.9)
        tm = ts["times"][1:]                                   # t = 0: not evaluated
        ax[1].plot(tm, ts["melt_Gt_per_yr"][1:], color=COL[m], marker=MARK[m], markevery=every(tm))
        ax[2].plot(ts["times"], ts["u_max_m_per_yr"], color=COL[m], marker=MARK[m],
                   markevery=every(ts["times"]))
    ax[0].axhline(0, color=GREY, lw=0.8, zorder=1)
    ax[0].set(ylabel=r"5-yr mean dVAF/dt (Gt yr$^{-1}$)", ylim=(-45, 12))   # headroom: letter
    ax[0].text(0.04, 0.05, rf"bars: $\pm 2{POST}$", transform=ax[0].transAxes)
    ax[1].set(ylabel=r"Sub-shelf melt (Gt yr$^{-1}$)")
    ax[2].set(ylabel=r"Peak speed (m yr$^{-1}$)", ylim=(2300, 4700))
    for a, k in zip(ax, "abc"):
        a.set(xlabel="Years after 2015", xlim=(0, T))
        fs.grid(a)
        fs.panel_letter(a, k)
    fs.top_legend(fig, [model_key(m) for m in MODELS])
    save(fig, "S6_trajectories")


# ── S7: grounding zone ──────────────────────────────────────────────────────
def fig_grounding(d, tri, area, n):
    box = gz_box(d)
    fig, axs, caxs = fs.map_grid(1, [2, 1], aspect(box), left=0.08)
    ex0 = d["ssa"]["ex"]
    for c, m in enumerate(MODELS):
        mp = fs.tri_field(axs[0][c], tri, d[m]["ex"]["speed_T"], "speed", LogNorm(10, 4000))
    fs.colorbar(fig, mp, caxs[0][0], r"Speed in 2065 (m yr$^{-1}$)")
    dh = d["hybrid"]["ex"]["h_T"] - d["ssa"]["ex"]["h_T"]
    mp = fs.tri_field(axs[0][2], tri, dh, "difference", Normalize(-200, 200))
    fs.colorbar(fig, mp, caxs[0][1], r"hybrid $-$ SSA thickness (m)")
    for c, m in enumerate(MODELS):
        gl2065(axs[0][c], tri, d, m)
    finish_maps(axs, box, d, tri)
    both = (haf(d["ssa"]["ex"]["h_T"], ex0["bed"]) > 0) & (haf(d["hybrid"]["ex"]["h_T"], ex0["bed"]) > 0)
    near = both & (gl_dist(d) < 30e3)
    n["thicker_m"] = float(np.sum((dh * area)[near]) / area[near].sum())
    fs.top_legend(fig, [GL_KEY] + [gl2065_key(m) for m in MODELS])
    save(fig, "S7_grounding")


# ── S8: spectrum ────────────────────────────────────────────────────────────
def fig_spectrum(d, tri, area, n):
    fig, ax = plt.subplots(1, 3, figsize=(fs.FULL, 2.45), layout="constrained")
    for m in MODELS:
        lam = np.sort(d[m]["lam"])[::-1]
        i = np.arange(1, len(lam) + 1)
        logm = np.unique(np.geomspace(1, len(lam), 12).astype(int)) - 1
        ax[0].loglog(i, lam, color=COL[m], marker=MARK[m], markevery=list(logm))
        cb = d[m]["contrib"]
        ax[1].scatter(cb[:, 0] + 1, cb[:, 3] / cb[:, 3].sum(), s=1.5, color=COL[m], marker=MARK[m],
                      linewidths=0, alpha=0.7, rasterized=True)
        ax[2].plot(cb[:, 0] + 1, cb[:, 4], color=COL[m], marker=MARK[m], markevery=150)
    ax[0].axhline(1, color=GREY, lw=0.8, ls="--")
    ax[0].text(1.3, 1.6, r"$\lambda$ = 1", color="0.3")
    ax[0].set(xlabel="Mode $i$", ylabel=r"Generalized eigenvalue $\lambda_i$", ylim=(0.3, 1e9))
    ax[1].set(xscale="log", yscale="log", ylim=(1e-8, 30), xlabel="Mode $i$",
              ylabel="Share of VAF variance reduction")
    ax[2].set(xlabel="Modes used, $K$", ylabel="Prior VAF variance removed", ylim=(0, 1.15))
    for a, k in zip(ax, "abc"):
        fs.grid(a)
        fs.panel_letter(a, k)
    fs.top_legend(fig, [model_key(m, f"{LABEL[m]}, $n_\\mathrm{{eff}}$ = {n[m]['n_eff']:.0f}")
                        for m in MODELS])
    save(fig, "S8_spectrum")


# ── main text: study area ───────────────────────────────────────────────────
BEDMACHINE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/bedmachine/"
                  "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")
APRES_FILE = DATA / "data" / "apres_legendre_fits.h5"


def antarctica(stride=20):
    r"""BedMachine v4 mask at 10 km (0 ocean, 1 land, 2 grounded, 3 floating, 4 Vostok),
    cached in results/paper so the figures do not reread the 500 m grid."""
    cache = RES / "paper" / "antarctica_mask_10km.npz"
    if not cache.exists():
        import netCDF4 as nc
        with nc.Dataset(str(BEDMACHINE)) as ds:
            np.savez_compressed(str(cache), x=ds["x"][::stride].astype(float),
                                y=ds["y"][::stride].astype(float),
                                mask=np.asarray(ds["mask"][::stride, ::stride]))
    return dict(np.load(str(cache)))


def fig_setting(d, tri, area, n):
    r"""Main-text Fig. 1: where the model domain and the ApRES line are."""
    box = full_box(d)
    fig, axs, caxs = fs.map_grid(1, [1, 1], aspect(box), group_gap=0.75)
    ex = d["ssa"]["ex"]
    mp = fs.tri_field(axs[0][0], tri, ex["speed_obs"], "speed", LogNorm(10, 4000))
    fs.colorbar(fig, mp, caxs[0][0], r"Observed speed (m yr$^{-1}$)")
    mp = fs.tri_field(axs[0][1], tri, ex["bed"], "elevation", Normalize(-2000, 0))
    fs.colorbar(fig, mp, caxs[0][1], "Bed elevation (m)")
    finish_maps(axs, box, d, tri, stations=True)
    st, _, _ = stations(d["hybrid"]["ex"])
    for ax in axs[0]:
        ax.annotate("ApRES line", st[np.argmax(st[:, 0])], xytext=(12, 10),
                    textcoords="offset points", ha="left", fontsize=fs.FS_TICK,
                    arrowprops=dict(arrowstyle="-", color=BLACK, lw=0.6))
    locator(fig, axs[0][1], ex["xy"])
    fs.top_legend(fig, [GL_KEY, STATION_KEY])
    save(fig, "F_setting")


def locator(fig, ax, xy, rect=(0.13, 0.66, 0.24, 0.32)):
    r"""Antarctica in grey with the model domain in red, in the empty upper left of ``ax``."""
    a = antarctica()
    loc = ax.inset_axes(list(rect))
    loc.contourf(a["x"], a["y"], a["mask"], [0.5, 2.5, 4.5], colors=["0.62", "0.85"], zorder=0)
    loc.contour(a["x"], a["y"], (a["mask"] > 0).astype(float), [0.5], colors=BLACK,
                linewidths=0.3, zorder=0)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    loc.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec=fs.RED, lw=1.0))
    loc.set(xlim=(-2.75e6, 2.85e6), ylim=(-2.3e6, 2.4e6), aspect="equal", xticks=[], yticks=[])
    loc.set_facecolor("white")
    loc.set_rasterization_zorder(0.5)
    for sp in loc.spines.values():
        sp.set_linewidth(0.6)


# ── main text: ApRES profiles ───────────────────────────────────────────────
WINDOW = (100.0, 800.0)            # assimilated depth range (observations.DEPTH_MIN/MAX)
# Muted pastel fills for the profile letters (b-e), repeated on their markers in (a) so
# each profile can be found on the line. Kept clear of the model red and blue.
PICK_FILL = ("#cadcb9", "#d9cee9", "#efdcb0", "#e8c9d3")   # sage, lavender, sand, mauve


def apres_sites():
    r"""Observed eps_zz profiles of every ApRES station and the modelled MAP profiles of both
    models, matched by station name. Model zeta is turned into depth with the radar's own ice
    thickness, as the observations were when they were assimilated."""
    import h5py
    prof = {m: dict(np.load(str(RES / "paper" / f"profiles_{m}.npz"))) for m in MODELS}
    names = list(prof["hybrid"]["station"])
    assert names == list(prof["ssa"]["station"]), "station order differs between models"
    sites = []
    with h5py.File(str(APRES_FILE), "r") as f:
        for i, nm in enumerate(names):
            g = f[f"sites/{nm}"]
            H = float(g.attrs["H_used"])
            ok = np.isfinite(g["eps_zz"][...]) & np.isfinite(g["eps_sigma"][...])
            depth, eps, sig = g["depth"][...], g["eps_zz"][...], g["eps_sigma"][...]
            site = dict(name=nm, xy=prof["hybrid"]["xy"][i], H=H, depth=depth[ok], eps=eps[ok],
                        sig=sig[ok])
            # The samples the inversion saw: observations.load_apres, verbatim selection.
            w = ok & (sig > 0) & (depth >= WINDOW[0]) & (depth <= WINDOW[1])
            stride = max(1, int(w.sum()) // 5)
            site["used"] = (depth[w][::stride], eps[w][::stride], sig[w][::stride])
            for m in MODELS:
                site[m] = ((1 - prof[m]["zeta"]) * H, prof[m]["eps_mod"][i])
            sites.append(site)
    xy = np.array([s["xy"] for s in sites])
    c = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - c)
    along = (xy - c) @ vt[0]
    if np.corrcoef(along, xy[:, 0])[0, 1] < 0:      # distance increases inland (east)
        along = -along
    along = (along - along.min()) / 1e3
    order = np.argsort(along)
    for k in order:
        sites[k]["s_km"] = float(along[k])
    return [sites[k] for k in order]


def window_mean(depth, eps, sig=None):
    r"""Mean over the assimilated window and, given sigma, the standard error of that mean
    for errors as correlated as the Legendre fit makes them (no 1/sqrt(N) reduction)."""
    w = (depth >= WINDOW[0]) & (depth <= WINDOW[1])
    if sig is None:
        return float(np.mean(eps[w]))
    return float(np.mean(eps[w])), float(np.sqrt(np.mean(sig[w] ** 2)))


def profile_picks(sites):
    r"""Indices into ``sites`` of the profiles drawn in full: in each quarter of the line, the
    station where the two models' window means differ most."""
    s_km = np.array([s["s_km"] for s in sites])
    gap = np.abs(np.array([window_mean(*s["hybrid"]) - window_mean(*s["ssa"]) for s in sites]))
    quarter = np.minimum((4 * s_km / s_km.max()).astype(int), 3)
    return [int(np.flatnonzero(quarter == q)[np.argmax(gap[quarter == q])]) for q in range(4)]


def fig_apres(d, tri, area, n):
    r"""Main-text Fig. 2: what the ApRES strain rates add. (a) Window-mean eps_zz along the
    line, observed and from both MAPs; (b-e) full profiles at the station in each quarter of
    the line where the two models differ most."""
    sites = apres_sites()
    fig = plt.figure(figsize=(fs.FULL, 5.0), layout="constrained")
    gs = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.55))
    top = fig.add_subplot(gs[0, :])
    s_km = np.array([s["s_km"] for s in sites])
    obs = np.array([window_mean(*s["used"]) for s in sites]) * 1e3
    top.errorbar(s_km, obs[:, 0], yerr=obs[:, 1], fmt="d", ms=3.0, color=BLACK, ecolor="0.6",
                 elinewidth=0.6, capsize=0, zorder=3)
    mods = {}
    for m in MODELS:
        mod = mods[m] = np.array([window_mean(*s[m]) for s in sites]) * 1e3
        top.plot(s_km, mod, color=COL[m], marker=MARK[m], ms=3.0, markevery=4, zorder=2)
        n.setdefault("apres", {})[m] = dict(
            rms_window=float(np.sqrt(np.mean((mod - obs[:, 0]) ** 2))))
    top.axhline(0, color=GREY, lw=0.8, zorder=1)
    top.set(xlabel="Distance along the ApRES line (km)", xlim=(-1.5, s_km.max() + 1.5),
            ylabel=rf"${EPS}$, 100$-$800 m mean" "\n" r"($10^{-3}$ yr$^{-1}$)")
    lo, hi = top.get_ylim()
    top.set_ylim(lo, hi + 0.35 * (hi - lo))           # headroom for the letter and picks
    fs.grid(top)
    fs.panel_letter(top, "a", x=0.022, y=0.86)

    idx = profile_picks(sites)
    n["apres"]["picks"] = [dict(name=sites[k]["name"], s_km=float(s_km[k])) for k in idx]
    axs =[fig.add_subplot(gs[1, j]) for j in range(4)]
    # One x range for all four profiles, wide enough for every assimilated sample and its sigma.
    xlim = 5 * np.ceil(1.08e3 * max(np.max(np.abs(sites[k]["used"][1]) + sites[k]["used"][2])
                                    for k in idx) / 5)
    # Pick circles start right of the panel letter; a pick nearer the left edge gets a
    # slanted leader instead of a circle on top of the "a".
    x0, x1 = top.get_xlim()
    s_min = x0 + 0.075 * (x1 - x0)
    for j, (ax, k) in enumerate(zip(axs, idx)):
        s = sites[k]
        letter = "bcde"[j]
        top.annotate(letter, (s_km[k], obs[k, 0]), xytext=(max(s_km[k], s_min), 0.9),
                     textcoords=("data", "axes fraction"), ha="center", va="center",
                     fontsize=fs.FS_TICK, zorder=1,
                     bbox=dict(fs.LETTER_BOX, facecolor=PICK_FILL[j]),
                     arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5, ls=":",
                                     shrinkA=0, shrinkB=4))
        ax.axhspan(*WINDOW, color=GREY, alpha=0.15, lw=0, zorder=0)
        ax.plot(s["eps"] * 1e3, s["depth"], color="0.65", lw=0.4, zorder=1)
        du, eu, su = s["used"]
        ax.errorbar(eu * 1e3, du, xerr=su * 1e3, fmt="d", ms=3.5, color=BLACK, capsize=1.5,
                    elinewidth=0.7, zorder=4)
        for m in MODELS:
            z, em = s[m]
            ax.plot(em * 1e3, z, color=COL[m], marker=MARK[m], ms=3.0, markevery=5, zorder=3)
        ax.axvline(0, color=GREY, lw=0.8, zorder=1)
        # Depth axis ends at the deepest ApRES sample (to the next 50 m): below it the
        # models are not tested.
        bottom = 50.0 * np.ceil(float(s["depth"].max()) / 50.0)
        ax.set(ylim=(bottom, -0.13 * bottom), xlabel=rf"${EPS}$ ($10^{{-3}}$ yr$^{{-1}}$)")
        if j == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_xlim(-xlim, xlim)
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= bottom])
        fs.grid(ax)
        fs.panel_letter(ax, letter, x=0.13, y=0.94, facecolor=PICK_FILL[j])
    fs.top_legend(fig, [Line2D([], [], ls="", marker="d", ms=3.5, color=BLACK,
                               label=r"ApRES assimilated $\pm\sigma$"),
                        Line2D([], [], color="0.65", lw=0.6, label="ApRES profile")]
                  + [model_key(m) for m in MODELS]
                  + [Patch(color=GREY, alpha=0.15, label="assimilated depths")], ncol=3)
    save(fig, "F_apres")


# ── main text: retreat rates ────────────────────────────────────────────────
def fig_retreat(d, tri, area, n):
    r"""Main-text figure: 5-yr VAF loss rate and cumulative sea-level contribution, +/- 2 sigma."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.FULL, 2.9), layout="constrained")
    tot = []
    for m in MODELS:
        p = d[m]["pcg"]
        edges = np.r_[0.0, p["times"]]
        loss = -np.array(p["rate_Gt_per_yr"])
        sp, sq = 2 * np.array(p["sigma_post_rate"]), 2 * np.array(p["sigma_prior_rate"])
        a.stairs(loss + sq, edges, baseline=loss - sq, fill=True, color=COL[m], alpha=0.12, lw=0)
        a.stairs(loss, edges, color=COL[m], lw=fs.LW, baseline=None)
        a.errorbar(p["rate_t_mid"], loss, yerr=sp, fmt=MARK[m], color=COL[m], capsize=2, lw=0.9)
        ts = d[m]["ts"]
        sle = -(ts["vaf_Gt"] - ts["vaf_Gt"][0]) / GT_PER_MM_SLE
        t = np.r_[0.0, p["times"]]
        mu = np.interp(t, ts["times"], sle)
        s = 2 * np.r_[0.0, p["sigma_post_Gt_t"]] / GT_PER_MM_SLE
        b.plot(ts["times"], sle, color=COL[m], marker=MARK[m], markevery=every(ts["times"]))
        b.fill_between(t, mu - s, mu + s, color=COL[m], alpha=0.3, lw=0)
        tot.append(rf"{SHORT[m]}, 2065: {sle[-1]:.2f} $\pm$ {s[-1]:.2f} mm")
    a.axhline(0, color=GREY, lw=0.8, zorder=1)
    a.set(xlabel="Years after 2015", ylabel=r"VAF loss rate, 5-yr mean (Gt yr$^{-1}$)", xlim=(0, T))
    a.text(0.97, 0.04, rf"bars: $\pm 2{POST}$" "\n" rf"shading: $\pm 2{PRIOR}$",
           transform=a.transAxes, ha="right", va="bottom")
    b.set(xlabel="Years after 2015", ylabel="Sea-level contribution (mm)", xlim=(0, T))
    b.set_ylim(bottom=min(-0.2, b.get_ylim()[0]))
    b.text(0.15, 0.94, "\n".join(tot), transform=b.transAxes, va="top")
    b.text(0.97, 0.04, rf"bands: $\pm 2{POST}$", transform=b.transAxes, ha="right", va="bottom")
    for ax, k in zip((a, b), "ab"):
        fs.grid(ax)
        fs.panel_letter(ax, k)
    fs.top_legend(fig, [model_key(m) for m in MODELS])
    save(fig, "F_retreat_rates")
    write_rate_table(d)


def write_rate_table(d):
    r"""LaTeX rows of the retreat-rate table (generated; do not edit the output by hand)."""
    p = {m: d[m]["pcg"] for m in MODELS}
    pm = r"\pm"
    rows = []
    for i, t1 in enumerate(p["ssa"]["times"]):
        t0 = 0.0 if i == 0 else p["ssa"]["times"][i - 1]
        cells = [f"{2015 + int(t0)}--{2015 + int(t1)}"]
        for m in MODELS:
            loss, s = -p[m]["rate_Gt_per_yr"][i], 2 * p[m]["sigma_post_rate"][i]
            cells.append(f"${loss:.1f} {pm} {s:.1f}$")
        for m in MODELS:
            loss, s = -p[m]["rate_Gt_per_yr"][i], 2 * p[m]["sigma_post_rate"][i]
            cells.append(f"${loss / GT_PER_MM_SLE:.3f} {pm} {s / GT_PER_MM_SLE:.3f}$")
        rows.append(" & ".join(cells) + r" \\")
    tot = ["2015--2065 total"]
    for m in MODELS:
        tot.append(f"${-p[m]['dvaf_Gt_t'][-1]:.0f} {pm} {2 * p[m]['sigma_post_Gt']:.0f}$ Gt")
    for m in MODELS:
        tot.append(f"${-p[m]['dvaf_Gt_t'][-1] / GT_PER_MM_SLE:.2f} {pm} "
                   f"{2 * p[m]['sigma_post_Gt'] / GT_PER_MM_SLE:.2f}$ mm")
    # The closing rule lives here too: Copernicus' \bottomhline is a \noalign and cannot
    # follow \input inside a tabular.
    body = ("\n".join(rows) + "\n\\middlehline\n" + " & ".join(tot) + r" \\" + "\n"
            + "\\bottomhline\n")
    head = ("% Generated by scripts/paper_figures.py (hoffmaao/vertical_velocity) from the PCG\n"
            "% UQ summaries; do not edit by hand. VAF loss is positive; +/- is 2 sigma_post.\n")
    (OUT / "retreat_rates_rows.tex").write_text(head + body)
    print(f"Saved {OUT / 'retreat_rates_rows.tex'}", flush=True)


FIGS = {"S1": fig_summary, "S2": fig_controls, "S3": fig_data_fit, "S4": fig_posterior_std,
        "S5": fig_sensitivity, "S6": fig_trajectories, "S7": fig_grounding, "S8": fig_spectrum,
        "F_setting": fig_setting, "F_apres": fig_apres, "F_retreat": fig_retreat}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=list(FIGS), default=list(FIGS))
    args = ap.parse_args()
    d, tri, area = load()
    n = numbers(d, area)
    OUT.mkdir(parents=True, exist_ok=True)
    for k in args.only:
        FIGS[k](d, tri, area, n)          # some figures add the numbers they measure
    json.dump(n, open(OUT / "numbers.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
