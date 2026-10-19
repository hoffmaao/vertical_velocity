r"""Assess the narrow streaks of high deformation share in a hybrid MAP, without changing it.

At some MAPs the hybrid model puts narrow, flow-aligned ridges of high friction
(theta_C) in the grounded trunk, where 50-80 % of the motion switches from
basal sliding to internal deformation while the surface speed hardly changes.
This script asks whether those features are constrained by the data, tied to
the bed, and robust to the prior, using only existing products:

  1. the deformation share 1 - |u_b| / |u_s| at the MAP (one forward solve),
     and the "streak" nodes where it exceeds --threshold in grounded ice;
  2. pointwise prior and posterior standard deviation of theta_C at the streak
     nodes and at random background nodes, from the eigendecomposition:
         Gamma_post = A^-1 - sum_i D_i w_i w_i^T,   D_i = lambda_i / (1 + lambda_i)
     (A^-1 columns by sparse LU; truncation makes the posterior value an upper
     bound). The ratio sigma_post / sigma_prior says how much the data inform
     theta_C there, and theta_C / sigma_post how far the MAP sits from the
     prior mean in posterior standard deviations;
  3. the bed: the correlation of the streaks with the high-passed bed
     b - b_smooth (Helmholtz over --bed-smooth m), i.e. whether they sit on
     bed bumps where form drag would be absorbed into the friction;
  4. theta_C at the streak nodes in every converged sweep MAP of the same
     configuration, against (delta, ELL): whether the ridges persist as the
     prior changes or appear only when it weakens.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/diagnose_deformation_streaks.py \
        --friction rc --map sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km --K 1500

Writes figures/diag_deformation_streaks_<map>.png and prints the numbers.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from firedrake import (
    CheckpointFile, Constant, Function, FunctionSpace, VertexOnlyMesh, derivative, dx, grad,
    inner, solve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from physical_prior import MESH_FILE  # noqa: E402
from prior_precond import Prior  # noqa: E402
from problem import build_problem  # noqa: E402

DATA = Path(__file__).resolve().parent.parent
N_BACKGROUND = 400
RNG = np.random.default_rng(0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--friction", choices=["rc", "weertman"], required=True)
    ap.add_argument("--map", required=True, help="the --name the hybrid inversion ran with")
    ap.add_argument("--K", type=int, default=1500)
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="deformation share above which a grounded node is a streak node")
    ap.add_argument("--bed-smooth", type=float, default=5000.0)
    return ap.parse_args()


def at_zeta(P, u, xy, zeta):
    vom = VertexOnlyMesh(P["mesh"], np.column_stack([xy, np.full(len(xy), zeta)]),
                         missing_points_behaviour="error", redundant=False)
    Δ, Δin = FunctionSpace(vom, "DG", 0), FunctionSpace(vom.input_ordering, "DG", 0)
    return np.hypot(*[Function(Δin).interpolate(Function(Δ).interpolate(u[i])).dat.data_ro
                      for i in range(2)])


def main():
    args = parse_args()
    P = build_problem("hybrid", args.friction, True)
    Q, Q_2d, mesh_2d = P["Q"], P["Q_2d"], P["mesh_2d"]
    n = Q.dim()
    with CheckpointFile(str(DATA / "mesh" / f"map_{args.map}.h5"), "r") as chk:
        m = chk.load_mesh()
        tA = chk.load_function(m, "log_fluidity").dat.data_ro.copy()
        tC = chk.load_function(m, "log_friction").dat.data_ro.copy()
    θA, θC = Function(Q), Function(Q)
    θA.dat.data[:], θC.dat.data[:] = tA, tC
    u = P["forward"](θA, θC)
    xy = mesh_2d.coordinates.dat.data_ro
    Us, Ub = at_zeta(P, u, xy, 1.0), at_zeta(P, u, xy, 0.0)
    share = 1 - Ub / np.maximum(Us, 1e-6)

    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mm = chk.load_mesh()
        b_m = chk.load_function(mm, "bed")
        h = chk.load_function(mm, "thickness").dat.data_ro.copy()
        b = b_m.dat.data_ro.copy()
    grounded = h - 1028.0 / 917.0 * np.maximum(0.0, -b) > 10.0
    streak = grounded & (share > args.threshold) & (Us > 10.0)
    bg_pool = np.flatnonzero(grounded & (share < 0.05) & (Us > 10.0))
    bg = RNG.choice(bg_pool, size=min(N_BACKGROUND, bg_pool.size), replace=False)
    st = np.flatnonzero(streak)
    print(f"{st.size} streak nodes (share > {args.threshold}) of {grounded.sum()} grounded nodes; "
          f"median share there {np.median(share[st]):.2f}", flush=True)

    # 2. pointwise prior / posterior sigma of theta_C.
    with open(DATA / "results" / f"eigendec_{args.map}_K{args.K}.json") as f:
        meta = json.load(f)
    eig = np.load(str(DATA / "results" / f"eigendec_{args.map}_K{args.K}.npz"))
    lam = eig["eigenvalues"]
    D = lam / (1 + lam)
    WC = eig["vecs"][n:, :]
    pr_C = Prior(Q_2d, meta["delta_C"], meta["gamma_C"], meta.get("prior_form", "h1"))
    nodes = np.r_[st, bg]
    var_prior = np.empty(nodes.size)
    for k, j in enumerate(nodes):
        e = np.zeros(n)
        e[j] = 1.0
        var_prior[k] = pr_C.apply_covariance(e)[j]
    var_post = var_prior - (D * WC[nodes, :] ** 2).sum(axis=1)
    sd_prior, sd_post = np.sqrt(var_prior), np.sqrt(np.maximum(var_post, 0))
    ns = st.size
    for lab, sl in (("streak", slice(0, ns)), ("background", slice(ns, None))):
        print(f"{lab:10s}: sigma_prior(theta_C) median {np.median(sd_prior[sl]):.3f}, sigma_post median "
              f"{np.median(sd_post[sl]):.3f} (ratio {np.median(sd_post[sl] / sd_prior[sl]):.2f}); "
              f"|theta_C| median {np.median(np.abs(tC[nodes[sl]])):.2f} = "
              f"{np.median(np.abs(tC[nodes[sl]]) / sd_post[sl]):.1f} posterior sigma", flush=True)

    # 3. the bed.
    bs = Function(b_m.function_space()).assign(b_m)
    L = Constant(args.bed_smooth)
    J = 0.5 * ((bs - b_m) ** 2 + L ** 2 * inner(grad(bs), grad(bs))) * dx
    solve(derivative(J, bs) == 0, bs)
    db = b - bs.dat.data_ro
    print(f"high-passed bed (b - b_{args.bed_smooth / 1e3:g}km): streak median {np.median(db[st]):+.0f} m, "
          f"background median {np.median(db[bg]):+.0f} m; corr(share, b - b_s) over grounded ice "
          f"{np.corrcoef(share[grounded], db[grounded])[0, 1]:+.2f}, corr(theta_C, b - b_s) "
          f"{np.corrcoef(tC[grounded], db[grounded])[0, 1]:+.2f}", flush=True)

    # 4. persistence across the sweep.
    m_ = re.match(r"sweep_(hybrid_[a-z]+)_d[^_]+_L[^_]+km(?P<sfx>.*)", args.map)
    rows = []
    for fj in sorted(glob.glob(str(DATA / "results" / f"map_sweep_hybrid_{args.friction}_d*_L*km{m_.group('sfx')}.json"))):
        d = json.load(open(fj))
        if not d["status"].startswith("converged"):
            continue
        name = Path(fj).stem[4:]
        with CheckpointFile(str(DATA / "mesh" / f"map_{name}.h5"), "r") as chk:
            mk = chk.load_mesh()
            tCk = chk.load_function(mk, "log_friction").dat.data_ro
        rows.append((d["delta"], d["ELL"] / 1e3, np.median(tCk[st]), np.median(tCk[bg]),
                     np.corrcoef(tCk[grounded], tC[grounded])[0, 1]))
    print("\n  delta    ELL   theta_C streak  theta_C background  corr with this MAP", flush=True)
    for r in sorted(rows):
        print(f"  {r[0]:.0e}  {r[1]:5g}   {r[2]:+8.2f}        {r[3]:+8.2f}          {r[4]:+.2f}", flush=True)

    tri = Triangulation(xy[:, 0], xy[:, 1], mesh_2d.coordinates.function_space().cell_node_list)
    x0 = np.percentile(xy[:, 0], 45)
    fig, axs = plt.subplots(2, 3, figsize=(17, 12), constrained_layout=True)
    for ax, (v, cm, nm, t) in zip(axs[0], [
            (share, "viridis", None, "deformation share 1 - |u_b|/|u_s|"),
            (tC, "RdBu_r", TwoSlopeNorm(0, -3, 3), "theta_C at the MAP"),
            (db, "BrBG_r", TwoSlopeNorm(0, -150, 150), f"bed - {args.bed_smooth / 1e3:g} km smoothed bed (m)")]):
        pc = ax.tripcolor(tri, v, cmap=cm, norm=nm, shading="gouraud", rasterized=True,
                          vmin=0 if nm is None else None, vmax=1 if nm is None else None)
        ax.tricontour(tri, share, levels=[args.threshold], colors="k", linewidths=0.5)
        fig.colorbar(pc, ax=ax, shrink=0.8)
        ax.set(xlim=(x0, xy[:, 0].max()), title=t)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax = axs[1, 0]
    ax.hist(sd_post[:ns] / sd_prior[:ns], bins=40, alpha=0.6, label="streak nodes", density=True)
    ax.hist(sd_post[ns:] / sd_prior[ns:], bins=40, alpha=0.6, label="background", density=True)
    ax.set(xlabel="sigma_post / sigma_prior of theta_C", title="data information at the nodes")
    ax.legend()
    ax = axs[1, 1]
    ax.hist(np.abs(tC[st]) / sd_post[:ns], bins=40, alpha=0.6, label="streak", density=True)
    ax.hist(np.abs(tC[bg]) / sd_post[ns:], bins=40, alpha=0.6, label="background", density=True)
    ax.set(xlabel="|theta_C| / sigma_post", title="MAP distance from the prior mean")
    ax.legend()
    ax = axs[1, 2]
    for dl in sorted({r[0] for r in rows}):
        rr = sorted([r for r in rows if r[0] == dl], key=lambda r: r[1])
        ax.plot([r[1] for r in rr], [r[2] for r in rr], "o-", label=f"delta={dl:.0e}")
    ax.set(xscale="log", xlabel="ELL (km)", ylabel="median theta_C at the streak nodes",
           title="persistence across the sweep")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.legend(fontsize=8)
    fig.suptitle(f"Deformation-share streaks at {args.map}", fontsize=13)
    out = DATA / "figures" / f"diag_deformation_streaks_{args.map}.png"
    fig.savefig(str(out), dpi=130)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
