r"""Plot eigendecomposition spectrum and leading mode spatial patterns.

Panels:
  (a) Eigenvalue spectrum λ_i
  (b) Data influence D_i = λ_i / (1+λ_i) and cumulative n_eff
  (c) Posterior weight 1/√(1+λ_i) — how much each mode contributes to uncertainty
  (d-g) Spatial patterns of the 4 leading modes (θ_A and θ_C components)

Usage:
    python plot_eigendec.py
"""
import numpy as np
import firedrake
from firedrake import Function, FunctionSpace, MixedFunctionSpace, CheckpointFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
EIGVAL_FILE = DATA_DIR / "results" / "eigenvalues.txt"
EIGDEC_FILE = DATA_DIR / "mesh" / "eigendec.h5"
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    # ── Load eigenvalues ───────────────────────────────────────────
    data = np.loadtxt(str(EIGVAL_FILE))
    idx = data[:, 0].astype(int)
    vals = data[:, 1]
    N = len(vals)

    D = vals / (1.0 + vals)             # data influence
    weights = 1.0 / np.sqrt(1.0 + vals) # posterior std weight
    cumD = np.cumsum(D)                  # cumulative n_eff

    print(f"Eigenvalues: {N} modes", flush=True)
    print(f"  range: [{vals.min():.2e}, {vals.max():.2e}]", flush=True)
    print(f"  n_eff = {D.sum():.1f}", flush=True)
    print(f"  All D_i > 0.999: all modes fully data-constrained", flush=True)

    # ── Load modes for spatial plots ───────────────────────────────
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    QQ = MixedFunctionSpace([Q_2d, Q_2d])

    modes_A = []
    modes_C = []
    n_show = min(4, N)
    with CheckpointFile(str(EIGDEC_FILE), "r") as chk:
        eig_mesh = chk.load_mesh()
        Q_eig = FunctionSpace(eig_mesh, "CG", 1)
        QQ_eig = MixedFunctionSpace([Q_eig, Q_eig])
        for i in range(n_show):
            mode = chk.load_function(eig_mesh, name=f"mode_{i:04d}")
            mA = Function(Q_2d, name=f"mode_{i}_A")
            mC = Function(Q_2d, name=f"mode_{i}_C")
            mA.dat.data[:] = mode.sub(0).dat.data_ro
            mC.dat.data[:] = mode.sub(1).dat.data_ro
            modes_A.append(mA)
            modes_C.append(mC)

    # ── Figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))

    # Top row: spectrum plots (3 panels)
    gs_top = fig.add_gridspec(1, 3, top=0.95, bottom=0.62, hspace=0.3, wspace=0.35,
                               left=0.05, right=0.95)

    # (a) Eigenvalue spectrum
    ax = fig.add_subplot(gs_top[0])
    ax.semilogy(idx, vals, "ko-", ms=5, lw=1)
    ax.set_xlabel("Mode index $i$", fontsize=11)
    ax.set_ylabel("Eigenvalue $\\lambda_i$", fontsize=11)
    ax.set_title("(a) Eigenvalue spectrum", fontsize=12)
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlim(-1, N)

    # (b) Data influence D_i and cumulative
    ax = fig.add_subplot(gs_top[1])
    ax2 = ax.twinx()
    ax.bar(idx, D, color="C0", alpha=0.7, label="$D_i = \\lambda_i/(1+\\lambda_i)$")
    ax2.plot(idx, cumD, "r-", lw=2, label="Cumulative $n_{\\mathrm{eff}}$")
    ax.set_xlabel("Mode index $i$", fontsize=11)
    ax.set_ylabel("Data influence $D_i$", fontsize=11, color="C0")
    ax2.set_ylabel("Cumulative $n_{\\mathrm{eff}}$", fontsize=11, color="r")
    ax.set_title("(b) Data influence per mode", fontsize=12)
    ax.set_ylim(0.999, 1.001)
    ax2.set_ylim(0, N + 2)
    ax.set_xlim(-1, N)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    # (c) Posterior weight
    ax = fig.add_subplot(gs_top[2])
    ax.semilogy(idx, weights, "s-", color="C2", ms=5, lw=1)
    ax.set_xlabel("Mode index $i$", fontsize=11)
    ax.set_ylabel("$1/\\sqrt{1+\\lambda_i}$", fontsize=11)
    ax.set_title("(c) Posterior std weight per mode", fontsize=12)
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlim(-1, N)

    # Bottom: spatial patterns of leading 4 modes (2 rows × 4 cols)
    gs_bot = fig.add_gridspec(2, n_show, top=0.52, bottom=0.02, hspace=0.25, wspace=0.3,
                               left=0.05, right=0.95)

    for i in range(n_show):
        # θ_A component
        ax = fig.add_subplot(gs_bot[0, i])
        vmax = max(abs(modes_A[i].dat.data_ro).max(), 1e-10)
        tc = firedrake.tripcolor(modes_A[i], axes=ax, cmap="RdBu_r",
                                  vmin=-vmax, vmax=vmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.03)
        fig.colorbar(tc, cax=cax)
        ax.set_title(f"Mode {i}: $\\theta_A$\n$\\lambda={vals[i]:.1e}$", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

        # θ_C component
        ax = fig.add_subplot(gs_bot[1, i])
        vmax = max(abs(modes_C[i].dat.data_ro).max(), 1e-10)
        tc = firedrake.tripcolor(modes_C[i], axes=ax, cmap="RdBu_r",
                                  vmin=-vmax, vmax=vmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.03)
        fig.colorbar(tc, cax=cax)
        ax.set_title(f"Mode {i}: $\\theta_C$", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    out = FIG_DIR / "eigendec_spectrum.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"Saved {out}", flush=True)

    # ── Print summary table ────────────────────────────────────────
    print(f"\n{'i':>3s}  {'λ_i':>12s}  {'D_i':>8s}  {'1/√(1+λ)':>12s}  {'cum n_eff':>10s}")
    print("-" * 55)
    for i in range(N):
        print(f"{i:3d}  {vals[i]:12.3e}  {D[i]:8.6f}  {weights[i]:12.3e}  {cumD[i]:10.1f}")


if __name__ == "__main__":
    main()
