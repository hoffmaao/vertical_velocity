r"""Figures for the MISMIP+ vertical-velocity OSSE: truth, recovery (surf vs vert),
and the θ_A identifiability signature (Δu vs Δw)."""
import numpy as np
import firedrake
import icepack
from firedrake import CheckpointFile, FunctionSpace, Function, project, tripcolor, sqrt, inner
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent
FIG = DATA / "figures"
FIG.mkdir(exist_ok=True)


def cbar(ax, obj, label):
    plt.colorbar(obj, ax=ax, fraction=0.025, pad=0.02).set_label(label, fontsize=9)


# ── Figure 1: the truth ──
with CheckpointFile(str(DATA / "mesh" / "mismip_truth.h5"), "r") as c:
    m2 = c.load_mesh()
    θA = c.load_function(m2, "log_fluidity_true")
    Cf = c.load_function(m2, "friction_true")
    us = c.load_function(m2, "u_surface")
with CheckpointFile(str(DATA / "mesh" / "mismip_truth_3d.h5"), "r") as c:
    m3 = c.load_mesh("firedrake_default_extruded")
    w3 = c.load_function(m3, "w_3d")
Q2 = FunctionSpace(m2, "CG", 1)
spd = Function(Q2, name="speed").interpolate(sqrt(inner(us, us)))
wbase = m3._base_mesh
w2 = Function(FunctionSpace(wbase, "CG", 1))
w2.dat.data[:] = icepack.depth_average(w3).dat.data_ro  # depth-avg vertical velocity, 2D map

fig, axs = plt.subplots(4, 1, figsize=(11, 11))
for ax, fld, lbl, cm in [(axs[0], θA, "true θ_A = log(A/A₀)", "viridis"),
                         (axs[1], Function(Q2).interpolate(Cf), "true friction C", "magma"),
                         (axs[2], spd, "surface speed (m/yr)", "Blues")]:
    cbar(ax, tripcolor(fld, axes=ax, cmap=cm), lbl); ax.set_title(lbl, fontsize=10); ax.set_aspect("equal")
cbar(axs[3], tripcolor(w2, axes=axs[3], cmap="PiYG"), "depth-avg w (m/yr)")
axs[3].set_title("vertical velocity w (the ApRES observable)", fontsize=10); axs[3].set_aspect("equal")
fig.suptitle("MISMIP+ OSSE truth (Hybrid, Weertman)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "mismip_osse_truth.png", dpi=130); plt.close(fig)
print(f"Saved {FIG/'mismip_osse_truth.png'}", flush=True)


# ── Figure 2: recovery, surf vs vert ──
def load_inv(tag):
    with CheckpointFile(str(DATA / "results" / f"mismip_inv_{tag}.h5"), "r") as c:
        m = c.load_mesh()
        return (m, c.load_function(m, "theta_A_rec"), c.load_function(m, "theta_A_true"),
                c.load_function(m, "theta_C_rec"))


try:
    ms, Arec_s, Atrue, Crec_s = load_inv("hybrid_surf_d20km")
    mv, Arec_v, Atrue_v, Crec_v = load_inv("hybrid_vert_d20km")
    s_npz = np.load(str(DATA / "results" / "mismip_inv_hybrid_surf_d20km.npz"))
    v_npz = np.load(str(DATA / "results" / "mismip_inv_hybrid_vert_d20km.npz"))
    expl_s = 100 * (1 - float(s_npz["rmse_A"]) / float(s_npz["truth_rms_A"]))
    expl_v = 100 * (1 - float(v_npz["rmse_A"]) / float(v_npz["truth_rms_A"]))

    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    vlim = float(np.abs(Atrue.dat.data_ro).max())
    for ax, fld, t in [(axs[0, 0], Atrue, "θ_A TRUTH"),
                       (axs[0, 1], Arec_s, f"recovered: surf only ({expl_s:.0f}%)"),
                       (axs[0, 2], Arec_v, f"recovered: surf + w ({expl_v:.0f}%)")]:
        cbar(ax, tripcolor(fld, axes=ax, cmap="viridis", vmin=-vlim, vmax=vlim), "θ_A")
        ax.set_title(t, fontsize=10); ax.set_aspect("equal")
    errs = Function(Arec_s.function_space(), name="es")
    errs.dat.data[:] = Arec_s.dat.data_ro - Atrue.dat.data_ro
    errv = Function(Arec_v.function_space(), name="ev")
    errv.dat.data[:] = Arec_v.dat.data_ro - Atrue_v.dat.data_ro
    for ax, fld, t in [(axs[1, 1], errs, "error: surf only"), (axs[1, 2], errv, "error: surf + w")]:
        cbar(ax, tripcolor(fld, axes=ax, cmap="coolwarm", vmin=-vlim, vmax=vlim), "θ_A err")
        ax.set_title(t, fontsize=10); ax.set_aspect("equal")
    axs[1, 0].axis("off")
    axs[1, 0].text(0.05, 0.5, f"θ_A explained variance\n\n surf only:  {expl_s:.0f}%\n surf + w:  {expl_v:.0f}%\n\n"
                   f"(joint θ_A+θ_C inversion,\n Δ=20 km, sliding-dominated:\n the fluidity/friction tradeoff\n"
                   f" dominates the global metric)", fontsize=11, va="center")
    fig.suptitle("MISMIP+ OSSE: fluidity recovery (Hybrid, joint θ_A+θ_C, Δ=20 km)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "mismip_osse_recovery.png", dpi=130); plt.close(fig)
    print(f"Saved {FIG/'mismip_osse_recovery.png'}", flush=True)
except Exception as e:
    print(f"recovery figure skipped: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    pass
