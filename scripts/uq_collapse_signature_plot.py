r"""Spatial map of the mean collapse θ-signature — WHERE does runaway get triggered?

Loads the mean collapse perturbation (δθ_A, δθ_C) saved by uq_collapse_dissect.py and
plots it on the Thwaites mesh, with the grounding line overlaid. A localized signal near
the grounding line / trunk = the parameter uncertainty that drives the collapse tail.

NOTE: with only a few collapse samples the mean is noisy; treat as indicative until more
tail events accrue. Writes figures/uq_collapse_signature.png.
"""
import numpy as np
import firedrake as fd
from firedrake import CheckpointFile, FunctionSpace, Function, Constant, max_value
from icepack.constants import ice_density as ρ_I, water_density as ρ_W
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
SIG = DATA / "results" / "collapse_signature.h5"
MESH = DATA / "mesh" / "thwaites.h5"


def main():
    with CheckpointFile(str(SIG), "r") as c:
        m = c.load_mesh()
        dA = c.load_function(m, "collapse_dthetaA")
        dC = c.load_function(m, "collapse_dthetaC")
    # grounding line from geometry (same base mesh)
    with CheckpointFile(str(MESH), "r") as c:
        mh = c.load_mesh()
        h = c.load_function(mh, "thickness"); s = c.load_function(mh, "surface"); b = c.load_function(mh, "bed")
    Q = FunctionSpace(mh, "CG", 1)
    s_float = Function(Q).interpolate(b + Constant(float(ρ_W / ρ_I)) * max_value(-b, Constant(0.0)))
    haf = Function(Q).interpolate(s - s_float)   # >0 grounded above flotation, ~0 at GL

    xy = m.coordinates.dat.data_ro
    tri = fd.utils.unique  # placeholder noop to avoid lint; real triangulation below
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    # build triangulation from mesh cells
    cells = m.coordinates.function_space().cell_node_list
    triang = mtri.Triangulation(xy[:, 0] / 1e3, xy[:, 1] / 1e3, cells)
    haf_v = haf.dat.data_ro

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, fld, name in [(axes[0], dA.dat.data_ro, "δθ_A  (log-fluidity)"),
                          (axes[1], dC.dat.data_ro, "δθ_C  (log-friction)")]:
        lim = np.percentile(np.abs(fld), 99)
        tpc = ax.tripcolor(triang, fld, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud")
        ax.tricontour(triang, haf_v, levels=[0.0], colors="k", linewidths=1.2)  # grounding line
        ax.set_aspect("equal"); ax.set_title(f"mean collapse perturbation: {name}")
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
        fig.colorbar(tpc, ax=ax, shrink=0.8, label="Δθ")
    fig.suptitle("Where collapse is triggered  (mean over collapse-tail samples; GL = black)", y=1.02)
    fig.tight_layout()
    fp = DATA / "figures" / "uq_collapse_signature.png"
    fig.savefig(str(fp), dpi=200, bbox_inches="tight")
    print(f"Saved {fp}")
    # quick quant: where is the signal strongest, relative to GL?
    near_gl = np.abs(haf_v) < 50.0   # within 50 m of flotation
    print(f"δθ_A: rms near-GL={np.sqrt((dA.dat.data_ro[near_gl]**2).mean()):.3f} vs overall {np.sqrt((dA.dat.data_ro**2).mean()):.3f}")
    print(f"δθ_C: rms near-GL={np.sqrt((dC.dat.data_ro[near_gl]**2).mean()):.3f} vs overall {np.sqrt((dC.dat.data_ro**2).mean()):.3f}")


if __name__ == "__main__":
    main()
