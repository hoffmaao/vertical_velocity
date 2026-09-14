r"""Demonstrate the working OSSE signal on the EXISTING reference (mismip-degree4.h5).

The proposal's observable is the higher-order vertical velocity w (icepack.utilities.
vertical_velocity). SSA (plug flow) gives a w that is linear in depth; the Hybrid (HO)
model adds a depth-varying shear correction. That HO−SSA difference is what ApRES
vertical-velocity observations can constrain. This loads the real spun-up reference,
computes w(x,ζ) along the centerline, and quantifies the HO signal + its depth structure.
"""
import numpy as np
import firedrake
import icepack
import icepack.utilities
from firedrake import (CheckpointFile, FunctionSpace, VertexOnlyMesh, Function,
                       project, Constant)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROP = Path("/media/andrew/wd1/projects/vertical_velocity_proposal/simulations")
OUT = Path("/media/andrew/wd1/projects/vertical_velocity/figures")
Lx, Ly = 640e3, 80e3


def main():
    with CheckpointFile(str(PROP / "mismip-degree4.h5"), "r") as chk:
        mesh = chk.load_mesh(name="mismip")
        h = chk.load_function(mesh, name="thickness")
        s = chk.load_function(mesh, name="surface")
        u = chk.load_function(mesh, name="velocity")
    vdeg = u.ufl_element().sub_elements[0].sub_elements[1].degree()
    print(f"reference velocity: CG{u.ufl_element().sub_elements[0].sub_elements[0].degree()} "
          f"x GLL{vdeg}; speed 0-{np.sqrt((u.dat.data_ro**2).sum(1)).max():.0f} m/yr", flush=True)

    # higher-order vertical velocity
    w = icepack.utilities.vertical_velocity(velocity=u, thickness=h, basal_mass_balance=Constant(0.0))
    Qw = FunctionSpace(mesh, "CG", 2, vfamily="GLL", vdegree=vdeg)
    wp = project(w, Qw)

    # sample w(x, ζ) along the centerline; compare to a depth-LINEAR (SSA plug) profile
    nx, nz = 120, 21
    xs = np.linspace(20e3, 520e3, nx)
    zetas = np.linspace(0.0, 1.0, nz)
    W = np.full((nx, nz), np.nan)
    for i, x in enumerate(xs):
        xyz = np.column_stack([np.full(nz, x), np.full(nz, Ly / 2), zetas])
        vom = VertexOnlyMesh(mesh, xyz, missing_points_behaviour="warn")
        co = vom.coordinates.dat.data_ro
        val = Function(FunctionSpace(vom, "DG", 0)).interpolate(wp).dat.data_ro
        if len(co) < 3:
            continue
        order = np.argsort(co[:, 2])
        W[i, :len(order)] = val[order]

    # HO signal = deviation of w(ζ) from the straight base→surface line (the SSA/plug shape)
    ho_dev = np.full((nx, nz), np.nan)
    for i in range(nx):
        col = W[i]
        if np.isnan(col).any():
            continue
        lin = np.linspace(col[0], col[-1], nz)   # SSA plug: w linear in depth
        ho_dev[i] = col - lin
    valid = ~np.isnan(ho_dev).any(axis=1)
    print(f"w(HO) range over centerline: [{np.nanmin(W):.4e}, {np.nanmax(W):.4e}] m/yr", flush=True)
    print(f"HO depth-nonlinearity (w − plug-line): max |dev| = {np.nanmax(np.abs(ho_dev)):.4e} m/yr", flush=True)
    print(f"  (ApRES vertical-velocity precision ~1e-3 m/yr → signal is "
          f"{'MEASURABLE' if np.nanmax(np.abs(ho_dev))>1e-3 else 'marginal'})", flush=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    pcm = a1.pcolormesh(xs / 1e3, zetas, W.T, shading="auto", cmap="PiYG", vmin=-0.013, vmax=0.013)
    fig.colorbar(pcm, ax=a1, label=r"$w$ (m/yr)")
    a1.set_title("HO vertical velocity $w(x,\\zeta)$ — centerline")
    a1.set_xlabel("x (km)"); a1.set_ylabel(r"$\zeta$")
    pcm2 = a2.pcolormesh(xs / 1e3, zetas, ho_dev.T, shading="auto", cmap="coolwarm")
    fig.colorbar(pcm2, ax=a2, label=r"$w - w_{\rm plug}$ (m/yr)")
    a2.set_title("HO signal (deviation from SSA plug-flow $w$)")
    a2.set_xlabel("x (km)"); a2.set_ylabel(r"$\zeta$")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "mismip_vv_signal.png", dpi=130)
    print(f"Saved {OUT / 'mismip_vv_signal.png'}", flush=True)


if __name__ == "__main__":
    main()
