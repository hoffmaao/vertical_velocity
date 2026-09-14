r"""Validate the MISMIP+ OSSE premise: does the Hybrid truth carry a depth-varying
ε_zz (vertical shear) over the ridges that SSA structurally cannot represent?

ε_zz = -(∂u_x/∂x + ∂u_y/∂y). In SSA the velocity is depth-constant, so ε_zz(ζ) is
flat. In the Hybrid model vertical shear makes ε_zz vary with depth ζ. The OSSE
hypothesis is that ApRES-like ε_zz(ζ) profiles constrain that shear — so the truth
MUST show a depth gradient, strongest over ridge crests where bed slope drives shear.

Extracts ε_zz(ζ) profiles at ridge crests vs troughs (mid-channel), quantifies the
depth gradient, and plots profiles + a surface map. Run after mismip_reference.py.
"""
import numpy as np
import firedrake as fd
from firedrake import CheckpointFile, FunctionSpace, VertexOnlyMesh, Function
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent.parent
Lx, Ly = 640e3, 80e3
WAVELENGTH = 60e3   # transverse_ridges wavelength (must match mismip_reference.py)


def profile(m3, ez, x, y, nz=21):
    """ε_zz(ζ) at column (x,y); ζ=0 base → 1 surface."""
    zetas = np.linspace(0.02, 0.98, nz)
    xyz = np.column_stack([np.full(nz, x), np.full(nz, y), zetas])
    vom = VertexOnlyMesh(m3, xyz, missing_points_behaviour="warn")
    kept = vom.coordinates.dat.data_ro
    vals = Function(FunctionSpace(vom, "DG", 0)).interpolate(ez).dat.data_ro.copy()
    # VOM may reorder/drop; sort by ζ for a clean profile
    order = np.argsort(kept[:, 2])
    return kept[order, 2], vals[order]


def main():
    with CheckpointFile(str(DATA / "mesh" / "mismip_truth_3d.h5"), "r") as c:
        m3 = c.load_mesh("firedrake_default_extruded")   # functions live on the extruded mesh, not the base
        ez = c.load_function(m3, "eps_zz_3d")
    print(f"loaded eps_zz_3d: {ez.dat.data_ro.shape[0]} dofs, "
          f"range [{ez.dat.data_ro.min():.2e}, {ez.dat.data_ro.max():.2e}] 1/yr", flush=True)

    ymid = Ly / 2
    # crests at sin(2πx/λ)=+1 → x=λ/4 + kλ ; troughs at x=3λ/4 + kλ
    crests = [WAVELENGTH * 0.25 + k * WAVELENGTH for k in range(2, 7)]   # skip x<120km (thin/divide)
    troughs = [WAVELENGTH * 0.75 + k * WAVELENGTH for k in range(2, 7)]

    fig, (axp, axm) = plt.subplots(1, 2, figsize=(13, 5))
    grads = {"crest": [], "trough": []}
    for label, xs, col in [("crest", crests, "C3"), ("trough", troughs, "C0")]:
        for j, x in enumerate(xs):
            z, e = profile(m3, ez, x, ymid)
            if len(z) < 3:
                continue
            axp.plot(e, z, col, alpha=0.6, lw=1.5,
                     label=(f"{label}" if j == 0 else None))
            grads[label].append((e.max() - e.min()))   # depth span of ε_zz
    axp.set_xlabel(r"$\varepsilon_{zz}$  (1/yr)")
    axp.set_ylabel(r"$\zeta$  (0=base, 1=surface)")
    axp.set_title("Depth profiles of $\\varepsilon_{zz}$ (mid-channel)")
    axp.legend()
    axp.grid(alpha=0.3)

    # surface map of depth-span |ε_zz(surf) − ε_zz(base)| sampled on a coarse grid
    xs = np.linspace(140e3, 460e3, 60)
    ys = np.linspace(8e3, Ly - 8e3, 16)
    GX, GY = np.meshgrid(xs, ys)
    span = np.full(GX.shape, np.nan)
    for iy in range(GX.shape[0]):
        for ix in range(GX.shape[1]):
            z, e = profile(m3, ez, GX[iy, ix], GY[iy, ix], nz=9)
            if len(z) >= 3:
                span[iy, ix] = e.max() - e.min()
    pcm = axm.pcolormesh(GX / 1e3, GY / 1e3, span, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=axm, label=r"depth span of $\varepsilon_{zz}$ (1/yr)")
    for x in crests:
        if 140e3 < x < 460e3:
            axm.axvline(x / 1e3, color="w", ls=":", lw=0.8)
    axm.set_xlabel("x (km)")
    axm.set_ylabel("y (km)")
    axm.set_title("Vertical-shear signal (ridge crests dotted)")

    fig.tight_layout()
    out = DATA / "figures" / "mismip_truth_shear.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    cr = np.array(grads["crest"]); tr = np.array(grads["trough"])
    print(f"ε_zz depth-span (shear signal):", flush=True)
    print(f"  crests : mean {cr.mean():.2e}  (n={len(cr)})", flush=True)
    print(f"  troughs: mean {tr.mean():.2e}  (n={len(tr)})", flush=True)
    print(f"  crest/trough ratio: {cr.mean()/max(tr.mean(),1e-30):.2f}", flush=True)
    verdict = "DEPTH-VARYING (OSSE viable)" if cr.mean() > 1e-4 else "FLAT — premise FAILS"
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
