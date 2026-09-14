r"""Build a coarse Thwaites mesh (~few thousand verts) for fast inversion tests.

Same recipe as build_mesh.py but with coarser RES_FINE/RES_COARSE.
Output: mesh/thwaites_coarse.h5 with thickness, bed, surface, velocity, phi_eff.
"""
import sys
from pathlib import Path

# Reuse build_mesh.py functions but override resolution and output path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_mesh  # noqa

# Override resolutions
build_mesh.RES_FINE = 5_000      # m
build_mesh.RES_COARSE = 15_000   # m

# Patch interpolate_and_save to write to a different file name
_orig = build_mesh.interpolate_and_save
def interpolate_and_save_coarse(msh_path):
    import firedrake
    from firedrake import (
        Function, FunctionSpace, VectorFunctionSpace, Constant,
        max_value, inner, grad, dx,
    )
    import icepack
    import rasterio
    import numpy as np

    mesh = firedrake.Mesh(msh_path)
    print(f"Firedrake mesh: {mesh.num_vertices()} verts, {mesh.num_cells()} cells")

    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)

    b = icepack.interpolate(rasterio.open(f"netcdf:{build_mesh.BEDMACHINE}:bed"), Q)
    b.rename("bed")

    H_raw = icepack.interpolate(
        rasterio.open(f"netcdf:{build_mesh.BEDMACHINE}:thickness"), Q)
    H = Function(Q, name="thickness")
    H.interpolate(max_value(H_raw, Constant(build_mesh.H_MIN)))

    s = Function(Q, name="surface")
    s.interpolate(max_value(
        b + H,
        Constant(1 - build_mesh.ICE_DENSITY / build_mesh.WATER_DENSITY) * H))
    s0 = s.copy(deepcopy=True)
    smoothing_length = Constant(2e3)
    firedrake.solve(
        firedrake.derivative(
            0.5 * ((s - s0) ** 2 + smoothing_length ** 2 * inner(grad(s), grad(s))) * dx,
            s,
        ) == 0,
        s,
    )

    u_obs = icepack.interpolate(
        (rasterio.open(f"netcdf:{build_mesh.VEL_FILE}:VX"),
         rasterio.open(f"netcdf:{build_mesh.VEL_FILE}:VY")),
        V, fillvalue=0.0,
    )
    u_obs.rename("velocity")

    phi_eff = Function(Q, name="phi_eff")
    phi_eff.interpolate(
        max_value(
            Constant(1.0)
            - Constant(build_mesh.WATER_DENSITY / build_mesh.ICE_DENSITY)
            * max_value(Constant(0.0), -b) / H,
            Constant(0.01),
        )
    )

    print(f"  Thickness: [{H.dat.data.min():.0f}, {H.dat.data.max():.0f}] m")
    print(f"  Surface:   [{s.dat.data.min():.0f}, {s.dat.data.max():.0f}] m")
    speed = np.sqrt(u_obs.dat.data_ro[:, 0] ** 2 + u_obs.dat.data_ro[:, 1] ** 2)
    print(f"  Speed:     [{speed.min():.0f}, {speed.max():.0f}] m/yr")

    out = build_mesh.OUTPUT_DIR / "mesh" / "thwaites_coarse.h5"
    with firedrake.CheckpointFile(str(out), "w") as chk:
        chk.save_mesh(mesh)
        for f, name in [
            (H, "thickness"), (b, "bed"), (s, "surface"),
            (u_obs, "velocity"), (phi_eff, "phi_eff"),
        ]:
            chk.save_function(f, name=name)
    print(f"Saved {out}")
    return mesh, Q, V, {}

build_mesh.interpolate_and_save = interpolate_and_save_coarse

# Override .msh output path so we don't clobber the hires .msh
import gmsh as _gmsh  # noqa
_orig_build = build_mesh.build_gmsh_mesh
def build_gmsh_coarse(domain, haf_interp):
    msh = _orig_build(domain, haf_interp)
    # rename to thwaites_coarse.msh
    coarse_msh = str(build_mesh.OUTPUT_DIR / "mesh" / "thwaites_coarse.msh")
    import shutil
    shutil.move(msh, coarse_msh)
    return coarse_msh

build_mesh.build_gmsh_mesh = build_gmsh_coarse


if __name__ == "__main__":
    build_mesh.main()
