r"""Rebuild the flotation-consistent surface on an existing mesh, node for node.

The mesh builders derived the surface correctly and then Helmholtz-smoothed it,
which breaks s = max(b + h, (1 - rho_I/rho_W) h). Only the surface was corrupted:
thickness, bed, velocity and phi_eff were all stored unsmoothed, and phi_eff
depends on (b, H) alone. So the correct surface is recoverable exactly, by
re-deriving it from the stored thickness and bed.

Doing it this way, rather than re-running build_mesh_hires.py, keeps the SAME
mesh: gmsh does not reproduce its triangulation run to run (a rebuild here gave
67324 nodes against 67600), and a different mesh makes old and new inversions
incomparable. With the mesh held fixed, a MAP inverted before and after the
repair can be differenced dof by dof.

    python scripts/repair_geometry.py                       # thwaites -> thwaites_fixed
    python scripts/repair_geometry.py --in mesh/thwaites_coarse.h5 --out mesh/x.h5

See docs/geometry_consistency.md.
"""
import argparse
from pathlib import Path

import icepack
import numpy as np
from firedrake import CheckpointFile

DATA = Path(__file__).resolve().parent.parent
FIELDS = ("thickness", "bed", "velocity", "phi_eff")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="src", default=str(DATA / "mesh" / "thwaites.h5"))
    ap.add_argument("--out", dest="dst", default=str(DATA / "mesh" / "thwaites_fixed.h5"))
    args = ap.parse_args()

    with CheckpointFile(args.src, "r") as chk:
        mesh = chk.load_mesh()
        fields = {n: chk.load_function(mesh, n) for n in FIELDS}
        s_old = chk.load_function(mesh, "surface")

    h, b = fields["thickness"], fields["bed"]
    s = icepack.compute_surface(thickness=h, bed=b)
    s.rename("surface")
    d = s.dat.data_ro - s_old.dat.data_ro
    print(f"{args.src}: {mesh.num_vertices()} vertices", flush=True)
    print(f"  surface corrected by [{d.min():+.1f}, {d.max():+.1f}] m; "
          f"|change| > 10 m at {int((np.abs(d) > 10).sum())} of {d.size} nodes",
          flush=True)

    with CheckpointFile(args.dst, "w") as chk:
        chk.save_mesh(mesh)
        chk.save_function(s, name="surface")
        for name, f in fields.items():
            chk.save_function(f, name=name)
    print(f"  wrote {args.dst} (same mesh, {', '.join(FIELDS)} copied verbatim)",
          flush=True)
    print("  next: VV_MESH=<stem> python scripts/physical_prior.py, then invert",
          flush=True)


if __name__ == "__main__":
    main()
