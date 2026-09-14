r"""Calibrate basal friction so the MISMIP+ Hybrid truth has REAL vertical shear.

MISMIP+ is sliding-dominated (plug flow, ~0% shear) → no signal for ApRES ε_zz.
This solves the Hybrid velocity on the current geometry with scaled friction to find
the multiplier that yields a meaningful shear fraction (surf-base)/surf, especially
over the sticky ridges. Fast: velocity-only (no re-spin), so geometry isn't self-
consistent — this only picks the friction scale; the real truth is then re-spun.
"""
import numpy as np
import firedrake
import icepack
import icepack.models.friction
from firedrake import (Constant, Function, FunctionSpace, VectorFunctionSpace,
                       ExtrudedMesh, CheckpointFile, VertexOnlyMesh, interpolate,
                       max_value, exp)
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
A0 = Constant(20.0)
VDEGREE = 4
SOLVER = {"snes_type": "newtonls", "snes_linesearch_type": "bt",
          "snes_max_it": 100, "snes_rtol": 1e-7,
          "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def friction_weertman(**kw):
    u, h, s, C = kw["velocity"], kw["thickness"], kw["surface"], kw["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def shear_stats(m3, u3):
    """Sample surface-vs-basal speed over a set of columns; return mean shear fraction."""
    xs = np.linspace(140e3, 440e3, 25)
    fr = []
    for x in xs:
        z = np.linspace(0.02, 0.98, 7)
        xyz = np.column_stack([np.full(7, x), np.full(7, 40e3), z])
        vom = VertexOnlyMesh(m3, xyz, missing_points_behaviour="warn")
        uu = Function(VectorFunctionSpace(vom, "DG", 0, dim=2)).interpolate(u3).dat.data_ro
        if len(uu) < 3:
            continue
        spd = np.hypot(uu[:, 0], uu[:, 1])
        if spd.max() > 1:
            fr.append((spd.max() - spd.min()) / spd.max())
    return np.array(fr)


def main():
    with CheckpointFile(str(DATA / "mesh" / "mismip_truth.h5"), "r") as c:
        m2 = c.load_mesh()
        h = c.load_function(m2, "thickness")
        s = c.load_function(m2, "surface")
        θA = c.load_function(m2, "log_fluidity_true")
        C = c.load_function(m2, "friction_true")
    Q = FunctionSpace(m2, "CG", 1)
    A = interpolate(A0 * exp(θA), Q)

    mesh3d = ExtrudedMesh(m2, layers=1)
    Q3 = FunctionSpace(mesh3d, "CG", 1, vfamily="DG", vdegree=0)
    V3 = VectorFunctionSpace(mesh3d, "CG", 1, vfamily="GL", vdegree=VDEGREE, dim=2)
    V3l = VectorFunctionSpace(mesh3d, "CG", 1, vfamily="DG", vdegree=0, dim=2)
    h3 = icepack.utilities.lift3d(h, Q3)
    s3 = icepack.utilities.lift3d(s, Q3)
    A3 = icepack.utilities.lift3d(A, Q3)

    model = icepack.models.HybridModel(friction=friction_weertman)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[1], ice_front_ids=[2], side_wall_ids=[3, 4],
        diagnostic_solver_type="petsc", diagnostic_solver_parameters=SOLVER)

    print(f"{'scale':>6} {'C_eff range':>20} {'surfspd':>9} {'shear% mean':>12} {'shear% max':>11}", flush=True)
    for scale in [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        C3 = icepack.utilities.lift3d(interpolate(C * Constant(scale), Q), Q3)
        u3 = Function(V3)
        try:
            u3 = solver.diagnostic_solve(velocity=u3, thickness=h3, surface=s3, fluidity=A3, friction=C3)
        except firedrake.exceptions.ConvergenceError:
            print(f"{scale:>6.0f}  (diverged)", flush=True)
            continue
        fr = shear_stats(mesh3d, u3)
        ua = icepack.depth_average(u3)
        smax = float(np.sqrt((ua.dat.data_ro ** 2).sum(axis=1)).max())
        cv = (C.dat.data_ro.min() * scale, C.dat.data_ro.max() * scale)
        print(f"{scale:>6.0f}  [{cv[0]:.2e},{cv[1]:.2e}] {smax:>9.0f} "
              f"{100*fr.mean():>11.1f}% {100*fr.max():>10.1f}%", flush=True)


if __name__ == "__main__":
    main()
