r"""Audit a mesh file for flotation consistency between thickness, surface and bed.

The model's own surface relation is

    s = max(b + h, (1 - rho_I/rho_W) h)        (icepack.compute_surface)

and a prognostic run re-imposes it at every step. If the stored surface does not
satisfy it, the first step applies the whole difference at once, which shocks the
driving stress and can destroy the run. The mesh builders in this repo derive s
that way and then Helmholtz-smooth it over 2 km, which breaks the relation: the
smoothing is exactly the inconsistency.

The second symptom is negative basal drag. The friction laws scale the
coefficient by the flotation factor

    phi = 1 - rho_W g max(0, h - s) / (rho_I g h),

which is 0 for ice in hydrostatic balance and negative for ice sitting deeper
than flotation. A smoothed surface puts many nodes below their hydrostatic
height, so phi goes negative there and the basal stress becomes a source of
momentum rather than a sink.

    python scripts/check_geometry.py                     # mesh/thwaites.h5
    python scripts/check_geometry.py mesh/thwaites_coarse.h5
"""
import sys
from pathlib import Path

import icepack
import numpy as np
from firedrake import (
    CheckpointFile, Constant, Function, TestFunction, TrialFunction,
    VectorFunctionSpace, dx, grad, inner, solve, sqrt,
)
from icepack.constants import gravity as g, ice_density as ρ_I

from friction_laws import flotation_factor

DATA = Path(__file__).resolve().parent.parent
SMOOTH_LENGTH = 2000.0     # m, the Helmholtz length physical_prior.py uses for tau_d


def driving_stress(h, s, Q, V, length=None):
    r"""|tau_d| = |rho_I g h grad(s)|, Helmholtz-smoothed over ``length`` if given."""
    if length is None:
        τ = -ρ_I * g * h * grad(s)
        return Function(Q).interpolate(sqrt(inner(τ, τ))).dat.data_ro.copy()
    τ, v = TrialFunction(V), TestFunction(V)
    a = (inner(τ, v) + Constant(length) ** 2 * inner(grad(τ), grad(v))) * dx
    L = -inner(ρ_I * g * h * grad(s), v) * dx
    τ_s = Function(V)
    solve(a == L, τ_s, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})
    return Function(Q).interpolate(sqrt(inner(τ_s, τ_s))).dat.data_ro.copy()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "mesh" / "thwaites.h5"
    with CheckpointFile(str(path), "r") as chk:
        mesh = chk.load_mesh()
        h = chk.load_function(mesh, "thickness")
        s = chk.load_function(mesh, "surface")
        b = chk.load_function(mesh, "bed")
    Q = h.function_space()
    V = VectorFunctionSpace(mesh, "CG", 1)
    s_model = icepack.compute_surface(thickness=h, bed=b)
    d = s_model.dat.data_ro - s.dat.data_ro
    n = d.size

    print(f"{path}  ({n} nodes)\n")
    print("stored surface vs compute_surface(h, b):")
    print(f"  difference [{d.min():+.1f}, {d.max():+.1f}] m, mean |diff| {np.abs(d).mean():.2f} m")
    for thr in (1, 10, 50, 100):
        print(f"  |diff| > {thr:3d} m at {int((np.abs(d) > thr).sum()):6d} nodes "
              f"({100 * (np.abs(d) > thr).mean():5.1f}%)")

    print("\nflotation factor phi (negative phi = negative basal drag):")
    for label, ss in (("stored surface", s), ("compute_surface(h, b)", s_model)):
        φ = Function(Q).interpolate(flotation_factor(h, ss)).dat.data_ro
        print(f"  {label:22s}: min {φ.min():+.4f}, phi < -0.01 at "
              f"{int((φ < -0.01).sum()):6d} nodes")

    print("\ndriving stress |tau_d| (MPa), raw and Helmholtz-smoothed over 2 km:")
    print(f"  {'surface':24s} {'raw median':>11s} {'raw p99':>9s} {'sm median':>10s} "
          f"{'sm p99':>8s}")
    for label, ss in (("stored surface", s), ("compute_surface(h, b)", s_model)):
        raw = driving_stress(h, ss, Q, V)
        sm = driving_stress(h, ss, Q, V, SMOOTH_LENGTH)
        print(f"  {label:24s} {np.median(raw):11.4f} {np.percentile(raw, 99):9.4f} "
              f"{np.median(sm):10.4f} {np.percentile(sm, 99):8.4f}")
    print("\nSmoothing tau_d (which physical_prior.py already does) recovers the same "
          "\nstress statistics from the consistent surface, so smoothing the surface "
          "\nitself buys nothing that the stress smoothing does not already provide.")


if __name__ == "__main__":
    main()
