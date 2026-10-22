r"""Where each model dissipates its energy: in the ice or at the bed.

At the MAP of each production run (rc friction, bilaplacian prior delta 4e-4,
ELL 3 km) the gravitational power the ice releases is dissipated as heat in
two places, by viscous deformation of the ice and by sliding over the bed:

    Phi_ice = tau : eps          (per unit area, integrated over the column)
    Phi_bed = tau_b . u_b

Both are taken from the model's own action functional rather than written out
again: the power of a term E(u) homogeneous of degree p in u is the Gateaux
derivative dE(u)[u] = p E(u), so

    Phi_ice = d(viscosity)[u],   Phi_bed = d(friction)[u]

whatever the flow and friction laws (here Glen n = 3 and the regularized
Coulomb law of friction_laws.py, which is not homogeneous).  The same holds
for the driving power, d(gravity)[u] = gravity, and the calving-front term.
The velocity is fixed on the inflow boundary, so the balance

    Phi_ice + Phi_bed = P_gravity + P_terminus + P_inflow

closes with the power P_inflow the upstream ice does on the domain, which is
reported as the remainder.

For the hybrid model the ice term splits exactly into membrane stretching and
vertical shear, because both enter Glen's law through one effective strain
rate with one viscosity mu:

    Phi_ice = 2 mu h (eps_x : eps_x + tr(eps_x)^2) + 4 mu h |eps_z|^2

(hybrid.stresses and hybrid._effective_strain_rate), integrated over zeta.
SSA has no vertical shear, so all of its ice term is membrane stretching.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/dissipation_partition.py --model ssa
    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/dissipation_partition.py --model hybrid

Writes results/paper/dissipation_<model>.npz with per-cell powers (W) and
cell centroids, areas and flotation state on the 2D mesh, and prints the
domain totals in GW.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from firedrake import (CheckpointFile, Constant, Function, FunctionSpace, SpatialCoordinate,
                       TestFunction, VectorFunctionSpace, assemble, derivative, ds, ds_b, ds_v,
                       dx, inner)

from icepack.constants import glen_flow_law, strain_rate_min
from icepack.models import hybrid as hy
from friction_laws import flotation_factor
from plot_map import to_2d
from problem import build_problem

DATA = Path(__file__).resolve().parent.parent
RUN = "sweep_{model}_rc_bilap_d4e-04_L3km_thwaites_fixed_tau5km"
TO_W = 1e6 / 3.15576e7          # MPa m^3/yr -> W  (and MPa m/yr -> W/m^2)
FRONT_IDS = (1,)                # problem.make_solver: ice_front_ids=[1]
GROUNDED_PHI = 1e-3             # floating cells carry phi at floating-point roundoff


def cell_values(form):
    """Assemble a form linear in a DG0 test function: one value per cell (W)."""
    return assemble(form).dat.data_ro.copy() * TO_W


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    args = ap.parse_args()
    name = RUN.format(model=args.model)
    hybrid = args.model == "hybrid"

    print(f"[{name}] building problem", flush=True)
    P = build_problem(args.model, "rc", hybrid)
    mesh, Q_2d, Q = P["mesh"], P["Q_2d"], P["Q"]
    with CheckpointFile(str(DATA / "mesh" / f"map_{name}.h5"), "r") as chk:
        mm = chk.load_mesh()
        θ_A_2d = to_2d(chk.load_function(mm, "log_fluidity"), Q_2d)
        θ_C_2d = to_2d(chk.load_function(mm, "log_friction"), Q_2d)
    θ_A, θ_C = Function(Q), Function(Q)
    θ_A.dat.data[:] = θ_A_2d.dat.data_ro
    θ_C.dat.data[:] = θ_C_2d.dat.data_ro

    print(f"[{name}] diagnostic solve at the MAP", flush=True)
    u = P["forward"](θ_A, θ_C)
    A, C = P["fields"](θ_A, θ_C)
    h, s = P["h"], P["s"]
    model = P["solver"].model
    kw = dict(velocity=u, thickness=h, surface=s, fluidity=A, friction=C)

    # The model's own quadrature, as in its action().
    md = {"quadrature_degree": model.quadrature_degree(**kw)}
    D = FunctionSpace(mesh, "DG", 0)
    w = TestFunction(D)
    vol = dx(metadata=md)
    bed = ds_b(domain=mesh, metadata=md) if hybrid else vol

    out = {}
    out["ice"] = cell_values(derivative(model.viscosity(**kw) * w * vol, u, u))
    out["bed"] = cell_values(derivative(model.friction(**kw) * w * bed, u, u))
    out["gravity"] = cell_values(model.gravity(**kw) * w * vol)
    if hybrid:
        ε_x = hy.horizontal_strain_rate(velocity=u, surface=s, thickness=h)
        ε_z = hy.vertical_strain_rate(velocity=u, thickness=h)
        ε_e = hy._effective_strain_rate(ε_x, ε_z, Constant(strain_rate_min))
        n = glen_flow_law
        μ = 0.5 * A ** (-1 / n) * ε_e ** (1 / n - 1)
        out["shear"] = cell_values(4 * μ * h * inner(ε_z, ε_z) * w * vol)
        xdeg, zdeg = u.ufl_element().degree()
        front = ds_v(FRONT_IDS, metadata={"quadrature_degree": (xdeg + 1, 2 * zdeg + 1)})
    else:
        out["shear"] = np.zeros_like(out["ice"])
        front = ds(FRONT_IDS, domain=mesh, metadata=md)
    out["membrane"] = out["ice"] - out["shear"]
    terminus = float(assemble(model.terminus(**kw) * front)) * TO_W

    # Cell geometry and flotation state, on the 2D footprint for plotting.
    X = Function(VectorFunctionSpace(mesh, "DG", 0)).interpolate(SpatialCoordinate(mesh))
    xyz = X.dat.data_ro.copy()
    area = assemble(w * vol).dat.data_ro.copy()        # extruded: column of unit zeta height
    phi = Function(D).interpolate(flotation_factor(h, s)).dat.data_ro.copy()

    tot = {k: float(v.sum()) / 1e9 for k, v in out.items()}
    inflow = tot["ice"] + tot["bed"] - tot["gravity"] - terminus / 1e9
    g = phi > GROUNDED_PHI
    summary = dict(
        model=args.model, run=name, GW=tot, terminus_GW=terminus / 1e9, inflow_GW=inflow,
        grounded_GW={k: float(v[g].sum()) / 1e9 for k, v in out.items()},
    )
    print(json.dumps(summary, indent=2), flush=True)
    res = DATA / "results" / "paper"
    res.mkdir(parents=True, exist_ok=True)
    np.savez(res / f"dissipation_{args.model}.npz", xy=xyz[:, :2], area=area, phi=phi,
             **{k: v for k, v in out.items()})
    json.dump(summary, open(res / f"dissipation_{args.model}.json", "w"), indent=2)
    print(f"Saved {res / f'dissipation_{args.model}.npz'}", flush=True)


if __name__ == "__main__":
    main()
