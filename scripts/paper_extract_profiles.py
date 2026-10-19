r"""Modelled ApRES eps_zz depth profiles at the MAP, for the main-text profile figure.

``paper_extract.py`` keeps the model only at the ~5 assimilated depths per
station. This evaluates eps_zz = -(eps_xx + eps_yy) on a full zeta grid at every
ApRES station inside the mesh, so a modelled profile can be drawn against the
observed one over the whole ice column.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/paper_extract_profiles.py --model hybrid
    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/paper_extract_profiles.py --model ssa

Writes results/paper/profiles_<model>.npz with the station names, positions,
the zeta grid and eps_zz[station, zeta] in 1/yr. The SSA strain rate does not
vary with depth, so its rows are constant.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from firedrake import CheckpointFile, Function, FunctionSpace, VertexOnlyMesh

from observations import APRES_FILE
from plot_map import eps_zz, to_2d, vals
from problem import build_problem

DATA = Path(__file__).resolve().parent.parent
RUN = "sweep_{model}_rc_bilap_d4e-04_L3km_thwaites_fixed_tau5km"
N_ZETA = 41


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    args = ap.parse_args()
    name = RUN.format(model=args.model)
    hybrid = args.model == "hybrid"
    print(f"[{name}] building problem", flush=True)
    P = build_problem(args.model, "rc", hybrid)
    Q_2d, Q = P["Q_2d"], P["Q"]
    with CheckpointFile(str(DATA / "mesh" / f"map_{name}.h5"), "r") as chk:
        m = chk.load_mesh()
        θ_A_2d = to_2d(chk.load_function(m, "log_fluidity"), Q_2d)
        θ_C_2d = to_2d(chk.load_function(m, "log_friction"), Q_2d)
    θ_A, θ_C = Function(Q), Function(Q)
    θ_A.dat.data[:] = θ_A_2d.dat.data_ro
    θ_C.dat.data[:] = θ_C_2d.dat.data_ro
    print(f"[{name}] diagnostic solve at the MAP", flush=True)
    u = P["forward"](θ_A, θ_C)

    with h5py.File(str(APRES_FILE), "r") as f:
        names = np.array([n.decode() for n in f["summary/names"][...]])
        xy = np.column_stack([f["summary/x"][...], f["summary/y"][...]])
    c = P["mesh_2d"].coordinates.dat.data_ro
    inside = ((xy[:, 0] > c[:, 0].min()) & (xy[:, 0] < c[:, 0].max())
              & (xy[:, 1] > c[:, 1].min()) & (xy[:, 1] < c[:, 1].max()))
    names, xy = names[inside], xy[inside]
    # Stay a hair inside the column so no point sits on the top or bottom facet.
    zeta = np.linspace(1e-3, 1 - 1e-3, N_ZETA)
    pts = (np.column_stack([np.repeat(xy, N_ZETA, axis=0), np.tile(zeta, len(xy))])
           if hybrid else xy)
    # "error": a station outside the domain must not silently shift every row.
    vom = VertexOnlyMesh(Q.mesh() if hybrid else P["mesh_2d"], pts,
                         missing_points_behaviour="error", redundant=False)
    Δ, Δ_in = FunctionSpace(vom, "DG", 0), FunctionSpace(vom.input_ordering, "DG", 0)
    e = vals(Function(Δ_in).interpolate(Function(Δ).interpolate(eps_zz(u, P, args.model))))
    e = e.reshape(len(xy), N_ZETA) if hybrid else np.repeat(e[:, None], N_ZETA, axis=1)

    out = DATA / "results" / "paper" / f"profiles_{args.model}.npz"
    np.savez_compressed(str(out), name=name, model=args.model, station=names, xy=xy,
                        zeta=zeta, eps_mod=e)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
