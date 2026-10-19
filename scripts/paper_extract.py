r"""Extract the fields behind the paper's summary and supplement figures.

One Firedrake pass per production run (rc friction, bilaplacian prior
delta 4e-4, ELL 3 km, ISMIP7-forced 50-yr forward) that writes everything
``paper_figures.py`` draws into ``results/paper/extract_<model>.npz``, so the
figures themselves are plain matplotlib on archived arrays.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/paper_extract.py --model ssa
    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/paper_extract.py --model hybrid

Per run it stores: the 2D mesh; the MAP controls and the fields they map to;
observed and modelled surface speed; the normalized velocity residual at the
observation points; observed, modelled and sigma ApRES eps_zz (assimilated for
hybrid, withheld for SSA); the t = 50 yr thickness, speed and dVAF/dtheta (as a
density, i.e. divided by the lumped mass); and the pointwise prior and K-mode
posterior standard deviation of theta_A and theta_C.

Posterior variance: diag(Gamma_post) = diag(Gamma) - sum_i lambda_i/(1+lambda_i) w_i^2
with the Gamma^-1-orthonormal GN eigenvectors w_i from eigendec.py. The K modes
miss the variance reduction of the unresolved tail, so this is an upper bound.
diag(Gamma) is estimated from --n-samples prior draws theta = A^-1 D^1/2 xi
(lumped mass D, as in the optimizer's whitening).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from firedrake import CheckpointFile, Function, FunctionSpace, VertexOnlyMesh, inner, sqrt

from observations import load_apres
from plot_map import eps_zz, onto_vom, to_2d, vals
from prior_precond import Prior
from problem import build_problem
from physical_prior import MESH_FILE

DATA = Path(__file__).resolve().parent.parent
RUN = "sweep_{model}_rc_bilap_d4e-04_L3km_thwaites_fixed_tau5km"
T = 50


def lumped_mass(Q_2d):
    from firedrake import TestFunction, assemble, dx
    return assemble(TestFunction(Q_2d) * dx).dat.data_ro.copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    ap.add_argument("--n-samples", type=int, default=1000)
    args = ap.parse_args()
    name = RUN.format(model=args.model)
    apres = args.model == "hybrid"
    out_dir = DATA / "results" / "paper"
    out_dir.mkdir(exist_ok=True)
    meta = json.load(open(DATA / "results" / f"map_{name}.json"))
    print(f"[{name}] building problem", flush=True)

    P = build_problem(args.model, "rc", apres)
    mesh_2d, Q_2d, Q = P["mesh_2d"], P["Q_2d"], P["Q"]
    with CheckpointFile(str(DATA / "mesh" / f"map_{name}.h5"), "r") as chk:
        m = chk.load_mesh()
        θ_A_2d = to_2d(chk.load_function(m, "log_fluidity"), Q_2d)
        θ_C_2d = to_2d(chk.load_function(m, "log_friction"), Q_2d)
    θ_A, θ_C = Function(Q), Function(Q)
    θ_A.dat.data[:] = θ_A_2d.dat.data_ro
    θ_C.dat.data[:] = θ_C_2d.dat.data_ro

    print(f"[{name}] diagnostic solve at the MAP", flush=True)
    u = P["forward"](θ_A, θ_C)
    A, C = P["fields"](θ_A, θ_C)
    xy = mesh_2d.coordinates.dat.data_ro.copy()
    cells = mesh_2d.coordinates.function_space().cell_node_list.copy()

    # Surface speed (the hybrid is compared at zeta = 1, as in the misfit).
    speed_obs = Function(Q_2d).interpolate(sqrt(inner(P["u_obs_2d"], P["u_obs_2d"])))
    if args.model == "ssa":
        speed_mod = Function(Q_2d).interpolate(sqrt(inner(u, u))).dat.data_ro.copy()
    else:
        vom_s = VertexOnlyMesh(P["mesh"], np.column_stack([xy, np.ones(len(xy))]),
                               missing_points_behaviour="error", redundant=False)
        Δs, Δs_in = FunctionSpace(vom_s, "DG", 0), FunctionSpace(vom_s.input_ordering, "DG", 0)
        us = [Function(Δs_in).interpolate(Function(Δs).interpolate(u[i])).dat.data_ro
              for i in range(2)]
        speed_mod = np.hypot(*us)

    # Normalized velocity residual at the observation points.
    o = P["obs"]
    Δv = o["Δv"]
    xy_v = np.array(Δv.mesh().coordinates.dat.data_ro, dtype=float)
    ui, vi = Function(Δv).interpolate(u[0]), Function(Δv).interpolate(u[1])
    χx = vals(Function(Δv).interpolate((ui - o["u"]) / o["σx"]))
    χy = vals(Function(Δv).interpolate((vi - o["v"]) / o["σy"]))

    # ApRES eps_zz: assimilated (hybrid) or withheld (SSA).
    pts, e_obs_in, e_sig_in = load_apres(mesh_2d)
    vom_a = VertexOnlyMesh(Q.mesh(), pts if apres else pts[:, :2], missing_points_behaviour="warn")
    Δa = FunctionSpace(vom_a, "DG", 0)
    xy_a = np.array(vom_a.coordinates.dat.data_ro, dtype=float)[:, :2]
    e_obs, e_sig = onto_vom(Δa, e_obs_in), onto_vom(Δa, e_sig_in)
    zeta = vals(onto_vom(Δa, pts[:, 2]))
    e_mod = vals(Function(Δa).interpolate(eps_zz(u, P, args.model)))

    # 50-yr forward: final state and the VAF sensitivity density.
    Mlump = lumped_mass(Q_2d)
    with CheckpointFile(str(DATA / "results" / f"forward_{name}_{T}yr.h5"), "r") as chk:
        mf = chk.load_mesh()
        assert np.allclose(mf.coordinates.dat.data_ro, xy), "forward mesh ordering differs"
        h_T = chk.load_function(mf, "thickness_final").dat.data_ro.copy()
        u_T = chk.load_function(mf, "velocity_final").dat.data_ro.copy()
        g_A = chk.load_function(mf, "dVAF_dthetaA").dat.data_ro / Mlump
        g_C = chk.load_function(mf, "dVAF_dthetaC").dat.data_ro / Mlump
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        m0 = chk.load_mesh()
        h0 = chk.load_function(m0, "thickness").dat.data_ro.copy()
        bed = chk.load_function(m0, "bed").dat.data_ro.copy()

    # Pointwise prior and K-mode posterior standard deviation of theta.
    print(f"[{name}] prior variance from {args.n_samples} draws", flush=True)
    rng = np.random.default_rng(0)
    std = {}
    for key, δ, γ in (("A", meta["delta_A"], meta["gamma_A"]), ("C", meta["delta_C"], meta["gamma_C"])):
        pr = Prior(Q_2d, δ, γ, meta["prior_form"])
        acc = np.zeros(pr.n)
        for _ in range(args.n_samples):
            acc += pr.to_theta(rng.standard_normal(pr.n)) ** 2
        std[f"prior_{key}"] = np.sqrt(acc / args.n_samples)
    print(f"[{name}] posterior variance from the eigenvectors", flush=True)
    eig = np.load(str(DATA / "results" / f"eigendec_{name}_K1500.npz"))
    lam, n = eig["eigenvalues"], int(eig["n_dof"])
    D = lam / (1.0 + lam)
    red = np.zeros(2 * n)
    W = eig["vecs"]
    for i0 in range(0, W.shape[1], 100):
        Wb = W[:, i0:i0 + 100]
        red += (Wb ** 2) @ D[i0:i0 + 100]
    for j, key in enumerate("AC"):
        var_post = std[f"prior_{key}"] ** 2 - red[j * n:(j + 1) * n]
        std[f"post_{key}"] = np.sqrt(np.maximum(var_post, 0.0))

    out = out_dir / f"extract_{args.model}.npz"
    np.savez_compressed(
        str(out), name=name, model=args.model, xy=xy, cells=cells, bed=bed, h0=h0,
        theta_A=θ_A_2d.dat.data_ro.copy(), theta_C=θ_C_2d.dat.data_ro.copy(),
        A=to_2d(A, Q_2d).dat.data_ro.copy(), C=to_2d(C, Q_2d).dat.data_ro.copy(),
        A_prior=P["A_prior_2d"].dat.data_ro.copy(), C_anchor=P["C_anchor_2d"].dat.data_ro.copy(),
        speed_obs=speed_obs.dat.data_ro.copy(), speed_mod=np.asarray(speed_mod),
        xy_v=xy_v.astype(np.float32), chi_vel=np.hypot(χx, χy).astype(np.float32),
        xy_a=xy_a, zeta_a=zeta, eps_obs=vals(e_obs), eps_sig=vals(e_sig), eps_mod=e_mod,
        h_T=h_T, speed_T=np.sqrt((u_T ** 2).sum(axis=1)), dVAF_dthetaA=g_A, dVAF_dthetaC=g_C,
        n_samples=args.n_samples, **{f"std_{k}": v for k, v in std.items()})
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
