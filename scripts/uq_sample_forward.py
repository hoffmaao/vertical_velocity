r"""Run the 50-yr Recinos-melt forward on one posterior θ-sample → VAF(T).

Reads results/posterior_samples.npz (from uq_sample_posterior.py), sets
(θ_A, θ_C) to sample --idx, runs the same forward as uq_directional_forward.py
(Recinos Eq.5 melt), and records the VAF trajectory. The ensemble of these gives
the true non-Gaussian posterior VAF distribution.

Run:  OMP_NUM_THREADS=4 python uq_sample_forward.py --idx 0
"""
import argparse
import numpy as np
import netCDF4 as nc
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, ExtrudedMesh,
    sqrt, inner, max_value, exp, tanh, dx, conditional, assemble,
)
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
import icepack
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
MESH_FILE = DATA / "mesh" / "thwaites.h5"
SAMPLES = DATA / "results" / "posterior_samples.npz"
RACMO_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/racmo/"
                  "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc")

DT = 1.0 / 12.0
MMAX = 30.0   # Recinos et al. (2023) Eq. 5
ZTH = 600.0
H_MIN = 10.0
SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 100, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}
PROG_PARAMS = {
    "snes_type": "ksponly", "ksp_type": "preonly",
    "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def load_racmo_smb(mesh_2d, Q_2d):
    ds = nc.Dataset(str(RACMO_FILE))
    lon_2d = ds.variables["lon"][:]; lat_2d = ds.variables["lat"][:]
    smb_monthly = np.array(ds.variables["smbgl"][:, 0, :, :]); ds.close()
    smb_ice = (smb_monthly.mean(axis=0) * 12.0) / 917.0
    from pyproj import Transformer
    from scipy.interpolate import griddata
    tr = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    coords = mesh_2d.coordinates.dat.data_ro
    lons, lats = tr.transform(coords[:, 0], coords[:, 1])
    vals = griddata((lon_2d.ravel(), lat_2d.ravel()), smb_ice.ravel(),
                    (lons, lats), method="linear", fill_value=0.0)
    a = Function(Q_2d, name="smb"); a.dat.data[:] = vals
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--samples", default=str(SAMPLES), help="posterior samples npz")
    ap.add_argument("--label", default="", help="output tag, e.g. 'b' → sample_vaf_b###.npz")
    args = ap.parse_args()
    num_steps = max(1, int(round(args.T / DT)))
    out_file = DATA / "results" / f"sample_vaf_{args.label}{args.idx:03d}.npz"
    print(f"=== sample {args.label}{args.idx}: T={args.T} yr, {num_steps} steps ===", flush=True)

    d = np.load(str(args.samples))
    thA = d["theta_A"][args.idx]; thC = d["theta_C"][args.idx]

    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        b_2d = chk.load_function(mesh_2d, "bed")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")
    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    smb_2d = load_racmo_smb(mesh_2d, Q_2d)

    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    b = icepack.utilities.lift3d(b_2d, Q_lift)
    u = Function(V).project(icepack.utilities.lift3d(u_obs_2d, V_lift))

    θ_A = Function(Q_lift); θ_C = Function(Q_lift)
    θ_A.dat.data[:] = thA; θ_C.dat.data[:] = thC

    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q_lift).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
                    Constant(1.0), Constant(0.0)))
    smb_3d = icepack.utilities.lift3d(smb_2d, Q_lift)
    A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))

    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS,
        prognostic_solver_parameters=PROG_PARAMS)

    try:
        u = solver.diagnostic_solve(velocity=u, thickness=h, surface=s, fluidity=A, friction=C)
    except fd.exceptions.ConvergenceError:
        print("  initial solve FAILED", flush=True)
        np.savez(str(out_file), idx=args.idx, failed=True)
        return

    h_inflow = h.copy(deepcopy=True)
    s_float = b + Constant(float(ρ_W / ρ_I)) * max_value(-b, Constant(0.0))

    def vaf_gt():
        return float(assemble(max_value(s - s_float, Constant(0.0)) * dx)) * 917.0 / 1e12

    times = [0.0]; vaf_ts = [vaf_gt()]; n_fail = 0
    vaf0 = vaf_ts[0]
    print(f"  sample {args.idx}: VAF(0)={vaf0:.1f} Gt, "
          f"θ_A∈[{thA.min():.2f},{thA.max():.2f}], θ_C∈[{thC.min():.2f},{thC.max():.2f}]", flush=True)
    for step in range(1, num_steps + 1):
        t = step * DT
        try:
            floating = Function(Q_lift).interpolate(
                conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) < 0.01,
                            Constant(1.0), Constant(0.0)))
            z_b = h - s
            m_melt = Constant(0.5 * MMAX) * (
                1 + tanh(Constant(2.0) * (z_b - Constant(ZTH)) / Constant(ZTH)))
            accum = Function(Q_lift).interpolate(smb_3d - floating * m_melt)
            h = solver.prognostic_solve(DT, thickness=h, velocity=u,
                                        accumulation=accum, thickness_inflow=h_inflow)
            h = Function(Q_lift).interpolate(max_value(h, Constant(H_MIN)))
            s = icepack.compute_surface(thickness=h, bed=b)
            u = solver.diagnostic_solve(velocity=u, thickness=h, surface=s, fluidity=A, friction=C)
        except fd.exceptions.ConvergenceError:
            n_fail += 1
        times.append(t); vaf_ts.append(vaf_gt())
        if step % max(1, num_steps // 10) == 0:
            print(f"    t={t:5.1f}: VAF={vaf_ts[-1]:.1f} (ΔVAF={vaf_ts[-1]-vaf0:+.1f}), {n_fail} fail", flush=True)

    dVAF = vaf_ts[-1] - vaf0
    print(f"  DONE sample {args.idx}: VAF(0)={vaf0:.1f}, VAF(T)={vaf_ts[-1]:.1f}, ΔVAF={dVAF:+.2f} Gt, {n_fail} fails", flush=True)
    np.savez(str(out_file), idx=args.idx, failed=False, time=np.array(times),
             vaf_Gt=np.array(vaf_ts), vaf0=vaf0, vafT=vaf_ts[-1], dVAF=dVAF, n_fail=n_fail)


if __name__ == "__main__":
    main()
