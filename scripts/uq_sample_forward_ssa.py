r"""Run the 50-yr SSA (IceStream) Recinos-melt forward on one posterior θ-sample.

SSA companion to uq_sample_forward.py. 2D IceStream + Schoof-Coulomb friction
(matching inversion_ssa_recinos.py / run_eigendec_ssa.py: A_0=20, C_0=0.5 MPa, U_0=300),
same Recinos Eq.5 sub-shelf melt and VAF QoI. Reads posterior_samples_ssa*.npz.

Run:  OMP_NUM_THREADS=4 python uq_sample_forward_ssa.py --idx 0
"""
import argparse
import numpy as np
import netCDF4 as nc
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, CheckpointFile,
    sqrt, inner, max_value, exp, tanh, dx, conditional, assemble, as_vector,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
    weertman_sliding_law as m,
)
import icepack
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
MESH_FILE = DATA / "mesh" / "thwaites.h5"
SAMPLES = DATA / "results" / "posterior_samples_ssa.npz"
RACMO_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/racmo/"
                  "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc")

DT = 1.0 / 12.0
MMAX, ZTH = 30.0, 600.0     # Recinos Eq.5 melt
H_MIN = 10.0
A_0 = Constant(20.0)
U_0 = Constant(300.0)
SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 250, "snes_rtol": 1e-6,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}
PROG_PARAMS = {"snes_type": "ksponly", "ksp_type": "preonly",
               "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}


def friction_coulomb(**kwargs):
    u, h, s, τ_0 = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W / ρ_I * max_value(0, h - s) / h
    U = sqrt(inner(u, u))
    return τ_0 * ϕ * ((U_0 ** (1 / m + 1) + U ** (1 / m + 1)) ** (m / (m + 1)) - U_0)


def friction_weertman(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


FRICTION = {"coulomb": (friction_coulomb, 0.5), "weertman": (friction_weertman, 0.01)}


def load_racmo_smb(mesh, Q):
    ds = nc.Dataset(str(RACMO_FILE))
    lon, lat = ds.variables["lon"][:], ds.variables["lat"][:]
    smb_monthly = np.array(ds.variables["smbgl"][:, 0, :, :]); ds.close()
    smb_ice = (smb_monthly.mean(axis=0) * 12.0) / 917.0
    from pyproj import Transformer
    from scipy.interpolate import griddata
    tr = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    c = mesh.coordinates.dat.data_ro
    lons, lats = tr.transform(c[:, 0], c[:, 1])
    vals = griddata((lon.ravel(), lat.ravel()), smb_ice.ravel(), (lons, lats), method="linear", fill_value=0.0)
    a = Function(Q, name="smb"); a.dat.data[:] = vals
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--samples", default=str(SAMPLES))
    ap.add_argument("--label", default="ssa")
    ap.add_argument("--friction", choices=["coulomb", "weertman"], default="coulomb")
    args = ap.parse_args()
    fric, C0v = FRICTION[args.friction]
    C_0 = Constant(C0v)
    num_steps = max(1, int(round(args.T / DT)))
    out_file = DATA / "results" / f"sample_vaf_{args.label}{args.idx:03d}.npz"
    print(f"=== SSA sample {args.label}{args.idx}: T={args.T} yr, {num_steps} steps ===", flush=True)

    d = np.load(str(args.samples))
    thA, thC = d["theta_A"][args.idx], d["theta_C"][args.idx]

    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        h = chk.load_function(mesh, "thickness")
        s = chk.load_function(mesh, "surface")
        b = chk.load_function(mesh, "bed")
        u_obs = chk.load_function(mesh, "velocity")
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    smb = load_racmo_smb(mesh, Q)

    θ_A = Function(Q); θ_C = Function(Q)
    θ_A.dat.data[:] = thA; θ_C.dat.data[:] = thC
    grounded_mask = Function(Q).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01, Constant(1.0), Constant(0.0)))
    A = Function(Q).interpolate(A_0 * exp(θ_A))
    C = Function(Q).interpolate(C_0 * exp(θ_C * grounded_mask))

    model = icepack.models.IceStream(friction=fric)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS, prognostic_solver_parameters=PROG_PARAMS)

    u = Function(V).interpolate(u_obs)
    u.interpolate(conditional(sqrt(inner(u_obs, u_obs)) < 1.0,
                              as_vector([Constant(1.0), Constant(0.0)]), u_obs))
    try:
        u = solver.diagnostic_solve(velocity=u, thickness=h, surface=s, fluidity=A, friction=C)
    except fd.exceptions.ConvergenceError:
        print("  initial solve FAILED", flush=True)
        np.savez(str(out_file), idx=args.idx, failed=True); return

    h_inflow = h.copy(deepcopy=True)
    s_float = b + Constant(float(ρ_W / ρ_I)) * max_value(-b, Constant(0.0))

    def vaf_gt():
        return float(assemble(max_value(s - s_float, Constant(0.0)) * dx)) * 917.0 / 1e12

    times = [0.0]; vaf_ts = [vaf_gt()]; vaf0 = vaf_ts[0]; n_fail = 0
    print(f"  sample {args.idx}: VAF(0)={vaf0:.1f}, θ_A∈[{thA.min():.2f},{thA.max():.2f}], "
          f"θ_C∈[{thC.min():.2f},{thC.max():.2f}]", flush=True)
    for step in range(1, num_steps + 1):
        t = step * DT
        try:
            floating = Function(Q).interpolate(
                conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) < 0.01, Constant(1.0), Constant(0.0)))
            z_b = h - s
            m_melt = Constant(0.5 * MMAX) * (1 + tanh(Constant(2.0) * (z_b - Constant(ZTH)) / Constant(ZTH)))
            accum = Function(Q).interpolate(smb - floating * m_melt)
            h = solver.prognostic_solve(DT, thickness=h, velocity=u, accumulation=accum, thickness_inflow=h_inflow)
            h = Function(Q).interpolate(max_value(h, Constant(H_MIN)))
            s = icepack.compute_surface(thickness=h, bed=b)
            u = solver.diagnostic_solve(velocity=u, thickness=h, surface=s, fluidity=A, friction=C)
        except fd.exceptions.ConvergenceError:
            n_fail += 1
        times.append(t); vaf_ts.append(vaf_gt())
        if step % max(1, num_steps // 10) == 0:
            print(f"    t={t:5.1f}: VAF={vaf_ts[-1]:.1f} (ΔVAF={vaf_ts[-1]-vaf0:+.1f}), {n_fail} fail", flush=True)

    dVAF = vaf_ts[-1] - vaf0
    print(f"  DONE sample {args.label}{args.idx}: VAF(T)={vaf_ts[-1]:.1f}, ΔVAF={dVAF:+.2f} Gt, {n_fail} fails", flush=True)
    np.savez(str(out_file), idx=args.idx, failed=False, time=np.array(times),
             vaf_Gt=np.array(vaf_ts), vaf0=vaf0, vafT=vaf_ts[-1], dVAF=dVAF, n_fail=n_fail)


if __name__ == "__main__":
    main()
