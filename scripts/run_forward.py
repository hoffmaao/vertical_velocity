r"""Taped 50-yr (configurable) prognostic forward with VAF gradient.

Tapes a HybridModel + RACMO + prescribed melt forward run through tlm_adjoint
and computes ∂VAF(T)/∂(θ_A, θ_C) via adjoint. The gradient feeds the
Isaac et al. (2015) posterior-variance formula in uq_vaf.py.

Run:
    OMP_NUM_THREADS=4 python run_forward.py --T 1.0    # smoke test
    OMP_NUM_THREADS=4 python run_forward.py --T 50.0   # production
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
from firedrake.adjoint import *  # noqa
from tlm_adjoint.firedrake import (
    reset_manager, start_manager, stop_manager,
    compute_gradient, Functional,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)
import icepack
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
MAP_FILE = DATA_DIR / "mesh" / "inversion_hires_apres_vd1.h5"
RACMO_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/racmo/"
                  "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc")

DT = 1.0 / 12.0
MMAX = 30.0   # Recinos et al. (2023) Eq. 5 max sub-shelf melt rate (m/yr ice eq.)
ZTH = 600.0   # Recinos et al. (2023) Eq. 5 thermocline depth (m)
H_MIN = 10.0

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 100, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
PROG_PARAMS = {
    "snes_type": "ksponly",
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def load_racmo_smb(mesh_2d, Q_2d):
    ds = nc.Dataset(str(RACMO_FILE))
    lon_2d = ds.variables["lon"][:]
    lat_2d = ds.variables["lat"][:]
    smb_monthly = np.array(ds.variables["smbgl"][:, 0, :, :])
    ds.close()
    smb_annual = smb_monthly.mean(axis=0) * 12.0
    smb_ice = smb_annual / 917.0
    from pyproj import Transformer
    from scipy.interpolate import griddata
    tr = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    coords = mesh_2d.coordinates.dat.data_ro
    lons, lats = tr.transform(coords[:, 0], coords[:, 1])
    smb_at_mesh = griddata(
        (lon_2d.ravel(), lat_2d.ravel()), smb_ice.ravel(),
        (lons, lats), method="linear", fill_value=0.0)
    a = Function(Q_2d, name="smb")
    a.dat.data[:] = smb_at_mesh
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=50.0,
                    help="Forward run length in years")
    ap.add_argument("--out", default=None,
                    help="Output checkpoint path (default: results/forward_vaf_<T>yr.h5)")
    args = ap.parse_args()

    T_YEARS = args.T
    num_steps = max(1, int(round(T_YEARS / DT)))
    out_file = Path(args.out) if args.out else (
        DATA_DIR / "results" / f"forward_vaf_{int(T_YEARS)}yr.h5")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f"Taped prognostic forward — VAF gradient", flush=True)
    print(f"  T={T_YEARS} yr, {num_steps} steps (dt={DT:.4f} yr)", flush=True)
    print(f"  Output: {out_file}", flush=True)
    print("=" * 60, flush=True)

    # ── Load mesh + fields ──
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        b_2d = chk.load_function(mesh_2d, "bed")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")
    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    print(f"Mesh: {mesh_2d.num_vertices()} verts", flush=True)

    # ── Load MAP controls ──
    with CheckpointFile(str(MAP_FILE), "r") as chk:
        m_inv = chk.load_mesh()
        θA_loaded = chk.load_function(m_inv, "log_fluidity")
        θC_loaded = chk.load_function(m_inv, "log_friction")

    # ── SMB ──
    print("Loading RACMO SMB...", flush=True)
    smb_2d = load_racmo_smb(mesh_2d, Q_2d)
    print(f"  SMB: [{smb_2d.dat.data.min():.3f}, {smb_2d.dat.data.max():.3f}] m/yr",
          flush=True)

    # ── Extrude ──
    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    b = icepack.utilities.lift3d(b_2d, Q_lift)
    u0 = Function(V).project(icepack.utilities.lift3d(u_obs_2d, V_lift))

    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    θ_A.dat.data[:] = θA_loaded.dat.data_ro
    θ_C.dat.data[:] = θC_loaded.dat.data_ro

    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q_lift).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
                    Constant(1.0), Constant(0.0)))
    smb_3d = icepack.utilities.lift3d(smb_2d, Q_lift)

    # ── Pre-tape prime velocity ──
    A_init = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_init = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))

    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS,
        prognostic_solver_parameters=PROG_PARAMS)
    u0 = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                  fluidity=A_init, friction=C_init)
    print(f"Primed (initial speed max: "
          f"{Function(Q_2d).interpolate(sqrt(inner(icepack.depth_average(u0), icepack.depth_average(u0)))).dat.data.max():.0f} m/yr)",
          flush=True)

    # Fixed inflow boundary thickness
    h_inflow = h.copy(deepcopy=True)

    # ── Start tape ──
    print(f"\nTime-stepping (taped, {num_steps} steps)...", flush=True)
    reset_manager()
    start_manager()

    A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))

    u = u0
    times = [0.0]
    vaf_ts = []

    # Initial VAF on 3D (h, s vdegree=0 → 3D ≡ 2D area integral)
    s_float = b + Constant(float(ρ_W / ρ_I)) * max_value(-b, Constant(0.0))
    haf = max_value(s - s_float, Constant(0.0))
    vaf0 = float(assemble(haf * dx)) * 917.0 / 1e12
    vaf_ts.append(vaf0)
    print(f"  t=0: VAF={vaf0:.1f} Gt", flush=True)

    for step in range(1, num_steps + 1):
        t = step * DT

        # Sub-shelf melt: Recinos et al. (2023) Eq. 5 depth-dependent rate
        #   m(z_b) = (Mmax/2)(1 + tanh(2(z_b - z_th)/z_th)),  z_b = ice draft = h - s.
        # m is SMOOTH in (h, s) and kept INSIDE the tape, so ∂melt/∂θ (the draft
        # feedback) is captured — unlike the old area-normalized melt, which was
        # frozen outside the tape and dropped a ~2× feedback. Only the floating
        # EXTENT mask is frozen per step (un-taped), as for the grounded mask.
        stop_manager()
        floating = Function(Q_lift).interpolate(
            conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) < 0.01,
                        Constant(1.0), Constant(0.0)))
        start_manager()

        z_b = h - s   # ice draft (depth of base below sea level); taped via h, s
        m_melt = Constant(0.5 * MMAX) * (
            1 + tanh(Constant(2.0) * (z_b - Constant(ZTH)) / Constant(ZTH)))
        accum = Function(Q_lift, name=f"accum_t{step}").interpolate(
            smb_3d - floating * m_melt)

        # Prognostic
        h = solver.prognostic_solve(
            DT, thickness=h, velocity=u,
            accumulation=accum, thickness_inflow=h_inflow)
        h = Function(Q_lift).interpolate(max_value(h, Constant(H_MIN)))
        s = icepack.compute_surface(thickness=h, bed=b)

        # Diagnostic
        u = solver.diagnostic_solve(velocity=u, thickness=h, surface=s,
                                     fluidity=A, friction=C)

        # Track VAF
        haf_k = max_value(s - s_float, Constant(0.0))
        vaf_k = float(assemble(haf_k * dx)) * 917.0 / 1e12
        times.append(t)
        vaf_ts.append(vaf_k)
        if step % max(1, num_steps // 20) == 0 or step == num_steps:
            stop_manager()
            melt_flux = float(assemble(floating * m_melt * dx)) * 917.0 / 1e12
            start_manager()
            print(f"  t={t:5.2f}: VAF={vaf_k:.1f} Gt "
                  f"(ΔVAF={vaf_k - vaf0:+.2f}), melt={melt_flux:.1f} Gt/yr", flush=True)

    # ── Final QoI + gradient ──
    print("\nAssembling final VAF as Functional...", flush=True)
    haf_final = max_value(s - s_float, Constant(0.0))
    J = Functional(name="VAF_T")
    J.assign(haf_final * dx)
    print(f"  J(VAF_T, raw m^3) = {float(J):.6e}", flush=True)

    print("Computing adjoint gradient ∂VAF/∂(θ_A, θ_C)...", flush=True)
    dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
    stop_manager()

    g_A_norm = np.linalg.norm(dJ_A.dat.data_ro)
    g_C_norm = np.linalg.norm(dJ_C.dat.data_ro)
    print(f"  ||∂VAF/∂θ_A|| = {g_A_norm:.4e}", flush=True)
    print(f"  ||∂VAF/∂θ_C|| = {g_C_norm:.4e}", flush=True)

    # ── Save ──
    # Project gradients back to 2D mesh (Q_lift CG1-DG0 has same DOFs as Q_2d CG1)
    gA_2d = Function(Q_2d, name="dVAF_dthetaA")
    gC_2d = Function(Q_2d, name="dVAF_dthetaC")
    gA_2d.dat.data[:] = dJ_A.dat.data_ro
    gC_2d.dat.data[:] = dJ_C.dat.data_ro

    with CheckpointFile(str(out_file), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(gA_2d, name="dVAF_dthetaA")
        chk.save_function(gC_2d, name="dVAF_dthetaC")
    print(f"Saved {out_file}", flush=True)

    np.savez(str(out_file).replace(".h5", "_ts.npz"),
             time=np.array(times), vaf_Gt=np.array(vaf_ts),
             J_raw_m3=float(J), T_years=T_YEARS)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
