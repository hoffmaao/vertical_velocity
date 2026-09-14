r"""Shared utilities for prognostic experiments on Thwaites.

Provides mesh loading, RACMO SMB interpolation, the hybrid model setup,
and the time-stepping loop. Each prognostic script imports this module
and calls `run_prognostic()` with the appropriate inversion file.

Melt forcing follows Joughin et al. (2024, TC): prescribed total melt
re-normalized to floating area each timestep. Flotation uses the
standard icepack approach (icepack.compute_surface).
"""
import numpy as np
import netCDF4 as nc
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, ExtrudedMesh,
    sqrt, inner, max_value, exp, dx, conditional,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
RACMO_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/racmo/"
                  "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc")

VDEGREE = 1
HDEGREE = 1
T_YEARS = 100.0
DT = 1.0 / 12.0     # monthly timesteps
TOTAL_MELT_GT = 40.0 # Gt/yr total melt applied to floating ice
H_MIN = 10.0         # m, thickness floor

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 100, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def load_racmo_smb(mesh_2d, Q_2d):
    """Interpolate RACMO mean annual SMB onto the 2D mesh.

    Returns a Function in m/yr ice equivalent.
    """
    ds = nc.Dataset(str(RACMO_FILE))
    lon_2d = ds.variables["lon"][:]
    lat_2d = ds.variables["lat"][:]
    smb_monthly = np.array(ds.variables["smbgl"][:, 0, :, :])
    ds.close()

    smb_annual = smb_monthly.mean(axis=0) * 12.0
    smb_ice = smb_annual / 917.0

    from pyproj import Transformer
    from scipy.interpolate import griddata
    transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    coords = mesh_2d.coordinates.dat.data_ro
    lons, lats = transformer.transform(coords[:, 0], coords[:, 1])

    smb_at_mesh = griddata(
        (lon_2d.ravel(), lat_2d.ravel()),
        smb_ice.ravel(),
        (lons, lats),
        method="linear",
        fill_value=0.0,
    )

    a = Function(Q_2d, name="smb")
    a.dat.data[:] = smb_at_mesh
    print(f"  SMB: [{a.dat.data.min():.3f}, {a.dat.data.max():.3f}] m/yr ice eq",
          flush=True)
    return a


def run_prognostic(case_name, inv_file):
    """Run a prognostic simulation from a given inversion result."""
    print("=" * 60, flush=True)
    print(f"Prognostic: {case_name}", flush=True)
    print(f"  {T_YEARS:.0f} yr, dt={DT:.4f} yr ({DT*12:.1f} mo), "
          f"melt={TOTAL_MELT_GT} Gt/yr", flush=True)
    print("=" * 60, flush=True)

    # ── Load mesh + fields ─────────────────────────────────────────
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        b_2d = chk.load_function(mesh_2d, "bed")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    print(f"Mesh: {mesh_2d.num_vertices()} verts", flush=True)

    # ── Load inversion result ──────────────────────────────────────
    with CheckpointFile(str(inv_file), "r") as chk:
        m_inv = chk.load_mesh()
        θ_A_2d = chk.load_function(m_inv, "log_fluidity")
        θ_C_2d = chk.load_function(m_inv, "log_friction")
    print(f"Loaded {inv_file.name}: "
          f"θ_A=[{θ_A_2d.dat.data.min():.2f}, {θ_A_2d.dat.data.max():.2f}], "
          f"θ_C=[{θ_C_2d.dat.data.min():.2f}, {θ_C_2d.dat.data.max():.2f}]",
          flush=True)

    # ── SMB forcing ────────────────────────────────────────────────
    print("Loading RACMO SMB...", flush=True)
    smb_2d = load_racmo_smb(mesh_2d, Q_2d)

    # ── Extrude ────────────────────────────────────────────────────
    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", HDEGREE,
                             vfamily="GL", vdegree=VDEGREE, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1,
                                  vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    b = icepack.utilities.lift3d(b_2d, Q_lift)
    u_obs_3d = icepack.utilities.lift3d(u_obs_2d, V_lift)
    u = Function(V).project(u_obs_3d)

    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    θ_A.dat.data[:] = θ_A_2d.dat.data_ro
    θ_C.dat.data[:] = θ_C_2d.dat.data_ro

    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q_lift).interpolate(
        conditional(
            1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
            Constant(1.0), Constant(0.0)))

    A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))

    # ── Model + solver ─────────────────────────────────────────────
    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    print("Initial diagnostic solve...", flush=True)
    u = solver.diagnostic_solve(velocity=u, thickness=h, surface=s,
                                 fluidity=A, friction=C)
    u_avg = icepack.depth_average(u)
    print(f"  initial speed max: "
          f"{Function(Q_2d).interpolate(sqrt(inner(u_avg, u_avg))).dat.data.max():.0f} m/yr",
          flush=True)

    # ── Accumulation ───────────────────────────────────────────────
    smb_3d = icepack.utilities.lift3d(smb_2d, Q_lift)

    # ── VAF helpers (on 2D mesh) ───────────────────────────────────
    def compute_vaf_gt(s_f, b_f):
        """Volume above flotation in Gt using icepack's flotation."""
        s_float = Function(Q_2d).interpolate(
            b_f + Constant(float(ρ_W / ρ_I)) * max_value(-b_f, 0.0))
        h_af = Function(Q_2d).interpolate(max_value(s_f - s_float, 0.0))
        return float(firedrake.assemble(h_af * dx(mesh_2d))) * 917.0 / 1e12

    def compute_vol_gt(h_f):
        """Total ice volume in Gt."""
        return float(firedrake.assemble(h_f * dx(mesh_2d))) * 917.0 / 1e12

    # ── Time-stepping ──────────────────────────────────────────────
    num_steps = int(T_YEARS / DT)
    h_inflow = h.copy(deepcopy=True)

    times = [0.0]
    vaf_ts = []
    vol_ts = []

    # 2D diagnostic fields (updated from 3D each step)
    s_2d_diag = Function(Q_2d, name="surface_diag")
    h_2d_diag = Function(Q_2d, name="thickness_diag")
    s_2d_diag.dat.data[:] = s_2d.dat.data_ro
    h_2d_diag.dat.data[:] = h_2d.dat.data_ro

    vaf_ts.append(compute_vaf_gt(s_2d_diag, b_2d))
    vol_ts.append(compute_vol_gt(h_2d_diag))
    print(f"  t=0: VAF={vaf_ts[0]:.1f} Gt, Vol={vol_ts[0]:.1f} Gt", flush=True)

    SAVE_EVERY = max(1, int(1.0 / DT))  # print every 1 year
    OUTPUT_FILE = DATA_DIR / "mesh" / f"prognostic_{case_name}_{int(TOTAL_MELT_GT)}Gt.h5"

    total_melt_m3_yr = TOTAL_MELT_GT * 1e12 / 917.0  # m³/yr ice equivalent
    n_diag_fail = 0

    print(f"\nTime-stepping: {num_steps} steps, dt={DT:.4f} yr...", flush=True)
    for step in range(1, num_steps + 1):
        t = step * DT

        try:
            # Floating mask (updated each step as geometry evolves)
            floating = Function(Q_lift).interpolate(
                conditional(
                    1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) < 0.01,
                    Constant(1.0), Constant(0.0)))

            # Normalize melt to floating area
            float_area = float(firedrake.assemble(
                icepack.depth_average(floating) * dx))
            melt_rate = total_melt_m3_yr / float_area if float_area > 0 else 0.0

            accum = Function(Q_lift).interpolate(
                smb_3d - floating * Constant(melt_rate))

            # Prognostic: thickness evolution
            h = solver.prognostic_solve(
                DT, thickness=h, velocity=u,
                accumulation=accum, thickness_inflow=h_inflow)

            # Floor thickness
            h.interpolate(max_value(h, Constant(H_MIN)))

            # Recompute surface using icepack's flotation
            s = icepack.compute_surface(thickness=h, bed=b)

            # Diagnostic: velocity from updated geometry
            u = solver.diagnostic_solve(
                velocity=u, thickness=h, surface=s, fluidity=A, friction=C)

        except firedrake.exceptions.ConvergenceError:
            n_diag_fail += 1
            if step % SAVE_EVERY == 0:
                print(f"  t={t:.2f}: diagnostic FAILED "
                      f"({n_diag_fail} total), using previous u", flush=True)
        except Exception as e:
            n_diag_fail += 1
            print(f"  t={t:.2f}: UNEXPECTED {type(e).__name__}: {e}",
                  flush=True)

        # 2D projections for VAF
        s_2d_diag.dat.data[:] = icepack.depth_average(s).dat.data_ro
        h_2d_diag.dat.data[:] = icepack.depth_average(h).dat.data_ro

        times.append(t)
        vaf_ts.append(compute_vaf_gt(s_2d_diag, b_2d))
        vol_ts.append(compute_vol_gt(h_2d_diag))

        if step % SAVE_EVERY == 0:
            dVAF = vaf_ts[-1] - vaf_ts[0]
            print(f"  t={t:5.1f}: VAF={vaf_ts[-1]:.1f} Gt "
                  f"(ΔVAF={dVAF:+.1f} Gt), "
                  f"Vol={vol_ts[-1]:.1f} Gt, "
                  f"melt={melt_rate:.1f} m/yr", flush=True)

    # ── Save results ───────────────────────────────────────────────
    print(f"\n{n_diag_fail} diagnostic failures in {num_steps} steps", flush=True)

    h_2d_out = Function(Q_2d, name="thickness_final")
    h_2d_out.dat.data[:] = icepack.depth_average(h).dat.data_ro
    s_2d_out = Function(Q_2d, name="surface_final")
    s_2d_out.dat.data[:] = icepack.depth_average(s).dat.data_ro
    u_avg = icepack.depth_average(u)
    speed_final = Function(Q_2d, name="speed_final").interpolate(
        sqrt(inner(u_avg, u_avg)))

    with CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(h_2d_out, name="thickness_final")
        chk.save_function(s_2d_out, name="surface_final")
        chk.save_function(speed_final, name="speed_final")
    print(f"Saved {OUTPUT_FILE}", flush=True)

    np.savez(str(OUTPUT_FILE).replace(".h5", "_timeseries.npz"),
             time=np.array(times), vaf=np.array(vaf_ts), vol=np.array(vol_ts))
    print(f"Saved timeseries ({len(times)} steps)", flush=True)
    print("Done!", flush=True)
