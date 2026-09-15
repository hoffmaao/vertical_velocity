r"""Prognostic experiment using the SSA (IceStream) inversion.

100-year forward simulation on the hires Thwaites 2D mesh with RACMO
SMB and sub-shelf melt, driven by fluidity and friction fields from
the SSA (shallow stream) inversion.  No vertical structure — this is
the baseline to compare against the hybrid model runs.

Usage:
    OMP_NUM_THREADS=4 python prognostic_ssa.py
"""
import numpy as np
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, sqrt, inner, max_value, exp, dx, conditional,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)
from prognostic_common import (
    DATA_DIR, MESH_FILE, T_YEARS, DT, TOTAL_MELT_GT, H_MIN,
    SOLVER_PARAMS, load_racmo_smb,
)

INV_FILE = DATA_DIR / "mesh" / "inversion_ssa.h5"


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def main():
    case_name = "ssa"
    print("=" * 60, flush=True)
    print(f"Prognostic: {case_name} (SSA / IceStream)", flush=True)
    print(f"  {T_YEARS:.0f} yr, dt={DT:.4f} yr ({DT*12:.1f} mo), "
          f"melt={TOTAL_MELT_GT} Gt/yr", flush=True)
    print("=" * 60, flush=True)

    # ── Load mesh + fields (2D only, no extrusion) ─────────────────
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        h = chk.load_function(mesh, "thickness")
        s = chk.load_function(mesh, "surface")
        b = chk.load_function(mesh, "bed")
        u_obs = chk.load_function(mesh, "velocity")

    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    print(f"Mesh: {mesh.num_vertices()} verts", flush=True)

    # ── Load SSA inversion result ──────────────────────────────────
    with CheckpointFile(str(INV_FILE), "r") as chk:
        m_inv = chk.load_mesh()
        θ_A_f = chk.load_function(m_inv, "log_fluidity")
        θ_C_f = chk.load_function(m_inv, "log_friction")
    print(f"Loaded {INV_FILE.name}: "
          f"θ_A=[{θ_A_f.dat.data.min():.2f}, {θ_A_f.dat.data.max():.2f}], "
          f"θ_C=[{θ_C_f.dat.data.min():.2f}, {θ_C_f.dat.data.max():.2f}]",
          flush=True)

    θ_A = Function(Q, name="log_fluidity")
    θ_C = Function(Q, name="log_friction")
    θ_A.dat.data[:] = θ_A_f.dat.data_ro
    θ_C.dat.data[:] = θ_C_f.dat.data_ro

    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q).interpolate(
        conditional(
            1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
            Constant(1.0), Constant(0.0)))

    A = Function(Q).interpolate(A_0 * exp(θ_A))
    C = Function(Q).interpolate(C_0 * exp(θ_C * grounded_mask))

    # ── SMB ────────────────────────────────────────────────────────
    print("Loading RACMO SMB...", flush=True)
    smb = load_racmo_smb(mesh, Q)

    # ── SSA model + solver ─────────────────────────────────────────
    model = icepack.models.IceStream(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    u = Function(V).interpolate(u_obs)
    # Avoid zero velocity
    u.interpolate(firedrake.conditional(
        sqrt(inner(u_obs, u_obs)) < 1.0,
        firedrake.as_vector([Constant(1.0), Constant(0.0)]), u_obs))

    print("Initial diagnostic solve...", flush=True)
    u = solver.diagnostic_solve(velocity=u, thickness=h, surface=s,
                                 fluidity=A, friction=C)
    speed = Function(Q).interpolate(sqrt(inner(u, u)))
    print(f"  initial speed max: {speed.dat.data.max():.0f} m/yr", flush=True)

    # ── VAF helpers ────────────────────────────────────────────────
    def compute_vaf_gt(s_f, b_f):
        s_float = Function(Q).interpolate(
            b_f + Constant(float(ρ_W / ρ_I)) * max_value(-b_f, 0.0))
        h_af = Function(Q).interpolate(max_value(s_f - s_float, 0.0))
        return float(firedrake.assemble(h_af * dx(mesh))) * 917.0 / 1e12

    def compute_vol_gt(h_f):
        return float(firedrake.assemble(h_f * dx(mesh))) * 917.0 / 1e12

    # ── Time-stepping ──────────────────────────────────────────────
    num_steps = int(T_YEARS / DT)
    h_inflow = h.copy(deepcopy=True)

    times = [0.0]
    vaf_ts = [compute_vaf_gt(s, b)]
    vol_ts = [compute_vol_gt(h)]
    print(f"  t=0: VAF={vaf_ts[0]:.1f} Gt, Vol={vol_ts[0]:.1f} Gt", flush=True)

    SAVE_EVERY = max(1, int(1.0 / DT))
    OUTPUT_FILE = DATA_DIR / "mesh" / f"prognostic_{case_name}_{int(TOTAL_MELT_GT)}Gt.h5"

    total_melt_m3_yr = TOTAL_MELT_GT * 1e12 / 917.0
    n_diag_fail = 0

    print(f"\nTime-stepping: {num_steps} steps, dt={DT:.4f} yr...", flush=True)
    for step in range(1, num_steps + 1):
        t = step * DT

        try:
            floating = Function(Q).interpolate(
                conditional(
                    1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) < 0.01,
                    Constant(1.0), Constant(0.0)))

            float_area = float(firedrake.assemble(floating * dx))
            melt_rate = total_melt_m3_yr / float_area if float_area > 0 else 0.0

            accum = Function(Q).interpolate(smb - floating * Constant(melt_rate))

            h = solver.prognostic_solve(
                DT, thickness=h, velocity=u,
                accumulation=accum, thickness_inflow=h_inflow)

            h.interpolate(max_value(h, Constant(H_MIN)))
            s = icepack.compute_surface(thickness=h, bed=b)

            u = solver.diagnostic_solve(
                velocity=u, thickness=h, surface=s, fluidity=A, friction=C)

        except firedrake.exceptions.ConvergenceError:
            n_diag_fail += 1
            if step % SAVE_EVERY == 0:
                print(f"  t={t:.2f}: FAILED ({n_diag_fail} total)",
                      flush=True)
        except Exception as e:
            n_diag_fail += 1
            print(f"  t={t:.2f}: {type(e).__name__}: {e}", flush=True)

        times.append(t)
        vaf_ts.append(compute_vaf_gt(s, b))
        vol_ts.append(compute_vol_gt(h))

        if step % SAVE_EVERY == 0:
            dVAF = vaf_ts[-1] - vaf_ts[0]
            print(f"  t={t:5.1f}: VAF={vaf_ts[-1]:.1f} Gt "
                  f"(ΔVAF={dVAF:+.1f} Gt), "
                  f"Vol={vol_ts[-1]:.1f} Gt, "
                  f"melt={melt_rate:.1f} m/yr", flush=True)

    # ── Save ───────────────────────────────────────────────────────
    print(f"\n{n_diag_fail} diagnostic failures in {num_steps} steps", flush=True)

    speed_final = Function(Q, name="speed_final").interpolate(sqrt(inner(u, u)))
    h_out = Function(Q, name="thickness_final").assign(h)
    s_out = Function(Q, name="surface_final").assign(s)

    with CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh)
        chk.save_function(h_out, name="thickness_final")
        chk.save_function(s_out, name="surface_final")
        chk.save_function(speed_final, name="speed_final")
    print(f"Saved {OUTPUT_FILE}", flush=True)

    np.savez(str(OUTPUT_FILE).replace(".h5", "_timeseries.npz"),
             time=np.array(times), vaf=np.array(vaf_ts), vol=np.array(vol_ts))
    print(f"Saved timeseries ({len(times)} steps)", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
