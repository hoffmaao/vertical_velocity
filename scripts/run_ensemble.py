r"""Ensemble prognostic with posterior samples from Hessian eigenmodes.

Draws N_SAMPLES posterior samples from the eigendecomposition:
  θ_k = θ_MAP + Σ_i (ξ_i / √(1+λ_i)) v_i,  ξ_i ~ N(0,1)

For each sample, runs a 100-year prognostic and records VAF time series.
Outputs an ensemble CSV + uncertainty plot.

Usage:
    OMP_NUM_THREADS=4 python run_ensemble.py
"""
import numpy as np
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, ExtrudedMesh, MixedFunctionSpace,
    sqrt, inner, max_value, exp, dx, conditional,
)
import icepack
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
from prognostic_common import (
    MESH_FILE, T_YEARS, DT, TOTAL_MELT_GT, H_MIN,
    SOLVER_PARAMS, friction, load_racmo_smb, DATA_DIR,
    RACMO_FILE,
)
from pathlib import Path
import json

MAP_FILE = DATA_DIR / "mesh" / "inversion_hires_apres_vd1.h5"
EIGDEC_FILE = DATA_DIR / "mesh" / "eigendec.h5"
EIGVAL_FILE = DATA_DIR / "results" / "eigenvalues.txt"

N_SAMPLES = 30
N_MODES = 40
SEED = 42


def main():
    print("=" * 60, flush=True)
    print(f"Ensemble prognostic: {N_SAMPLES} samples, {N_MODES} modes", flush=True)
    print(f"  {T_YEARS:.0f} yr, dt={DT:.4f} yr, melt={TOTAL_MELT_GT} Gt/yr", flush=True)
    print("=" * 60, flush=True)

    # ── Load eigenvalues ───────────────────────────────────────────
    eigvals = np.loadtxt(str(EIGVAL_FILE))[:, 1][:N_MODES]
    weights = 1.0 / np.sqrt(1.0 + eigvals)
    D = eigvals / (1.0 + eigvals)
    print(f"Eigenvalues: [{eigvals.min():.2e}, {eigvals.max():.2e}]", flush=True)
    print(f"Weights (1/√(1+λ)): [{weights.min():.2e}, {weights.max():.2e}]", flush=True)
    print(f"n_eff = {D.sum():.1f}", flush=True)

    # ── Load mesh + fields ─────────────────────────────────────────
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        b_2d = chk.load_function(mesh_2d, "bed")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    ndof = Q_2d.dof_dset.size

    # ── Load MAP ───────────────────────────────────────────────────
    with CheckpointFile(str(MAP_FILE), "r") as chk:
        m_inv = chk.load_mesh()
        θA_map = chk.load_function(m_inv, "log_fluidity")
        θC_map = chk.load_function(m_inv, "log_friction")
    θA_map_vals = θA_map.dat.data_ro.copy()
    θC_map_vals = θC_map.dat.data_ro.copy()

    # ── Load eigenmodes ────────────────────────────────────────────
    QQ = MixedFunctionSpace([Q_2d, Q_2d])
    modes_A = []
    modes_C = []
    with CheckpointFile(str(EIGDEC_FILE), "r") as chk:
        eig_mesh = chk.load_mesh()
        for i in range(N_MODES):
            mode = chk.load_function(eig_mesh, name=f"mode_{i:04d}")
            modes_A.append(mode.sub(0).dat.data_ro.copy())
            modes_C.append(mode.sub(1).dat.data_ro.copy())
    print(f"Loaded {N_MODES} eigenmodes", flush=True)

    # ── Generate posterior samples ─────────────────────────────────
    rng = np.random.default_rng(SEED)
    samples_A = [θA_map_vals.copy()]  # sample 0 = MAP
    samples_C = [θC_map_vals.copy()]
    for k in range(N_SAMPLES):
        xi = rng.standard_normal(N_MODES)
        θA_k = θA_map_vals.copy()
        θC_k = θC_map_vals.copy()
        for i in range(N_MODES):
            θA_k += xi[i] * weights[i] * modes_A[i]
            θC_k += xi[i] * weights[i] * modes_C[i]
        samples_A.append(θA_k)
        samples_C.append(θC_k)
    print(f"Generated {N_SAMPLES} posterior samples + MAP", flush=True)

    # ── SMB ────────────────────────────────────────────────────────
    print("Loading RACMO SMB...", flush=True)
    smb_2d = load_racmo_smb(mesh_2d, Q_2d)

    # ── Extrude ────────────────────────────────────────────────────
    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)

    h_init = icepack.utilities.lift3d(h_2d, Q_lift)
    s_init = icepack.utilities.lift3d(s_2d, Q_lift)
    b = icepack.utilities.lift3d(b_2d, Q_lift)
    u_obs_3d = icepack.utilities.lift3d(u_obs_2d, V_lift)
    smb_3d = icepack.utilities.lift3d(smb_2d, Q_lift)

    A_0, C_0 = Constant(20.0), Constant(0.01)

    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    # ── VAF helper ─────────────────────────────────────────────────
    def compute_vaf_gt(s_f, b_f):
        s_2d_f = Function(Q_2d)
        s_2d_f.dat.data[:] = icepack.depth_average(s_f).dat.data_ro
        s_float = Function(Q_2d).interpolate(
            b_2d + Constant(float(ρ_W / ρ_I)) * max_value(-b_2d, 0.0))
        h_af = Function(Q_2d).interpolate(max_value(s_2d_f - s_float, 0.0))
        return float(fd.assemble(h_af * dx(mesh_2d))) * 917.0 / 1e12

    # ── Time-stepping parameters ───────────────────────────────────
    num_steps = int(T_YEARS / DT)
    total_melt_m3_yr = TOTAL_MELT_GT * 1e12 / 917.0
    SAVE_EVERY = max(1, int(1.0 / DT))
    n_saves = int(T_YEARS) + 1

    # Storage: rows = samples (0=MAP, 1..N=posterior), cols = years
    all_vaf = np.zeros((N_SAMPLES + 1, n_saves))

    # ── Run ensemble ───────────────────────────────────────────────
    for sample_idx in range(N_SAMPLES + 1):
        label = "MAP" if sample_idx == 0 else f"sample {sample_idx}"
        print(f"\n{'─'*40}", flush=True)
        print(f"Running {label}...", flush=True)

        # Set controls
        θ_A = Function(Q_lift)
        θ_C = Function(Q_lift)
        θ_A.dat.data[:] = samples_A[sample_idx]
        θ_C.dat.data[:] = samples_C[sample_idx]

        grounded_mask = Function(Q_lift).interpolate(
            conditional(
                1 - ρ_W * g * max_value(0, h_init - s_init) / (ρ_I * g * h_init) > 0.01,
                Constant(1.0), Constant(0.0)))
        A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
        C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))

        # Reset state
        h = h_init.copy(deepcopy=True)
        s = s_init.copy(deepcopy=True)
        u = Function(V).project(u_obs_3d)

        # Initial diagnostic
        try:
            u = solver.diagnostic_solve(velocity=u, thickness=h, surface=s,
                                         fluidity=A, friction=C)
        except fd.exceptions.ConvergenceError:
            print(f"  initial solve FAILED — skipping", flush=True)
            all_vaf[sample_idx, :] = np.nan
            continue

        h_inflow = h.copy(deepcopy=True)
        yr_idx = 0
        all_vaf[sample_idx, 0] = compute_vaf_gt(s, b)
        n_fail = 0

        for step in range(1, num_steps + 1):
            t = step * DT
            try:
                floating = Function(Q_lift).interpolate(
                    conditional(
                        1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) < 0.01,
                        Constant(1.0), Constant(0.0)))
                float_area = float(fd.assemble(icepack.depth_average(floating) * dx))
                melt_rate = total_melt_m3_yr / float_area if float_area > 0 else 0.0
                accum = Function(Q_lift).interpolate(
                    smb_3d - floating * Constant(melt_rate))
                h = solver.prognostic_solve(
                    DT, thickness=h, velocity=u,
                    accumulation=accum, thickness_inflow=h_inflow)
                h.interpolate(max_value(h, Constant(H_MIN)))
                s = icepack.compute_surface(thickness=h, bed=b)
                u = solver.diagnostic_solve(
                    velocity=u, thickness=h, surface=s, fluidity=A, friction=C)
            except fd.exceptions.ConvergenceError:
                n_fail += 1
            except Exception:
                n_fail += 1

            if step % SAVE_EVERY == 0:
                yr_idx += 1
                if yr_idx < n_saves:
                    all_vaf[sample_idx, yr_idx] = compute_vaf_gt(s, b)

        dVAF = all_vaf[sample_idx, -1] - all_vaf[sample_idx, 0]
        print(f"  {label}: ΔVAF={dVAF:+.0f} Gt, {n_fail} failures", flush=True)

    # ── Save ───────────────────────────────────────────────────────
    times = np.arange(n_saves, dtype=float)
    out_csv = DATA_DIR / "results" / "ensemble_vaf.csv"
    header = "time_yr," + ",".join(
        ["MAP"] + [f"sample_{i}" for i in range(1, N_SAMPLES + 1)])
    np.savetxt(str(out_csv), np.column_stack([times, all_vaf.T]),
               delimiter=",", header=header, comments="")
    print(f"\nSaved {out_csv}", flush=True)

    np.savez(str(DATA_DIR / "results" / "ensemble_vaf.npz"),
             time=times, vaf=all_vaf, eigenvalues=eigvals[:N_MODES])
    print(f"Saved ensemble_vaf.npz", flush=True)

    # ── Quick plot ─────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    # Posterior samples
    for i in range(1, N_SAMPLES + 1):
        dv = all_vaf[i, :] - all_vaf[i, 0]
        ax.plot(times, dv, color="C0", alpha=0.15, lw=0.5)
    # MAP
    dv_map = all_vaf[0, :] - all_vaf[0, 0]
    ax.plot(times, dv_map, "k-", lw=2, label="MAP")
    # Percentiles
    dv_all = all_vaf[1:, :] - all_vaf[1:, 0:1]
    p5 = np.nanpercentile(dv_all, 5, axis=0)
    p95 = np.nanpercentile(dv_all, 95, axis=0)
    p25 = np.nanpercentile(dv_all, 25, axis=0)
    p75 = np.nanpercentile(dv_all, 75, axis=0)
    ax.fill_between(times, p5, p95, color="C0", alpha=0.15, label="5–95%")
    ax.fill_between(times, p25, p75, color="C0", alpha=0.3, label="25–75%")
    ax.set_xlabel("Time (yr)", fontsize=12)
    ax.set_ylabel("ΔVAF (Gt)", fontsize=12)
    ax.set_title(f"Ensemble VAF projection ({N_SAMPLES} posterior samples)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.savefig(str(DATA_DIR / "figures" / "ensemble_vaf.png"), dpi=200, bbox_inches="tight")
    print(f"Saved figures/ensemble_vaf.png", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
