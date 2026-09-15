r"""Hybrid model inversion using manual gradient descent.

Bypasses StatisticsProblem (which stalls with 3D controls) and uses
compute_gradient directly in a simple L-BFGS loop via scipy.

Usage:
    python inversion_hybrid_manual.py
"""
import numpy as np
import firedrake
import icepack
from firedrake import *
from firedrake.adjoint import *
from tlm_adjoint.firedrake import reset_manager, start_manager, stop_manager, compute_gradient, Functional
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
    weertman_sliding_law as m,
)
from scipy.optimize import minimize as scipy_minimize
from pathlib import Path
import sys

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
SSA_FILE = DATA_DIR / "mesh" / "inversion_hybrid.h5"  # warm start from vdeg=1 2D-control result
OUTPUT_FILE = DATA_DIR / "mesh" / "inversion_hybrid_3d.h5"

VDEGREE = 1
HDEGREE = 1
L_REG = 10e3
MAX_ITER = 200

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 200, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def main():
    print("=" * 60, flush=True)
    print(f"Hybrid inversion (manual L-BFGS, vdeg={VDEGREE})", flush=True)
    print("=" * 60, flush=True)

    # Load mesh
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", HDEGREE, vfamily="GL", vdegree=VDEGREE, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)

    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    u_obs_3d = icepack.utilities.lift3d(u_obs_2d, V_lift)
    u0 = Function(V).project(u_obs_3d)

    # 3D controls
    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    A_0, C_0 = Constant(20.0), Constant(0.01)

    # Grounded mask
    ϕ_mask = Function(Q_lift).interpolate(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h))
    grounded_mask = Function(Q_lift).interpolate(conditional(ϕ_mask > 0.01, Constant(1.0), Constant(0.0)))

    # Warm start — only works in serial (mesh partition mismatch in MPI)
    from firedrake.petsc import PETSc
    if SSA_FILE and Path(SSA_FILE).exists() and PETSc.COMM_WORLD.size == 1:
        with CheckpointFile(str(SSA_FILE), "r") as chk:
            m2 = chk.load_mesh()
            θ_A.dat.data[:] = chk.load_function(m2, "log_fluidity").dat.data_ro
            θ_C.dat.data[:] = chk.load_function(m2, "log_friction").dat.data_ro
        print(f"Warm start: θ_A=[{θ_A.dat.data.min():.2f}, {θ_A.dat.data.max():.2f}], "
              f"θ_C=[{θ_C.dat.data.min():.2f}, {θ_C.dat.data.max():.2f}]", flush=True)
    else:
        print(f"Cold start (θ=0, nprocs={PETSc.COMM_WORLD.size})", flush=True)

    # Model
    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(model, dirichlet_ids=[2], ice_front_ids=[1],
                                         diagnostic_solver_type="petsc",
                                         diagnostic_solver_parameters=SOLVER_PARAMS)

    # Forward solve to get good initial velocity
    A_init = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_init = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    u_test = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A_init, friction=C_init)
    u0.assign(u_test)
    print(f"Initial speed: [{Function(Q_2d).interpolate(sqrt(inner(icepack.depth_average(u_test), icepack.depth_average(u_test)))).dat.data.min():.0f}, "
          f"{Function(Q_2d).interpolate(sqrt(inner(icepack.depth_average(u_test), icepack.depth_average(u_test)))).dat.data.max():.0f}] m/yr", flush=True)

    from firedrake.petsc import PETSc
    comm = PETSc.COMM_WORLD

    n_dof_local = θ_A.dat.data.shape[0]
    area = assemble(Constant(1.0) * dx(mesh))
    L_REG_c = Constant(L_REG)
    σ_u = Constant(10.0)
    iteration = [0]

    def obj_grad(x):
        θ_A.dat.data[:] = x[:n_dof_local]
        θ_C.dat.data[:] = x[n_dof_local:]

        reset_manager(); start_manager()

        A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
        C = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
        u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A, friction=C)

        # 3D misfit (no depth_average)
        J_misfit = Functional(name="J")
        J_misfit.assign(0.5 / area * inner(u - u_obs_3d, u - u_obs_3d) / σ_u**2 * dx)
        J_val = float(J_misfit)

        dJ_A, dJ_C = compute_gradient(J_misfit, (θ_A, θ_C))
        stop_manager()

        # Regularization (not annotated)
        reg_A = float(assemble(0.5 / area * L_REG_c**2 * inner(grad(θ_A), grad(θ_A)) * dx))
        reg_C = float(assemble(0.5 / area * L_REG_c**2 * inner(grad(θ_C), grad(θ_C)) * dx))

        g_A = dJ_A.dat.data_ro.copy()
        g_C = dJ_C.dat.data_ro.copy()
        g_A += assemble(1.0 / area * L_REG_c**2 * inner(grad(θ_A), grad(TestFunction(Q_lift))) * dx).dat.data_ro
        g_C += assemble(1.0 / area * L_REG_c**2 * inner(grad(θ_C), grad(TestFunction(Q_lift))) * dx).dat.data_ro

        total = J_val + reg_A + reg_C
        iteration[0] += 1
        g_norm = float(np.sqrt(np.sum(g_A**2) + np.sum(g_C**2)))
        print(f"  iter {iteration[0]:3d}: J={J_val:.4e} reg={reg_A+reg_C:.4e} total={total:.4e} "
              f"|g|={g_norm:.4e}", flush=True)

        return total, np.concatenate([g_A, g_C])

    x0 = np.concatenate([θ_A.dat.data_ro.copy(), θ_C.dat.data_ro.copy()])
    print(f"\nStarting L-BFGS-B ({2*n_dof_local} DOFs, max {MAX_ITER} iterations)...", flush=True)

    result = scipy_minimize(
        obj_grad, x0, method="L-BFGS-B", jac=True,
        options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": 1e-8, "disp": True},
    )
    print(f"Result: {result.message}", flush=True)

    # Save
    θ_A.dat.data[:] = result.x[:n_dof_local]
    θ_C.dat.data[:] = result.x[n_dof_local:]

    θ_A_2d = Function(Q_2d, name="log_fluidity")
    θ_C_2d = Function(Q_2d, name="log_friction")
    θ_A_2d.dat.data[:] = θ_A.dat.data_ro
    θ_C_2d.dat.data[:] = θ_C.dat.data_ro

    print(f"θ_A: [{θ_A_2d.dat.data.min():.2f}, {θ_A_2d.dat.data.max():.2f}]", flush=True)
    print(f"θ_C: [{θ_C_2d.dat.data.min():.2f}, {θ_C_2d.dat.data.max():.2f}]", flush=True)

    # Final forward solve
    A_opt = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_opt = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    u_opt = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A_opt, friction=C_opt)
    u_avg = icepack.depth_average(u_opt)
    speed_opt = Function(Q_2d, name="speed_model").interpolate(sqrt(inner(u_avg, u_avg)))
    speed_obs = Function(Q_2d, name="speed_obs").interpolate(sqrt(inner(u_obs_2d, u_obs_2d)))

    with CheckpointFile(str(OUTPUT_FILE), "w") as chk:
        chk.save_mesh(mesh_2d)
        chk.save_function(θ_A_2d, name="log_fluidity")
        chk.save_function(θ_C_2d, name="log_friction")
        chk.save_function(speed_opt, name="speed_model")
        chk.save_function(speed_obs, name="speed_obs")
        chk.save_function(u_avg, name="velocity_avg")
    print(f"Saved {OUTPUT_FILE}", flush=True)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, f, title, cmap, vmin, vmax in [
        (axes[0,0], speed_obs, "Observed", "inferno", 0, 3000),
        (axes[0,1], speed_opt, "Model", "inferno", 0, 3000),
        (axes[1,0], θ_A_2d, "θ_A", "RdBu_r", -5, 5),
        (axes[1,1], θ_C_2d, "θ_C", "RdBu_r", -5, 5),
    ]:
        tc = firedrake.tripcolor(f, axes=ax, cmap=cmap, vmin=vmin, vmax=vmax)
        d = make_axes_locatable(ax); cax = d.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(tc, cax=cax)
        ax.set_title(title); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(DATA_DIR / "figures" / "inversion_hybrid_3d.png"), dpi=200)
    print("Saved figures/inversion_hybrid_3d.png", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
