r"""L-curve analysis for SSA inversion (manual L-BFGS).

Sweeps the regularization length scale L and records the misfit (E)
and regularization (R) at convergence. Warm-starts each L from the
previous converged result.

Uses the hires mesh + sparse obs (VertexOnlyMesh) for consistency
with the main inversion.

Usage:
    OMP_NUM_THREADS=4 python lcurve_ssa_manual.py
"""
import numpy as np
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    sqrt, inner, grad, max_value, exp, dx, conditional, assemble,
    TestFunction,
)
from firedrake.adjoint import *  # noqa
from tlm_adjoint.firedrake import (
    reset_manager, start_manager, stop_manager,
    compute_gradient, Functional,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
    weertman_sliding_law as m,
)
from scipy.optimize import minimize as scipy_minimize
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"

L_VALUES_KM = [1, 2, 5, 10, 20, 50]
MAX_ITER = 500
SIGMA_VEL = 10.0

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


def main():
    print("=" * 60, flush=True)
    print(f"L-curve: SSA (manual L-BFGS)", flush=True)
    print(f"L values: {L_VALUES_KM} km", flush=True)
    print("=" * 60, flush=True)

    with firedrake.CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        h = chk.load_function(mesh, "thickness")
        s = chk.load_function(mesh, "surface")
        u_obs = chk.load_function(mesh, "velocity")

    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    print(f"Mesh: {mesh.num_vertices()} verts", flush=True)

    A_0, C_0 = Constant(20.0), Constant(0.01)
    area = assemble(Constant(1.0) * dx(mesh))

    grounded_mask = Function(Q).interpolate(
        conditional(
            1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
            Constant(1.0), Constant(0.0)))

    model = icepack.models.IceStream(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    u_init = Function(V).interpolate(u_obs)
    u_init.interpolate(firedrake.conditional(
        sqrt(inner(u_obs, u_obs)) < 1.0,
        firedrake.as_vector([Constant(1.0), Constant(0.0)]), u_obs))

    # Cold start controls
    θ_A = Function(Q, name="log_fluidity")
    θ_C = Function(Q, name="log_friction")

    # Prime velocity
    A_init = Function(Q).interpolate(A_0 * exp(θ_A))
    C_init = Function(Q).interpolate(C_0 * exp(θ_C * grounded_mask))
    u0 = solver.diagnostic_solve(velocity=u_init, thickness=h, surface=s,
                                  fluidity=A_init, friction=C_init)
    print("Primed OK", flush=True)

    n_dof = θ_A.dat.data.shape[0]
    σ_u = Constant(SIGMA_VEL)
    results = []

    for L_km in L_VALUES_KM:
        L_REG = Constant(L_km * 1e3)
        print(f"\n{'=' * 40}", flush=True)
        print(f"L = {L_km} km", flush=True)
        print(f"{'=' * 40}", flush=True)

        iteration = [0]

        def obj_grad(x):
            θ_A.dat.data[:] = x[:n_dof]
            θ_C.dat.data[:] = x[n_dof:]

            reset_manager(); start_manager()
            A = Function(Q).interpolate(A_0 * exp(θ_A))
            C = Function(Q).interpolate(C_0 * exp(θ_C * grounded_mask))
            try:
                u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                             fluidity=A, friction=C)
            except firedrake.exceptions.ConvergenceError:
                stop_manager()
                iteration[0] += 1
                return 1e20, np.zeros_like(x)

            J = Functional(name="J")
            J.assign(0.5 / area * inner(u - u_obs, u - u_obs) / σ_u**2 * dx)
            J_val = float(J)

            try:
                dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
            except Exception:
                stop_manager()
                iteration[0] += 1
                return 1e20, np.zeros_like(x)
            stop_manager()

            reg_A = float(assemble(0.5 / area * L_REG**2 * inner(grad(θ_A), grad(θ_A)) * dx))
            reg_C = float(assemble(0.5 / area * L_REG**2 * inner(grad(θ_C), grad(θ_C)) * dx))

            g_A = dJ_A.dat.data_ro.copy()
            g_C = dJ_C.dat.data_ro.copy()
            g_A += assemble(1.0 / area * L_REG**2 * inner(grad(θ_A), grad(TestFunction(Q))) * dx).dat.data_ro
            g_C += assemble(1.0 / area * L_REG**2 * inner(grad(θ_C), grad(TestFunction(Q))) * dx).dat.data_ro

            total = J_val + reg_A + reg_C
            iteration[0] += 1
            if iteration[0] % 10 == 0:
                print(f"  iter {iteration[0]:3d}: E={J_val:.3e} R={reg_A+reg_C:.3e}", flush=True)

            u0.assign(u)
            return total, np.concatenate([g_A, g_C])

        x0 = np.concatenate([θ_A.dat.data_ro.copy(), θ_C.dat.data_ro.copy()])
        result = scipy_minimize(
            obj_grad, x0, method="L-BFGS-B", jac=True,
            options={"maxiter": MAX_ITER, "ftol": 0, "gtol": 0, "pgtol": 0},
        )

        # Evaluate final E and R separately
        θ_A.dat.data[:] = result.x[:n_dof]
        θ_C.dat.data[:] = result.x[n_dof:]
        A_f = Function(Q).interpolate(A_0 * exp(θ_A))
        C_f = Function(Q).interpolate(C_0 * exp(θ_C * grounded_mask))
        u_f = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                       fluidity=A_f, friction=C_f)
        u0.assign(u_f)

        E = float(assemble(0.5 / area * inner(u_f - u_obs, u_f - u_obs) / σ_u**2 * dx))
        R_A = float(assemble(0.5 / area * L_REG**2 * inner(grad(θ_A), grad(θ_A)) * dx))
        R_C = float(assemble(0.5 / area * L_REG**2 * inner(grad(θ_C), grad(θ_C)) * dx))
        R = R_A + R_C

        print(f"  CONVERGED: E={E:.4e}, R={R:.4e} ({iteration[0]} iters)", flush=True)
        results.append({"L_km": L_km, "E": E, "R": R, "R_A": R_A, "R_C": R_C,
                         "iters": iteration[0]})

        # Save per-L checkpoint
        with firedrake.CheckpointFile(
                str(DATA_DIR / "mesh" / f"lcurve_ssa_L{L_km}km.h5"), "w") as chk:
            chk.save_mesh(mesh)
            chk.save_function(Function(Q, name="log_fluidity").assign(θ_A),
                              name="log_fluidity")
            chk.save_function(Function(Q, name="log_friction").assign(θ_C),
                              name="log_friction")

        # Incremental JSON save
        out_inc = DATA_DIR / "data" / "lcurve_ssa_manual.json"
        out_inc.parent.mkdir(exist_ok=True)
        with open(str(out_inc), "w") as f:
            json.dump(results, f, indent=2)

    # Save
    out = DATA_DIR / "data" / "lcurve_ssa_manual.json"
    out.parent.mkdir(exist_ok=True)
    with open(str(out), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}", flush=True)

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Es = [r["E"] for r in results]
    Rs = [r["R"] for r in results]
    Ls = [r["L_km"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Rs, Es, "ko-", ms=8)
    for L, E, R in zip(Ls, Es, Rs):
        ax.annotate(f"L={L}km", (R, E), textcoords="offset points",
                    xytext=(10, 5), fontsize=10)
    ax.set_xlabel("Regularization R"); ax.set_ylabel("Misfit E")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("L-curve: SSA"); ax.grid(True, alpha=0.3)
    fig.savefig(str(DATA_DIR / "figures" / "lcurve_ssa_manual.png"), dpi=200, bbox_inches="tight")
    print("Saved figures/lcurve_ssa_manual.png", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
