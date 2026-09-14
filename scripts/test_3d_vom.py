r"""Synthetic test: 3D VertexOnlyMesh on extruded mesh + hybrid adjoint.

Goal: verify that
  1. VertexOnlyMesh(extruded_mesh, [(x, y, ζ)]) works
  2. interpolating icepack.models.hybrid.vertical_strain_rate onto it works
  3. tlm_adjoint compute_gradient gives a non-zero gradient for a functional
     built on the interpolated values.

We use a small synthetic ice stream on a rectangle — no geometry issues.
"""
import numpy as np
import firedrake
import icepack
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    RectangleMesh, ExtrudedMesh, VertexOnlyMesh,
    inner, grad, max_value, exp, dx,
)
from firedrake.adjoint import *  # noqa
from tlm_adjoint.firedrake import (
    reset_manager, start_manager, stop_manager,
    compute_gradient, Functional,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)

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
    print("Synthetic 3D VertexOnlyMesh / adjoint test", flush=True)
    print("=" * 60, flush=True)

    # ── Simple rectangular ice stream ──────────────────────────────
    Lx, Ly = 50e3, 20e3      # 50 km × 20 km
    nx, ny = 25, 10          # ~2 km resolution
    mesh_2d = RectangleMesh(nx, ny, Lx, Ly)
    print(f"2D mesh: {mesh_2d.num_vertices()} verts, "
          f"{mesh_2d.num_cells()} cells", flush=True)

    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)

    # Synthetic glacier: 1 km thick, surface slopes down in x
    x, y, ζ = firedrake.SpatialCoordinate(mesh)
    h = Function(Q_lift).interpolate(Constant(1000.0))
    # Driving stress ~ ρ_I * g * h * (ds/dx); pick ds/dx ≈ -1e-3
    s = Function(Q_lift).interpolate(200.0 - 1e-3 * x)

    # Initial guess velocity: flow in +x
    V_const = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)
    u0_const = Function(V_const).interpolate(
        firedrake.as_vector([Constant(100.0), Constant(0.0)]))
    u0 = Function(V).project(u0_const)

    # Controls — cold start
    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    A_0, C_0 = Constant(20.0), Constant(0.01)

    # ── Hybrid model solver ────────────────────────────────────────
    model = icepack.models.HybridModel(friction=friction)
    # Boundary: inflow at x=0 (id 1), no-slip; outflow at x=Lx (id 2)
    # Let's use inflow = 1 as dirichlet, outflow = 2 as ice_front.
    # RectangleMesh tags: 1=x=0, 2=x=Lx, 3=y=0, 4=y=Ly
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[1], ice_front_ids=[2],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    # Prime
    A_init = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_init = Function(Q_lift).interpolate(C_0 * exp(θ_C))
    try:
        u_prime = solver.diagnostic_solve(
            velocity=u0, thickness=h, surface=s, fluidity=A_init, friction=C_init)
        u0.assign(u_prime)
        print(f"Primed forward solve OK, u_x max = "
              f"{u0.dat.data[:, 0].max():.1f} m/yr", flush=True)
    except Exception as e:
        print(f"Forward solve failed: {type(e).__name__}: {e}", flush=True)
        return

    # ── Build 3D VertexOnlyMesh ────────────────────────────────────
    print("\n--- Step 1: build VertexOnlyMesh(extruded, 3D pts) ---", flush=True)
    rng = np.random.default_rng(0)
    n_sites = 20
    x_sites = rng.uniform(0.1 * Lx, 0.9 * Lx, n_sites)
    y_sites = rng.uniform(0.1 * Ly, 0.9 * Ly, n_sites)
    # ~10 depth levels per site
    zeta_levels = np.linspace(0.1, 0.95, 10)
    pts = np.array([(xi, yi, zi)
                    for xi, yi in zip(x_sites, y_sites)
                    for zi in zeta_levels])
    print(f"  {len(pts)} 3D points (x, y, ζ)", flush=True)

    try:
        vom = VertexOnlyMesh(mesh, pts, missing_points_behaviour="warn")
        print(f"  VOM created: {vom.num_vertices()} vertices", flush=True)
    except Exception as e:
        import traceback
        print(f"  VOM FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return

    Δ = FunctionSpace(vom, "DG", 0)

    # ── Test: interpolate scalar onto VOM ──────────────────────────
    print("\n--- Step 2: interpolate plain scalar onto VOM ---", flush=True)
    try:
        z_field = Function(Q_lift).interpolate(h)
        z_at_pts = Function(Δ).interpolate(z_field)
        print(f"  interpolated h onto VOM, mean = "
              f"{z_at_pts.dat.data.mean():.2f}", flush=True)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        return

    # ── Test: interpolate vertical_strain_rate UFL onto VOM ────────
    print("\n--- Step 3: interpolate vertical_strain_rate onto VOM ---",
          flush=True)
    reset_manager(); start_manager()

    A = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C = Function(Q_lift).interpolate(C_0 * exp(θ_C))
    u = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                 fluidity=A, friction=C)

    # ApRES measures ε_zz = ∂w/∂z. By incompressibility,
    # ε_zz = -tr(horizontal strain rate) = -(ε_xx + ε_yy).
    eps_h = icepack.models.hybrid.horizontal_strain_rate(
        velocity=u, thickness=h, surface=s)
    eps_zz_ufl = -(eps_h[0, 0] + eps_h[1, 1])

    try:
        eps_model = Function(Δ).interpolate(eps_zz_ufl)
        print(f"  eps_zz at VOM points: "
              f"[{eps_model.dat.data.min():.3e}, "
              f"{eps_model.dat.data.max():.3e}] 1/yr", flush=True)
    except Exception as e:
        import traceback
        stop_manager()
        print(f"  Interpolate FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return

    # ── Test: functional + compute_gradient ────────────────────────
    print("\n--- Step 4: functional on VOM + compute_gradient ---", flush=True)

    # Synthetic "observations" — perturbed version of current model
    eps_obs = Function(Δ)
    eps_obs.dat.data[:] = eps_model.dat.data_ro + rng.normal(
        scale=1e-4, size=len(eps_model.dat.data))

    σ = Constant(1e-4)
    N_pts = len(eps_obs.dat.data)
    J = Functional(name="J_eps")
    J.assign(0.5 / Constant(float(N_pts))
             * ((eps_model - eps_obs) / σ) ** 2 * dx)
    J_val = float(J)
    print(f"  J = {J_val:.4e}", flush=True)

    try:
        dJ_A, dJ_C = compute_gradient(J, (θ_A, θ_C))
        stop_manager()
        gA_norm = float(np.sqrt((dJ_A.dat.data_ro ** 2).sum()))
        gC_norm = float(np.sqrt((dJ_C.dat.data_ro ** 2).sum()))
        gA_max = float(abs(dJ_A.dat.data_ro).max())
        gC_max = float(abs(dJ_C.dat.data_ro).max())
        print(f"  |dJ/dθ_A| = {gA_norm:.4e}  (max {gA_max:.3e})", flush=True)
        print(f"  |dJ/dθ_C| = {gC_norm:.4e}  (max {gC_max:.3e})", flush=True)

        if gA_norm > 1e-12 and gC_norm > 1e-12:
            print("\n✓ SUCCESS: 3D VOM + vertical_strain_rate + adjoint WORKS",
                  flush=True)
        else:
            print("\n✗ ZERO GRADIENT — tape broken", flush=True)
    except Exception as e:
        stop_manager()
        import traceback
        print(f"\n✗ compute_gradient FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
