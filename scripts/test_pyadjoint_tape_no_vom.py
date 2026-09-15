r"""Does pyadjoint TLM/adjoint walk through HybridModel.diagnostic_solve alone?

This is the gating question for porting UQ_forward's `sparse_hessian.py`
(which bypasses the broken VOM-interpolate-in-tape problem by keeping VOM
outside the tape and using pyadjoint TLM/adj through the solve only).

We tape only:
    θ_A, θ_C → interpolate → diagnostic_solve → u

then test TLM (∂u/∂θ · v) by:
  (a) finite-difference of the taped forward
  (b) pyadjoint tape.evaluate_tlm()

Agreement to ~EPS_FD means pyadjoint TLM works through HybridModel solve
in isolation. Disagreement (or ~0) means even this stripped path fails.
"""
import argparse
import numpy as np
import netCDF4 as nc
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace,
    CheckpointFile, ExtrudedMesh,
    inner, max_value, exp, dx, conditional, assemble,
)
from firedrake.adjoint import (
    continue_annotation, pause_annotation,
    get_working_tape, set_working_tape, Tape,
)
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
)
import icepack
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
MAP_FILE = DATA_DIR / "mesh" / "inversion_hires_apres_vd1.h5"

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 50, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
EPS_FD = 1e-4


def friction(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def main():
    # ── Setup ──
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")

    mesh = ExtrudedMesh(mesh_2d, layers=1)
    Q_lift = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
    V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
    V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)
    h = icepack.utilities.lift3d(h_2d, Q_lift)
    s = icepack.utilities.lift3d(s_2d, Q_lift)
    u0 = Function(V).project(icepack.utilities.lift3d(u_obs_2d, V_lift))

    with CheckpointFile(str(MAP_FILE), "r") as chk:
        m_inv = chk.load_mesh()
        θA_loaded = chk.load_function(m_inv, "log_fluidity")
        θC_loaded = chk.load_function(m_inv, "log_friction")

    θ_A = Function(Q_lift, name="log_fluidity")
    θ_C = Function(Q_lift, name="log_friction")
    θ_A.dat.data[:] = θA_loaded.dat.data_ro
    θ_C.dat.data[:] = θC_loaded.dat.data_ro
    θA_map_vals = θ_A.dat.data_ro.copy()
    θC_map_vals = θ_C.dat.data_ro.copy()

    A_0, C_0 = Constant(20.0), Constant(0.01)
    grounded_mask = Function(Q_lift).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01,
                    Constant(1.0), Constant(0.0)))

    model = icepack.models.HybridModel(friction=friction)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc",
        diagnostic_solver_parameters=SOLVER_PARAMS)

    # Prime the solver state outside the tape
    A_pr = Function(Q_lift).interpolate(A_0 * exp(θ_A))
    C_pr = Function(Q_lift).interpolate(C_0 * exp(θ_C * grounded_mask))
    u_primed = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s,
                                        fluidity=A_pr, friction=C_pr)
    print("Primed.", flush=True)

    # ── Tape the forward θ → u ──
    n = θ_A.dat.data.shape[0]

    def forward(θA_in, θC_in):
        A = Function(Q_lift).interpolate(A_0 * exp(θA_in))
        C = Function(Q_lift).interpolate(C_0 * exp(θC_in * grounded_mask))
        u = solver.diagnostic_solve(velocity=u_primed, thickness=h, surface=s,
                                     fluidity=A, friction=C)
        return u

    set_working_tape(Tape())
    continue_annotation()
    θA_taped = Function(Q_lift); θA_taped.assign(θ_A)
    θC_taped = Function(Q_lift); θC_taped.assign(θ_C)
    u_at_map = forward(θA_taped, θC_taped)
    pause_annotation()
    tape = get_working_tape()
    print(f"Tape blocks: {len(tape.get_blocks())}", flush=True)

    rng = np.random.default_rng(0)
    vA_arr = rng.standard_normal(n)
    vC_arr = rng.standard_normal(n)

    # ── FD δu = u(θ+εv) - u(θ) all divided by ε ──
    print("\n[FD] computing u(θ_map + ε·v)...", flush=True)
    θA_p = Function(Q_lift); θA_p.dat.data[:] = θA_map_vals + EPS_FD * vA_arr
    θC_p = Function(Q_lift); θC_p.dat.data[:] = θC_map_vals + EPS_FD * vC_arr
    # Re-run forward at perturbed θ (annotation off)
    A_p = Function(Q_lift).interpolate(A_0 * exp(θA_p))
    C_p = Function(Q_lift).interpolate(C_0 * exp(θC_p * grounded_mask))
    u_p = solver.diagnostic_solve(velocity=u_primed, thickness=h, surface=s,
                                   fluidity=A_p, friction=C_p)
    δu_fd = (u_p.dat.data_ro - u_at_map.dat.data_ro) / EPS_FD
    print(f"     ||δu_FD|| = {np.linalg.norm(δu_fd):.4e}", flush=True)

    # ── TLM via pyadjoint tape ──
    print("\n[TLM] evaluate_tlm on the same tape...", flush=True)
    tape.reset_tlm_values()
    vA_f = Function(Q_lift); vA_f.dat.data[:] = vA_arr
    vC_f = Function(Q_lift); vC_f.dat.data[:] = vC_arr
    θA_taped.block_variable.tlm_value = vA_f
    θC_taped.block_variable.tlm_value = vC_f
    tape.evaluate_tlm()
    δu_tlm_var = u_at_map.block_variable.tlm_value
    if δu_tlm_var is None:
        print("     ||δu_TLM|| = (NONE — tape returned no TLM value)", flush=True)
        δu_tlm = np.zeros_like(δu_fd)
    else:
        δu_tlm = δu_tlm_var.dat.data_ro
        print(f"     ||δu_TLM|| = {np.linalg.norm(δu_tlm):.4e}", flush=True)

    # ── Compare ──
    fd_norm = np.linalg.norm(δu_fd)
    tlm_norm = np.linalg.norm(δu_tlm)
    if fd_norm > 0:
        rel = np.linalg.norm(δu_tlm - δu_fd) / fd_norm
    else:
        rel = float("inf")
    print(f"\n||δu_FD||  = {fd_norm:.4e}")
    print(f"||δu_TLM|| = {tlm_norm:.4e}")
    print(f"Relative diff (TLM - FD) / ||FD|| = {rel:.3e}")
    print(f"Expected (FD truncation): ~{EPS_FD:.0e}")
    if rel < 1e-2:
        print("RESULT: ✓ pyadjoint TLM works through HybridModel.diagnostic_solve")
        print("        → sparse_hessian.py port is viable")
    elif tlm_norm < 1e-6 * fd_norm:
        print("RESULT: ✗ pyadjoint TLM returns ≈ 0 — HybridModel solve TLM is broken")
    else:
        print(f"RESULT: ? TLM is nonzero but disagrees with FD ({rel:.2e})")


if __name__ == "__main__":
    main()
