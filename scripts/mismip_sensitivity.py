r"""Identifiability check: does the TRUE fluidity structure move the observables
above the noise floor? Independent of the inversion. If θ_A's signature in surface
velocity (σv) and vertical velocity w (σw) is below noise, NO inversion can recover
it — the regime, not the method, is the limit.

Solves the Hybrid forward with θ_A=truth vs θ_A=0 (friction fixed to truth in both),
and reports the resulting Δu_surf and Δw against the obs noise.
"""
import numpy as np
import firedrake
import icepack
import icepack.models.friction
import icepack.utilities
from firedrake import (Constant, Function, FunctionSpace, VectorFunctionSpace,
                       CheckpointFile, interpolate, project, exp, max_value, sqrt, inner, tripcolor)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
from pathlib import Path
from mismip_reference import true_fluidity

DATA = Path(__file__).resolve().parent.parent
A0, C0 = Constant(20.0), Constant(0.01)
VDEGREE = 4
SOLVER = {"snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_max_it": 150,
          "snes_rtol": 1e-6, "ksp_type": "preonly", "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}
SIGV, SIGW = 5.0, 5e-4


def friction_weertman(**kw):
    u, h, s, C = kw["velocity"], kw["thickness"], kw["surface"], kw["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


def main():
    with CheckpointFile(str(DATA / "mesh" / "mismip_truth_3d.h5"), "r") as c:
        m3 = c.load_mesh("firedrake_default_extruded")
        u3 = c.load_function(m3, "velocity_3d")
        h3 = c.load_function(m3, "thickness_3d")
        s3 = c.load_function(m3, "surface_3d")
        C3 = c.load_function(m3, "friction_3d")
    Qc = FunctionSpace(m3, "CG", 1, vfamily="R", vdegree=0)
    Qw = FunctionSpace(m3, "CG", 1, vfamily="GLL", vdegree=VDEGREE)
    Q2c = FunctionSpace(m3._base_mesh, "CG", 1)
    θA_true = icepack.utilities.lift3d(Function(Q2c).interpolate(true_fluidity(Q2c)), Qc)
    print(f"θ_A truth range on control: [{θA_true.dat.data_ro.min():.2f}, {θA_true.dat.data_ro.max():.2f}]", flush=True)

    model = icepack.models.HybridModel(friction=friction_weertman)
    solver = icepack.solvers.FlowSolver(model, dirichlet_ids=[1], ice_front_ids=[2], side_wall_ids=[3, 4],
                                        diagnostic_solver_type="petsc", diagnostic_solver_parameters=SOLVER)

    def solve(θA):
        A = interpolate(A0 * exp(θA), Qc)
        u = solver.diagnostic_solve(velocity=u3.copy(deepcopy=True), thickness=h3, surface=s3, fluidity=A, friction=C3)
        w = project(icepack.utilities.vertical_velocity(velocity=u, thickness=h3, basal_mass_balance=Constant(0.0)), Qw)
        return u, w

    u_t, w_t = solve(θA_true)
    u_0, w_0 = solve(Function(Qc))    # θ_A = 0 (uniform fluidity A0)

    uat, ua0 = icepack.depth_average(u_t), icepack.depth_average(u_0)
    Sb = FunctionSpace(uat.function_space().mesh(), "CG", 1)
    du_f = Function(Sb, name="du").interpolate(sqrt(inner(uat - ua0, uat - ua0)))
    spd_f = Function(Sb).interpolate(sqrt(inner(uat, uat)))
    dw2 = icepack.depth_average(Function(w_t.function_space()).interpolate(abs(w_t - w_0)))
    du, spd = du_f.dat.data_ro, spd_f.dat.data_ro
    dw = np.abs(w_t.dat.data_ro - w_0.dat.data_ro)   # per-3D-node, for SNR stats

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6))
    c1 = tripcolor(du_f, axes=a1, cmap="Reds"); plt.colorbar(c1, ax=a1, fraction=0.025).set_label("|Δu| (m/yr)")
    a1.set_title(f"θ_A signature in SURFACE VELOCITY (SNR max {du.max()/SIGV:.0f}) — broad", fontsize=10)
    a1.set_aspect("equal")
    c2 = tripcolor(dw2, axes=a2, cmap="Purples"); plt.colorbar(c2, ax=a2, fraction=0.025).set_label("|Δw| (m/yr)")
    a2.set_title(f"θ_A signature in VERTICAL VELOCITY w (SNR max {dw.max()/SIGW:.0f}) — over ridges", fontsize=10)
    a2.set_aspect("equal")
    fig.suptitle("Where the fluidity signal lives (true θ_A vs uniform)", fontsize=12)
    fig.tight_layout()
    fig.savefig(DATA / "figures" / "mismip_sensitivity.png", dpi=130)
    print(f"Saved {DATA/'figures'/'mismip_sensitivity.png'}", flush=True)
    print("=" * 56, flush=True)
    print(f"θ_A signature in SURFACE VELOCITY (σv={SIGV} m/yr):", flush=True)
    print(f"  |Δu| mean={du.mean():.2f}  max={du.max():.2f} m/yr   → SNR mean={du.mean()/SIGV:.1f}, max={du.max()/SIGV:.1f}", flush=True)
    print(f"  (as % of local speed: mean {100*du.mean()/max(spd.mean(),1):.1f}%)", flush=True)
    print(f"θ_A signature in VERTICAL VELOCITY w (σw={SIGW:.0e} m/yr):", flush=True)
    print(f"  |Δw| mean={dw.mean():.3e}  max={dw.max():.3e} m/yr   → SNR mean={dw.mean()/SIGW:.1f}, max={dw.max()/SIGW:.1f}", flush=True)
    print("=" * 56, flush=True)
    det_u = "DETECTABLE" if du.max() > SIGV else "below noise"
    det_w = "DETECTABLE" if dw.max() > SIGW else "below noise"
    print(f"VERDICT: θ_A signature is {det_u} in velocity, {det_w} in w.", flush=True)


if __name__ == "__main__":
    main()
