r"""Shared problem setup for the Thwaites SSA / hybrid configurations.

``build_problem(model, friction, apres)`` returns the mesh, control space, the
physical prior means (lifted for the hybrid), the observations on
VertexOnlyMeshes and a ``forward(theta_A, theta_C) -> u`` closure, so the MAP
inversion (invert.py), the Hessian eigendecomposition and the prognostic
forwards all evaluate exactly the same model:

    A = A_prior exp(theta_A),    C = C_w0 exp(theta_C * grounded).
"""
import icepack
import numpy as np
from firedrake import (
    CheckpointFile, Constant, ExtrudedMesh, Function, FunctionSpace,
    VectorFunctionSpace, VertexOnlyMesh, conditional, dx, exp,
)
from tlm_adjoint.firedrake import Functional

from friction_laws import FRICTION_LAWS, flotation_factor, weertman
from observations import load_apres, load_velocity
from physical_prior import MESH_FILE, load_physical_prior

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 250, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


def tag(model, friction, apres):
    return f"{model}_{friction}" + ("_noapres" if model == "hybrid" and not apres else "")


def build_problem(model, friction, apres):
    r"""Mesh, spaces, prior means, data and a ``forward(theta_A, theta_C) -> u``."""
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")
    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    C_w0_2d, A_prior_2d = load_physical_prior(mesh_2d)
    xs, ys, vx, vy, sx, sy = load_velocity(mesh_2d)

    if model == "ssa":
        mesh, Q = mesh_2d, Q_2d
        V = VectorFunctionSpace(mesh, "CG", 1)
        h, s, C_w0, A_prior = h_2d, s_2d, C_w0_2d, A_prior_2d
        u0 = Function(V).assign(u_obs_2d)
        vel_pts = np.column_stack([xs, ys])
        model_cls = icepack.models.IceStream
    else:
        mesh = ExtrudedMesh(mesh_2d, layers=1)
        Q = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
        V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
        V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)
        lift = icepack.utilities.lift3d
        h, s, C_w0, A_prior = (lift(f, Q) for f in (h_2d, s_2d, C_w0_2d, A_prior_2d))
        u0 = Function(V).project(lift(u_obs_2d, V_lift))
        vel_pts = np.column_stack([xs, ys, np.ones_like(xs)])
        model_cls = icepack.models.HybridModel
    assert Q.dim() == Q_2d.dim(), "controls must share the 2D CG1 dof ordering"

    def to_vom(space_in, space, vals):
        f_in = Function(space_in)
        f_in.dat.data[:] = vals[:len(f_in.dat.data)]
        return Function(space).interpolate(f_in)

    vom = VertexOnlyMesh(mesh, vel_pts, missing_points_behaviour="warn")
    Δv, Δv_in = FunctionSpace(vom, "DG", 0), FunctionSpace(vom.input_ordering, "DG", 0)
    obs = {"u": to_vom(Δv_in, Δv, vx), "v": to_vom(Δv_in, Δv, vy),
           "σx": to_vom(Δv_in, Δv, sx), "σy": to_vom(Δv_in, Δv, sy), "Δv": Δv}
    if apres:
        pts, eps, sig = load_apres(mesh_2d)
        vom_a = VertexOnlyMesh(mesh, pts, missing_points_behaviour="warn")
        Δa, Δa_in = FunctionSpace(vom_a, "DG", 0), FunctionSpace(vom_a.input_ordering, "DG", 0)
        obs.update(eps=to_vom(Δa_in, Δa, eps), σe=to_vom(Δa_in, Δa, sig), Δa=Δa)

    grounded = Function(Q).interpolate(
        conditional(flotation_factor(h, s) > 0.01, Constant(1.0), Constant(0.0)))

    def make_solver(friction):
        return icepack.solvers.FlowSolver(
            model_cls(friction=friction), dirichlet_ids=[2], ice_front_ids=[1],
            diagnostic_solver_type="petsc", diagnostic_solver_parameters=SOLVER_PARAMS)

    solver = make_solver(FRICTION_LAWS[friction])

    def fields(θ_A, θ_C):
        A = Function(Q).interpolate(A_prior * exp(θ_A))
        C = Function(Q).interpolate(C_w0 * exp(θ_C * grounded))
        return A, C

    def forward(θ_A, θ_C, slv=solver):
        A, C = fields(θ_A, θ_C)
        return slv.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A, friction=C)

    return dict(mesh_2d=mesh_2d, Q_2d=Q_2d, Q=Q, h=h, s=s, u0=u0, u_obs_2d=u_obs_2d,
                C_w0_2d=C_w0_2d, A_prior_2d=A_prior_2d, obs=obs, forward=forward,
                weertman_solver=make_solver(weertman), fields=fields)


def misfit(u, P, apres):
    r"""Taped misfit Functional and its (velocity, ApRES) parts."""
    o = P["obs"]
    ui = Function(o["Δv"]).interpolate(u[0])
    vi = Function(o["Δv"]).interpolate(u[1])
    J = Functional(name="J")
    J.assign(0.5 * (((ui - o["u"]) / o["σx"]) ** 2 + ((vi - o["v"]) / o["σy"]) ** 2) * dx)
    J_vel = float(J)
    J_apr = 0.0
    if apres:
        ε = icepack.models.hybrid.horizontal_strain_rate(velocity=u, thickness=P["h"], surface=P["s"])
        εi = Function(o["Δa"]).interpolate(-(ε[0, 0] + ε[1, 1]))
        J.addto(0.5 * ((εi - o["eps"]) / o["σe"]) ** 2 * dx)
        J_apr = float(J) - J_vel
    return J, J_vel, J_apr
