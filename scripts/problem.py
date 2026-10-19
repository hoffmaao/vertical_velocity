r"""Shared problem setup for the Thwaites SSA / hybrid configurations.

``build_problem(model, friction, apres)`` returns the mesh, control space, the
physical prior means (lifted for the hybrid), the observations on
VertexOnlyMeshes and a ``forward(theta_A, theta_C) -> u`` closure, so the MAP
inversion (invert.py), the Hessian eigendecomposition and the prognostic
forwards all evaluate exactly the same model:

    A = A_prior exp(theta_A),    C = anchor exp(theta_C * grounded),

with anchor = C_w0 (the Weertman balance coefficient) for the weertman and rc
laws and tau_c0 (the balance yield stress) for the schoof law
(friction_laws.ANCHOR, physical_prior.py); grounded is the smooth Heaviside He.

``A_const_T`` replaces the thermomechanical A_prior with the uniform fluidity
rate_factor(A_const_T), to see what the inversion does under a constant-rheology
assumption; the friction anchor C_w0 is unchanged.
"""
import firedrake.interpolation
import icepack
import numpy as np
from firedrake import (
    CheckpointFile, ExtrudedMesh, Function, FunctionSpace, VectorFunctionSpace,
    VertexOnlyMesh, dx, exp,
)
from tlm_adjoint.firedrake import Functional

from friction_laws import ANCHOR, FRICTION_LAWS, grounded_heaviside, weertman
from observations import load_apres, load_velocity
from physical_prior import MESH_FILE, load_physical_prior


def _cache_vom_extruded_map():
    r"""Cache firedrake's VOM -> extruded-parent node map (memory leak fix).

    Interpolating from an extruded mesh onto a VertexOnlyMesh builds the
    VOM-cell -> parent-node map with ``vom_cell_parent_node_map_extruded``,
    which returns a *new* ``op2.Map`` on every call. The interpolation matrix's
    ``op2.Sparsity`` is cached on the target DataSet under a key that includes
    that map, so every adjoint of a VOM interpolation adds a fresh sparsity
    that is never freed: ~20 MB per L-BFGS iteration for the hybrid ApRES
    inversion (three VOM interpolations per gradient), ~25 GB by iteration
    1400. The non-extruded path (SSA) goes through ``compose_map_and_cache``
    and does not leak. Present in both the 2025 and 2026.4 firedrake.

    The map depends only on (VOM, parent cell-node map), so cache it on the
    parent map exactly as ``compose_map_and_cache`` does.
    """
    fdi = firedrake.interpolation
    build = fdi.vom_cell_parent_node_map_extruded
    if getattr(build, "_vv_cached", False):
        return

    def cached(vertex_only_mesh, extruded_cell_node_map):
        key = (vertex_only_mesh.cell_set, "vom_cell_parent_node_map_extruded")
        cache = extruded_cell_node_map._cache
        if key not in cache:
            cache[key] = build(vertex_only_mesh, extruded_cell_node_map)
        return cache[key]

    cached._vv_cached = True
    fdi.vom_cell_parent_node_map_extruded = cached


_cache_vom_extruded_map()

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 250, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


def tag(model, friction, apres):
    return f"{model}_{friction}" + ("_noapres" if model == "hybrid" and not apres else "")


def build_problem(model, friction, apres, solver_parameters=None,
                  prognostic_parameters=None, prognostic_solver_type=None, A_const_T=None):
    r"""Mesh, spaces, prior means, data and a ``forward(theta_A, theta_C) -> u``.

    ``solver_parameters`` overrides the diagnostic PETSc options (default
    SOLVER_PARAMS, the direct MUMPS solve the inversions run with);
    ``prognostic_parameters`` and ``prognostic_solver_type`` set the
    mass-transport options and scheme, which only the prognostic runs in
    forward.py need. See forward_solvers.py for the presets. ``A_const_T`` (K)
    swaps the fluidity prior mean for the constant rate_factor(A_const_T).
    """
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh_2d = chk.load_mesh()
        h_2d = chk.load_function(mesh_2d, "thickness")
        s_2d = chk.load_function(mesh_2d, "surface")
        u_obs_2d = chk.load_function(mesh_2d, "velocity")
    Q_2d = FunctionSpace(mesh_2d, "CG", 1)
    C_w0_2d, A_prior_2d, τ_c0_2d = load_physical_prior(mesh_2d)
    if ANCHOR[friction] == "tau_c0" and τ_c0_2d is None:
        raise SystemExit("the prior file has no tau_c0: run physical_prior.py --append-schoof")
    # The friction control is theta_C = log(C / anchor): C_w0 for the laws whose
    # coefficient is the Weertman one, tau_c0 (the yield stress) for schoof.
    C_anchor_2d = τ_c0_2d if ANCHOR[friction] == "tau_c0" else C_w0_2d
    if A_const_T is not None:
        A_prior_2d.assign(float(icepack.rate_factor(float(A_const_T))))
    xs, ys, vx, vy, sx, sy = load_velocity(mesh_2d)

    if model == "ssa":
        mesh, Q = mesh_2d, Q_2d
        V = VectorFunctionSpace(mesh, "CG", 1)
        h, s, C_w0, C_anchor, A_prior = h_2d, s_2d, C_w0_2d, C_anchor_2d, A_prior_2d
        u0 = Function(V).assign(u_obs_2d)
        vel_pts = np.column_stack([xs, ys])
        model_cls = icepack.models.IceStream
    else:
        mesh = ExtrudedMesh(mesh_2d, layers=1)
        Q = FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)
        V = VectorFunctionSpace(mesh, "CG", 1, vfamily="GL", vdegree=1, dim=2)
        V_lift = VectorFunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0, dim=2)
        lift = icepack.utilities.lift3d
        h, s, C_w0, C_anchor, A_prior = (lift(f, Q) for f in (h_2d, s_2d, C_w0_2d, C_anchor_2d,
                                                                A_prior_2d))
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

    # Smooth Heaviside gate on the friction control (friction_laws.grounded_heaviside).
    grounded = Function(Q, name="He").interpolate(grounded_heaviside(h, s))

    def make_solver(friction_law=None, diagnostic=None, prognostic=None,
                    transport=None):
        r"""A FlowSolver for this configuration; arguments override the defaults.

        forward.py builds a second solver this way for the trust-region rescue,
        and the inversion uses the Weertman one to prime a stiff initial solve.
        """
        opts = {}
        prog = prognostic if prognostic is not None else prognostic_parameters
        scheme = transport if transport is not None else prognostic_solver_type
        if prog is not None:
            opts["prognostic_solver_parameters"] = prog
        if scheme is not None:
            opts["prognostic_solver_type"] = scheme
        return icepack.solvers.FlowSolver(
            model_cls(friction=friction_law or FRICTION_LAWS[friction]),
            dirichlet_ids=[2], ice_front_ids=[1], diagnostic_solver_type="petsc",
            diagnostic_solver_parameters=(
                diagnostic if diagnostic is not None
                else solver_parameters or SOLVER_PARAMS),
            **opts)

    solver = make_solver()

    weertman_solver = make_solver(weertman)

    def fields(θ_A, θ_C, anchor=C_anchor):
        A = Function(Q).interpolate(A_prior * exp(θ_A))
        C = Function(Q).interpolate(anchor * exp(θ_C * grounded))
        return A, C

    def forward(θ_A, θ_C, slv=solver, h=h, s=s):
        # The Weertman priming solve takes the Weertman coefficient whatever the
        # law being inverted; theta_C = 0 is the balance state under both anchors.
        A, C = fields(θ_A, θ_C, anchor=C_w0 if slv is weertman_solver else C_anchor)
        return slv.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A, friction=C)

    return dict(mesh=mesh, mesh_2d=mesh_2d, Q_2d=Q_2d, Q=Q, h=h, s=s, u0=u0,
                u_obs_2d=u_obs_2d, C_w0_2d=C_w0_2d, C_anchor_2d=C_anchor_2d,
                anchor_name=ANCHOR[friction], A_prior_2d=A_prior_2d, obs=obs,
                forward=forward, solver=solver, make_solver=make_solver,
                weertman_solver=weertman_solver, fields=fields)


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
