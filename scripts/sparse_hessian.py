r"""Manual Gauss-Newton Hessian-vector products for sparse-data UQ on Thwaites.

Adapted from UQ_forward/ismip-c/scripts/sparse_hessian.py for our setup:
  - Dual controls (θ_A, θ_C) on Q_lift (CG1 horiz × DG0 vert on extruded mesh)
  - HybridModel.diagnostic_solve (3D extruded velocity)
  - Two misfit terms: surface velocity at 2D points, ApRES strain rate at 3D points

The trick: keep VertexOnlyMesh.interpolate OUTSIDE the pyadjoint tape (it lacks
TLM annotation). Tape only θ → u via diagnostic_solve, then apply the sparse
weighting (point eval → diag(w/σ²) → point eval^T) as linear algebra outside
the tape using Interpolator(..., adjoint=True).

For ApRES, the misfit is on eps_zz(u) not u, so we use UFL with TestFunction(V)
to encode the linear operator u → eps_zz at apres points, then adjoint-interp
gives the adjoint of that chained operator.
"""
import numpy as np
import firedrake as fd
from firedrake import (
    Function, Cofunction, FunctionSpace, VectorFunctionSpace,
    TestFunction, Interpolate, Interpolator, assemble, dx,
    VertexOnlyMesh,
)
from firedrake.adjoint import (
    continue_annotation, pause_annotation,
    get_working_tape, set_working_tape, Tape,
)
import icepack


class TapedForward:
    r"""Pyadjoint tape with θ_A, θ_C → u(θ) annotated as a single sequence.

    Wraps a simulation closure and exposes tlm()/adj() to walk the tape.
    Construction does one forward solve at (θ_A_map, θ_C_map); subsequent
    Hv calls do not re-solve (they perturb on the tape).
    """

    def __init__(self, simulation, θ_A_value, θ_C_value):
        set_working_tape(Tape())
        continue_annotation()
        self.θ_A = Function(θ_A_value.function_space())
        self.θ_A.assign(θ_A_value)
        self.θ_C = Function(θ_C_value.function_space())
        self.θ_C.assign(θ_C_value)
        self.u = simulation(self.θ_A, self.θ_C)
        pause_annotation()
        self.tape = get_working_tape()

    def tlm(self, v_A, v_C):
        r"""Apply (∂u/∂θ) · [v_A, v_C]. Returns δu ∈ V."""
        self.tape.reset_tlm_values()
        self.θ_A.block_variable.tlm_value = v_A
        self.θ_C.block_variable.tlm_value = v_C
        self.tape.evaluate_tlm()
        du = self.u.block_variable.tlm_value
        out = Function(self.u.function_space())
        if du is not None:
            out.assign(du)
        return out

    def adj(self, r_cofunc):
        r"""Apply (∂u/∂θ)^T · r. r is a Cofunction on V.dual().

        Returns (adj_A, adj_C) — Cofunctions on Q_lift.dual().
        """
        for blk in self.tape.get_blocks():
            for bv in blk.get_outputs():
                bv.adj_value = None
            for bv in blk.get_dependencies():
                bv.adj_value = None
        self.u.block_variable.adj_value = r_cofunc
        self.tape.evaluate_adj(markings=False)
        return (self.θ_A.block_variable.adj_value,
                self.θ_C.block_variable.adj_value)


def make_Hv_gn_velocity(taped, vel_points, sigma_x_arr, sigma_y_arr, N_vel=None):
    r"""GN Hessian-vector product for surface-velocity misfit.

    Recinos convention (no 1/N):
    Misfit:  J_vel = 0.5 · Σ_q [(u_x[q] - obs_x[q])² / σ_x[q]²
                              + (u_y[q] - obs_y[q])² / σ_y[q]²]

    H_GN · v = (∂u/∂θ)^T · D · (∂u/∂θ) · v  where D = diag(1/σ²)

    sigma_x_arr, sigma_y_arr are in *input-point order* (matching vel_points
    rows). Internally we reorder them to match VOM-internal ordering via the
    input_ordering FunctionSpace, the same way obs_to_vom does in
    run_eigendec.py. N_vel arg kept for backwards-compatible API, unused.
    """
    V = taped.u.function_space()
    mesh = V.mesh()
    vom_vel = VertexOnlyMesh(mesh, vel_points, missing_points_behaviour="warn")
    P_space = VectorFunctionSpace(vom_vel, "DG", 0, dim=2)
    P_in = VectorFunctionSpace(vom_vel.input_ordering, "DG", 0, dim=2)

    # Sigma in input order → Function on P_in → interpolate to VOM-internal P_space
    sig_in = Function(P_in)
    n_in = len(sig_in.dat.data)
    sig_in.dat.data[:, 0] = sigma_x_arr[:n_in]
    sig_in.dat.data[:, 1] = sigma_y_arr[:n_in]
    sig_internal = Function(P_space).interpolate(sig_in)
    scale_arr = 1.0 / sig_internal.dat.data_ro ** 2

    interp = Interpolator(TestFunction(V), P_space)

    def Hv(v_A, v_C):
        δu = taped.tlm(v_A, v_C)
        δu_at_pts = assemble(Interpolate(δu, P_space))
        scaled = Function(P_space)
        scaled.dat.data[:] = δu_at_pts.dat.data_ro * scale_arr
        r_primal = interp._interpolate(scaled, adjoint=True)
        r_cofunc = Cofunction(V.dual())
        r_cofunc.dat.data[:] = r_primal.dat.data_ro
        adj_A, adj_C = taped.adj(r_cofunc)
        # Return as Functions with cotangent DOFs (to match run_eigendec.py's
        # FD path, which stays in cotangent space — its mass_inv is a no-op).
        out_A = Function(adj_A.function_space().dual())
        out_C = Function(adj_C.function_space().dual())
        out_A.dat.data[:] = adj_A.dat.data_ro
        out_C.dat.data[:] = adj_C.dat.data_ro
        return out_A, out_C
    return Hv


def make_Hv_gn_strainrate(taped, apres_points, sigma_eps_arr, N_apr=None, h=None, s=None):
    r"""GN Hessian-vector product for ApRES vertical-strain-rate misfit.

    eps_zz(u) = -(∂u_x/∂x + ∂u_y/∂y) (the horizontal divergence, neg)
    Recinos convention (no 1/N):
    Misfit:  J_apr = 0.5 · Σ_q [(eps_zz(u)[q] - eps_obs[q])² / σ_eps[q]²]

    The linear operator u → eps_zz at apres points is encoded via
    Interpolator(eps_test, P_space) where eps_test uses TestFunction(V).
    Its adjoint then gives the adjoint of point-eval-of-divergence.

    Parameters
    ----------
    h, s : firedrake.Function on Q_lift, the fixed thickness and surface.
    N_apr : unused, kept for backwards-compatible API.
    """
    V = taped.u.function_space()
    mesh = V.mesh()
    vom_apr = VertexOnlyMesh(mesh, apres_points, missing_points_behaviour="warn")
    P_space = FunctionSpace(vom_apr, "DG", 0)
    P_in = FunctionSpace(vom_apr.input_ordering, "DG", 0)

    # Sigma in input order → VOM-internal order (matches Interpolate(eps, P_space) output)
    sig_in = Function(P_in)
    n_in = len(sig_in.dat.data)
    sig_in.dat.data[:] = sigma_eps_arr[:n_in]
    sig_internal = Function(P_space).interpolate(sig_in)
    scale_arr = 1.0 / sig_internal.dat.data_ro ** 2

    # UFL expression for eps_zz as a function of a velocity test function
    v_test = TestFunction(V)
    eps_h_test = icepack.models.hybrid.horizontal_strain_rate(
        velocity=v_test, thickness=h, surface=s)
    eps_test = -(eps_h_test[0, 0] + eps_h_test[1, 1])
    interp = Interpolator(eps_test, P_space)

    def Hv(v_A, v_C):
        δu = taped.tlm(v_A, v_C)
        eps_h_δu = icepack.models.hybrid.horizontal_strain_rate(
            velocity=δu, thickness=h, surface=s)
        deps_form = -(eps_h_δu[0, 0] + eps_h_δu[1, 1])
        deps_at_pts = assemble(Interpolate(deps_form, P_space))
        scaled = Function(P_space)
        scaled.dat.data[:] = deps_at_pts.dat.data_ro * scale_arr
        r_primal = interp._interpolate(scaled, adjoint=True)
        r_cofunc = Cofunction(V.dual())
        r_cofunc.dat.data[:] = r_primal.dat.data_ro
        adj_A, adj_C = taped.adj(r_cofunc)
        # Return as Functions with cotangent DOFs (to match run_eigendec.py's
        # FD path, which stays in cotangent space — its mass_inv is a no-op).
        out_A = Function(adj_A.function_space().dual())
        out_C = Function(adj_C.function_space().dual())
        out_A.dat.data[:] = adj_A.dat.data_ro
        out_C.dat.data[:] = adj_C.dat.data_ro
        return out_A, out_C
    return Hv


def combine_Hv(*Hv_funcs):
    r"""Sum of multiple Hv operators returning (Hv_A, Hv_C) tuples."""
    def Hv_combined(v_A, v_C):
        out_A = None
        out_C = None
        for Hv in Hv_funcs:
            hA, hC = Hv(v_A, v_C)
            if out_A is None:
                out_A = hA.copy(deepcopy=True)
                out_C = hC.copy(deepcopy=True)
            else:
                out_A.dat.data[:] += hA.dat.data_ro
                out_C.dat.data[:] += hC.dat.data_ro
        return out_A, out_C
    return Hv_combined
