r"""Prior-whitened control coordinates for the inversions (fenics_ice metric).

The Recinos prior on each control is R(theta) = 0.5 theta^T A theta with the
precision A = delta M + gamma K. With the sparse Cholesky P A P^T = L L^T we
optimize in whitened coordinates xi,

    theta = P^T L^-T xi,    R = 0.5 xi^T xi,    dJ/dxi = L^-1 P dJ/dtheta,

where dJ/dtheta is the assembled (dual) gradient. In xi the prior Hessian is the
identity and the full Hessian is I + (prior-preconditioned misfit Hessian), so
L-BFGS convergence no longer depends on the mesh or on the mesh-scale roughness
the Euclidean dof metric over-weights. xi = 0 is the prior mean.
"""
import numpy as np
import scipy.sparse as sp
from cvxopt import cholmod, matrix, spmatrix
from firedrake import Constant, TestFunction, TrialFunction, assemble, dx, grad, inner


def precision_matrix(Q, delta, gamma):
    u, v = TrialFunction(Q), TestFunction(Q)
    a = (Constant(delta) * inner(u, v) + Constant(gamma) * inner(grad(u), grad(v))) * dx
    P = assemble(a).M.handle
    ai, aj, av = P.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=P.getSize())


class Whitening:
    r"""theta <-> xi maps for one control field with precision A = delta M + gamma K."""

    def __init__(self, Q, delta, gamma):
        A = precision_matrix(Q, delta, gamma).tocoo()
        Acvx = spmatrix(A.data.tolist(), A.row.tolist(), A.col.tolist(), A.shape)
        cholmod.options["supernodal"] = 2   # supernodal LL^T (simplicial would be LDL^T)
        self.F = cholmod.symbolic(Acvx)
        cholmod.numeric(Acvx, self.F)
        self.A = A.tocsr()
        self.n = A.shape[0]

    def _solve(self, x, *systems):
        w = matrix(np.ascontiguousarray(x, dtype=float))
        for sys_ in systems:
            cholmod.solve(self.F, w, sys=sys_)
        return np.array(w).ravel()

    def to_theta(self, xi):
        return self._solve(xi, 5, 8)        # L^T y = xi, theta = P^T y

    def to_xi(self, theta):
        # xi = L^T P theta = L^-1 P (A theta), since P A P^T = L L^T
        return self._solve(self.A @ theta, 7, 4)

    def grad_to_xi(self, g_theta):
        return self._solve(g_theta, 7, 4)   # y = P g, L z = y
