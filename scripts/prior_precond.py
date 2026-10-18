r"""The Gaussian prior on one control field, and the prior-whitened coordinates the inversions run in.

Both forms are built from the operator A = delta M + gamma K (M the CG1 mass
matrix, K the stiffness matrix), with gamma = delta ELL^2:

bilaplacian (default; fenics_ice / Recinos et al. 2023 / Isaac et al. 2015)
    R(theta) = 0.5 theta^T A M^-1 A theta,    covariance  Gamma = A^-1 M A^-1.
    This is fenics_ice's ``Laplacian`` prior: J_reg = 0.5 ||M^-1 A theta||^2_M,
    i.e. the squared Helmholtz operator, so the prior penalises curvature and a
    feature one or two elements wide is expensive. Its pointwise variance is
    1 / (4 pi delta gamma) in the continuum limit, independent of the mesh.

h1 (the form this code used before 2026-10-14)
    R(theta) = 0.5 theta^T A theta,            covariance  Gamma = A^-1.
    Pointwise variance ln(1 + gamma k_max^2 / delta) / (4 pi gamma), which grows
    with mesh resolution; narrow ridges are cheap. Kept so that MAPs inverted
    with it can still be read, and for comparison.

Note that delta and gamma do not mean the same thing under the two forms: for
the same (delta, ELL) the bilaplacian prior has a pointwise standard deviation
1 / (sqrt(4 pi) delta ELL), hundreds of times the h1 value at the delta ~ 1e-6
the h1 sweeps used. Hyperparameters must be chosen per form.

Whitened coordinates
--------------------
L-BFGS runs in coordinates xi in which the prior Hessian is (close to) the
identity and xi = 0 is the prior mean. With the sparse Cholesky P A P^T = L L^T:

h1:           theta = P^T L^-T xi,   R = 0.5 |xi|^2 exactly.
bilaplacian:  theta = A^-1 D^1/2 xi, with D the lumped mass matrix, so that
              R = 0.5 xi^T S xi, S = D^1/2 M^-1 D^1/2, which is the identity up
              to the lumped-vs-consistent mass ratio (eigenvalues within a
              factor ~2 of one). R and dR/dxi are evaluated with the consistent
              M, so the objective is exactly the fenics_ice one; only the
              optimizer metric uses the lumped mass.

In both cases dJ/dxi = (d theta / d xi)^T dJ/dtheta with dJ/dtheta the assembled
(dual) gradient.
"""
import numpy as np
import scipy.sparse as sp
from cvxopt import cholmod, matrix, spmatrix
from firedrake import Constant, TestFunction, TrialFunction, assemble, dx, grad, inner

FORMS = ("bilaplacian", "h1")
DEFAULT_FORM = "bilaplacian"


def precision_matrix(Q, delta, gamma):
    r"""A = delta M + gamma K on ``Q`` as scipy CSR."""
    u, v = TrialFunction(Q), TestFunction(Q)
    a = (Constant(delta) * inner(u, v) + Constant(gamma) * inner(grad(u), grad(v))) * dx
    P = assemble(a).M.handle
    ai, aj, av = P.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=P.getSize())


def pointwise_std(delta, gamma, form=DEFAULT_FORM, h_min=None):
    r"""Continuum pointwise prior standard deviation of theta (h1 needs the mesh scale h_min)."""
    if form == "bilaplacian":
        return 1.0 / np.sqrt(4 * np.pi * delta * gamma)
    k2 = (np.pi / h_min) ** 2
    return np.sqrt(np.log(1 + gamma * k2 / delta) / (4 * np.pi * gamma))


class _Cholesky:
    r"""Supernodal LL^T factorisation of a sparse SPD matrix (cvxopt cholmod), P A P^T = L L^T."""

    def __init__(self, A):
        A = A.tocoo()
        Acvx = spmatrix(A.data.tolist(), A.row.tolist(), A.col.tolist(), A.shape)
        cholmod.options["supernodal"] = 2
        self.F = cholmod.symbolic(Acvx)
        cholmod.numeric(Acvx, self.F)

    def _solve(self, x, *systems):
        w = matrix(np.ascontiguousarray(x, dtype=float))
        for s in systems:
            cholmod.solve(self.F, w, sys=s)
        return np.array(w).ravel()

    def solve(self, b):            # A^-1 b
        return self._solve(b, 0)

    def Lt_inv_Pt(self, x):        # P^T L^-T x
        return self._solve(x, 5, 8)

    def L_inv_P(self, x):          # L^-1 P x
        return self._solve(x, 7, 4)


class Prior:
    r"""Prior on one control field with precision built from A = delta M + gamma K; see the module docstring."""

    def __init__(self, Q, delta, gamma, form=DEFAULT_FORM):
        if form not in FORMS:
            raise ValueError(f"unknown prior form {form!r}; one of {FORMS}")
        self.form, self.delta, self.gamma = form, delta, gamma
        self.A = precision_matrix(Q, delta, gamma)
        self.n = self.A.shape[0]
        self.A_chol = _Cholesky(self.A)
        if form == "bilaplacian":
            self.M = precision_matrix(Q, 1.0, 0.0)
            self.M_chol = _Cholesky(self.M)
            self.sqrtD = np.sqrt(np.asarray(self.M.sum(axis=1)).ravel())   # lumped mass^1/2

    # -- the prior itself --------------------------------------------------------
    def apply_precision(self, theta):
        r"""Gamma^-1 theta."""
        if self.form == "h1":
            return self.A @ theta
        return self.A @ self.M_chol.solve(self.A @ theta)

    def apply_covariance(self, x):
        r"""Gamma x."""
        if self.form == "h1":
            return self.A_chol.solve(x)
        return self.A_chol.solve(self.M @ self.A_chol.solve(x))

    def R(self, theta):
        r"""0.5 theta^T Gamma^-1 theta."""
        return 0.5 * float(theta @ self.apply_precision(theta))

    # -- whitened coordinates ----------------------------------------------------
    def to_theta(self, xi):
        if self.form == "h1":
            return self.A_chol.Lt_inv_Pt(xi)
        return self.A_chol.solve(self.sqrtD * xi)

    def to_xi(self, theta):
        if self.form == "h1":
            return self.A_chol.L_inv_P(self.A @ theta)
        return (self.A @ theta) / self.sqrtD

    def grad_to_xi(self, g_theta):
        r"""(d theta / d xi)^T g for an assembled (dual) gradient g = dJ/dtheta."""
        if self.form == "h1":
            return self.A_chol.L_inv_P(g_theta)
        return self.sqrtD * self.A_chol.solve(g_theta)

    def R_and_grad_xi(self, xi):
        r"""(R, dR/dxi) at the whitened point xi."""
        if self.form == "h1":
            return 0.5 * float(xi @ xi), xi.copy()
        v = self.sqrtD * self.M_chol.solve(self.sqrtD * xi)    # S xi
        return 0.5 * float(xi @ v), v


# Backwards-compatible name for the h1 whitening the old scripts imported.
def Whitening(Q, delta, gamma):
    return Prior(Q, delta, gamma, form="h1")
