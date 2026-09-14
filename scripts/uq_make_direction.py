r"""Build the QoI posterior-uncertainty direction for the linearization check.

The linearized posterior VAF variance is σ²_post = gᵀ Γ_post g, with
  Γ_post = A⁻¹ − Σ_i [λ_i/(1+λ_i)] w_i w_iᵀ   (GHEP, A-orthonormal w_i).

The θ-direction that carries this QoI uncertainty is d = Γ_post g. Scaled to
  δθ_1σ = Γ_post g / σ_post     ⇒  g · δθ_1σ = σ_post,
so that a perturbation θ_MAP + n·δθ_1σ has *linearized* ΔVAF = n·σ_post.
Marching the true forward along ±δθ_1σ and comparing ΔVAF to n·σ_post tests
whether the Isaac et al. linearization holds at this (grounding-line-dominated) MAP.

Writes results/uq_linearity_direction.npz {dA, dC, sigma_post_m3, sigma_prior_m3}.
"""
import json
import numpy as np
import firedrake as fd
from firedrake import (
    Constant, Function, TrialFunction, TestFunction,
    inner, grad, dx, assemble, CheckpointFile,
)
from firedrake.petsc import PETSc
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
FWD = DATA / "results" / "forward_vaf_50yr.h5"
EIG = DATA / "results" / "eigenvectors.npz"
PRIOR = DATA / "results" / "prior_params.json"
OUT = DATA / "results" / "uq_linearity_direction.npz"
GT = 917.0 / 1e12


def cov_solver(Q, delta, gamma):
    """A⁻¹ action for A = δM + γK (fenics_ice prior precision)."""
    v, w = TrialFunction(Q), TestFunction(Q)
    a = (Constant(delta) * inner(v, w) + Constant(gamma) * inner(grad(v), grad(w))) * dx
    A = assemble(a).M.handle
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()

    def solve(xfun):
        y = Function(Q)
        with xfun.dat.vec_ro as xv, y.dat.vec as yv:
            ksp.solve(xv, yv)
        return y
    return solve


def main():
    with CheckpointFile(str(FWD), "r") as c:
        m = c.load_mesh()
        gA = c.load_function(m, "dVAF_dthetaA")
        gC = c.load_function(m, "dVAF_dthetaC")
    Q = gA.function_space()
    n = Q.dim()
    gA_v = gA.dat.data_ro.copy()
    gC_v = gC.dat.data_ro.copy()
    g_full = np.concatenate([gA_v, gC_v])

    pr = json.load(open(PRIOR))
    delta, gamma_A, gamma_C = pr["delta"], pr["gamma_A"], pr["gamma_C"]
    print(f"prior: delta={delta:.3e}, gamma_A={gamma_A:.3e}, gamma_C={gamma_C:.3e}", flush=True)

    solA = cov_solver(Q, delta, gamma_A)
    solC = cov_solver(Q, delta, gamma_C)
    Ainv_g = np.concatenate([solA(gA).dat.data_ro.copy(),
                             solC(gC).dat.data_ro.copy()])

    eig = np.load(EIG)
    vecs, vals = eig["vecs"], eig["eigenvalues"]
    assert vecs.shape[0] == 2 * n, f"{vecs.shape} vs 2*{n}"
    a_i = vecs.T @ g_full
    D = vals / (vals + 1.0)

    Gpost_g = Ainv_g - vecs @ (D * a_i)       # Γ_post g
    s2_prior = float(g_full @ Ainv_g)
    s2_post = float(g_full @ Gpost_g)
    sigma_prior = np.sqrt(max(s2_prior, 0.0))
    sigma_post = np.sqrt(max(s2_post, 0.0))
    print(f"sigma_prior = {sigma_prior * GT:.2f} Gt  (expect 127.0)", flush=True)
    print(f"sigma_post  = {sigma_post * GT:.2f} Gt  (expect 76.4)", flush=True)

    dtheta = Gpost_g / sigma_post              # g · dtheta = sigma_post
    dA, dC = dtheta[:n], dtheta[n:]
    print(f"||δθ_1σ||₂ = {np.linalg.norm(dtheta):.4f}", flush=True)
    print(f"  δθ_A: max|·|={np.abs(dA).max():.4f}, rms={np.sqrt((dA**2).mean()):.4f}", flush=True)
    print(f"  δθ_C: max|·|={np.abs(dC).max():.4f}, rms={np.sqrt((dC**2).mean()):.4f}", flush=True)
    print(f"  g·δθ_1σ = {float(g_full @ dtheta) * GT:.3f} Gt (should equal sigma_post)", flush=True)

    np.savez(str(OUT), dA=dA, dC=dC,
             sigma_post_m3=sigma_post, sigma_prior_m3=sigma_prior,
             sigma_post_Gt=sigma_post * GT, sigma_prior_Gt=sigma_prior * GT)
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
