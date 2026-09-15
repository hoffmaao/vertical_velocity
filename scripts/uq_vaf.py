r"""Posterior variance for VAF(T) using Isaac et al. (2015, Eq. 20).

σ²_Q = ∇Q · Γ_prior ∇Q  -  Σ_i (λ_i / (λ_i + 1)) (∇Q · w_i)²

where (fenics_ice / Recinos et al. 2023 convention)
  Γ_prior = A⁻¹           (prior covariance, A = δM + γK)
  w_i, λ_i                eigenpairs of GHEP H w = λ A w
                          (A-orthonormal: w_i^T A w_j = δ_ij)
  ∇Q                       VAF(T) gradient w.r.t. (θ_A, θ_C)
                          (from results/forward_vaf_<T>yr.h5)

The regularization R(θ) = ½ θᵀA θ = ½ ∫(δθ² + γ|∇θ|²)dx corresponds to
prior θ ~ N(0, A⁻¹), so σ²_prior = ∇Q · A⁻¹ · ∇Q (single inverse, no extra M).

Inner products use the Euclidean DOF-vector dot product. With A-orthonormal
eigenvectors, this gives σ²_post bounded in [0, σ²_prior].

Run:
    python uq_vaf.py --fwd_ckpt results/forward_vaf_50yr.h5
"""
import argparse
import json
import numpy as np
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, CheckpointFile,
    TrialFunction, TestFunction, inner, grad, dx, assemble,
)
from firedrake.petsc import PETSc
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"


def make_prior_cov_solver(Q, delta, gamma):
    """Return cov_action(x) → A⁻¹ x as a Function (fenics_ice convention).

    A = δM + γK is the prior precision (R(θ) = ½ θᵀA θ ⇒ θ ~ N(0, A⁻¹)).
    """
    v = TrialFunction(Q)
    w = TestFunction(Q)
    a_form = (Constant(delta) * inner(v, w)
              + Constant(gamma) * inner(grad(v), grad(w))) * dx

    A_mat = assemble(a_form).M.handle
    ksp_A = PETSc.KSP().create()
    ksp_A.setOperators(A_mat)
    ksp_A.setType("preonly")
    ksp_A.getPC().setType("lu")
    ksp_A.getPC().setFactorSolverType("mumps")
    ksp_A.setUp()

    def cov_action(x_func):
        # y = A^{-1} x
        y = Function(Q)
        with x_func.dat.vec_ro as xv, y.dat.vec as yv:
            ksp_A.solve(xv, yv)
        return y
    return cov_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd_ckpt", required=True,
                    help="Forward checkpoint with dVAF_dthetaA/C")
    ap.add_argument("--eig_npz", default=str(DATA_DIR / "results" / "eigenvectors.npz"))
    ap.add_argument("--prior_json", default=str(DATA_DIR / "results" / "prior_params.json"))
    ap.add_argument("--nmodes", type=int, default=None,
                    help="Limit to leading N modes (default: all positive)")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: alongside fwd_ckpt)")
    args = ap.parse_args()

    fwd_path = Path(args.fwd_ckpt)
    out_dir = Path(args.out) if args.out else fwd_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = fwd_path.stem  # e.g. "forward_vaf_50yr"

    print(f"UQ propagation for {fwd_path.name}", flush=True)

    # ── Load eigenpairs ──
    eig = np.load(args.eig_npz)
    vecs = eig["vecs"]            # shape (2n, n_pos)
    vals = eig["eigenvalues"]     # shape (n_pos,)
    n_dof = int(eig["n_dof"])
    n_pos = vals.size
    if args.nmodes is not None:
        n_use = min(args.nmodes, n_pos)
        vecs = vecs[:, :n_use]
        vals = vals[:n_use]
    else:
        n_use = n_pos
    D = vals / (vals + 1.0)
    print(f"  eigenpairs: {n_use} (of {n_pos} positive, λ_max={vals[0]:.3e}, λ_min={vals[-1]:.3e})",
          flush=True)
    print(f"  D = λ/(λ+1): D_max={D[0]:.6f}, D_min={D[-1]:.6f}", flush=True)

    # ── Load prior hyperparameters ──
    with open(args.prior_json) as f:
        priors = json.load(f)
    δ = priors["delta"]
    γ_A = priors["gamma_A"]
    γ_C = priors["gamma_C"]
    print(f"  prior: δ={δ:.4e}, γ_A={γ_A:.4e}, γ_C={γ_C:.4e}", flush=True)

    # ── Load mesh + gradient ──
    with CheckpointFile(str(fwd_path), "r") as chk:
        mesh = chk.load_mesh()
        gA = chk.load_function(mesh, "dVAF_dthetaA")
        gC = chk.load_function(mesh, "dVAF_dthetaC")
    Q = gA.function_space()
    assert Q.dim() == n_dof, f"DOF mismatch: gradient has {Q.dim()}, eigvecs expect {n_dof}"

    g_A_arr = gA.dat.data_ro.copy()
    g_C_arr = gC.dat.data_ro.copy()
    g_full = np.concatenate([g_A_arr, g_C_arr])  # shape (2n,)
    print(f"  ||g_A||={np.linalg.norm(g_A_arr):.4e}, ||g_C||={np.linalg.norm(g_C_arr):.4e}",
          flush=True)

    # ── Prior variance σ²_prior = g · Γ g ──
    cov_A = make_prior_cov_solver(Q, δ, γ_A)
    cov_C = make_prior_cov_solver(Q, δ, γ_C)

    ΓgA = cov_A(gA)
    ΓgC = cov_C(gC)

    with gA.dat.vec_ro as gv, ΓgA.dat.vec_ro as Γv:
        σ2_prior_A = gv.dot(Γv)
    with gC.dat.vec_ro as gv, ΓgC.dat.vec_ro as Γv:
        σ2_prior_C = gv.dot(Γv)
    σ2_prior = σ2_prior_A + σ2_prior_C
    σ_prior = np.sqrt(max(σ2_prior, 0.0))
    print(f"\n  σ²_prior(VAF) = {σ2_prior:.6e} m⁶ "
          f"(θ_A: {σ2_prior_A:.3e}, θ_C: {σ2_prior_C:.3e})", flush=True)
    print(f"  σ_prior(VAF)  = {σ_prior:.6e} m³ = {σ_prior * 917.0 / 1e12:.3f} Gt",
          flush=True)

    # ── Mode contributions: a_i = w_i · g ──
    a = vecs.T @ g_full                # shape (n_use,)
    contribs = D * a ** 2              # variance reduction per mode

    var_reduction = float(contribs.sum())
    σ2_post = σ2_prior - var_reduction
    σ_post = np.sqrt(max(σ2_post, 0.0))

    print(f"\n  Σ D_i a_i² = {var_reduction:.6e} m⁶", flush=True)
    print(f"  σ²_post = σ²_prior - Σ D_i a_i² = {σ2_post:.6e} m⁶", flush=True)
    print(f"  σ_post  = {σ_post:.6e} m³ = {σ_post * 917.0 / 1e12:.3f} Gt", flush=True)
    print(f"  Variance reduction: {100 * (1 - σ2_post / σ2_prior):.2f} %", flush=True)
    print(f"  σ ratio (post/prior): {σ_post / σ_prior:.4f}", flush=True)

    # ── Save ──
    out_summary = out_dir / f"uq_{tag}_summary.json"
    summary = {
        "tag": tag,
        "n_modes_used": int(n_use),
        "n_modes_positive": int(n_pos),
        "lambda_max": float(vals[0]),
        "lambda_min": float(vals[-1]),
        "sigma2_prior_m6": float(σ2_prior),
        "sigma2_prior_A_m6": float(σ2_prior_A),
        "sigma2_prior_C_m6": float(σ2_prior_C),
        "sigma2_post_m6": float(σ2_post),
        "sigma_prior_m3": float(σ_prior),
        "sigma_post_m3": float(σ_post),
        "sigma_prior_Gt": float(σ_prior * 917.0 / 1e12),
        "sigma_post_Gt": float(σ_post * 917.0 / 1e12),
        "variance_reduction_frac": float(1 - σ2_post / σ2_prior) if σ2_prior > 0 else 0.0,
        "g_A_norm": float(np.linalg.norm(g_A_arr)),
        "g_C_norm": float(np.linalg.norm(g_C_arr)),
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_summary}", flush=True)

    out_contribs = out_dir / f"uq_{tag}_contribs.txt"
    with open(out_contribs, "w") as f:
        f.write("# i  lambda_i  a_i  D_i*a_i^2  cumulative_reduction_frac\n")
        cum = np.cumsum(contribs)
        for i in range(n_use):
            cum_frac = cum[i] / σ2_prior if σ2_prior > 0 else 0.0
            f.write(f"{i} {vals[i]:.16e} {a[i]:.16e} "
                    f"{contribs[i]:.16e} {cum_frac:.6f}\n")
    print(f"Saved {out_contribs}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
