r"""Draw posterior samples of (θ_A, θ_C) for sampling-based (non-Gaussian) VAF UQ.

The linearized Isaac/Hessian UQ is invalid here (50-yr VAF is threshold-dominated;
see figures/uq_linearity.png). Instead we sample the Laplace posterior directly and
run the true forward on each sample.

Posterior  θ ~ N(θ_MAP, Γ_post),  Γ_post = A⁻¹ − Σ_i [λ_i/(1+λ_i)] w_i w_iᵀ
with GHEP eigenpairs (H w_i = λ_i A w_i, A-orthonormal). A correct sample is
    θ = θ_MAP + θ_pr − Σ_i (1 − 1/√(1+λ_i)) (w_iᵀ A θ_pr) w_i ,   θ_pr ~ N(0, A⁻¹),
which keeps full prior variance in the data-null space (the bulk of the variance —
the 1500 computed modes only reach λ_min≈1.24) and shrinks it by 1/√(1+λ_i) in the
data-informed modes. θ_pr is drawn with a CHOLMOD Cholesky of A = δM + γK:
θ_pr = Pᵀ L⁻ᵀ W (validated: reproduces gᵀA⁻¹g).

Writes results/posterior_samples.npz {theta_A (Ns×n), theta_C (Ns×n)}.
"""
import argparse
import json
import numpy as np
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, TrialFunction, TestFunction, inner, grad, dx, assemble, CheckpointFile,
)
import scipy.sparse as sp
from cvxopt import spmatrix, matrix, cholmod
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
FWD = DATA / "results" / "forward_vaf_50yr.h5"
EIG = DATA / "results" / "eigenvectors.npz"
PRIOR = DATA / "results" / "prior_params.json"
MAP_FILE = DATA / "mesh" / "inversion_hires_apres_vd1.h5"
GT = 917.0 / 1e12


def assemble_A_scipy(Q, delta, gamma):
    u, v = TrialFunction(Q), TestFunction(Q)
    a = (Constant(delta) * inner(u, v) + Constant(gamma) * inner(grad(u), grad(v))) * dx
    P = assemble(a).M.handle
    ai, aj, av = P.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=P.getSize())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=48)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--nval", type=int, default=4000, help="validation draws (cheap)")
    ap.add_argument("--out", default=str(DATA / "results" / "posterior_samples.npz"))
    ap.add_argument("--prior", action="store_true",
                    help="draw PRIOR samples (θ_MAP + θ_pr, no data shrink) for the reduction comparison")
    ap.add_argument("--eig", default=str(EIG))
    ap.add_argument("--prior_json", default=str(PRIOR))
    ap.add_argument("--map", default=str(MAP_FILE))
    ap.add_argument("--fwd", default=str(FWD),
                    help="forward gradient .h5 for σ-validation; 'none' to skip (e.g. SSA)")
    args = ap.parse_args()

    # ── Q + (optional) validation gradient.  --fwd none → no gradient (SSA). ──
    if args.fwd.lower() == "none":
        with CheckpointFile(str(DATA / "mesh" / "thwaites.h5"), "r") as c:
            m = c.load_mesh()
        Q = FunctionSpace(m, "CG", 1)
        g_full = None
    else:
        with CheckpointFile(args.fwd, "r") as c:
            m = c.load_mesh()
            gA = c.load_function(m, "dVAF_dthetaA")
            gC = c.load_function(m, "dVAF_dthetaC")
        Q = gA.function_space()
        g_full = np.concatenate([gA.dat.data_ro.copy(), gC.dat.data_ro.copy()])
    n = Q.dim()

    pr = json.load(open(args.prior_json))
    delta, gamma_A, gamma_C = pr["delta"], pr["gamma_A"], pr["gamma_C"]
    print(f"prior δ={delta:.2e}, γ_A={gamma_A:.1f}, γ_C={gamma_C:.1f}; n={n} per block", flush=True)

    # ── block prior precision A = diag(A_A, A_C) ──
    A_A = assemble_A_scipy(Q, delta, gamma_A)
    A_C = assemble_A_scipy(Q, delta, gamma_C)
    A_block = sp.block_diag([A_A, A_C], format="coo")
    Acvx = spmatrix(A_block.data.tolist(), A_block.row.tolist(), A_block.col.tolist(), A_block.shape)
    A_csr = A_block.tocsr()
    F = cholmod.symbolic(Acvx)
    cholmod.numeric(Acvx, F)
    print(f"CHOLMOD factored A_block {A_block.shape}, nnz={A_block.nnz}", flush=True)

    # ── eigenpairs + MAP ──
    eig = np.load(args.eig)
    W = eig["vecs"]                       # (2n, K), A-orthonormal
    lam = eig["eigenvalues"]
    shrink = 1.0 - 1.0 / np.sqrt(1.0 + lam)
    with CheckpointFile(args.map, "r") as c:
        mm = c.load_mesh()
        tA = c.load_function(mm, "log_fluidity")
        tC = c.load_function(mm, "log_friction")
    map_full = np.concatenate([tA.dat.data_ro.copy(), tC.dat.data_ro.copy()])

    rng = np.random.default_rng(args.seed)

    def prior_draw():
        w = matrix(rng.standard_normal(2 * n))
        cholmod.solve(F, w, sys=5)        # L' x = w  → L^{-T} w
        cholmod.solve(F, w, sys=8)        # P' x      → P^T (·)
        return np.array(w).ravel()

    def post_perturb(tp):
        # tp - Σ (1-1/√(1+λ)) (w^T A tp) w
        c = W.T @ (A_csr @ tp)
        return tp - W @ (shrink * c)

    # ── validate the sampler ──
    if g_full is not None:
        # against analytic σ_prior / σ_post (Hybrid pipeline)
        yp = np.empty(args.nval); yq = np.empty(args.nval)
        for k in range(args.nval):
            tp = prior_draw()
            yp[k] = g_full @ tp
            yq[k] = g_full @ post_perturb(tp)
        s_prior = np.sqrt((yp ** 2).mean()) * GT
        s_post = np.sqrt((yq ** 2).mean()) * GT
        print(f"\nVALIDATION ({args.nval} draws):", flush=True)
        print(f"  σ_prior sampled = {s_prior:8.2f} Gt   (analytic 127.2)", flush=True)
        print(f"  σ_post  sampled = {s_post:8.2f} Gt   (analytic  66.5)", flush=True)
        ok = abs(s_prior / 127.2 - 1) < 0.05 and abs(s_post / 66.5 - 1) < 0.06
        print(f"  → sampler {'OK' if ok else 'MISMATCH — check ordering/convention'}", flush=True)
    else:
        # No forward gradient (SSA): verify eigenvector A-orthonormality (wᵀA w ≈ I)
        s_prior = s_post = float("nan")
        K = min(8, W.shape[1])
        G = W[:, :K].T @ (A_csr @ W[:, :K])
        diag_err = float(np.abs(np.diag(G) - 1.0).max())
        offdiag = float(np.abs(G - np.diag(np.diag(G))).max())
        print(f"\nVALIDATION (no fwd gradient): A-orthonormality of first {K} eigenvectors", flush=True)
        print(f"  max|diag−1| = {diag_err:.2e}, max|offdiag| = {offdiag:.2e}", flush=True)
        print(f"  → eigendec {'OK (A-orthonormal)' if max(diag_err, offdiag) < 1e-3 else 'CHECK — NOT A-orthonormal'}", flush=True)

    # ── generate production samples ──
    theta_A = np.empty((args.nsamples, n)); theta_C = np.empty((args.nsamples, n))
    for k in range(args.nsamples):
        draw = prior_draw()
        th = map_full + (draw if args.prior else post_perturb(draw))
        theta_A[k] = th[:n]; theta_C[k] = th[n:]
    kind = "PRIOR" if args.prior else "posterior"
    out = Path(args.out)
    np.savez(str(out), theta_A=theta_A, theta_C=theta_C,
             seed=args.seed, sigma_prior_Gt=s_prior, sigma_post_lin_Gt=s_post)
    print(f"\nSaved {args.nsamples} {kind} samples → {out}", flush=True)
    print(f"  θ_A sample spread: rms perturb = {np.sqrt(((theta_A-map_full[:n])**2).mean()):.4f}", flush=True)
    print(f"  θ_C sample spread: rms perturb = {np.sqrt(((theta_C-map_full[n:])**2).mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
