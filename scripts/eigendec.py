r"""Gauss-Newton Hessian eigendecomposition at a MAP, for any (model, friction) configuration.

Successor to run_eigendec.py, which was wired to one hard-coded MAP file, the
Weertman law, the old mesh and a prior typed into the script. This one takes the
MAP by name, builds the model through problem.build_problem (so the forward
operator is byte-for-byte the one the inversion saw) and reads the prior
hyperparameters and prior form that MAP was inverted with from
results/map_<name>.json, so the generalised eigenproblem (fenics_ice / Recinos
et al. 2023 convention)

    H_GN w = lambda Gamma^-1 w,    Gamma^-1 = blockdiag over (theta_A, theta_C)

is posed with exactly the prior of that MAP (Gamma^-1 = A M^-1 A for the
bilaplacian form, A for the older h1 form, A = delta M + gamma K;
prior_precond.py), and the eigenvectors come out Gamma^-1-orthonormal, which is
what uq_vaf.py assumes.

    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/eigendec.py \
        --model hybrid --friction rc \
        --map sweep_hybrid_rc_d1e-05_L2km_thwaites_fixed_tau5km --K 20      # smoke test
    VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000 python scripts/eigendec.py \
        --model hybrid --friction weertman --K 1500                          # production MAP

--map is the --name the inversion ran with; without it the production MAP
map_<tag> is used. --prior-scale multiplies delta AND gamma of both controls,
which relaxes (or tightens) the prior at fixed correlation length without
re-inverting: Recinos et al. (2023) found the L-curve elbow too strong a prior
for UQ and relaxed it this way.

Hessian-vector products come from map_hessian.setup: exact Gauss-Newton
products by default (--method gn), finite differences of the misfit gradient
with --method fd, kept for validation (~10% spurious modes at K=1500).

Every matvec is cached in results/eigendec_cache/matvecs_<fingerprint>.h5, the
fingerprint covering the MAP, mesh, prior and method. Re-running with a larger
--K replays the cached products as long as ARPACK asks for the same vectors,
which it does until the smaller run's first implicit restart (after its ncv =
2K+1 Lanczos steps); from the first vector that differs the cache is truncated
and recomputed. Anything else changing starts a fresh cache.

Writes, with <name> the --map value and <K> the mode count requested:
  results/eigendec_<name>_K<K>.npz   eigenvectors (2n x n_pos, A-orthonormal), eigenvalues
  results/eigendec_<name>_K<K>.json  prior used, n_eff, spectrum summary
  mesh/eigendec_<name>_K<K>.h5       the first 100 modes as (theta_A, theta_C) fields
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
from firedrake import CheckpointFile, Function, MixedFunctionSpace
from scipy.sparse.linalg import LinearOperator, eigsh

from map_hessian import EPS_FD, load_prior, setup
from physical_prior import MESH_FILE, MESH_NAME, TAU_D_SMOOTH
from problem import tag

DATA = Path(__file__).resolve().parent.parent
N_MODES_H5 = 100        # modes written as fields for plotting
ARPACK_TOL = 1e-6
ARPACK_MAXITER = 300


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    ap.add_argument("--friction", required=True)
    ap.add_argument("--no-apres", action="store_true",
                    help="hybrid: the MAP came from a velocity-only inversion")
    ap.add_argument("--map", default=None,
                    help="the --name the inversion ran with (default: <tag>, the production "
                         "MAP); read from mesh/map_<name>.h5 and results/map_<name>.json")
    ap.add_argument("--K", type=int, default=1500, help="number of leading eigenmodes")
    ap.add_argument("--method", choices=["gn", "fd"], default="gn")
    ap.add_argument("--prior-scale", type=float, default=1.0,
                    help="multiply delta and gamma of both controls (keeps ELL); "
                         "< 1 relaxes the prior for UQ (Recinos et al. 2023)")
    ap.add_argument("--out", default=None, help="output stem (default: eigendec_<name>_K<K>)")
    args = ap.parse_args()
    args.apres = args.model == "hybrid" and not args.no_apres
    args.tag = tag(args.model, args.friction, args.apres)
    args.map = args.map or args.tag
    args.out = args.out or f"eigendec_{args.map}_K{args.K}"
    return args


def main():
    args = parse_args()
    t0 = time.time()
    δ_A, γ_A, δ_C, γ_C = load_prior(args.map, args.prior_scale)

    print("=" * 72, flush=True)
    print(f"GN Hessian eigendecomposition [{args.tag}] at {args.map}", flush=True)
    print(f"  mesh {MESH_NAME}, tau_d smoothing {TAU_D_SMOOTH:g} m, method {args.method}, "
          f"K={args.K}", flush=True)
    print(f"  prior: theta_A delta={δ_A:.3e} gamma={γ_A:.3e} (ELL {np.sqrt(γ_A / δ_A) / 1e3:g} km); "
          f"theta_C delta={δ_C:.3e} gamma={γ_C:.3e} (ELL {np.sqrt(γ_C / δ_C) / 1e3:g} km)"
          + (f"; scaled by {args.prior_scale:g}" if args.prior_scale != 1.0 else ""), flush=True)
    print("=" * 72, flush=True)

    S = setup(args.model, args.friction, args.apres, args.map, args.method, args.prior_scale)
    Q_2d, mesh_2d, n = S["P"]["Q_2d"], S["P"]["mesh_2d"], S["n"]
    Hv, N_vel, N_apr = S["Hv"], S["N_vel"], S["N_apr"]

    # The GHEP's metric Gamma^-1 and its inverse, as operators (the bilaplacian
    # precision A M^-1 A is not sparse; both are applied through factorisations
    # of A and M held by the Prior objects).
    N = 2 * n
    Rinv_op = LinearOperator((N, N), matvec=S["Rinv"], dtype=float)
    Rcov_op = LinearOperator((N, N), matvec=S["Rcov"], dtype=float)

    # Matvec cache keyed on everything that changes H or A (K is deliberately
    # not in it, so a larger K replays what a smaller one computed).
    fp_data = json.dumps({
        "map": args.map, "map_file": str(DATA / "mesh" / f"map_{args.map}.h5"),
        "mesh": str(MESH_FILE), "tau_d_smooth": TAU_D_SMOOTH, "tag": args.tag,
        "method": args.method, "eps_fd": EPS_FD if args.method == "fd" else None,
        "delta_A": δ_A, "gamma_A": γ_A, "delta_C": δ_C, "gamma_C": γ_C,
        "n": int(n), "N_vel": int(N_vel), "N_apr": int(N_apr),
        "prior_form": S["prior_form"],
        "convention": "recinos_GHEP_H_only",
    }, sort_keys=True)
    fp = hashlib.sha256(fp_data.encode()).hexdigest()[:12]
    cache_dir = DATA / "results" / "eigendec_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file, v0_file = cache_dir / f"matvecs_{fp}.h5", cache_dir / f"v0_{fp}.npy"
    if v0_file.exists():
        v0 = np.load(str(v0_file))
    else:
        v0 = np.random.default_rng(42).standard_normal(N)
        np.save(str(v0_file), v0)

    cache = h5py.File(str(cache_file), "a")
    if "y" not in cache:
        for k in ("x", "y"):
            cache.create_dataset(k, shape=(0, N), maxshape=(None, N), dtype="f8", chunks=(1, N))
        cache.attrs["fingerprint"] = fp_data
    n_cached = cache["y"].shape[0]
    print(f"Matvec cache {cache_file.name}: {n_cached} products stored", flush=True)
    calls = [0]
    stored = [n_cached]
    t_mv = [time.time()]

    def matvec(x):
        i = calls[0]
        calls[0] += 1
        if i < stored[0]:
            if np.abs(x - cache["x"][i, :]).max() <= 1e-8:
                return cache["y"][i, :].copy()
            # A smaller-K run restarted here; its later vectors are useless.
            print(f"  cache diverges at matvec {i} (an earlier run's ARPACK restart): "
                  f"dropping its remaining {stored[0] - i} products", flush=True)
            for k in ("x", "y"):
                cache[k].resize(i, axis=0)
            stored[0] = i
        y = Hv(x)
        for k, v in (("x", x), ("y", y)):
            cache[k].resize(i + 1, axis=0)
            cache[k][i, :] = v
        cache.flush()
        if calls[0] % 10 == 0 or i == stored[0]:
            now = time.time()
            print(f"  matvec {calls[0]}  ({(now - t_mv[0]) / (10 if i != stored[0] else 1):.1f} s each, "
                  f"{(now - t0) / 60:.1f} min)", flush=True)
            t_mv[0] = now
        return y

    print(f"\nARPACK: {args.K} modes of H w = lambda Gamma^-1 w ({S['prior_form']} prior) ...",
          flush=True)
    try:
        vals, vecs = eigsh(LinearOperator((N, N), matvec=matvec, dtype=float),
                           k=min(args.K, N - 2), M=Rinv_op, Minv=Rcov_op, which="LM",
                           maxiter=ARPACK_MAXITER, tol=ARPACK_TOL, v0=v0)
    finally:
        cache.close()
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    n_pos = int((vals > 0).sum())
    D = vals[:n_pos] / (1.0 + vals[:n_pos])
    n_eff = float(D.sum())
    print(f"\n{calls[0]} matvecs ({calls[0] - min(n_cached, stored[0])} new). Leading eigenvalues: "
          + " ".join(f"{v:.3e}" for v in vals[:6]), flush=True)
    print(f"Positive modes {n_pos}/{len(vals)}; lambda_min {vals[-1]:.3e}; "
          f"effective constrained parameters n_eff = {n_eff:.1f}", flush=True)
    if n_pos < len(vals):
        print(f"  WARNING: {len(vals) - n_pos} non-positive eigenvalues"
              + (" (the fd method's truncation noise)" if args.method == "fd"
                 else " - GN products should be positive semi-definite; check the tape"),
              flush=True)

    npz = DATA / "results" / f"{args.out}.npz"
    np.savez(str(npz), vecs=vecs[:, :n_pos], eigenvalues=vals[:n_pos], n_dof=n)
    summary = {
        "map": args.map, "tag": args.tag, "model": args.model, "friction": args.friction,
        "apres": args.apres, "mesh": MESH_NAME, "tau_d_smooth": TAU_D_SMOOTH,
        "method": args.method, "K": args.K, "n_positive": n_pos, "n_dof": int(n),
        "N_vel": int(N_vel), "N_apr": int(N_apr),
        "delta_A": δ_A, "gamma_A": γ_A, "ELL_A": float(np.sqrt(γ_A / δ_A)),
        "delta_C": δ_C, "gamma_C": γ_C, "ELL_C": float(np.sqrt(γ_C / δ_C)),
        "prior_scale": args.prior_scale, "prior_form": S["prior_form"],
        "lambda_max": float(vals[0]), "lambda_min_positive": float(vals[n_pos - 1]),
        "n_eff": n_eff, "n_matvecs": int(calls[0]), "cache": cache_file.name,
        "convention": ("recinos_GHEP: H w = lambda Gamma^-1 w, w Gamma^-1-orthonormal, "
                       + ("Gamma^-1 = A M^-1 A" if S["prior_form"] == "bilaplacian" else "Gamma^-1 = A")
                       + ", A = delta M + gamma K"),
        "wall_hours": (time.time() - t0) / 3600,
    }
    with open(DATA / "results" / f"{args.out}.json", "w") as f:
        json.dump(summary, f, indent=2)

    QQ = MixedFunctionSpace([Q_2d, Q_2d])
    h5 = DATA / "mesh" / f"{args.out}.h5"
    with CheckpointFile(str(h5), "w") as chk:
        chk.save_mesh(mesh_2d)
        for i in range(min(N_MODES_H5, n_pos)):
            mode = Function(QQ, name=f"mode_{i:04d}")
            mode.sub(0).dat.data[:] = vecs[:n, i]
            mode.sub(1).dat.data[:] = vecs[n:, i]
            chk.save_function(mode)
    print(f"\nSaved {npz.name}, {args.out}.json and {h5.name} in "
          f"{(time.time() - t0) / 3600:.2f} h wall", flush=True)


if __name__ == "__main__":
    main()
