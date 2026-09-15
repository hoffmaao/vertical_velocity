r"""Gauss-Newton Hessian eigendecomposition for the SSA (IceStream) inversion.

SSA companion to run_eigendec.py for the SSA-vs-Hybrid UQ comparison. Velocity-only
(SSA has no ApRES), 2D (no extrusion), IceStream + Schoof-Coulomb friction, MAP from
inversion_ssa_recinos.h5. Same Recinos GHEP prior as the Hybrid eigendec (δ=1e-5,
ELL=2km). Reuses sparse_hessian.TapedForward + make_Hv_gn_velocity (model-agnostic).

Writes results/eigenvectors_ssa.npz, eigenvalues_ssa.txt, prior_params_ssa.json,
mesh/eigendec_ssa.h5 (does NOT touch the unsuffixed Hybrid files).
"""
import argparse
import sys
import numpy as np
import h5py
import hashlib
import netCDF4 as nc
import firedrake as fd
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, MixedFunctionSpace,
    CheckpointFile, VertexOnlyMesh,
    sqrt, inner, grad, max_value, exp, dx, conditional, assemble,
    TestFunction, TrialFunction,
)
from firedrake.adjoint import *  # noqa
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu
from icepack.constants import (
    ice_density as ρ_I, water_density as ρ_W, gravity as g,
    weertman_sliding_law as m,
)
import icepack
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent
MESH_FILE = DATA_DIR / "mesh" / "thwaites.h5"
VEL_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/velocity/"
                "antarctica_ice_velocity_450m_v2.nc")

K_LEADING = 1500
ELL = 2000.0            # matches Hybrid eigendec
DELTA_PRIOR = 1.0e-5
GAMMA_A = 1.0
GAMMA_C = 1.0
VEL_STEP = 1
A_0v = Constant(20.0)
U_0 = Constant(300.0)

SOLVER_PARAMS = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 250, "snes_rtol": 1e-6,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}


def friction_coulomb(**kwargs):
    u, h, s, τ_0 = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W / ρ_I * max_value(0, h - s) / h
    U = sqrt(inner(u, u))
    return τ_0 * ϕ * ((U_0 ** (1 / m + 1) + U ** (1 / m + 1)) ** (m / (m + 1)) - U_0)


def friction_weertman(**kwargs):
    u, h, s, C = kwargs["velocity"], kwargs["thickness"], kwargs["surface"], kwargs["friction"]
    ϕ = 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)
    return icepack.models.friction.bed_friction(velocity=u, friction=C * ϕ)


# friction → (function, C_0, MAP filename, output suffix)
FRICTION = {
    "coulomb": (friction_coulomb, 0.5, "inversion_ssa_recinos.h5", ""),
    "weertman": (friction_weertman, 0.01, "inversion_ssa_weertman.h5", "w"),
}


def main():
    global K_LEADING
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--friction", choices=["coulomb", "weertman"], default="coulomb")
    args = ap.parse_args()
    if args.K is not None:
        K_LEADING = args.K
    fric, C0v, map_name, fsuf = FRICTION[args.friction]
    C_0v = Constant(C0v)
    MAP_FILE = DATA_DIR / "mesh" / map_name
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sparse_hessian as sh

    print("=" * 60, flush=True)
    print(f"SSA Hessian eigendecomposition (velocity-only GN, {args.friction})", flush=True)
    print(f"  K={K_LEADING}, ELL={ELL:.0f} m, δ={DELTA_PRIOR:.1e}", flush=True)
    print("=" * 60, flush=True)

    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        h = chk.load_function(mesh, "thickness")
        s = chk.load_function(mesh, "surface")
        u_obs = chk.load_function(mesh, "velocity")
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    area_val = float(assemble(Constant(1.0) * dx(mesh)))
    print(f"Mesh: {mesh.num_vertices()} verts, ndof={Q.dim()}", flush=True)

    # ── Sparse velocity (2D VOM) ──
    from scipy.spatial import Delaunay
    ds = nc.Dataset(str(VEL_FILE))
    x_vel = ds.variables["x"][:]; y_vel = ds.variables["y"][:]
    coords = mesh.coordinates.dat.data_ro
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    ix0 = max(0, np.searchsorted(x_vel, xmin) - 5); ix1 = min(len(x_vel), np.searchsorted(x_vel, xmax) + 5)
    iy0 = max(0, np.searchsorted(-y_vel, -ymax) - 5); iy1 = min(len(y_vel), np.searchsorted(-y_vel, -ymin) + 5)
    sl = np.s_[iy0:iy1:VEL_STEP, ix0:ix1:VEL_STEP]
    vx = np.nan_to_num(np.array(ds.variables["VX"][sl]).astype(float), nan=0.0)
    vy = np.nan_to_num(np.array(ds.variables["VY"][sl]).astype(float), nan=0.0)
    stdx = np.nan_to_num(np.array(ds.variables["STDX"][sl]).astype(float), nan=1e6)
    stdy = np.nan_to_num(np.array(ds.variables["STDY"][sl]).astype(float), nan=1e6)
    ds.close()
    YY, XX = np.meshgrid(y_vel[iy0:iy1:VEL_STEP], x_vel[ix0:ix1:VEL_STEP], indexing="ij")
    xs_f, ys_f = XX.ravel(), YY.ravel()
    inside = Delaunay(coords).find_simplex(np.column_stack([xs_f, ys_f])) >= 0
    has = (stdx.ravel() < 1e5) & (stdy.ravel() < 1e5) & (stdx.ravel() > 0) & (stdy.ravel() > 0)
    keep = inside & has
    _vel_xy = np.column_stack([xs_f[keep], ys_f[keep]])
    _σx_k = np.maximum(stdx.ravel()[keep], 1.0); _σy_k = np.maximum(stdy.ravel()[keep], 1.0)
    print(f"Sparse velocity: {keep.sum()} points", flush=True)

    # ── MAP + primed velocity ──
    θ_A = Function(Q, name="log_fluidity"); θ_C = Function(Q, name="log_friction")
    u0 = Function(V)
    with CheckpointFile(str(MAP_FILE), "r") as chk:
        mm = chk.load_mesh()
        θ_A.dat.data[:] = chk.load_function(mm, "log_fluidity").dat.data_ro
        θ_C.dat.data[:] = chk.load_function(mm, "log_friction").dat.data_ro
        u0.dat.data[:] = chk.load_function(mm, "velocity").dat.data_ro
    θA_map = θ_A.dat.data_ro.copy(); θC_map = θ_C.dat.data_ro.copy()
    print(f"MAP: θ_A=[{θA_map.min():.2f},{θA_map.max():.2f}], θ_C=[{θC_map.min():.2f},{θC_map.max():.2f}]", flush=True)

    grounded_mask = Function(Q).interpolate(
        conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) > 0.01, Constant(1.0), Constant(0.0)))
    model = icepack.models.IceStream(friction=fric)
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[2], ice_front_ids=[1],
        diagnostic_solver_type="petsc", diagnostic_solver_parameters=SOLVER_PARAMS)
    A_map = Function(Q).interpolate(A_0v * exp(θ_A))
    C_map = Function(Q).interpolate(C_0v * exp(θ_C * grounded_mask))
    u0 = solver.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A_map, friction=C_map)
    print("Primed at MAP", flush=True)

    # ── GN Hessian-vector product (velocity-only) ──
    def simulation(θA_in, θC_in):
        A = Function(Q).interpolate(A_0v * exp(θA_in))
        C = Function(Q).interpolate(C_0v * exp(θC_in * grounded_mask))
        return solver.diagnostic_solve(velocity=u0, thickness=h, surface=s, fluidity=A, friction=C)

    print("Building TapedForward at MAP...", flush=True)
    taped = sh.TapedForward(simulation, θ_A, θ_C)
    print(f"  tape blocks: {len(taped.tape.get_blocks())}", flush=True)
    Hv_vel = sh.make_Hv_gn_velocity(taped, _vel_xy, _σx_k, _σy_k)

    def Hv_pair(v_A, v_C):
        vA_f = Function(Q); vA_f.dat.data[:] = v_A
        vC_f = Function(Q); vC_f.dat.data[:] = v_C
        hvA, hvC = Hv_vel(vA_f, vC_f)
        return hvA.dat.data_ro.copy(), hvC.dat.data_ro.copy()

    # ── Prior A = δM + γK (Recinos GHEP) ──
    delta = Constant(DELTA_PRIOR)
    gamma_A_val = GAMMA_A * ELL**2 * DELTA_PRIOR
    gamma_C_val = GAMMA_C * ELL**2 * DELTA_PRIOR
    print(f"Prior: δ={DELTA_PRIOR:.2e}, γ_A={gamma_A_val:.1f}, γ_C={gamma_C_val:.1f}, ELL={ELL:.0f}", flush=True)

    def prior_matrix(gamma_v):
        tq, vq = TrialFunction(Q), TestFunction(Q)
        a = (delta * inner(tq, vq) + Constant(gamma_v) * inner(grad(tq), grad(vq))) * dx
        P = assemble(a).M.handle
        ai, aj, av = P.getValuesCSR()
        return sp.csr_matrix((av, aj, ai), shape=P.getSize())
    A_block = sp.block_diag([prior_matrix(gamma_A_val), prior_matrix(gamma_C_val)], format="csr")
    print(f"A_block {A_block.shape}, nnz={A_block.nnz}; factorizing...", flush=True)
    A_lu = splu(A_block.tocsc())
    Minv_op = LinearOperator(A_block.shape, matvec=lambda b: A_lu.solve(b), dtype=float)

    # ── ARPACK GHEP eigensolve with matvec cache ──
    n = θ_A.dat.data.shape[0]; N = 2 * n
    fp = hashlib.sha256(json.dumps({
        "model": f"SSA_IceStream_{args.friction}", "ELL": ELL, "DELTA_PRIOR": DELTA_PRIOR,
        "GAMMA_A": GAMMA_A, "GAMMA_C": GAMMA_C, "VEL_STEP": VEL_STEP, "n": int(n),
        "MAP_FILE": str(MAP_FILE), "convention": "recinos_GHEP_velocity_only",
    }, sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = DATA_DIR / "results" / "eigendec_cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"matvecs_ssa_{fp}.h5"; v0_file = cache_dir / f"v0_ssa_{fp}.npy"
    if v0_file.exists():
        v0 = np.load(str(v0_file))
    else:
        v0 = np.random.default_rng(42).standard_normal(N); np.save(str(v0_file), v0)
    h5c = h5py.File(str(cache_file), "a")
    if "y" not in h5c:
        h5c.create_dataset("x", shape=(0, N), maxshape=(None, N), dtype="f8", chunks=(1, N))
        h5c.create_dataset("y", shape=(0, N), maxshape=(None, N), dtype="f8", chunks=(1, N))
    n_cached = h5c["y"].shape[0]
    print(f"Matvec cache: {n_cached} computed → {cache_file.name}\nARPACK for {K_LEADING} modes...", flush=True)
    cc = [0]

    def matvec(x):
        i = cc[0]
        if i < n_cached:
            if np.abs(x - h5c["x"][i, :]).max() > 1e-8:
                raise RuntimeError(f"cache mismatch at {i}; delete {cache_file.name}")
            cc[0] += 1; return h5c["y"][i, :].copy()
        hv_A, hv_C = Hv_pair(x[:n], x[n:])
        y = np.concatenate([hv_A, hv_C])
        h5c["x"].resize(i + 1, axis=0); h5c["x"][i, :] = x
        h5c["y"].resize(i + 1, axis=0); h5c["y"][i, :] = y; h5c.flush()
        cc[0] += 1
        if cc[0] % 25 == 0:
            print(f"  matvec {cc[0]}", flush=True)
        return y

    H_op = LinearOperator((N, N), matvec=matvec, dtype=float)
    try:
        vals, vecs = eigsh(H_op, k=min(K_LEADING, N - 2), M=A_block, Minv=Minv_op,
                           which="LM", maxiter=300, tol=1e-6, v0=v0)
    finally:
        h5c.close()
    idx = np.argsort(-vals); vals = vals[idx]; vecs = vecs[:, idx]
    n_pos = int(np.sum(vals > 0))
    n_eff = float(np.sum(vals[:n_pos] / (1.0 + vals[:n_pos])))
    print(f"\nLeading eigenvalues: {vals[:8]}\nPositive: {n_pos}/{len(vals)}, n_eff={n_eff:.1f}", flush=True)

    rd = DATA_DIR / "results"
    with open(rd / f"eigenvalues_ssa{fsuf}.txt", "w") as f:
        for i, lam in enumerate(vals[:n_pos]):
            f.write(f"{i} {lam:.16e}\n")
    np.savez(str(rd / f"eigenvectors_ssa{fsuf}.npz"), vecs=vecs[:, :n_pos], eigenvalues=vals[:n_pos], n_dof=n)
    QQ = MixedFunctionSpace([Q, Q])
    with CheckpointFile(str(DATA_DIR / "mesh" / f"eigendec_ssa{fsuf}.h5"), "w") as chk:
        chk.save_mesh(mesh)
        for i in range(min(100, n_pos)):
            md = Function(QQ, name=f"mode_{i:04d}")
            md.sub(0).dat.data[:] = vecs[:n, i]; md.sub(1).dat.data[:] = vecs[n:, i]
            chk.save_function(md)
    with open(rd / f"prior_params_ssa{fsuf}.json", "w") as f:
        json.dump({"delta": DELTA_PRIOR, "gamma_A": gamma_A_val, "gamma_C": gamma_C_val,
                   "area": area_val, "ELL": ELL, "K_LEADING": K_LEADING, "n_eff": n_eff,
                   "model": f"SSA_{args.friction}", "normalization": "recinos_no_1_over_N"}, f, indent=2)
    print(f"Saved results/eigenvectors_ssa{fsuf}.npz, eigenvalues_ssa{fsuf}.txt, "
          f"prior_params_ssa{fsuf}.json, mesh/eigendec_ssa{fsuf}.h5", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
