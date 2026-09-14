r"""Sample synthetic observations from the MISMIP+ Hybrid truth at a given density.

The density/spacing Δ is the OSSE sweep knob. Produces:
  • surface velocity (u_x,u_y) at ζ≈1 (the satellite observable — sampled from the
    3D field, NOT the depth-average) + surface elevation, on a grid at spacing Δ
  • ApRES-style ε_zz(ζ) DEPTH PROFILES at sparser points (+ noise) — the shear signal
    that distinguishes Hybrid (depth-varying) from SSA (depth-constant).

All per-point samples are aligned to the original grid order via `input_ordering`,
then masked to grounded ice (h>H_OBS_MIN). Writes results/mismip_obs_d{Δ}km.npz.

Run:  python mismip_observe.py --spacing 20 --apres-spacing 40
"""
import argparse
import numpy as np
import firedrake as fd
from firedrake import (CheckpointFile, FunctionSpace, VectorFunctionSpace,
                       VertexOnlyMesh, Function)
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
Lx, Ly = 640e3, 80e3
H_OBS_MIN = 20.0    # only place obs where ice thickness exceeds this
ZSURF = 0.999       # ζ for "surface" sampling (avoid the exact top boundary)


def grid(spacing_km):
    d = spacing_km * 1e3
    xs = np.arange(d, Lx, d)
    ys = np.arange(d, Ly, d)
    GX, GY = np.meshgrid(xs, ys)
    return np.column_stack([GX.ravel(), GY.ravel()])


def sample_input_order(mesh, pts, fns):
    """Interpolate scalar/vector `fns` at `pts`, return values in INPUT order
    (length = len(pts); points outside the mesh read ~0 and are masked later)."""
    vom = VertexOnlyMesh(mesh, pts, missing_points_behaviour="warn")
    out = []
    for fn in fns:
        shp = fn.ufl_shape
        if shp == ():
            P = FunctionSpace(vom, "DG", 0)
            Pin = FunctionSpace(vom.input_ordering, "DG", 0)
        else:
            P = VectorFunctionSpace(vom, "DG", 0, dim=shp[0])
            Pin = VectorFunctionSpace(vom.input_ordering, "DG", 0, dim=shp[0])
        f_vom = Function(P).interpolate(fn)
        out.append(Function(Pin).interpolate(f_vom).dat.data_ro.copy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=20.0, help="surface vel/elev grid spacing (km)")
    ap.add_argument("--apres-spacing", type=float, default=40.0, help="ApRES point spacing (km)")
    ap.add_argument("--ndepth", type=int, default=5, help="ApRES ε_zz depth (ζ) samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-vel", type=float, default=5.0, help="velocity noise σ (m/yr)")
    ap.add_argument("--sigma-elev", type=float, default=2.0, help="surface-elevation noise σ (m)")
    ap.add_argument("--sigma-eps", type=float, default=2e-3, help="ε_zz noise σ (1/yr)")
    ap.add_argument("--sigma-w", type=float, default=5e-4, help="vertical-velocity noise σ (m/yr, ~ApRES precision)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    with CheckpointFile(str(DATA / "mesh" / "mismip_truth.h5"), "r") as c:
        m2 = c.load_mesh()
        s = c.load_function(m2, "surface")
        h = c.load_function(m2, "thickness")
    with CheckpointFile(str(DATA / "mesh" / "mismip_truth_3d.h5"), "r") as c:
        m3 = c.load_mesh("firedrake_default_extruded")   # functions live on the extruded mesh, not the base
        u3 = c.load_function(m3, "velocity_3d")
        w3 = c.load_function(m3, "w_3d")                 # vertical velocity (primary ApRES observable)
        ez = c.load_function(m3, "eps_zz_3d")

    # ── surface velocity (3D @ ζ≈1) + elevation + thickness, all in grid order ──
    pts = grid(args.spacing)
    xyz_surf = np.column_stack([pts, np.full(len(pts), ZSURF)])
    (uvec,) = sample_input_order(m3, xyz_surf, [u3])          # (N,2) true surface velocity
    h_at, s_at = sample_input_order(m2, pts, [h, s])          # (N,), (N,)
    keep = h_at > H_OBS_MIN
    vel_pts = pts[keep]
    spd = np.hypot(uvec[keep, 0], uvec[keep, 1])
    vel_obs = uvec[keep] + rng.normal(0, args.sigma_vel, uvec[keep].shape)
    elev_obs = s_at[keep] + rng.normal(0, args.sigma_elev, s_at[keep].shape)
    print(f"surface obs: {keep.sum()} points at Δ={args.spacing}km "
          f"(surface speed {spd.min():.0f}–{spd.max():.0f} m/yr)", flush=True)

    # ── ApRES ε_zz depth profiles (sparser columns × ζ levels) ──
    apts = grid(args.apres_spacing)
    (ah,) = sample_input_order(m2, apts, [h])
    apts = apts[ah > H_OBS_MIN]
    zetas = np.linspace(0.05, 0.95, args.ndepth)
    xyz = np.array([[ax, ay, z] for (ax, ay) in apts for z in zetas])
    col_id = np.repeat(np.arange(len(apts)), args.ndepth)     # which column each point belongs to
    zeta_of = np.tile(zetas, len(apts))
    eps_at, w_at = sample_input_order(m3, xyz, [ez, w3])
    eps_obs = eps_at + rng.normal(0, args.sigma_eps, eps_at.shape)
    w_obs = w_at + rng.normal(0, args.sigma_w, w_at.shape)
    print(f"ApRES obs: {len(apts)} columns × {args.ndepth} depths = {len(xyz)} points at Δ={args.apres_spacing}km "
          f"(w {w_at.min():.2e}–{w_at.max():.2e} m/yr, ε_zz {eps_at.min():.2e}–{eps_at.max():.2e})", flush=True)

    out = DATA / "results" / f"mismip_obs_d{args.spacing:g}km.npz"
    np.savez(str(out),
             vel_pts=vel_pts, vel_obs=vel_obs, sigma_vel=args.sigma_vel,
             elev_pts=vel_pts, elev_obs=elev_obs, sigma_elev=args.sigma_elev,
             apres_xyz=xyz, apres_col=col_id, apres_zeta=zeta_of,
             w_obs=w_obs, sigma_w=args.sigma_w,
             eps_obs=eps_obs, sigma_eps=args.sigma_eps,
             spacing_km=args.spacing, apres_spacing_km=args.apres_spacing, ndepth=args.ndepth)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
