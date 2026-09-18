r"""Observation loaders shared by the SSA and hybrid inversions.

Surface velocity: MEaSUREs 450 m v2 grid points inside the mesh with valid
STDX/STDY (sigma floored at 1 m/yr). ApRES: Legendre-fitted eps_zz between
DEPTH_MIN and DEPTH_MAX, ~5 points per site, as (x, y, zeta) with zeta = 1 - d/H.
Both loaders are lifted verbatim from inversion_ssa_recinos.py and
inversion_hires_apres.py so the new MAPs see the same data as the old ones.
"""
import h5py
import netCDF4 as nc
import numpy as np
from pathlib import Path
from scipy.spatial import Delaunay

DATA = Path(__file__).resolve().parent.parent
VEL_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/velocity/"
                "antarctica_ice_velocity_450m_v2.nc")
APRES_FILE = DATA / "data" / "apres_legendre_fits.h5"
DEPTH_MIN, DEPTH_MAX = 100.0, 800.0


def load_velocity(mesh_2d, step=1):
    r"""(x, y, vx, vy, sigma_x, sigma_y) arrays for grid points inside ``mesh_2d``."""
    ds = nc.Dataset(str(VEL_FILE))
    x_vel = ds.variables["x"][:]; y_vel = ds.variables["y"][:]
    coords = mesh_2d.coordinates.dat.data_ro
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    ix0 = max(0, np.searchsorted(x_vel, xmin) - 5); ix1 = min(len(x_vel), np.searchsorted(x_vel, xmax) + 5)
    iy0 = max(0, np.searchsorted(-y_vel, -ymax) - 5); iy1 = min(len(y_vel), np.searchsorted(-y_vel, -ymin) + 5)
    sl = np.s_[iy0:iy1:step, ix0:ix1:step]
    x_sub, y_sub = x_vel[ix0:ix1:step], y_vel[iy0:iy1:step]
    vx = np.nan_to_num(np.array(ds.variables["VX"][sl]).astype(float), nan=0.0)
    vy = np.nan_to_num(np.array(ds.variables["VY"][sl]).astype(float), nan=0.0)
    stdx = np.nan_to_num(np.array(ds.variables["STDX"][sl]).astype(float), nan=1e6)
    stdy = np.nan_to_num(np.array(ds.variables["STDY"][sl]).astype(float), nan=1e6)
    ds.close()
    YY, XX = np.meshgrid(y_sub, x_sub, indexing="ij")
    xs, ys = XX.ravel(), YY.ravel()
    inside = Delaunay(coords).find_simplex(np.column_stack([xs, ys])) >= 0
    has = (stdx.ravel() < 1e5) & (stdy.ravel() < 1e5) & (stdx.ravel() > 0) & (stdy.ravel() > 0)
    keep = inside & has
    return (xs[keep], ys[keep], vx.ravel()[keep], vy.ravel()[keep],
            np.maximum(stdx.ravel()[keep], 1.0), np.maximum(stdy.ravel()[keep], 1.0))


def load_apres(mesh_2d):
    r"""(points (x, y, zeta), eps_zz, sigma) for ApRES samples inside the mesh bbox."""
    pts, eps_obs, eps_sig = [], [], []
    with h5py.File(str(APRES_FILE), "r") as f:
        names = [n.decode() for n in f["summary/names"][...]]
        for name, xi, yi, Hi in zip(names, f["summary/x"][...], f["summary/y"][...],
                                    f["summary/H_apres"][...]):
            depth = f[f"sites/{name}/depth"][...]
            eps = f[f"sites/{name}/eps_zz"][...]
            sig = f[f"sites/{name}/eps_sigma"][...]
            window = ((depth >= DEPTH_MIN) & (depth <= DEPTH_MAX)
                      & np.isfinite(eps) & np.isfinite(sig) & (sig > 0))
            d_w = depth[window]
            if d_w.size == 0:
                continue
            stride = max(1, d_w.size // 5)
            d_w = d_w[::stride]
            pts.append(np.column_stack([np.full(d_w.size, xi), np.full(d_w.size, yi), 1.0 - d_w / Hi]))
            eps_obs.append(eps[window][::stride])
            eps_sig.append(sig[window][::stride])
    pts, eps_obs, eps_sig = np.vstack(pts), np.concatenate(eps_obs), np.concatenate(eps_sig)
    c = mesh_2d.coordinates.dat.data_ro
    inside = ((pts[:, 0] > c[:, 0].min()) & (pts[:, 0] < c[:, 0].max())
              & (pts[:, 1] > c[:, 1].min()) & (pts[:, 1] < c[:, 1].max()))
    return pts[inside], eps_obs[inside], eps_sig[inside]
