r"""Reduce the raw ApRES, velocity and BedMachine data for the GRL manuscript figures.

``manuscript_figures.py`` only plots; everything that reads a large or raw file
happens here, once, and lands in results/manuscript/:

  apres.npz   per station (the 91 GHOST sites of data/apres_legendre_fits.h5):
              position, distance along the line, ice thickness, BedMachine surface and
              bed, the vertical velocity profile w(depth) and its sigma, Legendre fits of
              order 1-4 with F-test p-values, the shear share, depth-averaged and
              100-400 m eps_zz, the surface eps_zz = -div(u) from MEaSUREs, an eps_zz(depth)
              profile for the section, and the 2022-23 return SNR(depth)
  maps.npz    MEaSUREs speed and the BedMachine mask over the regional map (Fig. 1) and
              BedMachine bed over the ApRES line (Fig. 5)

    python scripts/manuscript_prepare.py

Choices (stated in the captions):
  * Legendre basis: orthonormal shifted Legendre polynomials sqrt(2n+1) P_n(2 zeta - 1),
    zeta = depth / H_ApRES, fitted to w over 100-1100 m (``shear_analysis.py``) by weighted
    least squares; sigma floored at 1% of the profile's range, as there.
  * F-test: order n against n-1 at p < 0.05; the best order is the highest significant one.
  * Shear share A_s / (A_s + A_p): A_p = |c_1| (the linear, plug-flow part of w),
    A_s = sqrt(sum_{n>=2} c_n^2) of the best-order fit (0 where the best order is 1).
  * eps_zz(depth) for the section: weighted linear fits of w in 60 m windows every 20 m,
    dropped where the mean sigma_w of the window exceeds 1.5 mm/yr.
  * Surface eps_zz = -(du/dx + dv/dy) of the MEaSUREs 450 m mosaic after a Gaussian filter of
    2 km (about one ice thickness), sampled at each station.
  * Return SNR: power of the 2022-23 burst (first attenuator setting) in 2 m bins, relative to
    the median power 300-600 m below the ApRES bed.
"""
from pathlib import Path

import h5py
import netCDF4 as nc
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import f as f_dist

from apres_dat import power_profile

DATA = Path(__file__).resolve().parent.parent
OUT = DATA / "results" / "manuscript"
APRES_FILE = DATA / "data" / "apres_legendre_fits.h5"
RAW_2223 = DATA / "data" / "2022-2023"
VELOCITY = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/velocity/"
                "antarctica_ice_velocity_450m_v2.nc")
BEDMACHINE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/bedmachine/"
                  "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")

FIT = (100.0, 1100.0)           # Legendre fit range below the surface (shear_analysis.py)
NEAR = (100.0, 400.0)           # near-surface eps_zz window (Fig. 4)
MAX_ORDER = 4
P_SIG = 0.05
WIN, STEP, SIG_MAX = 60.0, 20.0, 1.5e-3       # eps_zz section windows; sigma_w cut (m/yr)
SMOOTH = 2000.0                 # velocity filter for the surface strain rate (m)
BIN = 2.0                       # return power bins (m)
R_MAX = 3200.0                  # deep enough for the noise window below the thickest ice
REGION = (-1633e3, -1263e3, -569e3, -276e3)   # Fig. 1 map: the whole model domain + 8 km
LINE_PAD = 6e3                  # Fig. 5 map: margin around the stations


def basis(n, zeta):
    from numpy.polynomial.legendre import legval
    c = np.zeros(n + 1)
    c[n] = 1.0
    return np.sqrt(2 * n + 1) * legval(2 * zeta - 1, c)


def wls(zeta, y, sig, n_modes):
    r"""Weighted least squares on the first ``n_modes`` basis functions: (coeffs, chi2, dof)."""
    G = np.column_stack([basis(k, zeta) for k in range(n_modes)])
    Gw, yw = G / sig[:, None], y / sig
    c = np.linalg.lstsq(Gw, yw, rcond=None)[0]
    return c, float(np.sum((yw - Gw @ c) ** 2)), len(y) - n_modes


def legendre(depth, w, sig, H):
    r"""Fits of order 1..MAX_ORDER, F-test p-values of each order against the one below."""
    k = (depth >= FIT[0]) & (depth <= min(FIT[1], 0.98 * H)) & np.isfinite(w) & np.isfinite(sig)
    z, y = depth[k] / H, w[k]
    s = np.maximum(sig[k], max(0.01 * np.ptp(y), 1e-4))
    coef = np.full((MAX_ORDER, MAX_ORDER + 1), np.nan)
    chi, dof = {}, {}
    for o in range(1, MAX_ORDER + 1):
        c, chi[o], dof[o] = wls(z, y, s, o + 1)
        coef[o - 1, :o + 1] = c
    p = np.full(MAX_ORDER - 1, np.nan)
    for o in range(2, MAX_ORDER + 1):
        F = (chi[o - 1] - chi[o]) / (chi[o] / dof[o])
        p[o - 2] = 1 - f_dist.cdf(F, 1, dof[o])
    sig_orders = [o for o in range(2, MAX_ORDER + 1) if p[o - 2] < P_SIG]
    best = max(sig_orders, default=1)
    c = coef[best - 1]
    a_p = abs(c[1])
    a_s = float(np.sqrt(np.nansum(c[2:best + 1] ** 2))) if best >= 2 else 0.0
    return dict(coef=coef, p=p, best=best, share=a_s / (a_s + a_p), n_fit=int(k.sum()))


def slope(depth, w, sig, lo, hi):
    r"""Weighted linear dw/d(depth) over [lo, hi] and its standard error (residual-scaled)."""
    k = (depth >= lo) & (depth <= hi) & np.isfinite(w) & np.isfinite(sig)
    if k.sum() < 5:
        return np.nan, np.nan
    x, y, s = depth[k], w[k], np.maximum(sig[k], 1e-6)
    G = np.column_stack([np.ones_like(x), x - x.mean()]) / s[:, None]
    c, *_ = np.linalg.lstsq(G, y / s, rcond=None)
    r = y / s - G @ c
    cov = np.linalg.inv(G.T @ G) * max(np.sum(r ** 2) / (k.sum() - 2), 1.0)
    return float(c[1]), float(np.sqrt(cov[1, 1]))


def eps_section(depth, w, sig):
    r"""eps_zz(depth) from windowed linear fits of w, NaN where sigma_w is too large."""
    centres = np.arange(FIT[0] + WIN / 2, depth.max() - WIN / 2 + 1e-6, STEP)
    e = np.full(centres.size, np.nan)
    for i, zc in enumerate(centres):
        k = (depth >= zc - WIN / 2) & (depth <= zc + WIN / 2) & np.isfinite(w)
        if k.sum() >= 5 and np.mean(sig[k]) <= SIG_MAX:
            e[i] = slope(depth, w, sig, zc - WIN / 2, zc + WIN / 2)[0]
    return centres, e


def along_line(xy):
    r"""Distance along the principal axis of the stations (km, increasing east) and the
    perpendicular offset (km)."""
    c = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - c)
    s, t = (xy - c) @ vt[0], (xy - c) @ vt[1]
    if np.corrcoef(s, xy[:, 0])[0, 1] < 0:
        s = -s
    return (s - s.min()) / 1e3, np.abs(t - np.median(t)) / 1e3


def raw_burst(name):
    r"""The 2022-23 burst of a station: the plain (unsuffixed) acquisition if there are several."""
    files = sorted(p for p in (RAW_2223 / name).glob("*") if p.suffix.lower() == ".dat"
                   and not p.name.startswith("._"))
    plain = [p for p in files if not any(t in p.stem for t in ("_a", "_g"))]
    return (plain or files)[0] if files else None


def grid(path, names, box, stride=1):
    r"""Variables of a polar-stereographic grid inside ``box``, rows ordered by increasing y."""
    with nc.Dataset(str(path)) as ds:
        x, y = ds["x"][:].astype(float), ds["y"][:].astype(float)
        ix = np.flatnonzero((x >= box[0]) & (x <= box[1]))[::stride]
        iy = np.flatnonzero((y >= box[2]) & (y <= box[3]))[::stride]
        sl = (slice(iy.min(), iy.max() + 1, stride), slice(ix.min(), ix.max() + 1, stride))
        out = {k: np.ma.filled(ds[k][sl].astype(float), np.nan) for k in names}
        gx, gy = x[sl[1]], y[sl[0]]
    if gy[0] > gy[-1]:
        gy = gy[::-1]
        out = {k: v[::-1] for k, v in out.items()}
    return gx, gy, out


def sample(gx, gy, v, pts):
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator((gy, gx), v, bounds_error=False)(pts[:, ::-1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(APRES_FILE), "r") as f:
        names = [n.decode() for n in f["summary/names"][...]]
        xy = np.column_stack([f["summary/x"][...], f["summary/y"][...]])
        prof = {}
        for nm in names:
            g = f[f"sites/{nm}"]
            prof[nm] = dict(depth=g["depth"][...], w=g["w"][...], sig=g["w_sigma"][...],
                            H=float(g.attrs["H_used"]))
    s_km, off_km = along_line(xy)
    n, L = len(names), max(p["depth"].size for p in prof.values())
    pad = lambda: np.full((n, L), np.nan)       # noqa: E731
    depth, w, sig = pad(), pad(), pad()
    coef = np.full((n, MAX_ORDER, MAX_ORDER + 1), np.nan)
    pval = np.full((n, MAX_ORDER - 1), np.nan)
    best, share, n_fit = np.zeros(n, int), np.zeros(n), np.zeros(n, int)
    eps_avg, eps_avg_s, eps_near, eps_near_s = (np.zeros(n) for _ in range(4))
    zc = np.arange(FIT[0] + WIN / 2, 2600.0, STEP)
    eps_sec = np.full((n, zc.size), np.nan)
    rr = np.arange(0.0, R_MAX, BIN) + BIN / 2
    snr = np.full((n, rr.size), np.nan)
    H = np.array([prof[nm]["H"] for nm in names])
    for i, nm in enumerate(names):
        p = prof[nm]
        m = p["depth"].size
        depth[i, :m], w[i, :m], sig[i, :m] = p["depth"], p["w"], p["sig"]
        lg = legendre(p["depth"], p["w"], p["sig"], p["H"])
        coef[i], pval[i], best[i], share[i], n_fit[i] = (lg["coef"], lg["p"], lg["best"],
                                                         lg["share"], lg["n_fit"])
        eps_avg[i], eps_avg_s[i] = slope(p["depth"], p["w"], p["sig"], FIT[0],
                                         min(FIT[1], 0.98 * p["H"]))
        eps_near[i], eps_near_s[i] = slope(p["depth"], p["w"], p["sig"], *NEAR)
        c, e = eps_section(p["depth"], p["w"], p["sig"])
        eps_sec[i, :c.size] = e
        burst = raw_burst(nm)
        if burst is not None:
            r, db = power_profile(burst, max_range=R_MAX + BIN)
            idx = np.minimum((r / BIN).astype(int), rr.size)
            pw = 10 ** (db / 10)
            prof_bin = np.bincount(idx, weights=pw, minlength=rr.size + 1)[:rr.size]
            cnt = np.bincount(idx, minlength=rr.size + 1)[:rr.size]
            pdb = 10 * np.log10(prof_bin / np.maximum(cnt, 1))
            noise = (rr > p["H"] + 300) & (rr < p["H"] + 600)
            snr[i] = pdb - np.median(pdb[noise])
        print(f"{nm}: best order {best[i]}, share {share[i]:.2f}, "
              f"eps {eps_avg[i] * 1e3:+.2f}e-3/yr, burst {burst.name if burst else '-'}",
              flush=True)

    # Surface strain rate from MEaSUREs, sampled at the stations.
    lo, hi = xy.min(axis=0) - 20e3, xy.max(axis=0) + 20e3
    gx, gy, v = grid(VELOCITY, ["VX", "VY"], (lo[0], hi[0], lo[1], hi[1]))
    sm = SMOOTH / abs(gx[1] - gx[0])
    vx, vy = (gaussian_filter(np.nan_to_num(v[k]), sm) for k in ("VX", "VY"))
    div = np.gradient(vx, gx, axis=1) + np.gradient(vy, gy, axis=0)
    eps_surf = -sample(gx, gy, div, xy)

    # BedMachine at the stations.
    bx, by, bm = grid(BEDMACHINE, ["surface", "bed"], (lo[0], hi[0], lo[1], hi[1]))
    surf, bed = sample(bx, by, bm["surface"], xy), sample(bx, by, bm["bed"], xy)
    np.savez_compressed(
        str(OUT / "apres.npz"), name=np.array(names), xy=xy, s_km=s_km, off_km=off_km, H=H,
        surface=surf, bed=bed, depth=depth, w=w, w_sigma=sig, coef=coef, p=pval, best=best,
        share=share, n_fit=n_fit, eps_avg=eps_avg, eps_avg_sigma=eps_avg_s, eps_near=eps_near,
        eps_near_sigma=eps_near_s, eps_surf=eps_surf, sec_depth=zc, eps_sec=eps_sec,
        snr_range=rr, snr=snr, fit=np.array(FIT), near=np.array(NEAR), smooth=SMOOTH)
    print(f"Saved {OUT / 'apres.npz'}", flush=True)

    # Map backgrounds.
    mx, my, mv = grid(VELOCITY, ["VX", "VY"], REGION)
    _, _, mb = grid(BEDMACHINE, ["mask"], REGION)
    kx, ky, _ = grid(BEDMACHINE, ["mask"], REGION)
    box5 = (xy[:, 0].min() - LINE_PAD, xy[:, 0].max() + LINE_PAD,
            xy[:, 1].min() - LINE_PAD, xy[:, 1].max() + LINE_PAD)
    fx, fy, fb = grid(BEDMACHINE, ["bed"], box5)
    np.savez_compressed(str(OUT / "maps.npz"), speed_x=mx, speed_y=my,
                        speed=np.hypot(mv["VX"], mv["VY"]), mask_x=kx, mask_y=ky,
                        mask=mb["mask"], bed_x=fx, bed_y=fy, bed=fb["bed"])
    print(f"Saved {OUT / 'maps.npz'}", flush=True)


if __name__ == "__main__":
    main()
