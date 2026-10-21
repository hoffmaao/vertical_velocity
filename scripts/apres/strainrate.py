"""Vertical strain rate with depth from a repeat ApRES velocity profile.

The vertical strain rate is the depth derivative of the vertical velocity,

    eps_zz(z) = dw/dz,

and it, rather than w itself, is the quantity that enters the flow model:
w depends on the reference segment the profile was tied to, whereas its
gradient does not.

Differentiating a measured profile amplifies noise, so the derivative is taken
as the slope of a weighted local linear fit over a depth window rather than by
differencing adjacent samples.  Two properties of these data motivate that
choice:

* fine-alignment segments are 6 m wide and stepped every 2 m, so neighbouring
  samples share two thirds of their data.  Central differences between them
  difference largely the same measurement and return mostly noise.
* the formal phase uncertainty on w is of order 1e-4 m/yr, four orders of
  magnitude below the signal, and badly understates the real scatter.  The
  slope uncertainty is therefore taken from the *observed* residuals about the
  local fit, not propagated from the phase error, and is inflated for the
  oversampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["StrainRateProfile", "strain_rate_profile"]

# Fine-alignment windows are 6 m wide stepped every 2 m.
OVERSAMPLING = 3.0


@dataclass
class StrainRateProfile:
    """Vertical strain rate and its uncertainty on the sampled depth axis."""

    depth_m: np.ndarray
    strain_rate_yr: np.ndarray
    strain_rate_err_yr: np.ndarray
    n_used: np.ndarray
    window_m: float


def strain_rate_profile(
    z: np.ndarray,
    w: np.ndarray,
    *,
    sigma_w: Optional[np.ndarray] = None,
    window_m: float = 60.0,
    min_points: int = 8,
) -> StrainRateProfile:
    """Local-slope estimate of ``dw/dz`` at every sampled depth.

    Parameters
    ----------
    z, w
        Depth below the surface (m) and vertical velocity (m/yr).
    sigma_w
        Optional per-sample weights.  Used for the slope estimate only; the
        reported uncertainty always comes from the residual scatter.
    window_m
        Full width of the depth window used for each local fit.  60 m spans
        ten independent fine-alignment segments, which is enough to stabilise
        the slope without smoothing away the curvature that vertical shear
        produces.
    min_points
        Windows with fewer samples than this return NaN rather than a slope
        fitted to almost nothing.

    Returns
    -------
    StrainRateProfile
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    good = np.isfinite(z) & np.isfinite(w)
    z, w = z[good], w[good]
    weights = (
        np.ones_like(w)
        if sigma_w is None
        else 1.0 / np.clip(np.asarray(sigma_w, dtype=float)[good], 1e-12, None)
    )

    order = np.argsort(z)
    z, w, weights = z[order], w[order], weights[order]

    n = z.size
    slope = np.full(n, np.nan)
    slope_err = np.full(n, np.nan)
    used = np.zeros(n, dtype=int)
    half = 0.5 * float(window_m)

    # Sample spacing sets how many of the points in a window are independent.
    spacing = float(np.median(np.diff(z))) if n > 1 else np.nan

    lo = np.searchsorted(z, z - half, side="left")
    hi = np.searchsorted(z, z + half, side="right")

    for i in range(n):
        sl = slice(lo[i], hi[i])
        zz, ww, vv = z[sl], w[sl], weights[sl]
        m = zz.size
        if m < min_points:
            continue

        # Weighted least squares for the slope of w against z.
        sw = vv.sum()
        zbar = float((vv * zz).sum() / sw)
        wbar = float((vv * ww).sum() / sw)
        dz = zz - zbar
        szz = float((vv * dz * dz).sum())
        if szz <= 0:
            continue
        b = float((vv * dz * (ww - wbar)).sum() / szz)

        resid = ww - (wbar + b * dz)
        # Effective sample size: the window holds m samples but only about
        # m/OVERSAMPLING independent ones.
        n_eff = max(m / OVERSAMPLING, 3.0)
        dof = max(n_eff - 2.0, 1.0)
        var_resid = float((vv * resid * resid).sum() / sw) * m / max(m - 2, 1)
        # Standard error of the slope, with the window's leverage recomputed
        # at the effective sample size.
        szz_eff = szz * (n_eff / m)
        slope[i] = b
        slope_err[i] = float(np.sqrt(max(var_resid * sw / m, 0.0) / max(szz_eff / m * n_eff, 1e-30)))
        slope_err[i] *= np.sqrt(max(dof, 1.0) / max(dof, 1.0))
        used[i] = m

    _ = spacing
    return StrainRateProfile(
        depth_m=z,
        strain_rate_yr=slope,
        strain_rate_err_yr=slope_err,
        n_used=used,
        window_m=float(window_m),
    )
