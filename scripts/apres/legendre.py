"""Shifted-Legendre decomposition of a vertical-velocity profile.

A vertical-velocity profile w(z) measured by repeat ApRES is decomposed on the
shifted Legendre polynomials of the normalised depth coordinate
zeta = z / H in [0, 1].  The basis is orthogonal on [0, 1], so each
coefficient is an independent description of one mode of the deformation:

  * degree 0  - uniform vertical motion of the whole column,
  * degree 1  - a depth-uniform vertical strain rate, the signature of
                plug flow, in which the column thins or thickens at a rate that
                does not vary with depth,
  * degree >=2 - curvature in w(z), which under incompressibility requires a
                depth-varying horizontal velocity, i.e. vertical shear.

The ratio of the power in degrees >=2 to the power in degrees 0-1 is therefore
a direct, basis-independent measure of how far the column departs from plug
flow.  Because the basis is orthonormal, that ratio can be read straight off
the coefficients without refitting.

Whether the departure is *significant* is decided by a joint test of the
degree->=2 coefficients against zero.

That test has to account for the fact that neighbouring samples in a repeat
ApRES profile are not independent.  Fine-alignment windows are 6 m wide and
stepped every 2 m, so adjacent samples already share two thirds of their data,
and the displacement of the ice varies smoothly with depth on top of that.
Treating the few hundred samples in a profile as independent makes every
conceivable departure from plug flow overwhelmingly significant: the nested
F-test on the raw sample count rejects plug flow at every site in this survey,
which makes it useless as a discriminator.

The covariance of the coefficients is therefore inflated by the integrated
autocorrelation time of the fit residuals, estimated from the residuals
themselves.  On this survey that time is about 9 samples, so a profile of 300
samples carries roughly 35 independent ones, and the test starts to
distinguish sites instead of rejecting all of them.  The nested F-test is still
reported for continuity, but ``shear_significant`` is the one to use.

Fitting window
--------------
The basis is orthogonal on the interval it is defined over, and *only* there.
A repeat ApRES profile never spans the whole column: the return falls into the
noise somewhere above the bed, and on this survey the deepest tracked segment
ranges from 0.43 to 1.14 of the ice thickness between sites.  Fitting each site
over whatever range it happens to reach, against a basis orthonormal on
[0, 1], destroys both the orthogonality and any comparison between sites,
because a coefficient then depends on how deep that particular acquisition
saw.

``fit_profile`` therefore takes an explicit ``window`` in normalised depth and
rescales the basis onto it, so that every site is described on the same
interval by a basis that is genuinely orthonormal there.  The cost is that the
decomposition describes only the part of the column that every site resolves,
which is the honest limit of the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

__all__ = [
    "LegendreFit",
    "autocorrelation_time",
    "shifted_legendre",
    "design_matrix",
    "fit_profile",
]


def autocorrelation_time(residual: np.ndarray) -> float:
    """Integrated autocorrelation time of a residual series, in samples.

    Uses the standard self-consistent (Sokal) truncation: the sum is
    accumulated until the window reaches six times the running estimate, which
    stops it before the noisy tail of the autocorrelation function dominates.
    Returns 1.0 for series that are too short or constant, so that an
    uninformative profile is never credited with extra independence.
    """
    r = np.asarray(residual, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 8:
        return 1.0
    r = r - r.mean()
    if not np.any(r):
        return 1.0
    spectrum = np.fft.rfft(r, 2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n].real
    if acf[0] <= 0:
        return 1.0
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]
        if k >= 6.0 * tau:
            break
    return float(min(max(tau, 1.0), n / 4.0))


def shifted_legendre(degree: int, zeta: np.ndarray) -> np.ndarray:
    """Shifted Legendre polynomial :math:`L_n` on the interval [0, 1].

    Uses the Bonnet recurrence on the mapped coordinate ``x = 2*zeta - 1`` so
    that arbitrary degrees are available without hard-coding coefficients.
    """
    x = 2.0 * np.asarray(zeta, dtype=float) - 1.0
    if degree == 0:
        return np.ones_like(x)
    p_prev, p = np.ones_like(x), x
    for n in range(1, degree):
        p_prev, p = p, ((2 * n + 1) * x * p - n * p_prev) / (n + 1)
    return p


def _norm_weight(degree: int) -> float:
    """Weight that makes :math:`L_n` orthonormal on [0, 1].

    :math:`\\int_0^1 L_n^2 \\, d\\zeta = 1/(2n+1)`, so the weight is
    :math:`\\sqrt{2n+1}`.
    """
    return float(np.sqrt(2 * degree + 1))


def design_matrix(
    zeta: np.ndarray, degree: int, window: tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """Orthonormal shifted-Legendre design matrix on ``window``.

    ``window`` is an interval of the normalised depth coordinate.  The
    coordinate is mapped affinely onto [0, 1] before the basis is evaluated,
    which makes the columns orthonormal over the window itself rather than over
    the full column.
    """
    a, b = window
    if not b > a:
        raise ValueError(f"window must be increasing, got {window}")
    u = (np.asarray(zeta, dtype=float) - a) / (b - a)
    return np.column_stack(
        [_norm_weight(n) * shifted_legendre(n, u) for n in range(degree + 1)]
    )


@dataclass
class LegendreFit:
    """Result of decomposing one vertical-velocity profile."""

    degree: int
    coefficients: np.ndarray          # c_0 .. c_degree, orthonormal basis
    coefficient_errors: np.ndarray
    covariance: np.ndarray
    zeta: np.ndarray
    w_observed: np.ndarray
    w_model: np.ndarray
    residual_rms: float
    n_samples: int

    # Partition of the profile
    shear_power: float                # sum of c_n^2 for n >= 2
    plug_power: float                 # c_1^2
    shear_fraction: float             # shear / (shear + plug)
    shear_to_plug: float              # shear / plug

    # Depth-uniform strain rate implied by the degree-1 term (1/yr)
    strain_rate_linear: float
    strain_rate_linear_err: float

    # Nested F-test of degree-1 against the full fit, on the raw sample count.
    # Retained for continuity; it rejects plug flow essentially everywhere.
    f_statistic: float
    p_value: float
    significant: bool

    # Joint test of the degree->=2 coefficients against zero, with the
    # coefficient covariance inflated by the residual autocorrelation time.
    residual_autocorr_time: float
    shear_chi2: float
    shear_p_value: float
    shear_significant: bool
    shear_amplitude: float
    shear_amplitude_err: float
    plug_amplitude: float

    thickness_m: float = float("nan")
    extras: dict = field(default_factory=dict)


def fit_profile(
    z: np.ndarray,
    w: np.ndarray,
    *,
    thickness_m: float,
    sigma_w: Optional[np.ndarray] = None,
    degree: int = 4,
    window: tuple[float, float] = (0.0, 1.0),
    alpha: float = 0.05,
    min_samples: int = 20,
) -> Optional[LegendreFit]:
    """Fit ``w(z)`` on the orthonormal shifted-Legendre basis.

    Parameters
    ----------
    z, w
        Depth below the surface (m, positive down) and vertical velocity
        (m/yr, positive upward relative to the shallowest tracked segment).
    thickness_m
        Ice thickness used to normalise depth to ``zeta = z / H``.
    sigma_w
        Per-sample uncertainty; when given the fit is weighted by
        ``1/sigma_w**2`` and the reported coefficient errors are propagated
        from it.
    degree
        Highest polynomial degree retained.  Degree 4 resolves the curvature
        that vertical shear produces without fitting radar noise.
    window
        Interval of normalised depth over which the basis is orthonormalised
        and the fit performed.  Samples outside it are discarded.  Passing the
        same window for every site is what makes coefficients comparable
        between sites.
    alpha
        Significance level of the nested F-test.

    Returns
    -------
    LegendreFit or None
        ``None`` when fewer than ``min_samples`` finite samples are available.
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    good = np.isfinite(z) & np.isfinite(w)
    if sigma_w is not None:
        sigma_w = np.asarray(sigma_w, dtype=float)
        good &= np.isfinite(sigma_w) & (sigma_w > 0)
    if good.sum() < max(min_samples, degree + 2):
        return None

    z, w = z[good], w[good]
    zeta = z / float(thickness_m)
    sig = None if sigma_w is None else sigma_w[good]

    # Restrict to the common window before fitting.
    a, b = window
    inside = (zeta >= a) & (zeta <= b)
    if inside.sum() < max(min_samples, degree + 2):
        return None
    z, w, zeta = z[inside], w[inside], zeta[inside]
    if sig is not None:
        sig = sig[inside]

    weights = np.ones_like(w) if sig is None else 1.0 / np.clip(sig, 1e-12, None)

    def _solve(deg: int):
        g = design_matrix(zeta, deg, window)
        gw = g * weights[:, None]
        coef, *_ = np.linalg.lstsq(gw, w * weights, rcond=None)
        model = g @ coef
        rss = float(np.sum(((w - model) * weights) ** 2))
        return g, coef, model, rss

    g_full, coef, model, rss_full = _solve(degree)
    _ = g_full
    _, _, _, rss_linear = _solve(1)

    n = int(w.size)
    dof = n - (degree + 1)
    if dof <= 0:
        return None

    # Coefficient covariance from the weighted normal equations, scaled by the
    # reduced chi-square so that the errors remain meaningful when sigma_w is
    # an incomplete description of the noise.
    gw = g_full * weights[:, None]
    try:
        cov = np.linalg.inv(gw.T @ gw) * (rss_full / dof)
    except np.linalg.LinAlgError:
        return None
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    # Nested F-test: degree-1 (plug flow) against the full fit.
    d_params = degree - 1
    f_stat = (
        ((rss_linear - rss_full) / d_params) / (rss_full / dof)
        if rss_full > 0 and d_params > 0
        else np.inf
    )
    p_value = float(stats.f.sf(f_stat, d_params, dof)) if np.isfinite(f_stat) else 0.0

    shear_power = float(np.sum(coef[2:] ** 2))
    plug_power = float(coef[1] ** 2)
    total = shear_power + plug_power

    # Joint significance of the shear terms, against an effective sample size.
    residual = w - model
    tau = autocorrelation_time(residual)
    idx = np.arange(2, coef.size)
    shear_chi2 = float("nan")
    shear_p = 1.0
    shear_err = float("nan")
    if idx.size:
        sub = cov[np.ix_(idx, idx)] * tau
        shear_err = float(np.sqrt(np.sum(np.diag(sub))))
        try:
            shear_chi2 = float(coef[idx] @ np.linalg.solve(sub, coef[idx]))
            shear_p = float(stats.chi2.sf(shear_chi2, idx.size))
        except np.linalg.LinAlgError:
            pass

    # dw/dz from the degree-1 term alone.  With u = (zeta - a)/(b - a) the
    # term is c1*sqrt(3)*(2u - 1), so dw/dz = 2*sqrt(3)*c1 / (H*(b - a)).
    scale = 2.0 * np.sqrt(3.0) / (float(thickness_m) * (b - a))

    return LegendreFit(
        degree=degree,
        coefficients=coef,
        coefficient_errors=err,
        covariance=cov,
        zeta=zeta,
        w_observed=w,
        w_model=model,
        residual_rms=float(np.sqrt(np.mean((w - model) ** 2))),
        n_samples=n,
        shear_power=shear_power,
        plug_power=plug_power,
        shear_fraction=float(shear_power / total) if total > 0 else float("nan"),
        shear_to_plug=float(shear_power / plug_power) if plug_power > 0 else float("inf"),
        strain_rate_linear=float(coef[1] * scale),
        strain_rate_linear_err=float(err[1] * scale),
        f_statistic=float(f_stat),
        p_value=p_value,
        significant=bool(p_value < alpha),
        residual_autocorr_time=float(tau),
        shear_chi2=shear_chi2,
        shear_p_value=shear_p,
        shear_significant=bool(shear_p < alpha),
        shear_amplitude=float(np.sqrt(shear_power)),
        shear_amplitude_err=shear_err,
        plug_amplitude=float(abs(coef[1])),
        thickness_m=float(thickness_m),
    )
