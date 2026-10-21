"""Single-shot scattering diagnostics for ApRES profiles.

These metrics target the "what is happening in the 10s of m above the bed"
question raised by Hills, Siegfried & Schroeder (2024, GRL). They operate on a
single PreprocessedProfile (no time series required) and are intentionally
calibration-robust: they characterise *relative* phase/amplitude structure
within depth windows, so a constant offset in absolute power does not bias
them.

Methods
-------
- pulse_abruptness:
    Sliding-window peak-power / integrated-power ratio (Dowdeswell & Evans
    2004 RoP; Christianson et al. 2012 EPSL). Sharp specular reflector → high
    abruptness; diffuse/volume scattering → low abruptness.
    Note: the Schroeder/Peters "specularity index" (coherent / incoherent
    multilook power) is implemented in `apres.preprocess` as
    `PreprocessedProfile.chirp_coherence` — the natural ApRES analog uses
    the chirps within a burst as the multilook ensemble.

- range_autocorrelation_length:
    Complex range autocorrelation of the spectrum in a depth window.
    Returns the lag (in metres) at which |R(d)| drops below a threshold (default 1/e).
    Quantifies the "thickness" of the scattering layer Hills described visually.

- basal_amplitude_pdf_fit:
    RSR-style Rice/Rayleigh fit (Grima et al. 2012, 2014) on |S| within a depth
    window. Decomposes returned power into a coherent component (ν^2) and a
    diffuse component (2σ^2) without needing absolute calibration.

- subband_reflectivity_slope:
    Fits mean dB power in a depth window across the (already-computed)
    sub-band spectra. Slope vs centre frequency is sensitive to Mie scattering
    regime and to conductivity-driven attenuation (Vaňková et al. 2021;
    MacGregor et al. 2007; Hills et al. 2024 Eqs 8, 10–12).

A convenience entry point `scattering_fingerprint()` runs all four and returns
a flat dictionary suitable for one CSV row per shot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# =============================================================================
# Helpers
# =============================================================================


def _window_indices(range_m: np.ndarray, depth_window_m: Tuple[float, float]) -> np.ndarray:
    """Indices of *range_m* falling within depth_window_m. range_m may be
    non-monotonic (when called with the phase-corrected total range axis),
    so we use a boolean mask rather than searchsorted."""
    z0, z1 = float(depth_window_m[0]), float(depth_window_m[1])
    if z1 <= z0:
        raise ValueError("depth_window_m must be (z_low, z_high) with z_high > z_low")
    return np.where((range_m >= z0) & (range_m <= z1))[0]


def _resolve_range_axis(profile_or_array, range_m_arg):
    """Allow callers to pass either a PreprocessedProfile (uses range_coarse_m)
    or an explicit range axis array. The coarse axis is monotonic; the total
    `range_m` (= coarse + fine) is not, so we always prefer coarse for
    window/spacing operations."""
    if hasattr(profile_or_array, "range_coarse_m"):
        return np.asarray(profile_or_array.range_coarse_m, dtype=float)
    if range_m_arg is None:
        raise ValueError("range axis required")
    return np.asarray(range_m_arg, dtype=float)


def _safe_db(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))


# =============================================================================
# 1. Pulse abruptness (Dowdeswell & Evans 2004; Christianson et al. 2012)
# =============================================================================
#
# Why not "depth-windowed specularity"?
#   The Schroeder/Peters specularity index = |<S>|² / <|S|²> requires an
#   ensemble of *independent looks at the same target*. In airborne SAR that
#   ensemble comes from along-track samples / Doppler bins; for a stationary
#   ApRES burst the natural ensemble is the **chirps within a burst**, which
#   is computed in `preprocess.py` and stored as `chirp_coherence(z)`. Trying
#   to compute |<S>|²/<|S|²> over a depth window is not the same thing —
#   the spectrum has a structured ~λ/2 phase rotation per range bin, so the
#   coherent average over more than ~λ bins washes out for *any* reflector,
#   specular or not. We therefore expose pulse abruptness here, which IS a
#   well-defined depth-window scattering metric.


@dataclass
class AbruptnessResult:
    """Sliding-window pulse-shape abruptness."""

    range_centers_m: np.ndarray         # (nw,) window centres (m)
    window_width_m: float
    n_bins_per_window: int
    peak_power_db: np.ndarray           # max |S|² in each window (dB)
    integrated_power_db: np.ndarray     # mean |S|² in each window (dB)
    abruptness: np.ndarray              # peak / integral ∈ [1, n_bins_per_window]
    abruptness_normalized: np.ndarray   # abruptness / n_bins_per_window ∈ (0, 1]


def pulse_abruptness(
    spec_cor: np.ndarray,
    range_m: np.ndarray,
    *,
    window_width_m: float = 4.0,
    step_m: Optional[float] = None,
    depth_window_m: Optional[Tuple[float, float]] = None,
) -> AbruptnessResult:
    """Sliding-window peak-to-mean power ratio.

    For each window of width W (in metres), compute:
        peak_power      = max |S(z)|²  in window
        integrated_power = mean |S(z)|²  in window
        abruptness      = peak_power / integrated_power  ∈ [1, n_bins_per_window]

    A single discrete reflector concentrated in one bin → abruptness ≈
    n_bins_per_window. Diffuse / volume-scattered window with energy spread
    over many bins → abruptness near 1. The normalised variant divides by
    n_bins_per_window so values are dimensionless ∈ (0, 1].

    Parameters
    ----------
    spec_cor:
        Complex spectrum, shape (nr,). Use PreprocessedProfile.spec_cor.
    range_m:
        Range axis (m), shape (nr,). Must be **monotonically increasing** —
        pass PreprocessedProfile.range_coarse_m, NOT range_m (the total
        phase-corrected range is non-monotonic at the bin level).
    window_width_m:
        Width of each window in metres. Default 4 m (~ a few λ in ice).
    step_m:
        Step between window centres. Defaults to window_width_m / 2.
    depth_window_m:
        Optional (z_low, z_high) over which to evaluate. Defaults to full axis.
    """

    spec_cor = np.asarray(spec_cor)
    range_m = np.asarray(range_m, dtype=float)
    if spec_cor.shape != range_m.shape:
        raise ValueError("spec_cor and range_m must have the same shape")

    step_m = float(window_width_m / 2.0 if step_m is None else step_m)
    if step_m <= 0:
        raise ValueError("step_m must be positive")

    dz = float(np.median(np.diff(range_m)))
    if dz <= 0:
        raise ValueError("range_m must be strictly increasing — pass range_coarse_m")

    half_bins = max(int(round(0.5 * window_width_m / dz)), 1)
    n_bins_per_window = 2 * half_bins + 1

    if depth_window_m is None:
        z_lo, z_hi = float(range_m[half_bins]), float(range_m[-1 - half_bins])
    else:
        z_lo, z_hi = float(depth_window_m[0]), float(depth_window_m[1])
    if z_hi <= z_lo:
        raise ValueError("depth_window_m must contain at least one window centre")

    centres = np.arange(z_lo, z_hi + 0.5 * step_m, step_m)
    pk_db = np.full(centres.size, np.nan)
    int_db = np.full(centres.size, np.nan)
    abr = np.full(centres.size, np.nan)

    pwr = np.abs(spec_cor) ** 2

    for i, c in enumerate(centres):
        c_idx = int(np.searchsorted(range_m, c))
        lo = max(c_idx - half_bins, 0)
        hi = min(c_idx + half_bins + 1, range_m.size)
        if hi - lo < 2:
            continue
        w = pwr[lo:hi]
        peak = float(np.max(w))
        integ = float(np.mean(w))
        if integ > 0:
            abr[i] = peak / integ
            pk_db[i] = 10.0 * np.log10(max(peak, 1e-30))
            int_db[i] = 10.0 * np.log10(integ)

    return AbruptnessResult(
        range_centers_m=centres,
        window_width_m=float(window_width_m),
        n_bins_per_window=int(n_bins_per_window),
        peak_power_db=pk_db,
        integrated_power_db=int_db,
        abruptness=abr,
        abruptness_normalized=abr / float(n_bins_per_window),
    )


# =============================================================================
# 2. Range autocorrelation length
# =============================================================================


@dataclass
class AutocorrelationResult:
    """Complex range autocorrelation over a depth window."""

    lags_m: np.ndarray                # (nl,)  positive lags only
    R_complex: np.ndarray             # (nl,)  complex
    R_magnitude: np.ndarray           # (nl,)  |R| / |R(0)|
    R_phase_rad: np.ndarray           # (nl,)  angle(R)
    autocorr_length_m_1e: float       # lag at which |R| drops below 1/e
    autocorr_length_m_half: float     # lag at which |R| drops below 0.5


def range_autocorrelation_length(
    spec_cor: np.ndarray,
    range_m: np.ndarray,
    depth_window_m: Tuple[float, float],
    *,
    max_lag_m: float = 20.0,
) -> AutocorrelationResult:
    """Complex range autocorrelation of S(z) in a depth window.

    Sharp specular reflector → R(d) drops to noise within one wavelength.
    Volume scattering layer → R(d) decays slowly; the half-width of |R(d)|
    measures the effective vertical extent of correlated scattering.
    """

    spec_cor = np.asarray(spec_cor)
    range_m = np.asarray(range_m, dtype=float)
    idx = _window_indices(range_m, depth_window_m)
    if idx.size < 4:
        raise ValueError("depth_window_m contains <4 range bins; widen the window")

    s = spec_cor[idx] - np.mean(spec_cor[idx])  # demean complex
    z = range_m[idx]
    dz = float(np.median(np.diff(z)))

    n_lag = max(int(round(max_lag_m / dz)), 2)
    n_lag = min(n_lag, s.size - 1)

    # Unbiased complex autocorrelation at lags 0..n_lag-1
    R = np.empty(n_lag, dtype=complex)
    for k in range(n_lag):
        if k == 0:
            R[0] = np.mean(s * np.conj(s))
        else:
            R[k] = np.mean(s[:-k] * np.conj(s[k:]))

    R0 = R[0].real if R[0].real != 0 else 1.0
    Rmag = np.abs(R) / abs(R0)
    Rphi = np.angle(R)
    lags = dz * np.arange(n_lag)

    def _crossing(threshold: float) -> float:
        below = np.where(Rmag < threshold)[0]
        if below.size == 0:
            return float(lags[-1])  # never drops below threshold within max_lag
        k = int(below[0])
        if k == 0:
            return 0.0
        # linear interpolation between k-1 and k
        x0, x1 = Rmag[k - 1], Rmag[k]
        if x1 == x0:
            return float(lags[k])
        frac = (x0 - threshold) / (x0 - x1)
        return float(lags[k - 1] + frac * (lags[k] - lags[k - 1]))

    return AutocorrelationResult(
        lags_m=lags,
        R_complex=R,
        R_magnitude=Rmag,
        R_phase_rad=Rphi,
        autocorr_length_m_1e=_crossing(1.0 / np.e),
        autocorr_length_m_half=_crossing(0.5),
    )


# =============================================================================
# 3. RSR-style amplitude PDF fit (Rice distribution)
# =============================================================================


@dataclass
class RsrResult:
    """Rice-distribution decomposition of |S| in a depth window."""

    n_samples: int
    coherent_power: float          # ν^2  (linear)
    diffuse_power: float           # 2σ^2 (linear)
    coherent_db: float             # 10·log10(ν^2)
    diffuse_db: float              # 10·log10(2σ^2)
    coherent_fraction: float       # ν^2 / (ν^2 + 2σ^2)  ∈ [0, 1]
    rice_K_db: float               # 10·log10(ν^2 / (2σ^2)) — Rician K-factor
    fit_method: str                # 'mom' (method of moments) or 'mle'


def basal_amplitude_pdf_fit(
    spec_cor: np.ndarray,
    range_m: np.ndarray,
    depth_window_m: Tuple[float, float],
    *,
    method: str = "mle",
) -> RsrResult:
    """RSR-style decomposition of |S(z)| samples in [z_low, z_high] into
    coherent (specular) and diffuse (Rayleigh) components via a Rice fit.

    Following Grima et al. (2012, 2014), the amplitude PDF of a coherent
    reflection contaminated by Rayleigh scatter is:

        f(x | ν, σ) = (x/σ²) exp[−(x² + ν²)/(2σ²)] · I₀(xν/σ²)

    with mean returned power ⟨|S|²⟩ = ν² + 2σ². ν² is the coherent component
    (Hagfors specular reflector) and 2σ² the diffuse component.

    Parameters
    ----------
    method:
        'mle' tries scipy.stats.rice fit; falls back to 'mom' if scipy is
        absent or the MLE optimization misbehaves.
    """

    spec_cor = np.asarray(spec_cor)
    range_m = np.asarray(range_m, dtype=float)
    idx = _window_indices(range_m, depth_window_m)
    if idx.size < 8:
        raise ValueError("depth_window_m contains <8 samples; widen the window")

    x = np.abs(spec_cor[idx]).astype(float)
    x = x[np.isfinite(x) & (x > 0)]
    n = int(x.size)
    if n < 8:
        raise ValueError("Too few finite positive samples to fit Rice")

    fit_method = method
    nu2 = sigma2 = None

    if method == "mle":
        try:
            from scipy.stats import rice  # type: ignore

            # scipy parameterisation: rice(b, scale) where b = ν/σ, scale = σ
            b, loc, scale = rice.fit(x, floc=0.0)
            sigma = float(scale)
            nu = float(b * scale)
            nu2 = nu * nu
            sigma2 = sigma * sigma
            if not (np.isfinite(nu2) and np.isfinite(sigma2) and sigma2 > 0):
                raise ValueError("MLE returned invalid Rice parameters")
        except Exception:
            fit_method = "mom"

    if fit_method == "mom" or nu2 is None:
        # Method of moments on |S|² (noncentral χ² with 2 dof, λ = ν², scale = σ²)
        # E[|S|²]  = ν² + 2σ²
        # Var[|S|²] = 4σ²(ν² + σ²)
        # Solve a 2x2 system.
        m1 = float(np.mean(x ** 2))
        v1 = float(np.var(x ** 2))
        # Let A = ν², B = σ²:  A + 2B = m1;  4B(A + B) = v1
        # → A = m1 − 2B,  4B(m1 − B) = v1  → 4B² − 4 m1 B + v1 = 0
        disc = (4.0 * m1) ** 2 - 16.0 * v1
        if disc < 0:
            disc = 0.0
        B_root = (4.0 * m1 - np.sqrt(disc)) / 8.0
        sigma2 = max(B_root, 1e-30)
        nu2 = max(m1 - 2.0 * sigma2, 0.0)
        fit_method = "mom"

    coh_power = float(nu2)
    diff_power = float(2.0 * sigma2)
    total_power = coh_power + diff_power if coh_power + diff_power > 0 else 1e-30

    return RsrResult(
        n_samples=n,
        coherent_power=coh_power,
        diffuse_power=diff_power,
        coherent_db=float(_safe_db(np.array([coh_power]))[0]),
        diffuse_db=float(_safe_db(np.array([diff_power]))[0]),
        coherent_fraction=float(coh_power / total_power),
        rice_K_db=float(_safe_db(np.array([coh_power / max(diff_power, 1e-30)]))[0]),
        fit_method=fit_method,
    )


# =============================================================================
# 4. Sub-band reflectivity slope
# =============================================================================


@dataclass
class SubbandSlopeResult:
    """Fit of mean depth-window power vs sub-band centre frequency."""

    fc_hz: np.ndarray                  # (n_sub,) sub-band centre frequencies
    mean_power_db: np.ndarray          # (n_sub,) mean 10·log10(|S|²) in window
    coherent_power_db: np.ndarray      # (n_sub,) 10·log10(|<S>|²) in window
    slope_db_per_octave: float         # linear fit in dB vs log2(f)
    slope_db_per_GHz: float            # linear fit in dB vs f (Hz)
    slope_residual_rms_db: float
    n_subbands: int


def subband_reflectivity_slope(
    spec_cor_subbands: np.ndarray,     # (n_sub, nr) complex
    subband_fc_hz: np.ndarray,
    range_m: np.ndarray,
    depth_window_m: Tuple[float, float],
) -> SubbandSlopeResult:
    """Fit the mean dB power in a depth window across sub-bands.

    Mie-regime volume scatterers and conductivity-driven attenuation both
    leave characteristic signatures in dR/df (Vaňková et al. 2021; Hills et
    al. 2024 Eq 12). Slope ≈ 0 ⇒ frequency-flat reflector; positive slope
    ⇒ Rayleigh-regime particles smaller than λ; negative slope ⇒ attenuation
    or particles approaching λ.
    """

    if spec_cor_subbands is None:
        raise ValueError("spec_cor_subbands is None — re-run preprocess with subbands_enabled")
    sb = np.asarray(spec_cor_subbands)
    fc = np.asarray(subband_fc_hz, dtype=float)
    range_m = np.asarray(range_m, dtype=float)
    if sb.ndim != 2 or sb.shape[1] != range_m.size or sb.shape[0] != fc.size:
        raise ValueError("Inconsistent shapes between spec_cor_subbands, subband_fc_hz, range_m")

    idx = _window_indices(range_m, depth_window_m)
    if idx.size < 4:
        raise ValueError("depth_window_m contains <4 range bins; widen the window")

    n_sub = sb.shape[0]
    mean_pow = np.full(n_sub, np.nan)
    coh_pow = np.full(n_sub, np.nan)
    for k in range(n_sub):
        s = sb[k, idx]
        s = s[np.isfinite(s)]
        if s.size < 2:
            continue
        mean_pow[k] = float(np.mean(np.abs(s) ** 2))
        coh_pow[k] = float(np.abs(np.mean(s)) ** 2)

    mean_db = _safe_db(mean_pow)
    coh_db = _safe_db(coh_pow)

    # Linear fit dB vs log2(f) → slope in dB/octave
    good = np.isfinite(mean_db) & np.isfinite(fc) & (fc > 0)
    slope_db_per_oct = np.nan
    slope_db_per_GHz = np.nan
    rms = np.nan
    if good.sum() >= 2:
        x_oct = np.log2(fc[good])
        y = mean_db[good]
        m_oct, b_oct = np.polyfit(x_oct, y, 1)
        slope_db_per_oct = float(m_oct)
        rms = float(np.sqrt(np.mean((y - (m_oct * x_oct + b_oct)) ** 2)))

        x_hz = fc[good] / 1e9  # GHz
        m_hz, _ = np.polyfit(x_hz, y, 1)
        slope_db_per_GHz = float(m_hz)

    return SubbandSlopeResult(
        fc_hz=fc,
        mean_power_db=mean_db,
        coherent_power_db=coh_db,
        slope_db_per_octave=slope_db_per_oct,
        slope_db_per_GHz=slope_db_per_GHz,
        slope_residual_rms_db=rms,
        n_subbands=int(n_sub),
    )


# =============================================================================
# Bed-centric metrics (Hills-corrected reflectivity, FWHM, etc.)
# =============================================================================
#
# These replace the older window-mean diagnostics for the continuous-time-
# series analysis: each one is computed AT (or relative to) the picked bed
# bin so the result tracks bed physics rather than the noise floor that
# happens to dominate a wider window.


def hills_relative_reflectivity(
    peak_power_db: float,
    bed_depth_m: float,
    *,
    eps_r: float = 3.15,
    atten_db_per_km: float = 10.0,
) -> float:
    """Hills 2024 Eqs 1–4: convert measured bed-echo power to a relative
    reflectivity by adding back the geometric-spreading and attenuation losses
    that depend on bed depth.

    For ground-based ApRES (aircraft height h = 0):
        G_dB = -20·log10(2·d/√εr)
        L_dB = -2·d·N̄    (N̄ in dB/km, d in km)
        R_rel = P − G − L  =  P + |G| + |L|

    Parameters
    ----------
    peak_power_db
        Bed peak power in dB (raw amplitude squared, 10·log10).
    bed_depth_m
        Picked bed depth (m).
    eps_r
        Relative permittivity of ice (default 3.15 per Hills).
    atten_db_per_km
        One-way depth-averaged attenuation rate in dB/km. Hills used 10.
    """
    if not np.isfinite(bed_depth_m) or bed_depth_m <= 0:
        return float("nan")
    G = -20.0 * np.log10(2.0 * bed_depth_m / np.sqrt(eps_r))
    L = -2.0 * (bed_depth_m / 1000.0) * atten_db_per_km
    return float(peak_power_db - G - L)


def bed_peak_fwhm_m(
    spec_cor: np.ndarray,
    range_coarse_m: np.ndarray,
    bed_depth_m: float,
    *,
    search_half_m: float = 10.0,
    drop_db: float = 3.0,
) -> Tuple[float, int, float]:
    """Find the bed peak inside ±search_half_m of the picked depth and
    return (FWHM in metres, peak bin index, peak power dB).

    A sharp specular bed yields FWHM ≈ λ/2 (≈ 0.28 m at 300 MHz in ice).
    A diffuse / volume-scattered bed broadens FWHM well beyond that.
    `drop_db` defaults to 3 (-3 dB = half power).
    """
    z = np.asarray(range_coarse_m, dtype=float)
    amp_db = 20.0 * np.log10(np.abs(np.asarray(spec_cor)) + 1e-30)
    sel = np.where((z >= bed_depth_m - search_half_m) & (z <= bed_depth_m + search_half_m))[0]
    if sel.size < 3:
        return float("nan"), -1, float("nan")
    pk_local = sel[int(np.argmax(amp_db[sel]))]
    pk_amp = float(amp_db[pk_local])
    threshold = pk_amp - drop_db

    left = pk_local
    while left > 0 and amp_db[left] >= threshold:
        left -= 1
    right = pk_local
    while right < z.size - 1 and amp_db[right] >= threshold:
        right += 1

    # Sub-bin linear interpolation of the threshold-crossing
    if left + 1 < z.size and amp_db[left + 1] != amp_db[left]:
        frac = (threshold - amp_db[left]) / (amp_db[left + 1] - amp_db[left])
        left_z = z[left] + frac * (z[left + 1] - z[left])
    else:
        left_z = z[left]
    if right - 1 >= 0 and amp_db[right] != amp_db[right - 1]:
        frac = (threshold - amp_db[right - 1]) / (amp_db[right] - amp_db[right - 1])
        right_z = z[right - 1] + frac * (z[right] - z[right - 1])
    else:
        right_z = z[right]

    return float(right_z - left_z), int(pk_local), pk_amp


def bed_bin_chirp_coherence(
    chirp_coherence: Optional[np.ndarray],
    range_coarse_m: np.ndarray,
    bed_depth_m: float,
    *,
    half_window_bins: int = 2,
) -> float:
    """Schroeder/Peters specularity (coherent / incoherent multilook power)
    evaluated AT the bed peak — taken as the max of γ_chirp(z) over a small
    window centred on the picked bed.

    Returns NaN if chirp_coherence wasn't computed.
    """
    if chirp_coherence is None:
        return float("nan")
    cc = np.asarray(chirp_coherence, dtype=float)
    z = np.asarray(range_coarse_m, dtype=float)
    if cc.shape != z.shape:
        return float("nan")
    bed_idx = int(np.argmin(np.abs(z - bed_depth_m)))
    lo = max(bed_idx - half_window_bins, 0)
    hi = min(bed_idx + half_window_bins + 1, z.size)
    if hi <= lo:
        return float("nan")
    win = cc[lo:hi]
    return float(np.max(win[np.isfinite(win)])) if np.any(np.isfinite(win)) else float("nan")


def above_bed_scattering_thickness_m(
    spec_cor: np.ndarray,
    range_coarse_m: np.ndarray,
    bed_depth_m: float,
    *,
    smooth_m: float = 10.0,
    threshold_db_above_noise: float = 6.0,
    noise_search_offset_m: float = 100.0,
    noise_search_extent_m: float = 400.0,
) -> Tuple[float, float]:
    """Hills' "10s of m above the bed" volume-scattering thickness, quantified.

    Smooths |S(z)| with a `smooth_m` rolling average, estimates the noise
    floor from a band BELOW the bed (z ∈ [bed+offset, bed+offset+extent]),
    then walks UP from the bed until the smoothed amplitude drops below
    noise_floor + threshold_db_above_noise. Returns the height above the bed
    where that crossing occurs (m), plus the noise-floor estimate (dB).
    """
    z = np.asarray(range_coarse_m, dtype=float)
    amp = 20.0 * np.log10(np.abs(np.asarray(spec_cor)) + 1e-30)
    dz = float(np.median(np.diff(z)))
    w = max(int(round(smooth_m / max(dz, 1e-9))), 1)
    amp_sm = np.convolve(amp, np.ones(w) / w, mode="same")

    # Noise floor estimate — search BELOW the bed (z > bed + offset)
    nf_lo = bed_depth_m + noise_search_offset_m
    nf_hi = nf_lo + noise_search_extent_m
    nf_mask = (z >= nf_lo) & (z <= min(nf_hi, z[-1]))
    if nf_mask.sum() < 5:
        return float("nan"), float("nan")
    nf = float(np.median(amp_sm[nf_mask]))
    threshold = nf + threshold_db_above_noise

    bed_idx = int(np.argmin(np.abs(z - bed_depth_m)))
    # Walk upward (decreasing index = shallower) until we DROP below threshold
    i = bed_idx
    while i > 0 and amp_sm[i] > threshold:
        i -= 1
    if i == bed_idx:
        # bed itself doesn't exceed threshold (bed buried in noise) — return 0
        return 0.0, nf
    height = float(bed_depth_m - z[i])
    return height, nf


# =============================================================================
# Mie/Rayleigh volume-scattering inversion (Hills 2024 Eq 8)
# =============================================================================


def _rayleigh_scattering_factor(eps_scatterer: float, eps_host: float) -> float:
    """|f|^2 for the Rayleigh scattering cross-section:
        f = (m^2 − 1) / (m^2 + 2),   m = n_scatterer / n_host = √(ε_s / ε_h)

    For water (ε ≈ 81) in ice (ε ≈ 3.15), m ≈ 5.07 and |f|^2 ≈ 0.78.
    """
    m2 = eps_scatterer / eps_host
    num = m2 - 1.0
    den = m2 + 2.0
    return float((num / den) ** 2)


def mie_rayleigh_invert_radius(
    suppression_db: float,
    scattering_thickness_m: float,
    porosity: float,
    *,
    frequency_hz: float = 3.0e8,
    eps_ice: float = 3.15,
    eps_scatterer: float = 81.0,
) -> float:
    """Invert the Rayleigh-regime scatterer radius from the observed volume-
    scattering suppression and the thickness of the scattering layer.

    Formulation (following Hills 2024 Eq 8 with Rayleigh Q):

        ρ_volume = exp(-Q · (3φ / 4πr) · z)                       (Hills 8)
        Q_Rayleigh = (8/3) · (kr)^4 · |f|^2,  f = (m² − 1)/(m² + 2)

    Substituting Q into the loss expression, the dB loss is

        dB = (10/ln10) · (8/3)·(kr)^4·|f|^2 · (3φ / 4πr) · z
           = (10/ln10) · 2 · k⁴ · r³ · |f|² · φ · z / π

    which solves in closed form for r (the only unknown):

        r³ = dB · π · ln10 / (20 · k⁴ · |f|² · φ · z)

    Parameters
    ----------
    suppression_db
        Observed one-way power suppression through the scattering layer (dB,
        positive). Typically computed as:
            suppression_db = (R_rel_reference) − (R_rel_observed)
        where R_rel_reference is the expected water-ice or till-ice reflection.
    scattering_thickness_m
        Vertical extent of the scattering layer (m), from
        `above_bed_scattering_thickness_m`.
    porosity
        Assumed volume fraction of scatterers (0 < φ < 1). Hills used 0-0.3.
    frequency_hz, eps_ice, eps_scatterer
        Radar centre frequency (Hz) and dielectric constants of the ice host
        and the scatterer (e.g. 81 for water at low frequency; ~3.15 for
        sediment/till; 1.0 for air pores).

    Returns
    -------
    r_m : float
        Effective scatterer radius in metres. NaN if inputs are non-physical
        (negative suppression, zero thickness, etc.).

    Caveats
    -------
    - Assumes Rayleigh regime (r ≪ λ). For r > ~λ/10 ≈ 5 cm at 300 MHz the
      actual Q deviates from the (kr)^4 approximation — results should be
      interpreted as "effective Rayleigh-equivalent radius".
    - Assumes scattering-dominated loss (ignores conductivity attenuation).
    """
    if suppression_db <= 0 or scattering_thickness_m <= 0 or porosity <= 0:
        return float("nan")
    c = 299_792_458.0
    c_ice = c / np.sqrt(eps_ice)
    lam = c_ice / frequency_hz
    k = 2.0 * np.pi / lam
    f_sq = _rayleigh_scattering_factor(eps_scatterer, eps_ice)
    # Solve for r³
    numer = float(suppression_db) * np.pi * np.log(10.0)
    denom = 20.0 * (k ** 4) * f_sq * float(porosity) * float(scattering_thickness_m)
    if denom <= 0 or numer <= 0:
        return float("nan")
    r3 = numer / denom
    return float(r3 ** (1.0 / 3.0))


# =============================================================================
# Convenience: one fingerprint per shot
# =============================================================================


def scattering_fingerprint(
    profile,
    *,
    basal_window_m: Optional[Tuple[float, float]] = None,
    above_basal_window_m: Optional[Tuple[float, float]] = None,
    spec_window_width_m: float = 4.0,
    autocorr_max_lag_m: float = 20.0,
) -> Dict[str, float]:
    """Run all four diagnostics in the basal window and return a flat dict
    suitable for a per-shot CSV row.

    If `basal_window_m` is None and the profile has a usable bed_depth_m,
    defaults to (bed-40, bed+5). Otherwise the caller must supply it.
    """

    # Use the monotonic coarse axis for windowing — `profile.range_m` is the
    # phase-corrected total range and is not monotonic at the bin level.
    range_m = np.asarray(profile.range_coarse_m, dtype=float)
    spec_cor = np.asarray(profile.spec_cor)

    if basal_window_m is None:
        bed = float(getattr(profile, "bed_depth_m", 0.0) or 0.0)
        if bed <= 0:
            raise ValueError(
                "basal_window_m not supplied and profile has no usable bed_depth_m"
            )
        basal_window_m = (max(bed - 40.0, range_m[0]), min(bed + 5.0, range_m[-1]))

    if above_basal_window_m is None:
        z0, z1 = basal_window_m
        depth = z1 - z0
        above_basal_window_m = (max(z0 - 4 * depth, range_m[0]), z0)

    out: Dict[str, float] = {
        "basal_z_lo_m": float(basal_window_m[0]),
        "basal_z_hi_m": float(basal_window_m[1]),
        "abovebasal_z_lo_m": float(above_basal_window_m[0]),
        "abovebasal_z_hi_m": float(above_basal_window_m[1]),
    }

    # 1: pulse abruptness over the basal window
    abr = pulse_abruptness(
        spec_cor, range_m,
        window_width_m=spec_window_width_m,
        depth_window_m=basal_window_m,
    )
    finite = np.isfinite(abr.abruptness)
    if finite.any():
        out["basal_abruptness_max"] = float(np.max(abr.abruptness[finite]))
        out["basal_abruptness_median"] = float(np.median(abr.abruptness[finite]))
        out["basal_abruptness_norm_max"] = float(np.max(abr.abruptness_normalized[finite]))
        out["basal_peak_power_db_max"] = float(np.max(abr.peak_power_db[finite]))
        out["basal_integrated_power_db_mean"] = float(np.mean(abr.integrated_power_db[finite]))
    else:
        for k in ("basal_abruptness_max", "basal_abruptness_median",
                  "basal_abruptness_norm_max", "basal_peak_power_db_max",
                  "basal_integrated_power_db_mean"):
            out[k] = np.nan

    # 2: range autocorrelation length in basal window
    try:
        ac = range_autocorrelation_length(
            spec_cor, range_m, basal_window_m, max_lag_m=autocorr_max_lag_m,
        )
        out["basal_autocorr_len_m_1e"] = float(ac.autocorr_length_m_1e)
        out["basal_autocorr_len_m_half"] = float(ac.autocorr_length_m_half)
    except Exception:
        out["basal_autocorr_len_m_1e"] = np.nan
        out["basal_autocorr_len_m_half"] = np.nan

    # 3: RSR Rice fit in basal window
    try:
        rsr = basal_amplitude_pdf_fit(spec_cor, range_m, basal_window_m, method="mle")
        out["basal_rsr_coherent_db"] = rsr.coherent_db
        out["basal_rsr_diffuse_db"] = rsr.diffuse_db
        out["basal_rsr_coherent_fraction"] = rsr.coherent_fraction
        out["basal_rsr_K_db"] = rsr.rice_K_db
        out["basal_rsr_fit_method"] = rsr.fit_method  # type: ignore[assignment]
    except Exception:
        out["basal_rsr_coherent_db"] = np.nan
        out["basal_rsr_diffuse_db"] = np.nan
        out["basal_rsr_coherent_fraction"] = np.nan
        out["basal_rsr_K_db"] = np.nan
        out["basal_rsr_fit_method"] = "failed"  # type: ignore[assignment]

    # Same RSR fit on the *above-basal* window for context (Hills' "10s of m
    # above bed" volume-scattering zone vs the bed itself)
    try:
        rsr_a = basal_amplitude_pdf_fit(spec_cor, range_m, above_basal_window_m, method="mle")
        out["abovebasal_rsr_coherent_db"] = rsr_a.coherent_db
        out["abovebasal_rsr_diffuse_db"] = rsr_a.diffuse_db
        out["abovebasal_rsr_coherent_fraction"] = rsr_a.coherent_fraction
        out["abovebasal_rsr_K_db"] = rsr_a.rice_K_db
    except Exception:
        out["abovebasal_rsr_coherent_db"] = np.nan
        out["abovebasal_rsr_diffuse_db"] = np.nan
        out["abovebasal_rsr_coherent_fraction"] = np.nan
        out["abovebasal_rsr_K_db"] = np.nan

    # 4: sub-band slope in basal window
    sb_spec = getattr(profile, "spec_cor_subbands", None)
    sb_fc = getattr(profile, "subband_fc_hz", None)
    if sb_spec is not None and sb_fc is not None:
        try:
            sl = subband_reflectivity_slope(sb_spec, sb_fc, range_m, basal_window_m)
            out["basal_subband_slope_db_per_octave"] = sl.slope_db_per_octave
            out["basal_subband_slope_db_per_GHz"] = sl.slope_db_per_GHz
            out["basal_subband_fit_rms_db"] = sl.slope_residual_rms_db
            out["basal_subband_n_subbands"] = float(sl.n_subbands)
        except Exception:
            out["basal_subband_slope_db_per_octave"] = np.nan
            out["basal_subband_slope_db_per_GHz"] = np.nan
            out["basal_subband_fit_rms_db"] = np.nan
            out["basal_subband_n_subbands"] = 0.0
    else:
        out["basal_subband_slope_db_per_octave"] = np.nan
        out["basal_subband_slope_db_per_GHz"] = np.nan
        out["basal_subband_fit_rms_db"] = np.nan
        out["basal_subband_n_subbands"] = 0.0

    # Chirp-coherence over the basal window (kept for backwards compatibility
    # with the attended-fingerprint CSV; for the bed-centric work below we
    # use bed_bin_chirp_coherence which evaluates AT the bed peak instead).
    cc = getattr(profile, "chirp_coherence", None)
    if cc is not None:
        cc = np.asarray(cc, dtype=float)
        idx = _window_indices(range_m, basal_window_m)
        cc_win = cc[idx]
        cc_win = cc_win[np.isfinite(cc_win)]
        if cc_win.size > 0:
            out["basal_chirp_coherence_median"] = float(np.median(cc_win))
            out["basal_chirp_coherence_max"] = float(np.max(cc_win))
        else:
            out["basal_chirp_coherence_median"] = np.nan
            out["basal_chirp_coherence_max"] = np.nan
    else:
        out["basal_chirp_coherence_median"] = np.nan
        out["basal_chirp_coherence_max"] = np.nan

    # ------------------------------------------------------------------
    # Bed-centric metrics (Hills R_rel, FWHM, bed-bin γ_chirp,
    # above-bed scattering thickness). These are the metrics intended for
    # cross-shot continuous-time-series comparison.
    # ------------------------------------------------------------------
    bed_depth = float(getattr(profile, "bed_depth_m", float("nan")))
    if np.isfinite(bed_depth) and bed_depth > 0:
        # FWHM around the picked bed (also returns the actual peak power, which
        # is more reliable than max-in-window when the bed pick wandered)
        fwhm_m, pk_idx, pk_amp_db = bed_peak_fwhm_m(
            spec_cor, range_m, bed_depth, search_half_m=10.0, drop_db=3.0,
        )
        out["bed_fwhm_m"] = fwhm_m
        out["bed_peak_power_db"] = pk_amp_db

        # Hills 2024 relative reflectivity (requires bed power and depth)
        out["bed_R_rel_db"] = (
            hills_relative_reflectivity(pk_amp_db, bed_depth)
            if np.isfinite(pk_amp_db) else float("nan")
        )

        # Bed-bin chirp coherence (Schroeder/Peters specularity at the bed)
        out["bed_chirp_coherence"] = bed_bin_chirp_coherence(
            cc, range_m, bed_depth, half_window_bins=2,
        )

        # Above-bed scattering layer thickness (Hills "10s of m above bed")
        h, nf = above_bed_scattering_thickness_m(
            spec_cor, range_m, bed_depth,
            smooth_m=10.0, threshold_db_above_noise=6.0,
        )
        out["above_bed_scattering_height_m"] = h
        out["above_bed_noise_floor_db"] = nf
    else:
        for k in ("bed_fwhm_m", "bed_peak_power_db", "bed_R_rel_db",
                  "bed_chirp_coherence", "above_bed_scattering_height_m",
                  "above_bed_noise_floor_db"):
            out[k] = float("nan")

    return out
