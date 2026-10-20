"""Bed picking for thick, grounded ice.

The default picker in :mod:`apres.preprocess` takes the strongest return below
``ice_thickness_min_m`` (50 m).  That rule was written for the McMurdo ice-shelf
deployments, where the bed is the brightest thing in the profile.  On the
Thwaites trunk the ice is 2.0-2.4 km thick, the basal return sits only
10-30 dB above the noise floor, and the brightest return in every profile is a
shallow firn/englacial reflector.  Applied unchanged, the rule picks the bed at
40-80 m in all 321 Thwaites acquisitions, which then collapses the depth range
available to the strain fit.

This module picks the bed inside a depth window centred on an independent
thickness prior (BedMachine Antarctica), and reports the signal-to-noise ratio
of the pick so that unreliable picks can be screened rather than silently used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["BedPick", "noise_floor_db", "pick_bed", "coherence_noise_depth"]


@dataclass
class BedPick:
    """Result of a single bed pick."""

    depth_m: float
    index: int
    power_db: float
    noise_floor_db: float
    snr_db: float
    prominence_db: float
    prior_m: float
    residual_m: float  # pick minus prior
    method: str
    ok: bool


def noise_floor_db(
    range_m: np.ndarray,
    power_db: np.ndarray,
    *,
    deep_start_m: float = 1200.0,
    percentile: float = 20.0,
) -> float:
    """Robust receiver-noise reference for one acquisition.

    The ApRES range window (2.5 km here) extends past the bed everywhere on the
    Thwaites transect, so its deep half is dominated by receiver noise.  A low
    percentile of that half is used rather than the median of the very deepest
    bins: where the ice is thickest the basal return lands within 100 m of the
    end of the window, and a tail median then absorbs the very signal the noise
    floor is meant to be measured against.
    """
    deep = range_m >= deep_start_m
    if not np.any(deep):
        deep = range_m >= np.nanmedian(range_m)
    return float(np.nanpercentile(power_db[deep], percentile))


def _bins(range_m: np.ndarray, length_m: float) -> int:
    """Odd number of range bins spanning ``length_m``."""
    spacing = float(np.nanmedian(np.diff(range_m)))
    n = max(int(round(length_m / spacing)), 1)
    return n + 1 if n % 2 == 0 else n


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width < 3:
        return x
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(x, kernel, mode="same")


def coherence_noise_depth(
    range_m: np.ndarray,
    power_db: np.ndarray,
    *,
    floor_db: Optional[float] = None,
    margin_db: float = 6.0,
    smooth_bins: int = 201,
) -> float:
    """Deepest range at which the smoothed return is ``margin_db`` above noise.

    Used as the outer limit for the correlation windows so that the strain fit
    never reaches into the noise.
    """
    floor = noise_floor_db(range_m, power_db) if floor_db is None else floor_db
    smooth = _smooth(power_db, smooth_bins)
    above = np.where(smooth > floor + margin_db)[0]
    return float(range_m[above[-1]]) if above.size else float(np.nanmax(range_m))


def _running_median(x: np.ndarray, width: int) -> np.ndarray:
    """Wide running median, used as the local background level."""
    if width < 3 or width >= x.size:
        return np.full_like(x, float(np.nanmedian(x)))
    half = width // 2
    padded = np.pad(x, half, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.nanmedian(strided, axis=-1)


def pick_bed(
    range_m: np.ndarray,
    spec_cor: np.ndarray,
    *,
    prior_m: float,
    search_halfwidth_m: float = 400.0,
    smooth_m: float = 4.0,
    background_m: float = 300.0,
    edge_exclude_m: float = 60.0,
    min_snr_db: float = 6.0,
    min_prominence_db: float = 4.0,
    min_depth_m: float = 300.0,
) -> BedPick:
    """Pick the basal return nearest an independent thickness prior.

    The bed is identified as the most *prominent* local return in the search
    window rather than the strongest one.  Absolute power falls by ~40 dB
    between 800 m and the bed, so a plain windowed maximum is biased toward the
    shallow end of the window; and the last few tens of metres of the ApRES
    range record carry an end-of-window artefact that outranks a genuine
    10-20 dB basal return.  Prominence, measured against a wide running median
    of the same profile, is insensitive to both.

    Parameters
    ----------
    range_m, spec_cor
        Range axis and complex range profile of a preprocessed acquisition.
    prior_m
        Independent ice-thickness prior (BedMachine Antarctica) in metres.
    search_halfwidth_m
        Half-width of the depth window searched around ``prior_m``.  400 m
        exceeds the BedMachine error estimate on the Thwaites trunk while
        excluding the bright shallow firn and englacial reflectors.
    smooth_m, background_m
        Lengths, in metres, of the boxcar applied before peak detection and of
        the running median that defines the local background.  Range bins are
        ~5 cm here, so these must be set in metres rather than bins: a
        background window narrower than the basal return itself tracks the peak
        and erases its own prominence.
    edge_exclude_m
        Depth at the end of the range record excluded from the search.  The
        last few tens of metres of the record carry an end-of-window artefact.
    min_snr_db, min_prominence_db
        A pick is flagged ``ok=False`` unless it clears the noise floor by
        ``min_snr_db`` *and* stands ``min_prominence_db`` above the local
        background.  Screening on both is what separates a genuine basal
        return from the smooth tail of the englacial decay.

    Returns
    -------
    BedPick
        Always returned; inspect ``.ok`` rather than catching an exception.
    """
    range_m = np.asarray(range_m, dtype=float)
    power_db = 20.0 * np.log10(np.abs(spec_cor) + 1e-30)
    floor = noise_floor_db(range_m, power_db)

    failed = BedPick(
        depth_m=float("nan"), index=-1, power_db=float("nan"),
        noise_floor_db=floor, snr_db=float("nan"), prominence_db=float("nan"),
        prior_m=float(prior_m), residual_m=float("nan"),
        method="failed", ok=False,
    )
    if not np.isfinite(prior_m):
        return failed

    r_max = float(np.nanmax(range_m)) - float(edge_exclude_m)
    lo = max(prior_m - search_halfwidth_m, min_depth_m)
    hi = min(prior_m + search_halfwidth_m, r_max)
    window = (range_m >= lo) & (range_m <= hi)
    if not np.any(window):
        return failed

    smooth_bins = _bins(range_m, smooth_m)
    smooth = _smooth(power_db, smooth_bins)
    background = _running_median(smooth, _bins(range_m, background_m))
    prominence = smooth - background

    idx_window = np.where(window)[0]
    coarse = idx_window[int(np.argmax(prominence[window]))]

    # Refine onto the strongest raw bin within one smoothing width so that the
    # reported bin's phase is usable by the basal cross-correlation.
    half = max(smooth_bins // 2, 1)
    lo_i = max(coarse - half, 0)
    hi_i = min(coarse + half + 1, range_m.size)
    index = lo_i + int(np.argmax(power_db[lo_i:hi_i]))

    peak_db = float(power_db[index])
    snr = peak_db - floor
    prom = float(prominence[coarse])
    return BedPick(
        depth_m=float(range_m[index]),
        index=int(index),
        power_db=peak_db,
        noise_floor_db=floor,
        snr_db=float(snr),
        prominence_db=prom,
        prior_m=float(prior_m),
        residual_m=float(range_m[index] - prior_m),
        method="prior-windowed-prominence",
        ok=bool(snr >= min_snr_db and prom >= min_prominence_db),
    )
