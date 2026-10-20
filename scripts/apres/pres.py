"""pRES (VNA-era stepped-frequency) data ingestion for the apres pipeline.

The pRES instrument predates ApRES: it records complex S21(f) using a VNA
over a stepped frequency sweep. The MATLAB `read_stk_v3.m` in `scripts/pg5/`
parses the ASCII .STK file format and `freq2time.m` performs the windowed
IFFT that converts the frequency-domain measurement into a complex
range-domain trace.

This module ports just those two steps. It deliberately does *not* port the
pg5 "good reflector" picking strategy (`locate_internals.m`, `find_pulse_shape.m`)
because that approach is fragile. Instead we hand the resulting complex range
trace to the existing apres preprocess/strain code, which uses dense amplitude
+ complex cross-correlation across continuous depth windows.

File format (plain ASCII, space-separated):

    line 1                         : cal settings  (start_f stop_f num_points cal_power)
    next num_points lines          : calibration data (real imag)
    for each shot:
        path line                  : e.g.  C:\\Roosevelt13\\SL01P01V01D1312131234.STK
        date line                  : e.g.  12/13/2013 12:34:56 PM Shot 1
        settings line              : 7 numbers (start_f stop_f num_points avg power if_bw est_depth)
        next num_points lines      : complex response (real imag)

Frequencies in the settings lines are in MHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import ProcessingConfig
from .preprocess import PreprocessedProfile, estimate_ice_thickness


# Network Analyser automatic power correction (dBm). Matches read_stk_v3.m.
_POWER_CORRECTION_DB = 16.0

# pg5 uses C = 84e6 m/s as "two-way speed of light in ice" so that
# range = C * t_roundtrip. The apres module uses ci_m_s = one-way speed
# (c / sqrt(er_ice)) and range = ci_m_s * t_oneway. The two are equivalent:
# ci_m_s = 2 * 84e6 = 168e6 m/s, which corresponds to er_ice = (c/ci)^2 ~= 3.18.
_PRES_ONE_WAY_C_IN_ICE_M_S = 168_000_000.0


# ------------------------------------------------------------
# Low-level STK reader
# ------------------------------------------------------------


@dataclass
class STKShot:
    """One shot from a .STK file."""

    shot_index: int  # 1-indexed
    timestamp: datetime
    settings: np.ndarray  # (7,) floats: [start_f_Hz, stop_f_Hz, num_points, avg, power, if_bw, est_depth]
    data: np.ndarray  # (num_points,) complex, NOT calibrated


@dataclass
class STKFile:
    """Parsed contents of a .STK file."""

    file_path: Path
    cal_settings: np.ndarray  # (4,) [start_f_Hz, stop_f_Hz, num_points, cal_power]
    cal_data: np.ndarray  # (cal_num_points,) complex
    shots: List[STKShot]


def _parse_shot_date(the_date: str) -> datetime:
    """Port of the date-parsing block in read_stk_v3.m.

    Examples encountered in the field:
        "05/06/2003  6:04:41 PM Shot 1"          (matlab-era, 4-digit year)
        "04/27/02 08:38:37 PM Shot 1"            (labview-era, 2-digit year)

    We return a timezone-aware UTC datetime. No PM/AM adjustment beyond
    turning 12-hour time into 24-hour time.
    """

    s = the_date.strip()

    # Try the 4-digit-year form first: MM/DD/YYYY H:MM:SS AM/PM
    parts = s.split()
    # Keep only the first 4 tokens (date, time, AM/PM) — the rest is "Shot N".
    if len(parts) < 3:
        raise ValueError(f"Could not parse STK date line: {the_date!r}")
    date_tok, time_tok, ampm = parts[0], parts[1], parts[2]

    try:
        month, day, year = date_tok.split("/")
    except ValueError as exc:
        raise ValueError(f"Could not parse date token in STK: {date_tok!r}") from exc

    # Normalize 2-digit year -> 20xx (labview files were from the 2000s)
    year_i = int(year)
    if year_i < 100:
        year_i += 2000

    h, m, sec = time_tok.split(":")
    hour_i = int(h)
    minute_i = int(m)
    sec_i = int(sec)

    ampm_u = ampm.upper()
    if ampm_u == "PM" and hour_i != 12:
        hour_i += 12
    elif ampm_u == "AM" and hour_i == 12:
        hour_i = 0

    return datetime(year_i, int(month), int(day), hour_i, minute_i, sec_i, tzinfo=timezone.utc)


def _read_float_pairs(lines: List[str], start: int, n_pairs: int) -> Tuple[np.ndarray, int]:
    """Read exactly `n_pairs` (real, imag) lines starting at `start`.

    Returns the complex array plus the next line index.
    """

    out = np.empty(n_pairs, dtype=np.complex128)
    i = start
    end = start + n_pairs
    if end > len(lines):
        raise ValueError(f"Not enough data lines: need {n_pairs} from index {start}, have {len(lines) - start}")
    for k in range(n_pairs):
        parts = lines[i + k].split()
        if len(parts) < 2:
            raise ValueError(f"Malformed data line {i + k}: {lines[i + k]!r}")
        out[k] = complex(float(parts[0]), float(parts[1]))
    return out, end


def read_stk(path: Path) -> STKFile:
    """Parse an ASCII .STK file into cal + shot records.

    Corresponds to read_stk_v3.m but omits calibration division (which the
    MATLAB code also skipped in practice — it stores the *uncalibrated* data
    in DATA.data).
    """

    path = Path(path)
    raw = path.read_text(errors="replace")
    # Split into stripped non-empty lines; keep original order.
    lines = [ln.rstrip("\r\n") for ln in raw.splitlines()]
    # Drop blank lines (some .STK files have trailing whitespace lines)
    lines = [ln for ln in lines if ln.strip() != ""]

    if not lines:
        raise ValueError(f"Empty STK file: {path}")

    # --- Calibration block
    cal_toks = lines[0].split()
    if len(cal_toks) < 4:
        raise ValueError(f"Expected 4+ values on cal settings line of {path}, got: {lines[0]!r}")
    cal_start_f_mhz = float(cal_toks[0])
    cal_stop_f_mhz = float(cal_toks[1])
    cal_num_points = int(float(cal_toks[2]))
    cal_power = float(cal_toks[3])

    cal_data, idx = _read_float_pairs(lines, 1, cal_num_points)

    cal_ref = 10.0 ** ((cal_power - _POWER_CORRECTION_DB) / 20.0)
    cal_data = cal_data * cal_ref

    # Frequencies stored in MHz in the file header; convert to Hz.
    cal_settings = np.array(
        [cal_start_f_mhz * 1e6, cal_stop_f_mhz * 1e6, float(cal_num_points), float(cal_power)], dtype=float
    )

    # --- Shots
    shots: List[STKShot] = []
    shot_num = 0
    while idx < len(lines):
        # path line (ignore, but validate it doesn't look like a data pair)
        path_line = lines[idx]
        idx += 1
        if idx >= len(lines):
            break

        # date line
        date_line = lines[idx]
        idx += 1
        try:
            timestamp = _parse_shot_date(date_line)
        except Exception as exc:
            raise ValueError(f"Bad date line in {path} near shot {shot_num + 1}: {date_line!r} ({exc})")

        # settings line
        if idx >= len(lines):
            raise ValueError(f"Missing settings line for shot in {path}")
        settings_toks = lines[idx].split()
        idx += 1
        if len(settings_toks) < 7:
            raise ValueError(f"Settings line has <7 tokens in {path}: {settings_toks!r}")
        settings_7 = np.array([float(tok) for tok in settings_toks[:7]], dtype=float)

        start_f_hz = settings_7[0] * 1e6
        stop_f_hz = settings_7[1] * 1e6
        num_points = int(settings_7[2])
        shot_power = settings_7[4]

        data, idx = _read_float_pairs(lines, idx, num_points)

        ref = 10.0 ** ((shot_power - _POWER_CORRECTION_DB) / 20.0)
        data = data * ref

        # Convert settings[0:2] to Hz in the stored copy for parity with MATLAB output.
        settings_hz = settings_7.copy()
        settings_hz[0] = start_f_hz
        settings_hz[1] = stop_f_hz

        shot_num += 1
        shots.append(
            STKShot(
                shot_index=shot_num,
                timestamp=timestamp,
                settings=settings_hz,
                data=data,
            )
        )

    if not shots:
        raise ValueError(f"No shots found in {path}")

    return STKFile(file_path=path, cal_settings=cal_settings, cal_data=cal_data, shots=shots)


# ------------------------------------------------------------
# Frequency-domain -> time/range-domain
# ------------------------------------------------------------


def pres_freq2time(
    f_data: np.ndarray,
    start_f_hz: float,
    stop_f_hz: float,
    num_points: int,
    array_size: int,
    *,
    c_round_trip_m_s: float = _PRES_ONE_WAY_C_IN_ICE_M_S / 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port of freq2time.m.

    Takes the complex frequency-domain response of one shot, applies a
    Blackman window, builds a Hermitian-symmetric double-sided spectrum of
    length `array_size`, and IFFTs to a complex time/range-domain trace.

    Parameters
    ----------
    f_data
        Complex S21(f) for `num_points` frequencies.
    start_f_hz, stop_f_hz
        Frequency sweep limits in Hz.
    num_points
        Number of stepped frequencies (must be odd — pg5 assumes odd).
    array_size
        Length of the zero-padded IFFT (pad factor * num_points).
    c_round_trip_m_s
        Effective "two-way" speed of light in ice used to convert the
        IFFT time axis to range (default: c/sqrt(er_ice)/2 with er_ice ~ 3.18).

    Returns
    -------
    x_m
        Range axis in metres, length `array_size`.
    time_data
        Complex range-domain trace, length `array_size`.
    """

    f_data = np.asarray(f_data, dtype=np.complex128).reshape(-1)
    if f_data.size != num_points:
        raise ValueError(f"f_data length ({f_data.size}) != num_points ({num_points})")
    if num_points % 2 == 0:
        # pg5 assumes odd num_points so that (num_points+1)/2 is an integer.
        # We don't rewrite the algorithm for the even case — raise so the
        # caller knows.
        raise ValueError("pres_freq2time expects odd num_points (pg5 convention)")

    step_size = (stop_f_hz - start_f_hz) / (num_points - 1)
    max_range = c_round_trip_m_s / step_size
    dis_step = max_range / (array_size - 1)
    x_m = np.arange(array_size, dtype=float) * dis_step

    win = np.blackman(num_points)

    fw_sht = f_data * win
    # MATLAB: stop_pos_f = (num_points+1)/2; start_neg_f = stop_pos_f + 1
    # (MATLAB 1-based). In 0-based Python: stop_pos = (num_points+1)//2 entries,
    # start_neg = stop_pos (Python slice end-exclusive).
    stop_pos = (num_points + 1) // 2  # number of "positive-frequency" samples
    start_neg = stop_pos  # index where negative-frequency samples start

    freq_demod = np.zeros(array_size, dtype=np.complex128)
    # Positive half: conj(flip(fw_sht[1..stop_pos])), placed at indices 0..stop_pos-1
    freq_demod[:stop_pos] = 2.0 * np.conj(fw_sht[:stop_pos][::-1])
    # Negative half: conj(flip(fw_sht[start_neg..num_points-1])), placed at tail
    tail_len = num_points - start_neg
    if tail_len > 0:
        freq_demod[array_size - tail_len:] = 2.0 * np.conj(fw_sht[start_neg:num_points][::-1])

    time_data = np.fft.ifft(freq_demod)
    # Amplitude correction (Blackman CPG + double-sided normalization)
    time_data = time_data * (array_size / num_points)

    return x_m, time_data


# ------------------------------------------------------------
# Derived parameters for the pRES "profile"
# ------------------------------------------------------------


def _derive_pres_params(start_f_hz: float, stop_f_hz: float, er_ice: float) -> Tuple[float, float, float, float, float]:
    """Return (fc_hz, B_hz, ci_m_s, lambdac_m, f1_hz) for pRES."""

    fc_hz = 0.5 * (start_f_hz + stop_f_hz)
    B_hz = float(stop_f_hz - start_f_hz)
    ci_m_s = 299_792_458.0 / np.sqrt(er_ice)
    lambdac_m = ci_m_s / fc_hz
    return float(fc_hz), B_hz, float(ci_m_s), float(lambdac_m), float(stop_f_hz)


# ------------------------------------------------------------
# Build a PreprocessedProfile from a .STK file
# ------------------------------------------------------------


def _apply_vna_cal(
    shot_data: np.ndarray,
    cal_data: np.ndarray,
    *,
    amp_floor_frac: float = 0.02,
) -> np.ndarray:
    """Divide a frequency-domain shot by the in-file calibration sweep.

    Each .STK file begins with a VNA calibration sweep recorded seconds
    before the shots (at the same instrument temperature, cable seating,
    etc). Dividing shots by the cal normalizes out the VNA's own amplitude
    and phase response across frequency — the MATLAB `read_stk_v3.m`
    implements this division but the original code left it commented out.

    The division is guarded: where |cal| is below `amp_floor_frac * median(|cal|)`,
    we clamp to avoid amplification of noise at weakly-matched frequencies.

    Parameters
    ----------
    shot_data
        Complex S21(f) for the shot. Must have the same length as `cal_data`.
    cal_data
        Complex cal sweep from the same .STK file.
    amp_floor_frac
        Minimum |cal| expressed as a fraction of median(|cal|). Cal bins
        below this are clamped before division so that division does not
        blow up at band edges.

    Returns
    -------
    Corrected shot data, same shape as `shot_data`.
    """
    shot = np.asarray(shot_data, dtype=np.complex128)
    cal = np.asarray(cal_data, dtype=np.complex128)
    if shot.shape != cal.shape:
        raise ValueError(f"shot ({shot.shape}) and cal ({cal.shape}) must have the same shape")

    amp = np.abs(cal)
    floor = amp_floor_frac * float(np.median(amp))
    if floor <= 0:
        floor = 1e-30
    # Clamp amplitude while keeping phase
    weak = amp < floor
    if np.any(weak):
        phase = np.angle(cal[weak])
        cal = cal.copy()
        cal[weak] = floor * (np.cos(phase) + 1j * np.sin(phase))

    return shot / cal


def preprocess_stk_file(
    path: Path,
    cfg: ProcessingConfig,
    *,
    shot_selection: str = "mean",
    apply_vna_cal: bool = True,
) -> PreprocessedProfile:
    """Read a .STK file and return a PreprocessedProfile.

    `shot_selection` controls how multiple shots within one .STK are combined:
      - "mean"  : coherent mean of all shots in the file (default, best SNR)
      - "first" : use only shot 1 (matches what pg5's default UI does)

    `apply_vna_cal=True` divides each shot's frequency-domain data by the
    in-file VNA calibration sweep, removing common-mode instrument response
    (cable temperature, VNA drift, connector seating). This is the same
    operation that `read_stk_v3.m` implements but leaves commented out.

    All shots within one .STK are recorded at the same station within seconds,
    so averaging them coherently is safe.
    """

    if not getattr(cfg, "_validated", False):
        cfg.validate()

    stk = read_stk(path)

    # Check all shots have matching settings (start_f, stop_f, num_points)
    s0 = stk.shots[0].settings
    start_f_hz = float(s0[0])
    stop_f_hz = float(s0[1])
    num_points = int(s0[2])

    for sh in stk.shots[1:]:
        if int(sh.settings[2]) != num_points:
            raise ValueError(
                f"Inconsistent num_points across shots in {path}: {int(sh.settings[2])} vs {num_points}"
            )
        # Allow tiny float drift in start/stop, but warn if >1 kHz off.
        if abs(float(sh.settings[0]) - start_f_hz) > 1e3 or abs(float(sh.settings[1]) - stop_f_hz) > 1e3:
            raise ValueError(
                f"Shots in {path} disagree on sweep range: "
                f"shot1=({start_f_hz:.0f}..{stop_f_hz:.0f}), "
                f"shot{sh.shot_index}=({float(sh.settings[0]):.0f}..{float(sh.settings[1]):.0f})"
            )

    # Apply VNA calibration correction to each shot (divide by cal sweep)
    if apply_vna_cal:
        if stk.cal_data.size != num_points:
            # The MATLAB code requires cal settings to match shot settings;
            # our dataset satisfies this, but guard anyway.
            raise ValueError(
                f"cal_data length ({stk.cal_data.size}) != num_points ({num_points}); "
                "cal settings do not match shot settings — skip VNA cal with apply_vna_cal=False"
            )
        shot_list = [_apply_vna_cal(sh.data, stk.cal_data) for sh in stk.shots]
    else:
        shot_list = [sh.data for sh in stk.shots]

    # Combine shots
    data_matrix = np.stack(shot_list, axis=0)  # (nshots, num_points)
    if shot_selection == "first":
        f_data = data_matrix[0]
    elif shot_selection == "mean":
        f_data = np.mean(data_matrix, axis=0)
    else:
        raise ValueError(f"Unknown shot_selection: {shot_selection!r}")

    # Timestamp: first shot (all shots within a .STK were taken within seconds)
    timestamp = stk.shots[0].timestamp

    # Array size (zero-padded IFFT length). pg5 uses 25*num_points;
    # map cfg.pad_factor (default 8) to this role.
    pad_factor = int(cfg.pad_factor)
    array_size = pad_factor * num_points

    # ci_m_s is the one-way speed in ice; c_round_trip = ci/2 matches pg5's "C=84e6".
    er_ice = float(cfg.relative_permittivity_ice)
    fc_hz, B_hz, ci_m_s, lambdac_m, f1_hz = _derive_pres_params(start_f_hz, stop_f_hz, er_ice)
    c_round_trip = ci_m_s / 2.0

    # Range-domain trace
    x_m_full, time_data = pres_freq2time(
        f_data,
        start_f_hz=start_f_hz,
        stop_f_hz=stop_f_hz,
        num_points=num_points,
        array_size=array_size,
        c_round_trip_m_s=c_round_trip,
    )

    # Crop to cfg.max_range_m
    keep = x_m_full <= float(cfg.max_range_m)
    if not np.any(keep):
        # If max_range_m is shorter than the smallest bin, keep the first bin.
        keep = np.zeros_like(x_m_full, dtype=bool)
        keep[0] = True

    range_coarse_m = x_m_full[keep]
    spec_cor = time_data[keep]
    spec_raw = spec_cor.copy()  # pRES has no separate "phase reference" step

    # Fine range offset from phase.
    # For pRES there is no FMCW chirp (K=0), so the apres phase2range
    # reduces to Rfine = phi / (4*pi/lambdac) = phi * lambdac / (4*pi).
    range_fine_m = np.angle(spec_cor) * lambdac_m / (4.0 * np.pi)
    range_m = range_coarse_m + range_fine_m

    # Phase standard error: estimate from across shots when >1 available.
    nshots = int(data_matrix.shape[0])
    if nshots >= 2:
        # Process each shot to range domain and measure complex std / |mean|.
        all_specs = np.empty((nshots, range_coarse_m.size), dtype=np.complex128)
        for i in range(nshots):
            _, td_i = pres_freq2time(
                data_matrix[i],
                start_f_hz=start_f_hz,
                stop_f_hz=stop_f_hz,
                num_points=num_points,
                array_size=array_size,
                c_round_trip_m_s=c_round_trip,
            )
            all_specs[i, :] = td_i[keep]
        mean_abs = np.abs(np.mean(all_specs, axis=0))
        # MATLAB: phase_std_dev = std(F)/(sqrt(2)*|mean|); phase_std_error = that / sqrt(n-1)
        std_complex = np.std(all_specs, axis=0, ddof=1)
        denom = np.sqrt(2.0) * np.maximum(mean_abs, 1e-30)
        phase_std_dev = std_complex / denom
        phase_std_error_rad = np.real(phase_std_dev) / np.sqrt(max(nshots - 1, 1))
        phase_std_error_rad = np.asarray(np.abs(phase_std_error_rad), dtype=float)
    else:
        # No cross-shot info available; use a conservative constant floor.
        phase_std_error_rad = np.full(range_coarse_m.size, 0.1, dtype=float)

    range_error_m = phase_std_error_rad * lambdac_m / (4.0 * np.pi)

    # Build a minimal burst-like object just for estimate_ice_thickness
    # (it only needs range_coarse_m and the spec_cor amplitude).
    class _FakeBurst:  # tiny shim — estimate_ice_thickness doesn't use radar params here
        pass

    fake = _FakeBurst()
    fake.chirps_in_burst = 1

    bed_idx, bed_depth = estimate_ice_thickness(
        fake,  # type: ignore[arg-type]
        cfg,
        range_coarse_m=range_coarse_m,
        spec_cor_mean=spec_cor,
    )

    # Noise depth: fall back to bed depth (no chirp-half noise estimate
    # is available without multiple chirps).
    noise_depth = float(bed_depth)

    return PreprocessedProfile(
        file_path=Path(path),
        name=Path(path).stem,
        timestamp=timestamp,
        gps_on=None,
        latitude_deg=None,
        longitude_deg=None,
        range_coarse_m=range_coarse_m,
        range_fine_m=range_fine_m,
        range_m=range_m,
        spec_raw=spec_raw,
        spec_cor=spec_cor,
        phase_std_error_rad=phase_std_error_rad,
        range_error_m=range_error_m,
        bed_depth_m=float(bed_depth),
        bed_index=int(bed_idx),
        noise_depth_m=noise_depth,
        fs_hz=float(num_points),  # not physically meaningful for pRES; kept for dataclass completeness
        f0_hz=float(start_f_hz),
        f1_hz=float(f1_hz),
        B_hz=float(B_hz),
        fc_hz=float(fc_hz),
        K_rad_s2=0.0,  # no FMCW chirp — setting K=0 reduces phase2range to the stepped-frequency formula
        ci_m_s=float(ci_m_s),
        lambdac_m=float(lambdac_m),
        er_ice=float(er_ice),
        pad_factor=int(pad_factor),
        subband_edges_hz=None,
        subband_fc_hz=None,
        spec_cor_subbands=None,
    )


def find_stk_files(root: Path) -> List[Path]:
    """Recursively find .STK files under *root* (case-insensitive)."""

    root = Path(root)
    files: List[Path] = []
    for pat in ("*.STK", "*.stk"):
        files.extend(p for p in root.rglob(pat) if p.is_file())
    # Drop macOS AppleDouble files
    files = [p for p in files if not p.name.startswith("._")]
    return sorted(files)


# ------------------------------------------------------------
# Within-visit shot noise diagnostics
# ------------------------------------------------------------


@dataclass
class ShotNoiseDiagnostics:
    """Per-depth shot-to-shot noise statistics for one .STK file.

    Captures the achievable noise floor at this station/visit, independent
    of what any other visit measured. This is the hard limit on how small
    a motion an inter-visit pair could detect.
    """

    file_path: Path
    timestamp: "datetime"
    nshots: int
    range_coarse_m: np.ndarray  # (nr,)

    # Across-shot complex mean and std at each depth (shape nr)
    amp_mean: np.ndarray  # |<x>| = coherent mean amplitude
    amp_std_incoh: np.ndarray  # std(|x|) = incoherent amplitude std
    complex_std: np.ndarray  # std(x) across shots (complex std, |·|)

    # Phase std (radians). MATLAB convention:
    #   σφ = |std(x)| / (sqrt(2) * |<x>|)
    phase_std_rad: np.ndarray  # per-shot phase σ at each depth
    phase_stderr_rad: np.ndarray  # phase σ of the MEAN of N shots = phase_std / sqrt(N-1)

    # Equivalent range-domain uncertainty (m) for the mean-profile
    # (what gets stored in the PreprocessedProfile.range_error_m)
    range_stderr_m: np.ndarray

    # Coherent SNR = |<x>| / |std(x)|
    snr: np.ndarray  # dimensionless

    def summary(self, ref_depth_m: float = 60.0, ref_half_win_m: float = 20.0,
                bed_depth_m: Optional[float] = None, bed_half_win_m: float = 5.0) -> dict:
        """Return a dict summarising noise in the ref and bed windows."""
        r = self.range_coarse_m
        out = {"nshots": int(self.nshots)}

        def _stats(mask: np.ndarray) -> dict:
            if not mask.any():
                return {"phase_stderr_rad": float("nan"),
                        "range_stderr_mm": float("nan"),
                        "snr_db": float("nan")}
            pse = float(np.median(self.phase_stderr_rad[mask]))
            rse_mm = 1000.0 * float(np.median(self.range_stderr_m[mask]))
            snr_db = 20.0 * np.log10(np.median(self.snr[mask]) + 1e-30)
            return {"phase_stderr_rad": pse, "range_stderr_mm": rse_mm, "snr_db": float(snr_db)}

        ref_mask = (r >= ref_depth_m - ref_half_win_m) & (r <= ref_depth_m + ref_half_win_m)
        for k, v in _stats(ref_mask).items():
            out[f"ref_{k}"] = v

        if bed_depth_m is not None:
            bed_mask = (r >= bed_depth_m - bed_half_win_m) & (r <= bed_depth_m + bed_half_win_m)
            for k, v in _stats(bed_mask).items():
                out[f"bed_{k}"] = v
        return out


def shot_noise_diagnostics(
    path: Path,
    cfg: ProcessingConfig,
    *,
    apply_vna_cal: bool = True,
) -> ShotNoiseDiagnostics:
    """Compute per-depth shot-to-shot noise for one .STK file.

    Each shot in the file is processed independently to a complex range
    trace; we then take the across-shot statistics at every range bin.

    Returns a `ShotNoiseDiagnostics`; the relevant numbers are
    `phase_stderr_rad` (per-depth phase std of the mean profile),
    `range_stderr_m` (equivalent range uncertainty), and `snr` (coherent
    SNR = |mean| / |std|).
    """
    if not getattr(cfg, "_validated", False):
        cfg.validate()

    stk = read_stk(path)

    s0 = stk.shots[0].settings
    start_f_hz = float(s0[0])
    stop_f_hz = float(s0[1])
    num_points = int(s0[2])

    if apply_vna_cal:
        shot_list = [_apply_vna_cal(sh.data, stk.cal_data) for sh in stk.shots]
    else:
        shot_list = [sh.data for sh in stk.shots]

    nshots = len(shot_list)
    pad_factor = int(cfg.pad_factor)
    array_size = pad_factor * num_points

    er_ice = float(cfg.relative_permittivity_ice)
    _, _, ci_m_s, lambdac_m, _ = _derive_pres_params(start_f_hz, stop_f_hz, er_ice)
    c_round_trip = ci_m_s / 2.0

    # Range-domain per shot
    all_td = None
    x_m_full = None
    for i, f_data in enumerate(shot_list):
        x_m_full, td = pres_freq2time(
            f_data,
            start_f_hz=start_f_hz,
            stop_f_hz=stop_f_hz,
            num_points=num_points,
            array_size=array_size,
            c_round_trip_m_s=c_round_trip,
        )
        if all_td is None:
            all_td = np.empty((nshots, td.size), dtype=np.complex128)
        all_td[i] = td

    # Crop to max_range_m
    keep = x_m_full <= float(cfg.max_range_m)
    if not keep.any():
        keep = np.zeros_like(x_m_full, dtype=bool)
        keep[0] = True

    r = x_m_full[keep]
    specs = all_td[:, keep]  # (nshots, nr)

    amp_mean = np.abs(np.mean(specs, axis=0))
    amp_std_incoh = np.std(np.abs(specs), axis=0, ddof=1) if nshots >= 2 else np.zeros_like(amp_mean)
    complex_std = np.abs(np.std(specs, axis=0, ddof=1)) if nshots >= 2 else np.zeros_like(amp_mean)

    # Phase std (radians)
    if nshots >= 2:
        phase_std = complex_std / (np.sqrt(2.0) * np.maximum(amp_mean, 1e-30))
        phase_stderr = phase_std / np.sqrt(max(nshots - 1, 1))
    else:
        phase_std = np.full_like(amp_mean, np.nan)
        phase_stderr = np.full_like(amp_mean, np.nan)

    range_stderr_m = phase_stderr * lambdac_m / (4.0 * np.pi)
    snr = amp_mean / np.maximum(complex_std, 1e-30)

    return ShotNoiseDiagnostics(
        file_path=Path(path),
        timestamp=stk.shots[0].timestamp,
        nshots=int(nshots),
        range_coarse_m=r,
        amp_mean=np.asarray(amp_mean, dtype=float),
        amp_std_incoh=np.asarray(amp_std_incoh, dtype=float),
        complex_std=np.asarray(complex_std, dtype=float),
        phase_std_rad=np.asarray(phase_std, dtype=float),
        phase_stderr_rad=np.asarray(phase_stderr, dtype=float),
        range_stderr_m=np.asarray(range_stderr_m, dtype=float),
        snr=np.asarray(snr, dtype=float),
    )


def pair_precision_floor(
    sn1: ShotNoiseDiagnostics, sn2: ShotNoiseDiagnostics
) -> Tuple[np.ndarray, np.ndarray]:
    """Combined bed-shift precision floor (m) for a pair.

    Range uncertainty for measuring a shift between two visits is
    sqrt(σ_1² + σ_2²) at each depth. Returns (range_axis, sigma_combined_m).

    Assumes both profiles share the same range axis (they do by construction).
    """
    r = sn1.range_coarse_m
    n = min(r.size, sn2.range_coarse_m.size)
    sig = np.sqrt(sn1.range_stderr_m[:n] ** 2 + sn2.range_stderr_m[:n] ** 2)
    return r[:n], sig
