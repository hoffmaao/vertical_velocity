r"""Minimal reader for ApRES (BAS pRES) burst files and their power-depth profile.

Follows the BAS ``fmcw_load`` / ``fmcw_range`` processing (Brennan et al., 2014;
Nicholls et al., 2015) as far as the return power needs it:

    ASCII burst header ... "*** End Header ***", then NSubBursts * nAttenuators
    chirps of N_ADC_SAMPLES little-endian uint16 samples (Average = 0),
    attenuator settings interleaved chirp by chirp.

Each chirp is scaled to volts, mean-removed, Blackman-windowed, zero-padded by
``pad`` and Fourier transformed; the power profile is the mean of |FFT|^2 over
the chirps of the first attenuator setting. Range is c / (2 B sqrt(eps_r)) per
unpadded bin. Only the amplitude is used here, so the phase reference of
``fmcw_range`` is not needed.

    from apres_dat import power_profile
    r, p_db = power_profile(path)          # range below the antennas (m), power (dB)
"""
from pathlib import Path

import numpy as np

C0 = 299792458.0
END = b"*** End Header ***"
VOLTS_PER_COUNT = 2.5 / 2 ** 16


def _header(text):
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_burst(path):
    r"""First burst of an ApRES file: (header dict, chirps[n_chirp, n_sample] in volts)."""
    raw = Path(path).read_bytes()
    i = raw.find(END)
    if i < 0:
        raise ValueError(f"{path}: no burst header")
    hdr = _header(raw[:i].decode("latin-1"))
    if int(hdr.get("Average", "0")) != 0:
        raise ValueError(f"{path}: averaged bursts (Average != 0) are not supported")
    start = raw.find(b"\n", i) + 1
    n_s = int(hdr["N_ADC_SAMPLES"])
    n_att = int(hdr.get("nAttenuators", "1"))
    n_c = int(hdr["NSubBursts"]) * n_att
    data = np.frombuffer(raw, dtype="<u2", count=n_c * n_s, offset=start)
    chirps = data.reshape(n_c, n_s).astype(float) * VOLTS_PER_COUNT
    hdr["_n_att"] = n_att
    return hdr, chirps


def power_profile(path, pad=2, max_range=3000.0):
    r"""Mean power (dB re 1 V^2) against range below the antennas for the first attenuator."""
    hdr, chirps = read_burst(path)
    chirps = chirps[::hdr["_n_att"]]
    n_s = chirps.shape[1]
    B = float(hdr["StopFreq"]) - float(hdr["StartFreq"])
    er = float(hdr.get("ER_ICE", "3.18"))
    x = (chirps - chirps.mean(axis=1, keepdims=True)) * np.blackman(n_s)
    nfft = pad * n_s
    spec = np.fft.rfft(x, n=nfft, axis=1) / n_s
    p = np.mean(np.abs(spec) ** 2, axis=0)
    r = np.arange(p.size) * C0 / (2 * B * np.sqrt(er)) / pad
    keep = r <= max_range
    return r[keep], 10 * np.log10(p[keep] + 1e-30)
