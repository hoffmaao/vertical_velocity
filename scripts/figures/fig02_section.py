"""Figure 2 - the along-flow section of the repeat ApRES survey.

Two panels sharing a distance axis.  The upper panel is the return
signal-to-noise ratio from the 2022-2023 acquisitions, which documents how deep
the radar sees and therefore how deep layers can be tracked.  The lower panel
is the vertical strain rate, dw/dz, on a diverging ramp centred on zero so the
along-flow sign reversal is the thing the eye lands on.

The lower panel shows the strain rate rather than the vertical velocity it was
differentiated from.  w(z) is referenced to the shallowest tracked segment at
each site, so its absolute value carries that arbitrary choice and is not
comparable between sites; its depth gradient is, and it is the gradient that
enters the incompressibility condition and therefore the inferred horizontal
shear.  The velocity profiles themselves appear in Figure 4 and in the
supplement.

The upper panel shows power referenced to each acquisition's own receiver noise
floor rather than raw returned power.  Attenuator settings were changed
between sites in the field, so absolute power is not comparable along the
transect: plotted raw, the section shows the gain schedule as much as the ice.
Referencing each profile to its own noise floor removes that, and leaves a
quantity - how far the return stands above noise - that is both comparable
between sites and the thing that actually decides how deep layers can be
tracked below.

Only the ice surface is drawn over the return-power panel.  A modelled bed laid over a
measured radargram invites the reader to check one against the other, which is
not what that panel is for; the bed appears below it, where it provides
the geometric context for the strain rate, and the comparison between the
ApRES picks and BedMachine is made directly in Figure 3.

Each site is drawn as a column of constant width in distance, which is what
the data are: independent point occupations, not a continuously sampled line.

Usage
-----
    python scripts/figures/fig02_section.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

import paths  # noqa: E402
import style  # noqa: E402
import surface  # noqa: E402

COLUMN_WIDTH_KM = 0.55   # drawn half-width of each site column
DEPTH_STEP_M = 10.0      # vertical resampling of the power profile
SPECKLE_SMOOTH = 3       # bins of incoherent averaging applied after gain removal


def load_power_section(pairs: pd.DataFrame):
    """Return power vs elevation for every site, on a common depth grid."""
    from apres.bedpick import noise_floor_db

    depth = np.arange(0.0, 2500.0, DEPTH_STEP_M)
    power = np.full((len(pairs), depth.size), np.nan)
    for i, (_, row) in enumerate(pairs.iterrows()):
        f = paths.ROOT / row["file_1"]
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        r = z["range_m"]
        db = 20.0 * np.log10(np.abs(z["spec_cor_real"] + 1j * z["spec_cor_imag"]) + 1e-30)
        # Average within each output bin rather than sampling it, so the
        # displayed power is not aliased by the 5 cm native bin spacing.
        idx = np.clip(((r - depth[0]) / DEPTH_STEP_M).astype(int), 0, depth.size - 1)
        total = np.bincount(idx, weights=db, minlength=depth.size)
        count = np.bincount(idx, minlength=depth.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            binned = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        power[i] = binned - noise_floor_db(r, db)
    if not np.isfinite(power).any():
        # Otherwise the top panel is drawn blank and the figure still saves.
        raise FileNotFoundError(
            f"none of the {len(pairs)} preprocessed profiles in {paths.PREPROCESSED} were found; "
            "run compute_vertical_velocities.py first"
        )
    return depth, power


def main() -> None:
    style.use_paper_style()
    import xarray as xr

    pairs = pd.read_csv(paths.PAIR_CATALOG).sort_values("distance_km").reset_index(drop=True)
    ds = xr.open_dataset(paths.VV_PROFILES)

    surface_bm = pairs["bm_surface_m"].values
    bed_bm = pairs["bm_bed_m"].values
    dist = pairs["distance_km"].values

    fig, axs = plt.subplots(
        2, 1, figsize=(style.TWO_COL, 0.62 * style.TWO_COL),
        sharex=True, gridspec_kw={"hspace": 0.11},
    )

    # ---------------------------------------------------------------- (a) power
    depth, power = load_power_section(pairs)
    # The firn and the brightest near-surface reflector are 30-40 dB above
    # anything below them; letting them set the colour limits would render the
    # whole englacial column one flat tone.  The scale is therefore taken from
    # the ice below the firn, and the shallow band simply saturates.
    # Incoherent averaging in depth: the profile is speckled at the wavelength
    # scale, and without it the panel shows fading rather than layering.
    if SPECKLE_SMOOTH > 1:
        k = np.ones(SPECKLE_SMOOTH) / SPECKLE_SMOOTH
        power = np.apply_along_axis(
            lambda row: np.convolve(row, k, mode="same"), 1, power
        )
    norm_p = Normalize(0.0, float(np.nanpercentile(power, 99.5)))
    for xi, s, col in zip(dist, surface_bm, power):
        good = np.isfinite(col)
        if not good.any():
            continue
        axs[0].pcolormesh(
            np.array([xi - COLUMN_WIDTH_KM, xi + COLUMN_WIDTH_KM]),
            np.append(s - depth - DEPTH_STEP_M / 2, s - depth[-1] + DEPTH_STEP_M / 2)[::-1],
            col[::-1, None],
            cmap=style.SEQ_POWER, norm=norm_p, shading="flat", rasterized=True,
        )
    m_power = plt.cm.ScalarMappable(norm=norm_p, cmap=style.SEQ_POWER)
    style.thin_colorbar(fig, m_power, axs[0], "return SNR (dB)")

    # ------------------------------------------------------ (b) strain rate
    site_index = {s: i for i, s in enumerate(ds["site"].values)}
    order = [site_index.get(s, -1) for s in pairs["site"]]
    keep = [i for i, o in enumerate(order) if o >= 0]

    strain = xr.open_dataset(paths.VERTICAL_VELOCITY / "strain_profiles.nc")
    strain_sites = list(strain["site"].values)
    e_norm = Normalize(*style.symmetric_limits(strain["strain_rate"].values, 99.0))

    for i in keep:
        site = str(ds["site"].values[order[i]])
        if site not in strain_sites:
            continue
        k = strain_sites.index(site)
        zs = strain["depth"].values[k]
        es = strain["strain_rate"].values[k]
        gs = np.isfinite(zs) & np.isfinite(es)
        if not gs.any():
            continue
        e = surface_bm[i] - zs[gs]
        axs[1].pcolormesh(
            np.array([dist[i] - COLUMN_WIDTH_KM, dist[i] + COLUMN_WIDTH_KM]),
            np.append(e - 1.0, e[-1] + 1.0),
            es[gs][:, None],
            cmap=style.DIV_VELOCITY, norm=e_norm, shading="flat", rasterized=True,
        )

    style.thin_colorbar(
        fig, plt.cm.ScalarMappable(norm=e_norm, cmap=style.DIV_VELOCITY), axs[1],
        r"$\dot\varepsilon_{zz}$ (yr$^{-1}$)",
    )
    # ------------------------------------------------------------- geometry
    # The surface comes from REMA at 32 m along a densely sampled path, not
    # from the 500 m BedMachine surface read once per site: at one point per
    # occupation the ice top draws as a chain of straight segments between
    # radar stations rather than as a surface.
    rema_d, rema_z = surface.surface_profile(
        pairs["x_ps_m"].values, pairs["y_ps_m"].values, dist
    )
    if rema_z is None or not np.isfinite(rema_z).any():
        rema_d, rema_z = dist, surface_bm

    for ax in axs:
        ax.plot(rema_d, rema_z, color=style.COLOR_SURFACE_LINE, lw=1.0, zorder=5)
        ax.set_ylabel("elevation (m)")
        ax.set_ylim(np.nanmin(bed_bm) - 120, np.nanmax(rema_z) + 160)
        ax.margins(x=0.01)

    # The bed is drawn only under the strain rate, where it is the context for
    # the measurement rather than an overlay on it.  Neither line is keyed on
    # the panel: the surface and the bed are identified in the caption.
    axs[1].plot(dist, bed_bm, color=style.COLOR_BED_BEDMACHINE, lw=1.1, zorder=5)

    axs[1].set_xlabel("distance along flow (km)")

    style.save(fig, paths.FIGURES / "fig02_section.png")


if __name__ == "__main__":
    main()
