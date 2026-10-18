r"""ISMIP7 climate forcing for the prognostic forward: surface mass balance and sub-shelf melt.

The forcing follows the Antarctic ISMIP7 set-up of the sibling ismip7 project
(/media/andrew/wd1/projects/ismip7, antarctica/scripts/experiment.py and
icepack2_tools/forcing.py), whose data readers are reused here so that both
projects read the same files the same way.

Surface mass balance, as ismip7's ``smb_scheme`` builds it:

    SMB(t) = RACMO2.4p1 mean over CLIM_START..CLIM_END
             + [aSMB(year(t)) - mean aSMB over historical + CLIM_SCENARIO, CLIM_START..CLIM_END]

with aSMB the ISMIP7 ``acabf-anomaly`` of the chosen ESM and scenario, an annual
mean, nearest-neighbour onto the mesh. aSMB is in kg m^-2 s^-1 and is converted
with SMB [m ice/yr] = aSMB * seconds_per_year / rho_ice. (ismip7's own
``smb_kgm2s_to_myr`` multiplies by a further 1000/917, which over-scales the
anomaly by 9 %; that factor is deliberately not reproduced.)

Sub-shelf melt is the Burgard et al. (2022) quadratic mixed-slope law, local
variant, with the ISMIP7 per-basin calibration of K (ismip7
antarctica/results/calibrated_K_per_basin_2500.npz):

    m = K_basin * (rho_sw c_po / (rho_i L_i)) * (c_po / L_i) beta_S g / (2|f|) S
        * TF |TF| * sin(alpha) * seconds_per_year           [m ice/yr]

TF and S are the ISMIP7 ocean v3 thermal forcing and salinity of the year, at the
nearest of the 60 m levels to the ice draft. sin(alpha) is the slope of the ice
base, capped at SIN_ALPHA_CAP as in the K calibration (calibrate_melt.py). The
forward multiplies the melt by its own floating mask.

No apparent or reference mass balance is added anywhere: the state is not in
balance and that imbalance is part of the projection.

Split for the adjoint: everything that comes from data - K, S, TF at the current
draft, gathered into one coefficient field per step - is a plain field held off
the tape. The slope sin(alpha) is a UFL expression of the geometry, so the
melt's dependence on the shape of the ice base is differentiated. TF is a
nearest-level lookup, piecewise constant in the draft, so its derivative with
respect to the draft is zero almost everywhere and nothing is lost by keeping it
off the tape.
"""
import sys
from pathlib import Path

import numpy as np
from firedrake import Constant, inner, grad, min_value, sqrt

ISMIP7_ROOT = Path("/media/andrew/wd1/projects/ismip7")
if str(ISMIP7_ROOT) not in sys.path:
    sys.path.insert(0, str(ISMIP7_ROOT))
from icepack2_tools import forcing as F  # noqa: E402

SEC_PER_YEAR = F._SEC_PER_YEAR
RHO_ICE = F._RHO_ICE
CLIM_START, CLIM_END = 2000, 2029     # ismip7 icepack2_tools/climatology.py defaults
CLIM_SCENARIO = "ssp126"              # pooled with historical over the window
K_NPZ = ISMIP7_ROOT / "antarctica" / "results" / "calibrated_K_per_basin_2500.npz"
SIN_ALPHA_CAP = 5e-3                  # the cap the K calibration used
# K * this * S * TF|TF| * sin(alpha) is the melt in m ice/yr (forcing.quadratic_mixed_slope).
MELT_PREFACTOR = (F._MELT_FACTOR * (F._C_PO / F._L_I) * F._BETA_S
                  * (F._G / (2.0 * abs(F._F_CORIOLIS))) * SEC_PER_YEAR)


def kgm2s_to_myr(x):
    r"""kg m^-2 s^-1 of ice to m/yr of ice."""
    return np.asarray(x) * SEC_PER_YEAR / RHO_ICE


class Ismip7Forcing:
    r"""ISMIP7 SMB and ocean forcing sampled at the vertices of a 2D CG1 space.

    ``Q_2d`` is the footprint space; for the extruded hybrid model the forward
    copies these per-vertex arrays straight into its CG1 x DG0 space, which has
    one dof per footprint vertex in the same order.
    """

    def __init__(self, Q_2d, esm="CESM2-WACCM", scenario="ssp585", start_year=2015):
        self.esm, self.scenario, self.start_year = esm, scenario, int(start_year)
        xy = Q_2d.mesh().coordinates.dat.data_ro
        self.x, self.y = xy[:, 0].copy(), xy[:, 1].copy()

        self.atm = F.ISMIP7Atmosphere(esm=esm, scenario=scenario)
        self.ocean = F.ISMIP7Ocean(esm=esm, scenario=scenario)
        years = self.atm.available_years("acabf-anomaly")
        if not years:
            raise FileNotFoundError(f"no ISMIP7 acabf-anomaly files for {esm} {scenario}")
        self.smb_years = (min(years), max(years))

        # Baseline: RACMO climatology minus the pooled mean anomaly over the same window.
        pool = []
        for a in (F.ISMIP7Atmosphere(esm=esm, scenario="historical"),
                  F.ISMIP7Atmosphere(esm=esm, scenario=CLIM_SCENARIO)):
            pool += [(a, yr) for yr in a.available_years("acabf-anomaly")
                     if CLIM_START <= yr <= CLIM_END]
        if not pool:
            raise FileNotFoundError(f"no acabf-anomaly years in {CLIM_START}-{CLIM_END} "
                                    f"for {esm} historical/{CLIM_SCENARIO}")
        ref = sum(kgm2s_to_myr(a.get_field("acabf-anomaly", yr, self.x, self.y))
                  for a, yr in pool) / len(pool)
        racmo = F.load_racmo_smb_climatology(Q_2d, CLIM_START, CLIM_END).dat.data_ro
        self.smb_baseline = racmo - ref
        self.n_pool = len(pool)

        self.K = F.load_K_per_basin(str(K_NPZ), self.x, self.y, fill=0.0)
        self._smb_cache = {}

    def year(self, t):
        r"""Calendar year of model time ``t`` (years since the start of the run)."""
        return self.start_year + int(np.floor(t + 1e-9))

    def smb(self, t):
        r"""SMB [m ice/yr] at the vertices for the year containing model time ``t``."""
        yr = self.year(t)
        if yr not in self._smb_cache:
            if not self.smb_years[0] <= yr <= self.smb_years[1]:
                raise ValueError(f"year {yr} outside the {self.esm} {self.scenario} "
                                 f"acabf-anomaly coverage {self.smb_years}")
            anom = kgm2s_to_myr(self.atm.get_field("acabf-anomaly", yr, self.x, self.y))
            self._smb_cache = {yr: self.smb_baseline + anom}
        return self._smb_cache[yr]

    def melt_coefficient(self, t, draft):
        r"""K * prefactor * S * TF|TF| at the vertices [m ice/yr per unit sin(alpha)].

        ``draft`` is the elevation of the ice base, s - h (negative below sea level).
        """
        yr = self.year(t)
        d = np.minimum(np.asarray(draft), 0.0)
        tf = self.ocean.get_thermal_forcing(yr, self.x, self.y, d)
        so = self.ocean.get_salinity(yr, self.x, self.y, d)
        return self.K * MELT_PREFACTOR * so * tf * np.abs(tf)

    def describe(self):
        return (f"ISMIP7 {self.esm} {self.scenario} from {self.start_year}: SMB = RACMO2.4p1 "
                f"{CLIM_START}-{CLIM_END} + aSMB re-referenced over {self.n_pool} pooled years "
                f"(historical + {CLIM_SCENARIO}); melt = Burgard 2022 quadratic mixed-slope, "
                f"per-basin K, sin(alpha) capped at {SIN_ALPHA_CAP:g}")


def sin_slope(h, s):
    r"""sin of the ice-base slope as UFL, capped at SIN_ALPHA_CAP.

    |grad z_b| / sqrt(1 + |grad z_b|^2) with z_b = s - h; the tiny shift under the
    square root keeps the derivative finite where the base is flat.
    """
    gz = grad(s - h)
    n2 = inner(gz, gz)
    return min_value(sqrt(n2 + Constant(1e-24)) / sqrt(1 + n2), Constant(SIN_ALPHA_CAP))
