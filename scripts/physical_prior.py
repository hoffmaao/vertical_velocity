r"""Physical prior means for the Thwaites inversions (fenics_ice / ISMIP7 method).

The inversion controls are log-deviations from physical fields rather than from
constants, so the prior regularizes the DEVIATION and theta stays O(1):

    theta_C = log(C / C_w0),     theta_A = log(A / A_prior).

C_w0 : driving-stress balance anchor. At theta_C = 0 and u = u_obs the Weertman
       stress C_w0 phi |u_obs|^(1/m) equals the driving stress tau_d. tau_d is
       Helmholtz-smoothed over TAU_D_SMOOTH (the CG1 surface gradient is
       cell-wise and noisy); phi and |u_obs| are floored so the anchor stays
       finite at flotation and in stagnant ice.
tau_c0 : the same balance for the yield-stress law (friction_laws.schoof): at
       theta_C = 0 and u = u_obs the Schoof stress tau_c0 g(HAF) |u_obs|^(1/m) /
       (U_0^(4/3) + |u_obs|^(4/3))^(1/4) equals tau_d, g the grounding-line
       taper of the law, floored at GL_TAPER_MIN so the anchor stays finite
       where the observed ice is within a few metres of flotation.
A_prior : thermomechanical fluidity from a fixed-velocity depth-averaged enthalpy
       solve (thermo_model.py, vendored from ~/projects/ismip7) driven by the
       observed velocity, ISMIP7 tas mean-annual surface temperature, RACMO SMB,
       and frictional heating from C_w0.

Both are computed once on the 2D mesh and saved to mesh/physical_prior.h5; the
SSA and hybrid inversions load the same fields (the hybrid lifts them to its
extruded mesh), so the two models share one prior mean.

Run:  OMP_NUM_THREADS=1 python scripts/physical_prior.py
      OMP_NUM_THREADS=1 python scripts/physical_prior.py --append-schoof
          adds tau_c0 to an existing prior file without recomputing A_prior
          (the thermal solve), so a prior already inverted with is unchanged.
"""
import argparse
import numpy as np
import netCDF4 as nc
import xarray as xr
import firedrake as fd
from firedrake import (
    CheckpointFile, Constant, Function, FunctionSpace, VectorFunctionSpace,
    TestFunction, TrialFunction, dx, grad, inner, max_value, sqrt,
)
from icepack.constants import gravity as g, ice_density as ρ_I, weertman_sliding_law as m
import os
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator, griddata

from friction_laws import GL_THRESH, SCHOOF_U0, flotation_factor, gl_taper
from thermo_model import compute_fluidity_prior

DATA = Path(__file__).resolve().parent.parent
# VV_MESH selects the mesh every script builds on, so a corrected mesh can be
# built and inverted while another one is still in use. Default: the existing
# mesh, so nothing already running changes behaviour.
MESH_NAME = os.environ.get("VV_MESH", "thwaites")
MESH_FILE = DATA / "mesh" / f"{MESH_NAME}.h5"
# VV_TAU_SMOOTH (metres) overrides the Helmholtz length applied to the driving
# stress in the anchor, for experiments on how rough the friction prior mean
# should be. Non-default values get their own prior file so the production
# prior is never overwritten; invert.py records the value in its summary.
TAU_D_SMOOTH = float(os.environ.get("VV_TAU_SMOOTH", 2000.0))
_TAU_SUFFIX = "" if TAU_D_SMOOTH == 2000.0 else f"_tau{TAU_D_SMOOTH / 1000:g}km"
PRIOR_FILE = DATA / "mesh" / (
    ("physical_prior" if MESH_NAME == "thwaites" else f"physical_prior_{MESH_NAME}")
    + _TAU_SUFFIX + ".h5")
RACMO_FILE = Path("/media/andrew/wd1/projects/ismip7/antarctica/data/racmo/"
                  "smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc")
TAS_FILE = Path("/home/andrew/projects/ismip7/ISMIP7/AIS/CESM2-WACCM/climatology/"
                "SDBN1/tas/v1/tas_CESM2-WACCM_ltm_SDBN1_1960-1989.nc")

# TAU_D_SMOOTH (Helmholtz length for the driving stress, default 2000 m) is set
# above from VV_TAU_SMOOTH.
TAU_D_MIN = 1e-3        # MPa
PHI_MIN = 0.02          # flotation-factor floor for the anchor (PISM delta)
GL_TAPER_MIN = 3.0 / GL_THRESH   # floor on the schoof GL taper in the anchor (fastice: 3 m of HAF)
U_MIN = 1.0             # m/yr


def load_geometry():
    with CheckpointFile(str(MESH_FILE), "r") as chk:
        mesh = chk.load_mesh()
        fields = {k: chk.load_function(mesh, k) for k in ("thickness", "surface", "bed", "velocity")}
    return mesh, fields


def load_racmo_smb(Q):
    r"""RACMO2.4p1 1979-2023 mean SMB [m/yr ice eq.] on Q."""
    ds = nc.Dataset(str(RACMO_FILE))
    lon, lat = ds.variables["lon"][:], ds.variables["lat"][:]
    smb = np.array(ds.variables["smbgl"][:, 0, :, :]).mean(axis=0) * 12.0 / 917.0
    ds.close()
    from pyproj import Transformer
    xy = Q.mesh().coordinates.dat.data_ro
    lons, lats = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True).transform(xy[:, 0], xy[:, 1])
    f = Function(Q, name="smb")
    f.dat.data[:] = griddata((lon.ravel(), lat.ravel()), smb.ravel(), (lons, lats),
                             method="linear", fill_value=0.0)
    return f


def load_surface_temperature(Q, fill_K=260.0):
    r"""ISMIP7 CESM2-WACCM SDBN1 mean-annual tas [K] on Q (nearest neighbour)."""
    ds = xr.open_dataset(TAS_FILE)
    T = np.asarray(ds["tas"].mean(dim="month").transpose("y", "x").values, dtype=float)
    x, y = ds["x"].values.astype(float), ds["y"].values.astype(float)
    if y[0] > y[-1]:
        y, T = y[::-1], T[::-1, :]
    if x[0] > x[-1]:
        x, T = x[::-1], T[:, ::-1]
    T = np.where(T > 100.0, T, fill_K)
    interp = RegularGridInterpolator((y, x), T, method="nearest", bounds_error=False, fill_value=fill_K)
    f = Function(Q, name="T_srf")
    xy = Q.mesh().coordinates.dat.data_ro
    f.dat.data[:] = interp(np.column_stack([xy[:, 1], xy[:, 0]]))
    return f


def driving_stress(h, s, V, length=TAU_D_SMOOTH):
    r"""|tau_d| with tau_d = -rho_I g h grad(s), Helmholtz-smoothed over ``length``."""
    τ, v = TrialFunction(V), TestFunction(V)
    a = (inner(τ, v) + Constant(length) ** 2 * inner(grad(τ), grad(v))) * dx
    L = -inner(ρ_I * g * h * grad(s), v) * dx
    τ_s = Function(V)
    fd.solve(a == L, τ_s, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})
    return τ_s


def weertman_anchor(h, s, u_obs, Q, V):
    τ_s = driving_stress(h, s, V)
    τ_mag = Function(Q, name="tau_d").interpolate(max_value(sqrt(inner(τ_s, τ_s)), Constant(TAU_D_MIN)))
    speed = max_value(sqrt(inner(u_obs, u_obs)), Constant(U_MIN))
    ϕ = max_value(flotation_factor(h, s), Constant(PHI_MIN))
    C_w0 = Function(Q, name="C_w0").interpolate(τ_mag / (ϕ * speed ** (1 / m)))
    return C_w0, τ_mag


def schoof_anchor(h, s, u_obs, Q, V):
    r"""tau_c0 such that the schoof law at theta_C = 0 and u = u_obs balances tau_d."""
    τ_s = driving_stress(h, s, V)
    τ_mag = Function(Q, name="tau_d").interpolate(max_value(sqrt(inner(τ_s, τ_s)), Constant(TAU_D_MIN)))
    speed = max_value(sqrt(inner(u_obs, u_obs)), Constant(U_MIN))
    p = 1 / m + 1
    taper = max_value(gl_taper(h, s), Constant(GL_TAPER_MIN))
    τ_c0 = Function(Q, name="tau_c0").interpolate(
        τ_mag * (Constant(SCHOOF_U0) ** p + speed**p) ** (1 / (m + 1)) / (speed ** (1 / m) * taper))
    return τ_c0, τ_mag


def load_physical_prior(mesh_2d, names=("C_w0", "A_prior", "tau_c0")):
    r"""The prior-mean fields ``names`` on the 2D CG1 space of ``mesh_2d`` (same dof order as thwaites.h5).

    Returns one Function per name; a name the file does not hold gives None
    (prior files written before tau_c0 existed: physical_prior.py --append-schoof).
    """
    Q = FunctionSpace(mesh_2d, "CG", 1)
    out = []
    with CheckpointFile(str(PRIOR_FILE), "r") as chk:
        mp = chk.load_mesh()
        for name in names:
            try:
                src = chk.load_function(mp, name)
            except RuntimeError:
                out.append(None)
                continue
            f = Function(Q, name=name)
            f.dat.data[:] = src.dat.data_ro
            out.append(f)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--append-schoof", action="store_true",
                    help="only add tau_c0 to the existing prior file (no thermal solve)")
    args = ap.parse_args()
    mesh, F = load_geometry()
    h, s, b, u_obs = F["thickness"], F["surface"], F["bed"], F["velocity"]
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    print(f"Mesh: {mesh.num_vertices()} vertices; tau_d Helmholtz length {TAU_D_SMOOTH:.0f} m",
          flush=True)

    C_w0, τ_d = weertman_anchor(h, s, u_obs, Q, V)
    print(f"tau_d [MPa]: [{τ_d.dat.data.min():.3g}, {τ_d.dat.data.max():.3g}]  "
          f"C_w0: [{C_w0.dat.data.min():.3g}, {C_w0.dat.data.max():.3g}]", flush=True)
    τ_c0, _ = schoof_anchor(h, s, u_obs, Q, V)
    c = τ_c0.dat.data_ro
    print(f"tau_c0 [MPa] (U_0 = {SCHOOF_U0:g} m/yr, GL taper {GL_THRESH:g} m): median {np.median(c):.3g}, "
          f"[p1, p99] = [{np.percentile(c, 1):.3g}, {np.percentile(c, 99):.3g}], max {c.max():.3g}",
          flush=True)

    if args.append_schoof:
        with CheckpointFile(str(PRIOR_FILE), "r") as chk:
            mp = chk.load_mesh()
            if not np.allclose(chk.load_function(mp, "C_w0").dat.data_ro, C_w0.dat.data_ro):
                raise SystemExit(f"{PRIOR_FILE} holds a different C_w0 than this geometry and "
                                 f"tau_d smoothing give; regenerate it instead of appending")
            mesh_file = mp
        with CheckpointFile(str(PRIOR_FILE), "a") as chk:
            f = Function(FunctionSpace(mesh_file, "CG", 1), name="tau_c0")
            f.dat.data[:] = τ_c0.dat.data_ro
            chk.save_function(f)
        print(f"Appended tau_c0 to {PRIOR_FILE}", flush=True)
        return

    smb = load_racmo_smb(Q)
    T_srf = load_surface_temperature(Q)
    print(f"SMB [m/yr]: [{smb.dat.data.min():.3f}, {smb.dat.data.max():.3f}]  "
          f"T_srf [K]: [{T_srf.dat.data.min():.1f}, {T_srf.dat.data.max():.1f}]", flush=True)

    # Frictional heating uses the Weertman coefficient including flotation.
    C_heat = Function(Q).interpolate(C_w0 * max_value(flotation_factor(h, s), 0))
    A_prior = compute_fluidity_prior(u_obs, h, s, b, C_heat, smb, T_srf)
    A_prior.rename("A_prior")
    print(f"A_prior [MPa^-3/yr]: [{A_prior.dat.data.min():.2f}, {A_prior.dat.data.max():.2f}]", flush=True)

    with CheckpointFile(str(PRIOR_FILE), "w") as chk:
        chk.save_mesh(mesh)
        for f in (C_w0, τ_c0, A_prior, τ_d, smb, T_srf):
            chk.save_function(f)
    print(f"Saved {PRIOR_FILE}", flush=True)


if __name__ == "__main__":
    main()
