r"""Build a Thwaites Glacier mesh matching the BedMachine ice extent.

Uses the Thwaites GeoJSON outline as the domain boundary (no ocean buffer).
Boundary segments are classified as calving (floating, tag 1) or inflow
(grounded, tag 2) based on BedMachine height above flotation.

Data:
  - BedMachine Antarctica v4.1 (NSIDC-0756)
  - MEaSUREs Antarctica Ice Velocity v2 (450m)
  - Thwaites outline: /media/andrew/wd1/projects/thwaites/mesh/thwaites.geojson (EPSG:3031)
"""
import numpy as np
import netCDF4 as nc
import geojson
import gmsh
import firedrake
import icepack
import rasterio
from firedrake import (
    Function, FunctionSpace, VectorFunctionSpace, Constant,
    max_value, inner, grad, dx,
)
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BEDMACHINE = Path(
    "/media/andrew/wd1/projects/ismip7/data/bedmachine/"
    "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
)
VEL_FILE = Path(
    "/media/andrew/wd1/projects/ismip7/data/velocity/"
    "antarctica_ice_velocity_450m_v2.nc"
)
GEOJSON_PATH = Path("/media/andrew/wd1/projects/thwaites/mesh/thwaites.geojson")
OUTPUT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
RES_FINE = 2_000     # m, minimum element size
RES_COARSE = 10_000  # m, maximum element size

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
H_MIN = 10.0   # m, minimum thickness
ICE_DENSITY = 917.0
WATER_DENSITY = 1024.0


# ═══════════════════════════════════════════════════════════════════
# 1. Load outline
# ═══════════════════════════════════════════════════════════════════

def load_thwaites_outline():
    """Load Thwaites GeoJSON outline (EPSG:3031) as a Shapely Polygon."""
    with open(GEOJSON_PATH) as f:
        collection = geojson.load(f)

    all_coords = []
    for feature in collection["features"]:
        geom = feature["geometry"]
        if geom["type"] == "LineString":
            all_coords.extend(geom["coordinates"])
        elif geom["type"] == "MultiLineString":
            for line in geom["coordinates"]:
                all_coords.extend(line)

    if all_coords[0] != all_coords[-1]:
        all_coords.append(all_coords[0])

    poly = Polygon(all_coords)
    if not poly.is_valid:
        poly = poly.buffer(0)

    print(f"Thwaites outline: {poly.area / 1e9:.0f} x 10^3 km^2")
    return poly


# ═══════════════════════════════════════════════════════════════════
# 2. Build HAF interpolator for boundary classification
# ═══════════════════════════════════════════════════════════════════

def build_haf_interpolator(bounds):
    """Build a height-above-flotation interpolator from BedMachine."""
    xmin, ymin, xmax, ymax = bounds
    buf = 50_000

    ds = nc.Dataset(str(BEDMACHINE))
    x = ds.variables["x"][:]
    y = ds.variables["y"][:]

    ix0 = max(0, np.searchsorted(x, xmin - buf) - 1)
    ix1 = min(len(x), np.searchsorted(x, xmax + buf) + 1)
    iy0 = max(0, np.searchsorted(-y, -(ymax + buf)) - 1)
    iy1 = min(len(y), np.searchsorted(-y, -(ymin - buf)) + 1)

    x_sub = x[ix0:ix1]
    y_sub = y[iy0:iy1]
    surface = np.array(ds.variables["surface"][iy0:iy1, ix0:ix1]).astype(float)
    bed = np.array(ds.variables["bed"][iy0:iy1, ix0:ix1]).astype(float)
    ds.close()

    # Height above flotation: positive = grounded
    h_f = np.maximum(0, -bed * (WATER_DENSITY / ICE_DENSITY))
    haf = surface - h_f

    y_inc = y_sub[::-1]
    interp = RegularGridInterpolator(
        (y_inc, x_sub), haf[::-1, :], method="linear",
        bounds_error=False, fill_value=-9999.0,
    )
    return interp


# ═══════════════════════════════════════════════════════════════════
# 3. Mesh generation with boundary classification
# ═══════════════════════════════════════════════════════════════════

def build_gmsh_mesh(domain, haf_interp):
    """Create Gmsh mesh with calving (tag 1) and inflow (tag 2) boundaries."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("thwaites")

    simplified = domain.exterior.simplify(1000)
    coords = list(simplified.coords)[:-1]
    print(f"Boundary: {len(coords)} points after simplification")

    # Create Gmsh points
    points = []
    for x, y in coords:
        p = gmsh.model.geo.addPoint(x, y, 0, RES_COARSE)
        points.append(p)

    # Classify each edge by height above flotation at midpoint
    calving_lines = []
    inflow_lines = []
    for i in range(len(points)):
        j = (i + 1) % len(points)
        line = gmsh.model.geo.addLine(points[i], points[j])
        x0, y0 = coords[i]
        x1, y1 = coords[j % len(coords)]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        haf = float(haf_interp((my, mx)))
        if haf <= 0:
            calving_lines.append(line)
        else:
            inflow_lines.append(line)

    all_lines = calving_lines + inflow_lines
    loop = gmsh.model.geo.addCurveLoop(all_lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    if calving_lines:
        gmsh.model.addPhysicalGroup(1, calving_lines, tag=1, name="calving")
    if inflow_lines:
        gmsh.model.addPhysicalGroup(1, inflow_lines, tag=2, name="inflow")
    gmsh.model.addPhysicalGroup(2, [surf], tag=1, name="domain")

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MeshSizeMin", RES_FINE)
    gmsh.option.setNumber("Mesh.MeshSizeMax", RES_COARSE)

    gmsh.model.mesh.generate(2)
    msh_path = str(OUTPUT_DIR / "mesh" / "thwaites.msh")
    gmsh.write(msh_path)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_elems = len(gmsh.model.mesh.getElements(2)[1][0])
    print(f"Mesh: {n_nodes} nodes, {n_elems} elements")
    print(f"Boundary: {len(calving_lines)} calving, {len(inflow_lines)} inflow edges")

    gmsh.finalize()
    return msh_path


# ═══════════════════════════════════════════════════════════════════
# 4. Data interpolation and output
# ═══════════════════════════════════════════════════════════════════

def interpolate_and_save(msh_path):
    """Interpolate BedMachine + MEaSUREs velocity onto mesh and save to HDF5."""
    mesh = firedrake.Mesh(msh_path)
    print(f"Firedrake mesh: {mesh.num_vertices()} verts, {mesh.num_cells()} cells")

    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)

    # BedMachine fields via icepack.interpolate + rasterio
    b = icepack.interpolate(rasterio.open(f"netcdf:{BEDMACHINE}:bed"), Q)
    b.rename("bed")

    H_raw = icepack.interpolate(rasterio.open(f"netcdf:{BEDMACHINE}:thickness"), Q)
    H = Function(Q, name="thickness")
    H.interpolate(max_value(H_raw, Constant(H_MIN)))

    # Surface: max of (bed + thickness) and flotation height, then smooth
    s = Function(Q, name="surface")
    s.interpolate(max_value(b + H, Constant(1 - ICE_DENSITY / WATER_DENSITY) * H))
    s0 = s.copy(deepcopy=True)
    smoothing_length = Constant(2e3)
    firedrake.solve(
        firedrake.derivative(
            0.5 * ((s - s0) ** 2 + smoothing_length ** 2 * inner(grad(s), grad(s))) * dx,
            s,
        ) == 0,
        s,
    )

    # MEaSUREs velocity
    u_obs = icepack.interpolate(
        (rasterio.open(f"netcdf:{VEL_FILE}:VX"),
         rasterio.open(f"netcdf:{VEL_FILE}:VY")),
        V, fillvalue=0.0,
    )
    u_obs.rename("velocity")

    # Effective pressure fraction (for friction parameterization)
    phi_eff = Function(Q, name="phi_eff")
    phi_eff.interpolate(
        max_value(
            Constant(1.0) - Constant(WATER_DENSITY / ICE_DENSITY)
            * max_value(Constant(0.0), -b) / H,
            Constant(0.01),
        )
    )

    # Diagnostics
    print(f"  Thickness: [{H.dat.data.min():.0f}, {H.dat.data.max():.0f}] m")
    print(f"  Bed:       [{b.dat.data.min():.0f}, {b.dat.data.max():.0f}] m")
    print(f"  Surface:   [{s.dat.data.min():.0f}, {s.dat.data.max():.0f}] m")
    speed = np.sqrt(u_obs.dat.data_ro[:, 0] ** 2 + u_obs.dat.data_ro[:, 1] ** 2)
    print(f"  Speed:     [{speed.min():.0f}, {speed.max():.0f}] m/yr")
    print(f"  phi_eff:   [{phi_eff.dat.data.min():.3f}, {phi_eff.dat.data.max():.3f}]")
    n_thin = (H.dat.data_ro <= H_MIN + 1).sum()
    print(f"  Nodes at minimum thickness: {n_thin} / {H.dat.data.shape[0]}")

    # Save
    chk_path = str(OUTPUT_DIR / "mesh" / "thwaites.h5")
    with firedrake.CheckpointFile(chk_path, "w") as chk:
        chk.save_mesh(mesh)
        for f, name in [
            (H, "thickness"),
            (b, "bed"),
            (s, "surface"),
            (u_obs, "velocity"),
            (phi_eff, "phi_eff"),
        ]:
            chk.save_function(f, name=name)
    print(f"Saved {chk_path}")

    return mesh, Q, V, {"thickness": H, "bed": b, "surface": s,
                         "velocity": u_obs, "phi_eff": phi_eff}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Building Thwaites mesh (no ocean buffer)")
    print("=" * 60)

    print("\n1. Loading Thwaites outline...")
    outline = load_thwaites_outline()

    print("\n2. Building HAF interpolator for boundary classification...")
    haf_interp = build_haf_interpolator(outline.bounds)

    print("\n3. Generating mesh...")
    msh_path = build_gmsh_mesh(outline, haf_interp)

    print("\n4. Interpolating data...")
    mesh, Q, V, fields = interpolate_and_save(msh_path)

    # Quick plot
    print("\n5. Plotting...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    firedrake.triplot(mesh, axes=ax)
    ax.set_title("Mesh")
    ax.set_aspect("equal")

    ax = axes[1]
    tc = firedrake.tripcolor(fields["thickness"], axes=ax, cmap="viridis")
    fig.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title("Thickness (m)")
    ax.set_aspect("equal")

    ax = axes[2]
    speed = Function(Q, name="speed")
    speed.interpolate(
        firedrake.sqrt(inner(fields["velocity"], fields["velocity"]))
    )
    sc = firedrake.tripcolor(speed, axes=ax, cmap="inferno", vmin=0, vmax=3000)
    fig.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title("Speed (m/yr)")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig_path = str(OUTPUT_DIR / "figures" / "mesh.png")
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
