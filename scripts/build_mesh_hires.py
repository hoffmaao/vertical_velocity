r"""Build a Thwaites high-res mesh with calving front from the geojson outline.

Takes the high-resolution mesh from multi-layer/thwaites (strain-rate +
velocity adapted refinement) and clips it to the Thwaites geojson outline
(Goldberg/Morlighem domain). This removes the 30km ocean buffer while
keeping the ice shelf — the calving front matches the geojson boundary.

Data:
  - Source mesh: multi-layer/thwaites/mesh/thwaites_hires.msh
  - Calving front: /media/andrew/wd1/projects/thwaites/mesh/thwaites.geojson
  - Size field: multi-layer/thwaites/mesh/thwaites_hires_sizefield.pos
  - BedMachine Antarctica v4.1 (NSIDC-0756)
  - MEaSUREs velocity v2
"""
import numpy as np
import geojson
import gmsh
import firedrake
import icepack
import rasterio
from firedrake import (
    Function, FunctionSpace, VectorFunctionSpace, Constant,
    max_value, inner, grad, dx,
)
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_MSH = Path(
    "/media/andrew/wd1/projects/multi-layer/thwaites/mesh/thwaites_hires.msh"
)
SOURCE_SIZEFIELD = Path(
    "/media/andrew/wd1/projects/multi-layer/thwaites/mesh/thwaites_hires_sizefield.pos"
)
GEOJSON_PATH = Path("/media/andrew/wd1/projects/thwaites/mesh/thwaites.geojson")
BEDMACHINE = Path(
    "/media/andrew/wd1/projects/ismip7/data/bedmachine/"
    "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
)
VEL_FILE = Path(
    "/media/andrew/wd1/projects/ismip7/data/velocity/"
    "antarctica_ice_velocity_450m_v2.nc"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Resolution (match hires builder)
# ---------------------------------------------------------------------------
RES_GL = 500
RES_COARSE = 5000
H_MIN = 100.0
ICE_DENSITY = 917.0
WATER_DENSITY = 1024.0
OUTPUT_NAME = "thwaites"


# ═══════════════════════════════════════════════════════════════════
# 1. Build domain from intersection of hires mesh and geojson outline
# ═══════════════════════════════════════════════════════════════════

def extract_boundary(msh_path):
    """Extract boundary nodes classified as calving (tag 1) or inflow (tag 2)."""
    gmsh.initialize()
    gmsh.open(str(msh_path))
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords_map = {
        int(tag): (node_coords[3 * i], node_coords[3 * i + 1])
        for i, tag in enumerate(node_tags)
    }
    calving_nodes, inflow_nodes = set(), set()
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            for nodes in gmsh.model.mesh.getElements(1, ent)[1:][0]:
                for n in nodes:
                    (calving_nodes if tag == 1 else inflow_nodes).add(n)
    gmsh.finalize()
    return (
        np.array([coords_map[n] for n in calving_nodes | inflow_nodes]),
        np.array([coords_map[n] for n in calving_nodes]),
        np.array([coords_map[n] for n in inflow_nodes]),
    )


def load_geojson_outline():
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
    return poly


def build_trimmed_domain(all_boundary, geojson_poly):
    """Intersect the hires mesh domain with the geojson outline.

    The hires mesh has a 30km ocean buffer beyond the calving front.
    The geojson outline traces the actual calving front. Their intersection
    clips off the ocean buffer while keeping the ice shelf.
    """
    # Build polygon from ordered boundary points
    cx, cy = all_boundary.mean(axis=0)
    angles = np.arctan2(all_boundary[:, 1] - cy, all_boundary[:, 0] - cx)
    ordered = all_boundary[np.argsort(angles)]
    mesh_poly = Polygon(ordered)
    if not mesh_poly.is_valid:
        mesh_poly = mesh_poly.buffer(0)

    # Intersect with geojson
    domain = mesh_poly.intersection(geojson_poly)
    if isinstance(domain, MultiPolygon):
        domain = max(domain.geoms, key=lambda p: p.area)
    if not domain.is_valid:
        domain = domain.buffer(0)

    print(f"  Hires mesh domain: {mesh_poly.area / 1e9:.0f} x 10^3 km^2")
    print(f"  GeoJSON outline:   {geojson_poly.area / 1e9:.0f} x 10^3 km^2")
    print(f"  Trimmed domain:    {domain.area / 1e9:.0f} x 10^3 km^2")
    print(f"  Buffer removed:    {(mesh_poly.area - domain.area) / 1e9:.0f} x 10^3 km^2")

    return domain


# ═══════════════════════════════════════════════════════════════════
# 2. Mesh generation
# ═══════════════════════════════════════════════════════════════════

def build_mesh(domain, geojson_poly, inflow, sizefield_path):
    """Generate mesh with calving/inflow boundary tags and adaptive size field.

    Boundary classification:
      - Segments near the geojson outline (calving front) -> tag 1
      - Segments near the original inflow boundary -> tag 2
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add(OUTPUT_NAME)

    simplified = domain.exterior.simplify(500)
    coords = list(simplified.coords)[:-1]
    print(f"  Boundary: {len(coords)} points after simplification")

    # Build trees for boundary classification
    inflow_tree = cKDTree(inflow)
    geojson_boundary = np.array(geojson_poly.exterior.coords)
    geojson_tree = cKDTree(geojson_boundary)

    points = [gmsh.model.geo.addPoint(x, y, 0, RES_COARSE) for x, y in coords]
    calving_lines, inflow_lines = [], []
    for i in range(len(points)):
        j = (i + 1) % len(points)
        line = gmsh.model.geo.addLine(points[i], points[j])
        mx = (coords[i][0] + coords[j][0]) / 2
        my = (coords[i][1] + coords[j][1]) / 2
        d_geojson = geojson_tree.query([mx, my])[0]
        d_inflow = inflow_tree.query([mx, my])[0]
        if d_geojson < d_inflow:
            calving_lines.append(line)
        else:
            inflow_lines.append(line)

    loop = gmsh.model.geo.addCurveLoop(calving_lines + inflow_lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()

    if calving_lines:
        gmsh.model.addPhysicalGroup(1, calving_lines, tag=1, name="calving")
    if inflow_lines:
        gmsh.model.addPhysicalGroup(1, inflow_lines, tag=2, name="inflow")
    gmsh.model.addPhysicalGroup(2, [surf], tag=1, name="domain")

    # Use the existing hires size field
    gmsh.merge(str(sizefield_path))
    bg = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(bg, "ViewIndex", 0)
    gmsh.model.mesh.field.setAsBackgroundMesh(bg)
    for opt in [
        "MeshSizeExtendFromBoundary",
        "MeshSizeFromPoints",
        "MeshSizeFromCurvature",
    ]:
        gmsh.option.setNumber(f"Mesh.{opt}", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MeshSizeMin", RES_GL)
    gmsh.option.setNumber("Mesh.MeshSizeMax", RES_COARSE)

    gmsh.model.mesh.generate(2)
    msh_path = str(OUTPUT_DIR / "mesh" / f"{OUTPUT_NAME}.msh")
    gmsh.write(msh_path)

    n = len(gmsh.model.mesh.getNodes()[0])
    e = len(gmsh.model.mesh.getElements(2)[1][0])
    print(f"  Mesh: {n:,} nodes, {e:,} elements")
    print(f"  Boundary: {len(calving_lines)} calving, {len(inflow_lines)} inflow edges")

    gmsh.finalize()
    return msh_path


# ═══════════════════════════════════════════════════════════════════
# 3. Data interpolation
# ═══════════════════════════════════════════════════════════════════

def interpolate_and_save(msh_path):
    """Interpolate BedMachine + velocity onto the mesh."""
    mesh = firedrake.Mesh(msh_path)
    print(f"  Firedrake: {mesh.num_vertices():,} verts, {mesh.num_cells():,} cells")

    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)

    b = icepack.interpolate(rasterio.open(f"netcdf:{BEDMACHINE}:bed"), Q)
    b.rename("bed")

    H = Function(Q, name="thickness")
    H.interpolate(
        max_value(
            icepack.interpolate(rasterio.open(f"netcdf:{BEDMACHINE}:thickness"), Q),
            Constant(H_MIN),
        )
    )

    s = Function(Q, name="surface")
    s.interpolate(max_value(b + H, Constant(1 - ICE_DENSITY / WATER_DENSITY) * H))
    s0 = s.copy(deepcopy=True)
    firedrake.solve(
        firedrake.derivative(
            0.5 * ((s - s0) ** 2 + Constant(2e3) ** 2 * inner(grad(s), grad(s))) * dx,
            s,
        ) == 0,
        s,
    )

    u_obs = icepack.interpolate(
        (rasterio.open(f"netcdf:{VEL_FILE}:VX"),
         rasterio.open(f"netcdf:{VEL_FILE}:VY")),
        V, fillvalue=0.0,
    )
    u_obs.rename("velocity")

    phi_eff = Function(Q, name="phi_eff")
    phi_eff.interpolate(
        max_value(
            Constant(1.0) - Constant(WATER_DENSITY / ICE_DENSITY)
            * max_value(Constant(0.0), -b) / H,
            Constant(0.01),
        )
    )

    print(f"  H: [{H.dat.data.min():.0f}, {H.dat.data.max():.0f}] m")
    print(f"  Bed: [{b.dat.data.min():.0f}, {b.dat.data.max():.0f}] m")

    chk_path = str(OUTPUT_DIR / "mesh" / f"{OUTPUT_NAME}.h5")
    with firedrake.CheckpointFile(chk_path, "w") as chk:
        chk.save_mesh(mesh)
        for f, name in [
            (H, "thickness"), (b, "bed"), (s, "surface"),
            (u_obs, "velocity"), (phi_eff, "phi_eff"),
        ]:
            chk.save_function(f, name=name)
    print(f"  Saved {chk_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Building Thwaites mesh (hires clipped to geojson outline)")
    print("=" * 60)

    print("\n1. Extracting boundary from hires mesh...")
    boundary, calving, inflow = extract_boundary(SOURCE_MSH)
    print(f"  {len(boundary)} boundary nodes "
          f"({len(calving)} calving, {len(inflow)} inflow)")

    print("\n2. Loading geojson outline...")
    geojson_poly = load_geojson_outline()
    print(f"  GeoJSON: {len(geojson_poly.exterior.coords)} points")

    print("\n3. Clipping domain to geojson outline...")
    domain = build_trimmed_domain(boundary, geojson_poly)

    print("\n4. Building mesh with hires size field...")
    msh_path = build_mesh(domain, geojson_poly, inflow, SOURCE_SIZEFIELD)

    print("\n5. Interpolating data...")
    interpolate_and_save(msh_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
