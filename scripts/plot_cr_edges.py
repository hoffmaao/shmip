import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import firedrake as fd


def triangulation_from_mesh(mesh):
    # vertex coordinates
    xy = mesh.coordinates.dat.data_ro
    x, y = xy[:, 0], xy[:, 1]

    # triangle connectivity (vertex indices per cell)
    # Firedrake stores this in cell node map (arity == number of vertices per cell)
    cell_node_map = mesh.coordinates.function_space().cell_node_map()
    tris = cell_node_map.values

    return mtri.Triangulation(x, y, tris)

def sample_to_vertices(func):
    # For CG1, dofs are at vertices → just grab .dat
    return func.dat.data_ro.copy()

def _all_vertex_coords(dm):
    """Return array of vertex coordinates indexed by DMPlex vertex point number."""
    sec = dm.getCoordinateSection()
    coords_local = dm.getCoordinatesLocal()    # PETSc Vec
    arr = coords_local.array                  # 1D packed coords
    vStart, vEnd = dm.getDepthStratum(0)      # vertices
    gdim = sec.getDof(vStart)                 # 2 for 2D
    XY = np.empty((vEnd - vStart, gdim), dtype=float)
    for p in range(vStart, vEnd):
        off = sec.getOffset(p)
        XY[p - vStart, :] = arr[off:off + gdim]
    return XY, vStart, vEnd

def _edge_segments_and_indices(mesh):
    """Return (segments, edge_point_ids, (eStart,eEnd)) with segments.shape == (Nedge, 2, 2)."""
    dm = mesh.topology_dm
    XY, vStart, vEnd = _all_vertex_coords(dm)
    eStart, eEnd = dm.getDepthStratum(1)      # edges
    segments = []
    edge_ids = []
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)                  # 2 vertex point ids
        v0, v1 = int(cone[0]), int(cone[1])
        p0 = XY[v0 - vStart, :2]
        p1 = XY[v1 - vStart, :2]
        segments.append([p0, p1])
        edge_ids.append(e)
    return np.array(segments), np.array(edge_ids), (eStart, eEnd)

def _cr_values_in_dmplex_order(func):
    """
    Returns CR values ordered by DMPlex edge points [eStart:eEnd).
    Assumes Firedrake's CR dof ordering aligns with DMPlex edge ordering (true for standard simplicial meshes).
    """
    # sanity
    V = func.function_space()
    family = V.ufl_element().family()
    if family != "Crouzeix-Raviart":
        raise ValueError(f"Function {func.name()} not on CR space.")

    # Firedrake’s CR dofs are 1 per edge; we assume same order as plex edges.
    return func.dat.data_ro.copy()

def main(chkfile, field_name="S", out="cr_edges.png", cmap="viridis"):
    with fd.CheckpointFile(chkfile, "r") as chk:
        mesh = chk.load_mesh()
        f = chk.load_function(mesh, field_name)

    # Build the edge segment geometry
    segments, edge_ids, (eStart, eEnd) = _edge_segments_and_indices(mesh)

    # Values on edges (must be CR)
    vals = _cr_values_in_dmplex_order(f)

    if len(vals) != len(segments):
        # Fallback: project to CR so lengths match, if needed
        Vcr = fd.FunctionSpace(mesh, "CR", 1)
        f = fd.project(f, Vcr)
        vals = f.dat.data_ro.copy()
        assert len(vals) == len(segments), (
            f"Length mismatch even after project: vals={len(vals)} segments={len(segments)}"
        )

    # Build colored line collection
    lc = LineCollection(segments, linewidths=1.5, cmap=cmap)
    lc.set_array(vals)
    lc.set_linewidth(1.2)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{field_name} (CR on edges)")
    cbar = plt.colorbar(lc, ax=ax, shrink=0.9)
    cbar.set_label(field_name)
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--field", default="S")
    p.add_argument("--out", default="cr_edges.png")
    p.add_argument("--cmap", default="viridis")
    args = p.parse_args()
    main(args.checkpoint, args.field, args.out, args.cmap)
