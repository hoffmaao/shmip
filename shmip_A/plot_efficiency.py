#!/usr/bin/env python3
import argparse, os, numpy as np, matplotlib.pyplot as plt

# Parse args BEFORE importing firedrake (so PETSc doesn't eat your flags)
ap = argparse.ArgumentParser(description="Percent efficiency of drainage (channels vs sheet)")
ap.add_argument("checkpoints", nargs="+", help="Firedrake HDF5 checkpoints")
ap.add_argument("--labels", nargs="*", help="Bar labels (same length as checkpoints)")
ap.add_argument("--outfile", default="percent_efficiency.png")
ap.add_argument("--suite", choices=["A","E"], default="A",
                help="A = full width; E = centreline 200 m band")
ap.add_argument("--band-width-E", type=float, default=200.0)
args = ap.parse_args()

import firedrake as fd
from plot_cr_edges import triangulation_from_mesh, _edge_segments_and_indices, _cr_values_in_dmplex_order

# ---------- small helpers ----------
def cell_centroids(mesh):
    VdgV = fd.VectorFunctionSpace(mesh, "DG", 0)
    Xdg = fd.Function(VdgV).interpolate(fd.SpatialCoordinate(mesh))
    X = Xdg.dat.data_ro
    return X[:,0].copy(), X[:,1].copy()

def cell_areas(mesh):
    Vdg = fd.FunctionSpace(mesh, "DG", 0)
    A = fd.Function(Vdg).interpolate(fd.CellVolume(mesh))
    return A.dat.data_ro.copy()

def edge_lengths(segments):
    d = segments[:,1,:] - segments[:,0,:]
    return np.sqrt((d*d).sum(axis=1))

def dphi_ds_CR(mesh, phi):
    """CR function with |avg(grad phi)·t| per facet DOF (prevents sign cancellation)."""
    Vcr = fd.FunctionSpace(mesh, "CR", 1)
    w = fd.TestFunction(Vcr)
    n = fd.FacetNormal(mesh)
    t = fd.as_vector([n[1], -n[0]])

    mass = fd.Function(Vcr).assign(0.0)
    mass.dat.data[:] = (fd.assemble(w('+')*fd.dS).dat.data_ro
                        + fd.assemble(w*fd.ds).dat.data_ro)

    rhs_int = fd.assemble((fd.dot(fd.avg(fd.grad(phi)), t)) * w('+') * fd.dS)
    rhs_bnd = fd.assemble((fd.dot(fd.grad(phi), t)) * w * fd.ds)
    rhs = rhs_int.dat.data_ro + rhs_bnd.dat.data_ro

    out = fd.Function(Vcr)
    eps = 1e-16
    out.dat.data[:] = rhs / np.maximum(mass.dat.data_ro, eps)
    return out

def load_fields(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with fd.CheckpointFile(path, "r") as f:
        mesh = f.load_mesh()
        phi  = f.load_function(mesh, "phi")
        S    = f.load_function(mesh, "S")       # CR
        # sheet thickness h may be absent → use 0
        try:
            h = f.load_function(mesh, "h")
        except Exception:
            h = fd.interpolate(fd.Constant(0.0), fd.FunctionSpace(mesh, "CG", 1))
    tri = triangulation_from_mesh(mesh)
    segments, _, _ = _edge_segments_and_indices(mesh)
    return mesh, tri, segments, phi, S, h

def band_masks(mesh, tri, segments, suite, band_width_E):
    # Suite A: no mask (all True). Suite E: keep a 200 m centre band.
    ymin, ymax = float(tri.y.min()), float(tri.y.max())
    if suite == "E":
        yc = 0.5*(ymin+ymax)
        y0, y1 = yc - 0.5*band_width_E, yc + 0.5*band_width_E
    else:
        y0, y1 = -np.inf, np.inf
    # cells
    xc, yc = cell_centroids(mesh)
    cell_mask = (yc >= y0) & (yc <= y1)
    # edges: use segment midpoints
    emid = 0.5*(segments[:,0,:] + segments[:,1,:])
    eym  = emid[:,1]
    edge_mask = (eym >= y0) & (eym <= y1)
    return cell_mask, edge_mask

def percent_efficiency_one(path, suite="A", band_width_E=200.0):
    mesh, tri, segments, phi, S, h = load_fields(path)
    cell_mask, edge_mask = band_masks(mesh, tri, segments, suite, band_width_E)

    # Efficient (channels): sum S * |dphi/ds| * L over edges in band
    abs_dphi_ds = dphi_ds_CR(mesh, phi)
    S_vals   = _cr_values_in_dmplex_order(S).astype(float)
    dphi_val = _cr_values_in_dmplex_order(abs_dphi_ds).astype(float)
    L        = edge_lengths(np.asarray(segments))
    Qeff = float(np.nansum((S_vals * dphi_val * L)[edge_mask]))

    # Inefficient (sheet): sum h * |grad phi| * A over cells in band
    VdgV = fd.VectorFunctionSpace(mesh, "DG", 0)
    gradphi = fd.Function(VdgV).project(fd.grad(phi))
    gmag = fd.sqrt(fd.inner(gradphi, gradphi))
    Vdg = fd.FunctionSpace(mesh, "DG", 0)
    gmag_dg = fd.Function(Vdg); gmag_dg.project(gmag)
    h_dg    = fd.Function(Vdg); h_dg.project(h)
    A = cell_areas(mesh)
    Qineff = float(np.nansum((h_dg.dat.data_ro * gmag_dg.dat.data_ro * A)[cell_mask]))

    Qtot = Qeff + Qineff
    feff = (Qeff / Qtot) if Qtot > 0 else 0.0
    return feff * 100.0  # percent

# ---------- run + plot ----------
percents = [percent_efficiency_one(p, suite=args.suite, band_width_E=args.band_width_E)
            for p in args.checkpoints]
labels = args.labels if args.labels and len(args.labels)==len(percents) \
         else [os.path.splitext(os.path.basename(p))[0] for p in args.checkpoints]

fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
x = np.arange(len(percents))
ineff = 100 - np.array(percents)
ax.bar(x, ineff, color="0.8", label="inefficient (sheet)")
ax.bar(x, percents, bottom=ineff, color=(0.85,0.15,0.15), label="efficient (channels)")
ax.axhline(10, ls="--", lw=1.0, color="0.4")  # SHMIP 10% line
for i, p in enumerate(percents):
    ax.text(i, 102, f"{p:.1f}% ch.", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x, labels)
ax.set_ylim(0, 110)
ax.set_ylabel("Drainage partition (%)")
ax.legend(frameon=False, loc="upper right")
title = "Percent efficiency (channels share of drainage)"
if args.suite == "E":
    title += " — Suite E centre band (200 m)"
ax.set_title(title)
fig.savefig(args.outfile, dpi=200)
print(f"Saved {args.outfile}")
