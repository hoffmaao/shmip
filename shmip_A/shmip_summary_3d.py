#!/usr/bin/env python3
"""
SHMIP 'stacked ribbons':
  - Each experiment (A1, A2, ...) is a 2D N(x) curve drawn in a y=const plane.
  - The area under N(x) is filled with a color graded by f_eff(x):
       0% -> light blue, 10% -> white, 50%+ -> red (clamped above 50%).
  - Uses saved fluxes (Q_ch, q_s_mag/q_s) if available.

Usage:
  python plot_shmip_ribbons.py A1.h5 A2.h5 ... --labels A1 A2 ...
"""

import argparse, os, math, numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- CLI first (so Firedrake/PETSc don't eat flags) ----------
ap = argparse.ArgumentParser(description="SHMIP stacked ribbons: 2D N(x) per experiment with graded fill (3D view)")
ap.add_argument("checkpoints", nargs="+", help="Firedrake HDF5 checkpoint(s)")
ap.add_argument("--labels", nargs="*", help="Labels for experiments (A1 A2 ...)")
ap.add_argument("--suite", choices=["A","E"], default="A", help="A=full width, E=centre band")
ap.add_argument("--band-width-E", type=float, default=200.0, help="Suite E band width (m)")
ap.add_argument("--nbins", type=int, default=250, help="x-bins for width-averaging")
ap.add_argument("--outfile", default="shmip_ribbons.png", help="output image")
ap.add_argument("--no-proxy", action="store_true", help="require saved fluxes; error if missing")
ap.add_argument("--elev", type=float, default=23.0, help="3D elevation (deg)")
ap.add_argument("--azim", type=float, default=230.0, help="3D azimuth (deg)")
ap.add_argument("--print-diag", action="store_true", help="print mean % efficient per experiment")
args = ap.parse_args()

# ---------- Heavy deps after parsing ----------
import firedrake as fd
from plot_cr_edges import (
    triangulation_from_mesh,
    _edge_segments_and_indices,
    _cr_values_in_dmplex_order,
)

# ---------------- helpers: geometry & arrays ----------------
def dg0_array(mesh, expr_or_func):
    V = fd.FunctionSpace(mesh, "DG", 0)
    out = fd.Function(V); out.project(expr_or_func)
    return out.dat.data_ro

def cell_centroids(mesh):
    Vv = fd.VectorFunctionSpace(mesh, "DG", 0)
    X  = fd.Function(Vv).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro
    return X[:,0].copy(), X[:,1].copy()

def cell_areas(mesh):
    V = fd.FunctionSpace(mesh, "DG", 0)
    return fd.Function(V).interpolate(fd.CellVolume(mesh)).dat.data_ro.copy()

def edge_lengths(segments):
    d = segments[:,1,:] - segments[:,0,:]
    return np.sqrt((d*d).sum(axis=1))

def shmip_eff_cmap():
    # 0%->light blue, 10%->white, 50%+->red
    light_blue = "#9ecae1"; white = "#ffffff"; red = "#d7301f"; dark_red   = "#8B0000"
    return mcolors.LinearSegmentedColormap.from_list(
        "shmip_eff", [(0.00, light_blue), (0.10, white), (0.50, red), (1.00, dark_red)]
    )

# ---------------- load checkpoint (prefer saved fluxes) ----------------
def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with fd.CheckpointFile(path, "r") as f:
        mesh = f.load_mesh()
        N    = f.load_function(mesh, "N")
        phi  = f.load_function(mesh, "phi")
        S    = f.load_function(mesh, "S")
        try:   h = f.load_function(mesh, "h")
        except Exception:
               h = fd.interpolate(fd.Constant(0.0), fd.FunctionSpace(mesh,"CG",1))
        # saved fluxes
        try:   Q_ch = f.load_function(mesh, "Q_ch")
        except Exception:
               Q_ch = None
        try:   q_s_mag = f.load_function(mesh, "q_s_mag")
        except Exception:
               q_s_mag = None
        try:
            q_s = f.load_function(mesh, "q_s") if q_s_mag is None else None
        except Exception:
            q_s = None
    tri = triangulation_from_mesh(mesh)
    segments,_,_ = _edge_segments_and_indices(mesh)
    return dict(mesh=mesh, tri=tri, N=N, phi=phi, S=S, h=h,
                Q_ch=Q_ch, q_s_mag=q_s_mag, q_s=q_s,
                segments=np.asarray(segments))

# ---------------- width-averaged N(x) ----------------
def width_average_N_along_x(mesh, N, nbins=200, band=None):
    N0 = dg0_array(mesh, N); xc,yc = cell_centroids(mesh); A = cell_areas(mesh)
    tri = triangulation_from_mesh(mesh); x0,x1 = float(tri.x.min()), float(tri.x.max())
    mask = np.ones_like(N0, bool)
    if band is not None:
        y0,y1 = band; mask &= (yc>=y0)&(yc<=y1)
    nbins = int(min(nbins, max(32, int(np.sqrt(max(1, len(N0)))))))
    edges = np.linspace(x0, x1, nbins+1)
    xmid  = 0.5*(edges[:-1] + edges[1:])
    ix = np.clip(np.searchsorted(edges, xc[mask], side="right")-1, 0, nbins-1)
    Nsum = np.zeros(nbins); Asum = np.zeros(nbins)
    np.add.at(Nsum, ix, N0[mask]*A[mask]); np.add.at(Asum, ix, A[mask])
    bad = Asum==0; Nsum[bad]=np.nan; Asum[bad]=np.nan
    Nbar = Nsum/Asum; keep = ~np.isnan(Nbar)
    return xmid[keep], Nbar[keep], edges

# ---------------- along-x partition using saved fluxes ----------------
def partition_to_edges(mesh, tri, segments, edges, band,
                       Q_ch=None, q_s_mag=None, q_s=None,
                       phi=None, S=None, h=None, allow_proxy=True):
    # geometry & bands
    segs = np.asarray(segments); L = edge_lengths(segs)
    emid = 0.5*(segs[:,0,:] + segs[:,1,:]); xe, ye = emid[:,0], emid[:,1]
    xc, yc = cell_centroids(mesh); A = cell_areas(mesh)
    ymin, ymax = float(tri.y.min()), float(tri.y.max())
    edge_mask = np.ones(len(segs), bool); cell_mask = np.ones_like(xc, bool)
    if band is not None:
        y0,y1 = band; edge_mask &= (ye>=y0)&(ye<=y1); cell_mask &= (yc>=y0)&(yc<=y1)

    # channels per x-bin
    if Q_ch is not None:
        Qe = _cr_values_in_dmplex_order(Q_ch).astype(float)
    else:
        if not allow_proxy: raise RuntimeError("Q_ch missing and --no-proxy set.")
        # minimal fallback proxy
        Vv = fd.VectorFunctionSpace(mesh,"DG",0)
        g  = fd.Function(Vv).project(fd.grad(phi))
        gmag = dg0_array(mesh, fd.sqrt(fd.inner(g,g)))
        S_vals = _cr_values_in_dmplex_order(S).astype(float)
        g_edge = np.interp(ye, np.linspace(ymin, ymax, len(gmag)), np.sort(gmag))
        Qe = np.maximum(S_vals,0.0)*np.maximum(g_edge,0.0)
    QeL = Qe * L
    ie = np.clip(np.searchsorted(edges, xe, side="right")-1, 0, len(edges)-2)
    Qeff_x = np.zeros(len(edges)-1); np.add.at(Qeff_x, ie[edge_mask], QeL[edge_mask])

    # sheet per x-bin
    if q_s_mag is None and q_s is not None:
        Vdg = fd.FunctionSpace(mesh,"DG",0)
        tmp = fd.Function(Vdg); tmp.project(fd.sqrt(fd.inner(q_s,q_s) + fd.Constant(1e-30)))
        q_s_mag = tmp
    if q_s_mag is not None:
        mag = q_s_mag.dat.data_ro
    else:
        if not allow_proxy: raise RuntimeError("q_s/_mag missing and --no-proxy set.")
        Vv = fd.VectorFunctionSpace(mesh,"DG",0)
        g  = fd.Function(Vv).project(fd.grad(phi))
        mag = dg0_array(mesh, fd.max_value(h,0.0)**3 * fd.sqrt(fd.inner(g,g)))
    ic = np.clip(np.searchsorted(edges, xc, side="right")-1, 0, len(edges)-2)
    Qineff_x = np.zeros(len(edges)-1); np.add.at(Qineff_x, ic[cell_mask], (mag*A)[cell_mask])

    Qtot_x = Qeff_x + Qineff_x
    with np.errstate(divide="ignore", invalid="ignore"):
        feff_x = np.where(Qtot_x>0, Qeff_x/Qtot_x, np.nan)
    return feff_x

# ---------------- build colored ribbon polygons for one experiment ----------------
def build_ribbon_quads(x, Nbar, feff, y_pos, cmap):
    """
    Return list of 4-vertex quads (x km, y index, z MPa) and facecolors,
    splitting the strip into narrow trapezoids to enable a color gradient.
    """
    # interpolate feff to same grid as Nbar if needed
    if feff.shape != Nbar.shape:
        x_f = np.linspace(x[0], x[-1], max(len(feff), 2))
        feff = np.interp(x, x_f, feff, left=feff[0] if feff.size else 0.0,
                                    right=feff[-1] if feff.size else 0.0)
    quads = []; colors = []
    for i in range(len(x)-1):
        x0, x1 = x[i]/1000.0, x[i+1]/1000.0          # km
        z0, z1 = Nbar[i]/1e6, Nbar[i+1]/1e6          # MPa
        fmid = 0.5*(feff[i] + feff[i+1])
        col  = cmap(np.clip(fmid, 0.0, 1.0))
        # quad in the plane y = y_pos
        verts = [(x0, y_pos, 0.0),
                 (x0, y_pos, z0),
                 (x1, y_pos, z1),
                 (x1, y_pos, 0.0)]
        quads.append(verts); colors.append(col)
    return quads, colors

# ---------------- main ----------------
def main():
    runs = [load_checkpoint(p) for p in args.checkpoints]
    labels = args.labels if args.labels and len(args.labels)==len(runs) else \
             [os.path.splitext(os.path.basename(p))[0] for p in args.checkpoints]

    # choose band for each run
    bands = []
    xmins = []; xmaxs = []; zmax = 0.0
    for r in runs:
        tri = r["tri"]; ymin,ymax = float(tri.y.min()), float(tri.y.max())
        if args.suite.upper()=="E":
            yc = 0.5*(ymin+ymax); bands.append((yc-0.5*args.band_width_E, yc+0.5*args.band_width_E))
        else:
            bands.append(None)
        xmins.append(float(tri.x.min())); xmaxs.append(float(tri.x.max()))

    # Figure & axes
    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    ax  = fig.add_subplot(111, projection='3d')

    cmap = shmip_eff_cmap()
    y_positions = np.arange(len(runs))  # 0..nr-1; we’ll label with A1.. etc.

    # Build and draw ribbons
    for j,(r,band,label) in reversed(list(enumerate(zip(runs, bands, labels)))):
        # width-averaged N(x) on adaptive edges
        x, Nbar, edges = width_average_N_along_x(r["mesh"], r["N"], nbins=args.nbins, band=band)
        # partition on SAME edges
        feff_edges = partition_to_edges(
            r["mesh"], r["tri"], r["segments"], edges, band,
            Q_ch=r["Q_ch"], q_s_mag=r["q_s_mag"], q_s=r["q_s"],
            phi=r["phi"], S=r["S"], h=r["h"], allow_proxy=(not args.no_proxy)
        )
        # feff on centers corresponding to edges -> interpolate to x
        if np.all(np.isnan(feff_edges)):
            feff_on_x = np.zeros_like(x)
        else:
            x_edges_mid = 0.5*(edges[:-1] + edges[1:])
            feff_on_x = np.interp(x, x_edges_mid, feff_edges,
                                  left=feff_edges[~np.isnan(feff_edges)][0] if np.any(~np.isnan(feff_edges)) else 0.0,
                                  right=feff_edges[~np.isnan(feff_edges)][-1] if np.any(~np.isnan(feff_edges)) else 0.0)

        quads, colors = build_ribbon_quads(x, Nbar, feff_on_x, y_positions[j], cmap)
        coll = Poly3DCollection(quads, facecolors=colors, edgecolors='none')
        ax.add_collection3d(coll)
        ax.plot(x/1000.0, np.full_like(x, y_positions[j]), Nbar/1e6,
                color='k', lw=1.2, zorder=10)

        # track z max
        if Nbar.size:
            zmax = max(zmax, float(np.nanmax(Nbar))/1e6)

        if args.print_diag and feff_on_x.size:
            print(f"{label}: mean channel share ≈ {100*np.nanmean(feff_on_x):.1f}%")

    # Axes cosmetics
    x0km = min(xmins)/1000.0; x1km = max(xmaxs)/1000.0
    ax.set_xlim(x0km, x1km)
    ax.set_ylim(-0.5, len(runs)-0.5)
    ax.set_zlim(0, zmax*1.05 if zmax>0 else 1.0)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("experiment")
    ax.set_yticks(y_positions); ax.set_yticklabels(labels)
    ax.set_zlabel(r"$\bar N$ (MPa)")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title("$\\bar N(x)$ plotted with percentage of flux in the efficient system")

    # Colorbar for f_eff
    import matplotlib.cm as cm
    sm = cm.ScalarMappable(norm=mcolors.Normalize(0,1), cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.7)
    cb.set_label("Percentage of flux in the efficient system")
    cb.set_ticks([0.0, 0.10, 0.50, 1.0])
    cb.set_ticklabels(["0%", "10%", "50%", "100%"])

    fig.savefig(args.outfile, dpi=220)
    print(f"Saved {args.outfile}")

if __name__ == "__main__":
    if os.environ.get("OMP_NUM_THREADS","") not in ("","1"):
        print("Note: for post-processing consider: export OMP_NUM_THREADS=1")
    main()
