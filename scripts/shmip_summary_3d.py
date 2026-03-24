#!/usr/bin/env python3
"""
SHMIP 'stacked ribbons':
  - Each experiment (A1, A2, ...) is a 2D N(x) curve drawn in a y=const plane.
  - The area under N(x) is filled with a color graded by f_eff(x):
       0% -> light blue, 10% -> white, 50%+ -> red (clamped above 50%).
  - Uses saved fluxes (Q_ch, q_s_mag/q_s) if available.

Usage (standalone):
  python shmip_summary_3d.py A1.h5 A2.h5 ... --labels A1 A2 ...

Usage (importable):
  from shmip_summary_3d import make_ribbon_figure
  make_ribbon_figure(["outputs/checkpoints/A1.h5", ...], labels=["A1", ...])
"""

import os, numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- Heavy deps deferred until needed ----------
_fd = None
_cr = None

def _ensure_imports():
    global _fd, _cr
    if _fd is None:
        import firedrake as fd
        _fd = fd
        from plot_cr_edges import (
            triangulation_from_mesh,
            _edge_segments_and_indices,
            _cr_values_in_dmplex_order,
            _all_vertex_coords,
        )
        _cr = {
            "triangulation_from_mesh": triangulation_from_mesh,
            "_edge_segments_and_indices": _edge_segments_and_indices,
            "_cr_values_in_dmplex_order": _cr_values_in_dmplex_order,
            "_all_vertex_coords": _all_vertex_coords,
        }

# ---------------- helpers: geometry & arrays ----------------
def dg0_array(mesh, expr_or_func):
    fd = _fd
    V = fd.FunctionSpace(mesh, "DG", 0)
    out = fd.Function(V); out.project(expr_or_func)
    return out.dat.data_ro

def cell_centroids(mesh):
    fd = _fd
    Vv = fd.VectorFunctionSpace(mesh, "DG", 0)
    X  = fd.Function(Vv).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro
    return X[:,0].copy(), X[:,1].copy()

def cell_areas(mesh):
    fd = _fd
    V = fd.FunctionSpace(mesh, "DG", 0)
    return fd.Function(V).interpolate(fd.CellVolume(mesh)).dat.data_ro.copy()

def edge_lengths(segments):
    d = segments[:,1,:] - segments[:,0,:]
    return np.sqrt((d*d).sum(axis=1))


def _compute_dphi_ds(mesh, phi, segments):
    """
    Compute along-edge gradient of phi (CG1) at each CR1 edge midpoint.
    Uses coordinate matching to map DMPlex vertex indices to Firedrake DOF indices.
    Returns array of shape (n_edges,) in DMPlex edge order.
    """
    xy_fd  = mesh.coordinates.dat.data_ro
    phi_fd = phi.dat.data_ro
    coord_to_fd = {(round(float(x), 4), round(float(y), 4)): i
                   for i, (x, y) in enumerate(xy_fd)}
    segs  = np.asarray(segments)
    d01   = segs[:, 1, :] - segs[:, 0, :]
    L     = np.sqrt((d01 ** 2).sum(axis=1))
    dphi  = np.zeros(len(segs))
    for i, seg in enumerate(segs):
        k0 = (round(float(seg[0, 0]), 4), round(float(seg[0, 1]), 4))
        k1 = (round(float(seg[1, 0]), 4), round(float(seg[1, 1]), 4))
        i0 = coord_to_fd.get(k0)
        i1 = coord_to_fd.get(k1)
        if i0 is not None and i1 is not None and L[i] > 0:
            dphi[i] = (phi_fd[i1] - phi_fd[i0]) / L[i]
    return dphi

def shmip_eff_cmap():
    light_blue = "#9ecae1"; white = "#ffffff"; red = "#d7301f"; dark_red = "#8B0000"
    return mcolors.LinearSegmentedColormap.from_list(
        "shmip_eff", [(0.00, light_blue), (0.10, white), (0.50, red), (1.00, dark_red)]
    )

# ---------------- load checkpoint (prefer saved fluxes) ----------------
def load_checkpoint(path):
    fd = _fd
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with fd.CheckpointFile(path, "r") as f:
        mesh = f.load_mesh()
        N    = f.load_function(mesh, "N")
        phi  = f.load_function(mesh, "phi")
        try:   h = f.load_function(mesh, "h")
        except Exception:
               h = fd.interpolate(fd.Constant(0.0), fd.FunctionSpace(mesh,"CG",1))
        try:   q_s_mag = f.load_function(mesh, "q_s_mag")
        except Exception:
               q_s_mag = None
        try:
            q_s = f.load_function(mesh, "q_s") if q_s_mag is None else None
        except Exception:
            q_s = None

    CR   = fd.FunctionSpace(mesh, "CR", 1)
    S    = fd.Function(CR)
    Q_ch = None
    npz_path = path.replace(".h5", "_cr.npz")
    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        if "S" in npz:
            S.dat.data[:] = npz["S"]
    else:
        try:
            with fd.CheckpointFile(path, "r") as f:
                m2  = f.load_mesh()
                S_s = f.load_function(m2, "S")
            S.dat.data[:] = S_s.dat.data_ro
        except Exception as exc:
            print(f"  [warn] S load failed ({type(exc).__name__}); S=0", flush=True)
    try:
        with fd.CheckpointFile(path, "r") as f:
            m2   = f.load_mesh()
            Qf   = f.load_function(m2, "Q_ch")
        Q_ch = fd.Function(CR)
        Q_ch.dat.data[:] = Qf.dat.data_ro
    except Exception:
        Q_ch = None

    tri = _cr["triangulation_from_mesh"](mesh)
    segments,_,_ = _cr["_edge_segments_and_indices"](mesh)
    return dict(mesh=mesh, tri=tri, N=N, phi=phi, S=S, h=h,
                Q_ch=Q_ch, q_s_mag=q_s_mag, q_s=q_s,
                segments=np.asarray(segments))

# ---------------- width-averaged N(x) ----------------
def width_average_N_along_x(mesh, N, nbins=200, band=None):
    N0 = dg0_array(mesh, N); xc,yc = cell_centroids(mesh); A = cell_areas(mesh)
    tri = _cr["triangulation_from_mesh"](mesh)
    x0,x1 = float(tri.x.min()), float(tri.x.max())
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
    fd = _fd
    segs = np.asarray(segments); L = edge_lengths(segs)
    emid = 0.5*(segs[:,0,:] + segs[:,1,:]); xe, ye = emid[:,0], emid[:,1]
    xc, yc = cell_centroids(mesh); A = cell_areas(mesh)
    edge_mask = np.ones(len(segs), bool); cell_mask = np.ones_like(xc, bool)
    if band is not None:
        y0,y1 = band; edge_mask &= (ye>=y0)&(ye<=y1); cell_mask &= (yc>=y0)&(yc<=y1)

    if Q_ch is not None:
        Qe = np.abs(np.nan_to_num(
            np.array(Q_ch.at(emid.tolist()), dtype=float), nan=0.0))
    else:
        if not allow_proxy: raise RuntimeError("Q_ch missing and --no-proxy set.")
        k_c, alpha_exp, delta_exp, phi_reg = 0.1, 1.25, -0.5, 1.0
        S_vals = np.maximum(np.nan_to_num(
            np.array(S.at(emid.tolist()), dtype=float), nan=0.0), 0.0)
        dphi_ds = _compute_dphi_ds(mesh, phi, segs)
        abs_phi = (dphi_ds**2 + phi_reg**2)**(delta_exp / 2.0)
        Qe = np.abs(k_c * (S_vals ** alpha_exp) * abs_phi * dphi_ds)
    QeL = Qe * L
    ie = np.clip(np.searchsorted(edges, xe, side="right")-1, 0, len(edges)-2)
    Qeff_x = np.zeros(len(edges)-1); np.add.at(Qeff_x, ie[edge_mask], QeL[edge_mask])

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
    if feff.shape != Nbar.shape:
        x_f = np.linspace(x[0], x[-1], max(len(feff), 2))
        feff = np.interp(x, x_f, feff, left=feff[0] if feff.size else 0.0,
                                    right=feff[-1] if feff.size else 0.0)
    quads = []; colors = []
    for i in range(len(x)-1):
        x0, x1 = x[i]/1000.0, x[i+1]/1000.0
        z0, z1 = Nbar[i]/1e6, Nbar[i+1]/1e6
        fmid = 0.5*(feff[i] + feff[i+1])
        col  = cmap(np.clip(fmid, 0.0, 1.0))
        verts = [(x0, y_pos, 0.0),
                 (x0, y_pos, z0),
                 (x1, y_pos, z1),
                 (x1, y_pos, 0.0)]
        quads.append(verts); colors.append(col)
    return quads, colors


# ======================================================================
# Public API: importable function for generating the ribbon figure.
# ======================================================================

def make_ribbon_figure(
    checkpoints,
    labels=None,
    outfile="shmip_ribbons.png",
    suite="A",
    band_width_E=200.0,
    nbins=250,
    elev=23.0,
    azim=230.0,
    no_proxy=False,
    print_diag=False,
    dpi=220,
):
    """
    Generate a SHMIP stacked-ribbon figure from checkpoint files.

    Parameters
    ----------
    checkpoints : list of str
        Paths to Firedrake HDF5 checkpoint files.
    labels : list of str, optional
        Labels for each experiment. Auto-derived from filenames if omitted.
    outfile : str
        Output image path.
    suite : str
        "A" (full domain width) or "E" (centre band only).
    band_width_E : float
        Width of centre band for Suite E (metres).
    nbins : int
        Number of x-bins for width-averaging.
    elev, azim : float
        3D view angles (degrees).
    no_proxy : bool
        If True, require saved Q_ch/q_s; error if missing.
    print_diag : bool
        If True, print mean channel share per experiment.
    dpi : int
        Output image resolution.

    Returns
    -------
    fig : matplotlib Figure
    """
    _ensure_imports()

    runs = [load_checkpoint(p) for p in checkpoints]
    if labels is None or len(labels) != len(runs):
        labels = [os.path.splitext(os.path.basename(p))[0] for p in checkpoints]

    bands = []
    xmins = []; xmaxs = []; zmax = 0.0
    for r in runs:
        tri = r["tri"]; ymin, ymax = float(tri.y.min()), float(tri.y.max())
        if suite.upper() == "E":
            yc = 0.5*(ymin+ymax)
            bands.append((yc - 0.5*band_width_E, yc + 0.5*band_width_E))
        else:
            bands.append(None)
        xmins.append(float(tri.x.min())); xmaxs.append(float(tri.x.max()))

    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.0, right=0.85, bottom=0.05, top=0.95)

    cmap = shmip_eff_cmap()
    y_positions = np.arange(len(runs))

    for j, (r, band, label) in enumerate(zip(runs, bands, labels)):
        x, Nbar, edges = width_average_N_along_x(r["mesh"], r["N"], nbins=nbins, band=band)
        feff_edges = partition_to_edges(
            r["mesh"], r["tri"], r["segments"], edges, band,
            Q_ch=r["Q_ch"], q_s_mag=r["q_s_mag"], q_s=r["q_s"],
            phi=r["phi"], S=r["S"], h=r["h"], allow_proxy=(not no_proxy)
        )
        if np.all(np.isnan(feff_edges)):
            feff_on_x = np.zeros_like(x)
        else:
            x_edges_mid = 0.5*(edges[:-1] + edges[1:])
            valid = ~np.isnan(feff_edges)
            feff_on_x = np.interp(
                x, x_edges_mid, feff_edges,
                left=feff_edges[valid][0] if np.any(valid) else 0.0,
                right=feff_edges[valid][-1] if np.any(valid) else 0.0,
            )

        quads, colors = build_ribbon_quads(x, Nbar, feff_on_x, y_positions[j], cmap)
        coll = Poly3DCollection(quads, facecolors=colors, edgecolors='none')
        ax.add_collection3d(coll)
        ax.plot(x/1000.0, np.full_like(x, y_positions[j]), Nbar/1e6,
                color='k', lw=1.2, zorder=10)

        if Nbar.size:
            zmax = max(zmax, float(np.nanmax(Nbar))/1e6)

        if print_diag and feff_on_x.size:
            print(f"{label}: mean channel share = {100*np.nanmean(feff_on_x):.1f}%")

    x0km = min(xmins)/1000.0; x1km = max(xmaxs)/1000.0
    ax.set_xlim(x0km, x1km)
    ax.set_ylim(-0.5, len(runs)-0.5)
    ax.set_zlim(0, zmax*1.05 if zmax > 0 else 1.0)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("experiment")
    ax.set_yticks(y_positions); ax.set_yticklabels(labels)
    ax.set_zlabel(r"$\bar N$ (MPa)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(r"$\bar N(x)$ with percentage of flux in the efficient system")

    import matplotlib.cm as cm
    sm = cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.7)
    cb.set_label("Fraction of flux in the efficient system")
    cb.set_ticks([0.0, 0.10, 0.50, 1.0])
    cb.set_ticklabels(["0%", "10%", "50%", "100%"])

    fig.savefig(outfile, dpi=dpi)
    print(f"Saved {outfile}")
    return fig


# ======================================================================
# CLI entry point (standalone usage).
# ======================================================================

if __name__ == "__main__":
    import argparse, sys

    # Parse before importing Firedrake so PETSc doesn't consume our flags.
    ap = argparse.ArgumentParser(
        description="SHMIP stacked ribbons: 2D N(x) per experiment with graded fill (3D view)")
    ap.add_argument("checkpoints", nargs="+", help="Firedrake HDF5 checkpoint(s)")
    ap.add_argument("--labels", nargs="*", help="Labels (A1 A2 ...)")
    ap.add_argument("--suite", choices=["A","E"], default="A")
    ap.add_argument("--band-width-E", type=float, default=200.0)
    ap.add_argument("--nbins", type=int, default=250)
    ap.add_argument("--outfile", default="shmip_ribbons.png")
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--elev", type=float, default=23.0)
    ap.add_argument("--azim", type=float, default=230.0)
    ap.add_argument("--print-diag", action="store_true")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    # Clear sys.argv so PETSc/Firedrake don't choke on our flags.
    sys.argv = sys.argv[:1]

    if os.environ.get("OMP_NUM_THREADS", "") not in ("", "1"):
        print("Note: for post-processing consider: export OMP_NUM_THREADS=1")

    make_ribbon_figure(
        checkpoints=args.checkpoints,
        labels=args.labels,
        outfile=args.outfile,
        suite=args.suite,
        band_width_E=args.band_width_E,
        nbins=args.nbins,
        elev=args.elev,
        azim=args.azim,
        no_proxy=args.no_proxy,
        print_diag=args.print_diag,
        dpi=args.dpi,
    )
