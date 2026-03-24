#!/usr/bin/env python3
import argparse, os, math, numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import firedrake as fd
from plot_cr_edges import triangulation_from_mesh, _edge_segments_and_indices, _cr_values_in_dmplex_order


# -------------- CLI (parse before Firedrake) --------------
ap = argparse.ArgumentParser(description="SHMIP shaded N(x) with along-x flux partition (uses saved fluxes)")
ap.add_argument("checkpoints", nargs="+", help="Firedrake HDF5 checkpoint(s)")
ap.add_argument("--labels", nargs="*", help="Panel labels")
ap.add_argument("--suite", choices=["A","E"], default="A", help="A=full width, E=centre 200 m band")
ap.add_argument("--band-width-E", type=float, default=200.0, help="Suite E band width (m)")
ap.add_argument("--nbins", type=int, default=250, help="x-bins for width averages")
ap.add_argument("--outfile", default="shmip_shaded.png")
ap.add_argument("--no-proxy", action="store_true", help="require saved fluxes; error if missing")
ap.add_argument("--print-diag", action="store_true")
args = ap.parse_args()

# ---------------- small helpers ----------------


def shmip_eff_cmap():
    # 0.00 -> light blue, 0.10 -> white, 0.50 -> red, >=0.50 stays red
    # tweak the blue/red tones if you want to match the paper exactly
    light_blue = "#9ecae1"   # gentle light blue
    white      = "#ffffff"
    red        = "#d7301f"   # warm red
    dark_red   = "#8B0000"
    return mcolors.LinearSegmentedColormap.from_list(
        "shmip_eff",
        [(0.00, light_blue),
         (0.10, white),
         (0.5, red),
         (1.00, dark_red)]
    )

def dg0_array(mesh, expr_or_func):
    V = fd.FunctionSpace(mesh,"DG",0); out = fd.Function(V); out.project(expr_or_func); return out.dat.data_ro

def cell_centroids(mesh):
    Vv = fd.VectorFunctionSpace(mesh,"DG",0)
    X  = fd.Function(Vv).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro
    return X[:,0].copy(), X[:,1].copy()

def cell_areas(mesh):
    V = fd.FunctionSpace(mesh,"DG",0)
    return fd.Function(V).interpolate(fd.CellVolume(mesh)).dat.data_ro.copy()

def edge_lengths(segments):
    d = segments[:,1,:]-segments[:,0,:]
    return np.sqrt((d*d).sum(axis=1))

def width_average_N_along_x(mesh, N, nbins=200, band=None):
    N0 = dg0_array(mesh, N); xc,yc = cell_centroids(mesh); A = cell_areas(mesh)
    tri = triangulation_from_mesh(mesh); x0,x1 = float(tri.x.min()), float(tri.x.max())
    mask = np.ones_like(N0, bool)
    if band is not None:
        y0,y1 = band; mask &= (yc>=y0)&(yc<=y1)
    # avoid over-binning
    nbins = int(min(nbins, max(32, int(np.sqrt(max(1,len(N0)))))))
    edges = np.linspace(x0,x1,nbins+1); xmid = 0.5*(edges[:-1]+edges[1:])
    ix = np.clip(np.searchsorted(edges, xc[mask], side="right")-1, 0, nbins-1)
    Nsum = np.zeros(nbins); Asum = np.zeros(nbins)
    np.add.at(Nsum, ix, N0[mask]*A[mask]); np.add.at(Asum, ix, A[mask])
    bad = Asum==0; Nsum[bad]=np.nan; Asum[bad]=np.nan
    Nbar = Nsum/Asum; keep = ~np.isnan(Nbar)
    return xmid[keep], Nbar[keep]

def load_checkpoint(path):
    with fd.CheckpointFile(path,"r") as f:
        mesh = f.load_mesh()
        N    = f.load_function(mesh,"N")
        phi  = f.load_function(mesh,"phi")
        S    = f.load_function(mesh,"S")
        try:   h = f.load_function(mesh,"h")
        except Exception: h = fd.interpolate(fd.Constant(0.0), fd.FunctionSpace(mesh,"CG",1))
        try:   Q_ch = f.load_function(mesh,"Q_ch")
        except Exception: Q_ch = None
        try:   q_s_mag = f.load_function(mesh,"q_s_mag")
        except Exception: q_s_mag = None
        try:   q_s = f.load_function(mesh,"q_s") if q_s_mag is None else None
        except Exception: q_s = None
    tri = triangulation_from_mesh(mesh)
    segs,_,_ = _edge_segments_and_indices(mesh)
    return dict(mesh=mesh,tri=tri,N=N,phi=phi,S=S,h=h,Q_ch=Q_ch,q_s_mag=q_s_mag,q_s=q_s,segments=np.asarray(segs))

def partition_vs_x(mesh, tri, segments, nbins, band, Q_ch=None, q_s_mag=None, q_s=None,
                   phi=None, S=None, h=None, allow_proxy=True):
    # band
    ymin,ymax = float(tri.y.min()), float(tri.y.max())
    if band is None: y0,y1 = ymin-1.0, ymax+1.0
    else: y0,y1 = band
    # geometry
    xc,yc = cell_centroids(mesh); A = cell_areas(mesh)
    segs  = np.asarray(segments); L = edge_lengths(segs)
    emid  = 0.5*(segs[:,0,:]+segs[:,1,:]); xe,ye = emid[:,0], emid[:,1]
    # bins
    x0,x1 = float(tri.x.min()), float(tri.x.max())
    nbins = int(min(nbins, max(32, int(np.sqrt(max(1,len(xc)))))))
    edges = np.linspace(x0,x1,nbins+1); xmid = 0.5*(edges[:-1]+edges[1:])
    # channels per x-bin
    edge_band = (ye>=y0)&(ye<=y1)
    if Q_ch is not None:
        Qe = _cr_values_in_dmplex_order(Q_ch).astype(float)
    else:
        if not allow_proxy: raise RuntimeError("Q_ch missing and --no-proxy set")
        # crude fallback
        Vv = fd.VectorFunctionSpace(mesh,"DG",0)
        g  = fd.Function(Vv).project(fd.grad(phi))
        gmag = dg0_array(mesh, fd.sqrt(fd.inner(g,g)))
        Svals= _cr_values_in_dmplex_order(S).astype(float)
        # map gmag by y only (last resort)
        g_edge = np.interp(ye, np.linspace(ymin,ymax,len(gmag)), np.sort(gmag))
        Qe = np.maximum(Svals,0.0)*np.maximum(g_edge,0.0)
    QeL = Qe*L
    ie = np.clip(np.searchsorted(edges, xe, side="right")-1, 0, len(xmid)-1)
    Qeff_x = np.zeros_like(xmid); np.add.at(Qeff_x, ie[edge_band], QeL[edge_band])
    # sheet per x-bin
    cell_band = (yc>=y0)&(yc<=y1)
    if q_s_mag is None and q_s is not None:
        V = fd.FunctionSpace(mesh,"DG",0); tmp = fd.Function(V)
        tmp.project(fd.sqrt(fd.inner(q_s,q_s)+fd.Constant(1e-30))); q_s_mag = tmp
    if q_s_mag is not None:
        mag = q_s_mag.dat.data_ro
    else:
        if not allow_proxy: raise RuntimeError("q_s/_mag missing and --no-proxy set")
        Vv = fd.VectorFunctionSpace(mesh,"DG",0)
        g  = fd.Function(Vv).project(fd.grad(phi))
        mag = dg0_array(mesh, fd.max_value(h,0.0)**3 * fd.sqrt(fd.inner(g,g)))
    ic = np.clip(np.searchsorted(edges, xc, side="right")-1, 0, len(xmid)-1)
    Qineff_x = np.zeros_like(xmid); np.add.at(Qineff_x, ic[cell_band], (mag*A)[cell_band])
    Qtot_x = Qeff_x + Qineff_x
    with np.errstate(divide="ignore", invalid="ignore"):
        feff_x = np.where(Qtot_x>0, Qeff_x/Qtot_x, 0.0)
    keep = Qtot_x>0
    return xmid[keep], feff_x[keep], Qeff_x[keep], Qineff_x[keep]

# -------------------- shading renderer --------------------
def shade_under_N(ax, xkm, Nbar, feff, color_sheet="tab:blue", color_chan="tab:red", alpha=0.35):
    """
    Stack the area under N(x) into sheet (blue) and channel (red).
    sheet area = (1-feff)*N, channel area = feff*N.
    """
    Ns = Nbar/1e6  # MPa for plotting
    y_sheet = (1.0 - feff) * Ns
    y_chan  = feff * Ns
    ax.fill_between(xkm, 0.0, y_sheet, color=color_sheet, alpha=alpha, linewidth=0)
    ax.fill_between(xkm, y_sheet, y_sheet + y_chan, color=color_chan, alpha=alpha, linewidth=0)
    # outline N(x)
    ax.plot(xkm, Ns, color="k", lw=1.2)


def shaded_partition_under_N(ax, x_m, Nbar, feff, cmap=None, alpha=1.0, add_colorbar=False):
    """
    Shade the area under N(x) with a color determined by feff(x).
    feff in [0,1]; 0%->light blue, 10%->white, 50%+->red.
    """
    if cmap is None:
        cmap = shmip_eff_cmap()

    # ensure same grid for N and feff
    if feff.shape != Nbar.shape or np.any(np.diff(x_m) <= 0):
        # light resample of feff onto N grid
        x_f = np.linspace(x_m[0], x_m[-1], len(feff))
        feff = np.interp(x_m, x_f, feff, left=feff[0], right=feff[-1])

    # build trapezoids for each x-interval
    verts = []
    colors = []
    for i in range(len(x_m) - 1):
        x0, x1 = x_m[i], x_m[i+1]
        y0, y1 = Nbar[i]/1e6, Nbar[i+1]/1e6  # MPa on plot
        # trapezoid from (x0,0)-(x0,y0)-(x1,y1)-(x1,0)
        verts.append([(x0/1000.0, 0.0),
                      (x0/1000.0, y0),
                      (x1/1000.0, y1),
                      (x1/1000.0, 0.0)])
        # color by feff at segment midpoint
        fmid = 0.5*(feff[i] + feff[i+1])
        # clamp into [0,1]
        colors.append(cmap(np.clip(fmid, 0.0, 1.0)))

    coll = mcoll.PolyCollection(verts, facecolors=colors, edgecolors='none', alpha=alpha)
    ax.add_collection(coll)
    # thin outline of N(x) on top
    ax.plot(x_m/1000.0, Nbar/1e6, color="k", lw=1.0)

    if add_colorbar:
        # colorbar with ticks at 0%, 10%, 50%
        sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Efficient share (channels)")
        cbar.set_ticks([0.0, 0.10, 0.50, 1.0])
        cbar.set_ticklabels(["0%", "10%", "50%", "100%"])


# -------------------- main plotting --------------------
def main():
    runs = [load_checkpoint(p) for p in args.checkpoints]
    labels = args.labels if args.labels and len(args.labels)==len(runs) else \
             [os.path.splitext(os.path.basename(p))[0] for p in args.checkpoints]

    # bands
    bands = []
    for r in runs:
        tri = r["tri"]; ymin,ymax = float(tri.y.min()), float(tri.y.max())
        if args.suite.upper()=="E":
            yc = 0.5*(ymin+ymax); bands.append((yc-0.5*args.band_width_E, yc+0.5*args.band_width_E))
        else:
            bands.append(None)

    # figure: one panel per run (3 columns)
    n = len(runs); ncol = 1; nrow = math.ceil(n/ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5*ncol, 3.4*nrow), squeeze=False, constrained_layout=True)

    for i,(r,band,label) in enumerate(zip(runs, bands, labels)):
        ax = axes[i//ncol][i%ncol]
        # N(x)
        x, Nbar = width_average_N_along_x(r["mesh"], r["N"], nbins=args.nbins, band=band)
        # partition vs x (prefer saved fluxes)
        xfx, feffx, _, _ = partition_vs_x(
            r["mesh"], r["tri"], r["segments"], nbins=args.nbins, band=band,
            Q_ch=r["Q_ch"], q_s_mag=r["q_s_mag"], q_s=r["q_s"],
            phi=r["phi"], S=r["S"], h=r["h"], allow_proxy=not args.no_proxy
        )
        # align partitions to N(x) grid (light linear interp)
        feff_interp = np.interp(x, xfx, feffx, left=feffx[0] if feffx.size else 0.0,
                                              right=feffx[-1] if feffx.size else 0.0)

        shaded_partition_under_N(ax, x, Nbar, feff_interp, cmap=shmip_eff_cmap(), alpha=0.9, add_colorbar=True)
        ax.set_xlabel("x (km)")
        ax.set_ylabel(r"$\bar N$ (MPa)")
        ax.grid(ls=":", alpha=0.5)
        if args.print_diag:
            eff_mean = 100*np.nanmean(feff_interp) if feff_interp.size else 0.0
            print(f"{label}: mean channel share ≈ {eff_mean:.1f}%")

    # hide any empty subplots
    for j in range(n, nrow*ncol):
        fig.delaxes(axes[j//ncol][j%ncol])



    fig.savefig(args.outfile, dpi=200)
    print(f"Saved {args.outfile}")

if __name__ == "__main__":
    if os.environ.get("OMP_NUM_THREADS","") not in ("","1"):
        print("Note: for post-processing set OMP_NUM_THREADS=1")
    main()
