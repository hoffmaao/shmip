#!/usr/bin/env python3
"""
plot_werder_fig12.py  –  SHMIP Suite A analogue of Werder (2013) Figure 12.

For each experiment (A1–A6) shows:
  • Background fill  : N (effective pressure, MPa)  –  red = low N (near flotation),
                       blue = high N (far from flotation)  [RdBu colormap]
  • Channel network  : CR-edge lines coloured + thickened by |Q_ch|  (log scale)
  • φ contours       : dashed white lines at evenly-spaced hydraulic-potential levels

Usage (run from shmip_A/):
  python plot_werder_fig12.py
  python plot_werder_fig12.py --cases A1 A3 A5
  python plot_werder_fig12.py --no-phi-contours --outfile outputs/fig12_nophi.png
  python plot_werder_fig12.py --N-max 6.0          # fix N colourbar ceiling to 6 MPa
"""

import argparse, os, sys
import numpy as np

# ── parse args BEFORE importing Firedrake so PETSc doesn't consume our flags ──
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--cases", nargs="*", default=None,
                help="Cases to plot (e.g. A1 A2 A3); default: all found")
ap.add_argument("--chk-dir", default="outputs/checkpoints",
                help="Directory containing <CASE>.h5 checkpoint files")
ap.add_argument("--outfile", default="outputs/werder_fig12.png",
                help="Output image path")
ap.add_argument("--no-phi-contours", action="store_true",
                help="Omit hydraulic-potential (φ) contour lines")
ap.add_argument("--n-phi-levels", type=int, default=8,
                help="Number of φ contour levels per panel")
ap.add_argument("--N-max", type=float, default=None,
                help="Upper bound for N colourbar in MPa (default: 99.5th-pct of data)")
ap.add_argument("--Q-floor", type=float, default=None,
                help="Min |Q_ch| for log colourbar in m³/s (default: 1st-pct of non-zero)")
ap.add_argument("--Q-ceil", type=float, default=None,
                help="Max |Q_ch| for log colourbar in m³/s (default: 99.5th-pct)")
ap.add_argument("--dpi", type=int, default=220)
args = ap.parse_args()

# Clear sys.argv so PETSc (loaded with firedrake) doesn't treat our flags as its own options
import sys as _sys
_sys.argv = _sys.argv[:1]

# ── heavy imports ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import firedrake as fd
from hydropack.constants import pcs as _pcs
from hydropack.utilities import CRTools
from plot_cr_edges import (
    triangulation_from_mesh,
    _edge_segments_and_indices,
    _cr_values_in_dmplex_order,
)

# ── constants ─────────────────────────────────────────────────────────────────
CASES_ALL = ["A1", "A2", "A3", "A4", "A5", "A6"]
N_CMAP    = "RdBu"       # red = low N (near flotation); blue = high N
Q_CMAP    = "hot_r"      # white/yellow = small Q; dark orange/red = large Q

_K_C   = float(_pcs["k_c"])    # channel conductivity (m^(3-2α)/s)
_ALPHA = float(_pcs["alpha"])  # 1.25
_DELTA = float(_pcs["delta"])  # -0.5
_PHI_REG = 1.0                 # Pa/m, regularisation floor (same as hs_solver)


def _recompute_Q_ch(mesh, phi, S):
    """
    Recompute channel discharge Q_ch = k_c * max(S,0)^α * max(|∂φ/∂s|, φ_reg)^δ
    at CR1 edge midpoints, using the same formula as Glads2DModel.compute_flux_fields().
    Returns a numpy array in DMPlex edge order.
    """
    U  = fd.FunctionSpace(mesh, "CG", 1)
    CR = fd.FunctionSpace(mesh, "CR", 1)
    utilities = CRTools(mesh, U, CR)

    dphi_ds = fd.Function(CR)
    utilities.ds_assemble(phi, dphi_ds)

    S_vals       = np.maximum(_cr_values_in_dmplex_order(S).astype(float),      0.0)
    dphi_ds_vals = np.maximum(_cr_values_in_dmplex_order(dphi_ds).astype(float), _PHI_REG)

    return _K_C * S_vals ** _ALPHA * dphi_ds_vals ** _DELTA


# ── loader ────────────────────────────────────────────────────────────────────
def load_panel(path):
    """Return a dict with all arrays needed for one panel."""
    with fd.CheckpointFile(path, "r") as f:
        mesh = f.load_mesh()
        N   = f.load_function(mesh, "N")
        phi = f.load_function(mesh, "phi")
        S   = f.load_function(mesh, "S")

    # Recompute Q_ch from phi and S (direct load via CheckpointFile fails for
    # CR1 functions that were saved from a non-standard function-space instance)
    Q_cr, q_source = None, "none"
    try:
        Q_cr     = _recompute_Q_ch(mesh, phi, S)
        q_source = "Q_ch (recomputed)"
    except Exception as e:
        print(f"    Warning: Q_ch recomputation failed ({e}); falling back to S")

    if Q_cr is None:
        try:
            Q_cr     = np.maximum(_cr_values_in_dmplex_order(S).astype(float), 0.0)
            q_source = "S (proxy for Q_ch)"
        except Exception:
            pass

    tri  = triangulation_from_mesh(mesh)
    segs = np.asarray(_edge_segments_and_indices(mesh)[0])
    if Q_cr is None:
        Q_cr = np.zeros(len(segs))

    return dict(
        tri      = tri,
        segs     = segs,
        N_vals   = N.dat.data_ro.copy(),    # Pa, at CG1 vertices
        phi_vals = phi.dat.data_ro.copy(),  # Pa, at CG1 vertices
        Q_vals   = Q_cr,                    # m³/s, at CR edge midpoints
        q_source = q_source,
    )


# ── load data ─────────────────────────────────────────────────────────────────
cases_req = args.cases if args.cases else CASES_ALL
panels    = {}
for tag in cases_req:
    path = os.path.join(args.chk_dir, f"{tag}.h5")
    if not os.path.exists(path):
        print(f"  skip {tag}: {path} not found", flush=True)
        continue
    print(f"  loading {tag} …", flush=True)
    panels[tag] = load_panel(path)
    print(f"    discharge source: {panels[tag]['q_source']}")

loaded = [t for t in cases_req if t in panels]
if not loaded:
    sys.exit("No checkpoint files found — run the simulation first.")


# ── global colour bounds ──────────────────────────────────────────────────────
N_all = np.concatenate([panels[t]["N_vals"] for t in loaded])
N_vmax_MPa = (args.N_max
              if args.N_max is not None
              else float(np.nanpercentile(N_all, 99.5)) / 1e6)
N_norm = mcolors.Normalize(vmin=0.0, vmax=N_vmax_MPa)

Q_pos = np.concatenate([
    panels[t]["Q_vals"][panels[t]["Q_vals"] > 0] for t in loaded
    if (panels[t]["Q_vals"] > 0).any()
])
if Q_pos.size:
    Q_floor = (args.Q_floor if args.Q_floor is not None
               else max(float(np.percentile(Q_pos, 1)), 1e-10))
    Q_ceil  = (args.Q_ceil  if args.Q_ceil  is not None
               else float(np.percentile(Q_pos, 99.5)))
    Q_floor = max(Q_floor, 1e-10)
    Q_ceil  = max(Q_ceil, Q_floor * 10)
else:
    Q_floor, Q_ceil = 1e-6, 1e-3   # dummy range when no channels exist

Q_norm = mcolors.LogNorm(vmin=Q_floor, vmax=Q_ceil)


# ── figure layout ─────────────────────────────────────────────────────────────
# domain is 100 km × 20 km → 5:1 aspect ratio per panel
ncols = min(3, len(loaded))
nrows = int(np.ceil(len(loaded) / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 4.6, nrows * 1.85 + 1.5),
    constrained_layout=True,
)
axes = np.atleast_1d(axes).ravel()
for ax in axes[len(loaded):]:
    ax.set_visible(False)


# ── draw panels ───────────────────────────────────────────────────────────────
for ax, tag in zip(axes, loaded):
    p       = panels[tag]
    tri     = p["tri"]
    N_MPa   = p["N_vals"]   / 1e6     # convert to MPa for display
    phi_MPa = p["phi_vals"] / 1e6
    segs    = p["segs"]
    Q       = p["Q_vals"]

    # — N background —————————————————————————————————————————————————————
    ax.tripcolor(tri, N_MPa, shading="gouraud", norm=N_norm, cmap=N_CMAP)

    # — φ contours  (dashed white) ————————————————————————————————————————
    if not args.no_phi_contours:
        phi_lo, phi_hi = float(np.nanmin(phi_MPa)), float(np.nanmax(phi_MPa))
        if phi_hi > phi_lo:
            # inner levels only (drop the two extremes to avoid boundary artefacts)
            lvls = np.linspace(phi_lo, phi_hi, args.n_phi_levels + 2)[1:-1]
            try:
                ax.tricontour(tri, phi_MPa, levels=lvls,
                              colors="white", linewidths=0.55,
                              linestyles="--", alpha=0.60)
            except Exception:
                pass  # tricontour can rarely fail on degenerate meshes

    # — channel network  (|Q_ch|, log-scaled width + colour) ————————————
    active = Q > Q_floor * 0.05     # draw only edges carrying meaningful flux
    if active.any():
        Q_clamp = np.clip(Q[active], Q_floor, Q_ceil)
        # linewidth in [0.4, 3.2] pt, scaled by log(Q) within the colourbar range
        log_span = np.log10(Q_ceil) - np.log10(Q_floor)
        lw = 0.4 + 2.8 * (np.log10(Q_clamp) - np.log10(Q_floor)) / log_span
        lc = LineCollection(segs[active], linewidths=lw,
                            norm=Q_norm, cmap=Q_CMAP, zorder=5,
                            capstyle="round")
        lc.set_array(Q[active])
        ax.add_collection(lc)

    # — axes cosmetics ————————————————————————————————————————————————————
    ax.set_xlim(float(tri.x.min()), float(tri.x.max()))
    ax.set_ylim(float(tri.y.min()), float(tri.y.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(tag, fontsize=11, fontweight="bold")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# x-axis tick labels on the bottom row only
x_km_ticks = np.linspace(0, 100, 6)   # 0, 20, 40, 60, 80, 100 km
for ax, tag in zip(axes[(nrows-1)*ncols : len(loaded)],
                   loaded[(nrows-1)*ncols :]):
    ax.set_xticks(x_km_ticks * 1e3)
    ax.set_xticklabels([f"{v:.0f}" for v in x_km_ticks], fontsize=7)
    ax.set_xlabel("x  (km)", fontsize=8)


# ── shared colorbars ───────────────────────────────────────────────────────────
ax_list = axes[:len(loaded)].tolist()

# N colorbar  –  below the panels
sm_N = plt.cm.ScalarMappable(cmap=N_CMAP, norm=N_norm)
sm_N.set_array([])
cb_N = fig.colorbar(sm_N, ax=ax_list, location="bottom",
                    shrink=0.55, aspect=40, pad=0.03)
cb_N.set_label("Effective pressure  $N$  (MPa)", fontsize=9)
cb_N.ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=False, nbins=6))

# Q colorbar  –  right of the panels
sm_Q = plt.cm.ScalarMappable(cmap=Q_CMAP, norm=Q_norm)
sm_Q.set_array([])
cb_Q = fig.colorbar(sm_Q, ax=ax_list, location="right",
                    shrink=0.65, aspect=22, pad=0.01)
cb_Q.set_label(r"Channel discharge  $|Q_c|$  (m$^3$ s$^{-1}$)", fontsize=9)
cb_Q.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())

fig.suptitle(
    "SHMIP Suite A – Subglacial drainage system  "
    r"(after Werder et al. 2013, Fig. 12)"
    "\n"
    r"Background: $N$ [MPa]  ·  dashed lines: $\varphi$ contours  ·  "
    r"coloured lines: channel discharge $|Q_c|$",
    fontsize=9.5,
)


# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
fig.savefig(args.outfile, dpi=args.dpi, bbox_inches="tight")
print(f"Saved → {args.outfile}")
