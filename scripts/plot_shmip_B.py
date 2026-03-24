#!/usr/bin/env python3
"""
plot_shmipB.py — SHMIP Suite B visualisation, analogous to Figure 3 in
de Fleurian et al. (2018).

Figure layout
─────────────
  Upper subfigure : 2×3 grid of plan-view panels for B1–B5
                    • Background fill  : N (effective pressure, MPa), RdBu colourmap
                    • Coloured lines   : channel discharge |Q_c|, hot_r log scale
                    • Dashed white     : hydraulic-potential φ contours
                    • Black dots       : moulin positions (reproduced from the same
                                         random seed used in run_shmip_B.py)
  Lower subfigure : width-averaged N(x) profiles for B1–B5 + A5 reference line

Run from shmip_B/:
  python plot_shmipB.py

Outputs: outputs/shmip_B_fig3.png
"""

import sys
import os

# ── clear argv before Firedrake/PETSc initialises ────────────────────────────
sys.argv = sys.argv[:1]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import firedrake as fd
from hydropack.constants import pcs as _pcs
from hydropack.utilities import CRTools

# import the shared CR-edge plotting utilities from shmip_A/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "shmip_A"))
from plot_cr_edges import (
    triangulation_from_mesh,
    _edge_segments_and_indices,
    _cr_values_in_dmplex_order,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — edit here to adjust paths, colours, etc.
# ═══════════════════════════════════════════════════════════════════════════════

CHK_DIR_B  = "outputs/checkpoints"                           # B1–B5 checkpoints
CHK_DIR_A  = os.path.join("..", "shmip_A", "outputs",
                          "checkpoints")                     # A5 reference
OUTFILE    = "outputs/shmip_B_fig3.png"

CASES      = ["B1", "B2", "B3", "B4", "B5"]
N_MOULINS  = {"B1": 1, "B2": 10, "B3": 20, "B4": 50, "B5": 100}

# These MUST match the values used in run_shmip_B.py so moulin dots are correct
MOULIN_SEED   = 1
MOULIN_SIGMA  = 1000.0    # m  (Gaussian half-width — used only for reference)
MARGIN_BUFFER = 2e3       # m  (interior placement buffer on all four sides)

Lx, Ly     = 100e3, 20e3   # domain dimensions [m]
N_PHI_LVLS = 8             # number of φ contour levels per panel
DPI        = 220

# ── Channel visibility threshold ────────────────────────────────────────────
# CR1 DOFs are at *edge midpoints* — each represents a 1D channel conduit
# running along the interface between two adjacent triangular elements.
# Plotting "active" channels should therefore show only edges where water is
# actually flowing, not every edge that has residual S > 0 from initialisation.
#
# The correct filter is Q_ch directly (the quantity shown on the colorbar):
#   Q_ch = k_c · S^α · |∂φ/∂s|^δ   [m³/s]
#
# Q_VIS_FLOOR is the minimum discharge to be drawn.  At 1e-3 m³/s (1 L/s) the
# channels are genuinely hydraulically active; this gives ~800 edges for B1
# (single prominent channel) up to ~1400 for B5 (dense network) — a clean
# representation of the 1D conduit network on the triangular mesh.
Q_VIS_FLOOR = 1e-3   # m³/s  — minimum channel discharge to draw

# N(x) profile bins: must be ≤ nx to avoid x-node aliasing.
# Suite B uses nx=141 → 142 unique x-positions; 200 bins (> 142) creates empty
# bins and a sawtooth pattern.  Match to the mesh element count.
PROFILE_NBINS = 141

N_CMAP     = "RdBu"        # red = low N (near flotation); blue = high N
Q_CMAP     = "hot_r"       # dark = large discharge

# colours for N(x) profiles (one per B case)
PROFILE_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

# ═══════════════════════════════════════════════════════════════════════════════
# Physical constants (from hydropack)
# ═══════════════════════════════════════════════════════════════════════════════

_K_C   = float(_pcs["k_c"])    # channel conductivity
_ALPHA = float(_pcs["alpha"])  # 1.25
_DELTA = float(_pcs["delta"])  # −0.5
_PREG  = 1.0                   # Pa/m — regularisation floor (matches run_shmip_B)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def moulin_positions(n, xlo, xhi, ylo, yhi,
                     seed=MOULIN_SEED, margin=MARGIN_BUFFER):
    """Reproduce the random moulin locations used in run_shmip_B.py."""
    rng = np.random.default_rng(seed)
    xs  = rng.uniform(xlo + margin, xhi - margin, n)
    ys  = rng.uniform(ylo + margin, yhi - margin, n)
    return xs, ys


def load_panel(path):
    """Load one B checkpoint; return everything needed for one figure panel."""
    with fd.CheckpointFile(path, "r") as f:
        mesh = f.load_mesh()
        N    = f.load_function(mesh, "N")
        phi  = f.load_function(mesh, "phi")
        S    = f.load_function(mesh, "S")

    # Recompute Q_ch = k_c · max(S,0)^α · max(|∂φ/∂s|, φ_reg)^δ
    # (direct CheckpointFile load of Q_ch fails with a ConvergenceError on CR1
    # embedding reconstruction, so we recompute from phi and S instead)
    U  = fd.FunctionSpace(mesh, "CG", 1)
    CR = fd.FunctionSpace(mesh, "CR", 1)
    dphi_ds = fd.Function(CR)
    CRTools(mesh, U, CR).ds_assemble(phi, dphi_ds)

    S_v  = np.maximum(_cr_values_in_dmplex_order(S).astype(float), 0.0)
    dp_v = np.maximum(_cr_values_in_dmplex_order(dphi_ds).astype(float), _PREG)
    Q    = _K_C * S_v**_ALPHA * dp_v**_DELTA

    tri    = triangulation_from_mesh(mesh)
    segs   = np.asarray(_edge_segments_and_indices(mesh)[0])
    coords = mesh.coordinates.dat.data_ro

    return dict(
        tri      = tri,
        segs     = segs,
        N_vals   = N.dat.data_ro.copy(),      # Pa  (CG1 nodal)
        phi_vals = phi.dat.data_ro.copy(),    # Pa  (CG1 nodal)
        Q_vals   = Q,                         # m³/s (CR edge-midpoint)
        S_vals   = S_v,                       # m²  (CR edge-midpoint)
        x_coords = coords[:, 0].copy(),
        xlim     = (float(coords[:, 0].min()), float(coords[:, 0].max())),
        ylim     = (float(coords[:, 1].min()), float(coords[:, 1].max())),
    )


def width_averaged_Nx(N_vals, x_coords, nbins=PROFILE_NBINS):
    """Width-average N over x-bins; returns (x_centres_m, N_mean_Pa)."""
    bins  = np.linspace(0.0, Lx, nbins + 1)
    idx   = np.digitize(x_coords, bins) - 1
    Nx    = np.zeros(nbins)
    count = np.zeros(nbins, dtype=int)
    for i, v in zip(idx, N_vals):
        if 0 <= i < nbins:
            Nx[i] += v
            count[i] += 1
    mask     = count > 0
    Nx[mask] /= count[mask]
    return 0.5 * (bins[:-1] + bins[1:]), Nx


# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════

panels = {}
for tag in CASES:
    path = os.path.join(CHK_DIR_B, f"{tag}.h5")
    if not os.path.exists(path):
        print(f"  skip {tag}: {path} not found")
        continue
    print(f"  loading {tag} …", flush=True)
    panels[tag] = load_panel(path)

loaded = [t for t in CASES if t in panels]
if not loaded:
    sys.exit("No B checkpoints found — run run_shmip_B.py first.")

# A5 reference (Suite A; optional — skipped gracefully if absent)
a5_x, a5_Nx = None, None
path_a5 = os.path.join(CHK_DIR_A, "A5.h5")
if os.path.exists(path_a5):
    print("  loading A5 reference …", flush=True)
    with fd.CheckpointFile(path_a5, "r") as f:
        m_a5 = f.load_mesh()
        N_a5 = f.load_function(m_a5, "N")
    a5_x, a5_Nx = width_averaged_Nx(N_a5.dat.data_ro.copy(),
                                     m_a5.coordinates.dat.data_ro[:, 0].copy(),
                                     nbins=200)   # Suite A: nx=200, no aliasing


# ═══════════════════════════════════════════════════════════════════════════════
# Global colour bounds (consistent across all panels)
# ═══════════════════════════════════════════════════════════════════════════════

N_all   = np.concatenate([panels[t]["N_vals"] for t in loaded])
N_vmax  = float(np.nanpercentile(N_all, 99.5)) / 1e6
N_norm  = mcolors.Normalize(vmin=0.0, vmax=N_vmax)

# Colour bounds: floor = Q_VIS_FLOOR (the minimum we draw); ceiling from the
# 99.5th percentile of all active edges across all cases.  The percentile
# guards against the occasional corrupted outlier (e.g. a near-zero-gradient
# edge where Q ∝ |∂φ/∂s|^δ = |∂φ/∂s|^{-0.5} → ∞).
_Q_active = np.concatenate([
    panels[t]["Q_vals"][panels[t]["Q_vals"] >= Q_VIS_FLOOR]
    for t in loaded
    if (panels[t]["Q_vals"] >= Q_VIS_FLOOR).any()
]) if any((panels[t]["Q_vals"] >= Q_VIS_FLOOR).any() for t in loaded) \
  else np.array([Q_VIS_FLOOR])
Q_floor = Q_VIS_FLOOR
Q_ceil  = max(float(np.percentile(_Q_active, 99.5)), Q_floor * 10)
Q_norm  = mcolors.LogNorm(vmin=Q_floor, vmax=Q_ceil)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure layout — subfigures so each part can use constrained_layout
# ═══════════════════════════════════════════════════════════════════════════════

ncols = 3
nrows_map = int(np.ceil(len(loaded) / ncols))   # 2 rows for 5 panels

fig = plt.figure(figsize=(ncols * 4.6, nrows_map * 1.85 + 2.8),
                 layout="constrained")
# Two vertical subfigures: spatial maps on top, N(x) profiles on bottom
sf_top, sf_bot = fig.subfigures(2, 1,
                                 height_ratios=[nrows_map * 1.85,
                                                2.4])

map_axes = sf_top.subplots(nrows_map, ncols)
map_axes = np.atleast_2d(map_axes).ravel()
for ax in map_axes[len(loaded):]:
    ax.set_visible(False)

nx_ax = sf_bot.subplots(1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Draw spatial panels
# ═══════════════════════════════════════════════════════════════════════════════

for ax, tag in zip(map_axes, loaded):
    p       = panels[tag]
    tri     = p["tri"]
    N_MPa   = p["N_vals"]   / 1e6
    phi_MPa = p["phi_vals"] / 1e6
    segs    = p["segs"]
    Q       = p["Q_vals"]
    n_m     = N_MOULINS[tag]

    # — N background ——————————————————————————————————————————————————————
    ax.tripcolor(tri, N_MPa, shading="gouraud", norm=N_norm, cmap=N_CMAP)

    # — φ contours (dashed white) —————————————————————————————————————————
    phi_lo, phi_hi = float(np.nanmin(phi_MPa)), float(np.nanmax(phi_MPa))
    if phi_hi > phi_lo:
        lvls = np.linspace(phi_lo, phi_hi, N_PHI_LVLS + 2)[1:-1]
        try:
            ax.tricontour(tri, phi_MPa, levels=lvls, colors="white",
                          linewidths=0.5, linestyles="--", alpha=0.55)
        except Exception:
            pass

    # — channel network (|Q_c|) ———————————————————————————————————————————
    # Filter by Q >= Q_VIS_FLOOR: draw only edges where the 1D channel conduit
    # (running along the interface between two adjacent mesh elements) is
    # carrying physically meaningful discharge.  This is consistent with the
    # CR1 formulation: each active segment IS the line between two elements.
    active = Q >= Q_VIS_FLOOR
    if active.any():
        Q_c  = np.clip(Q[active], Q_floor, Q_ceil)
        span = np.log10(Q_ceil) - np.log10(Q_floor)
        lw   = 0.4 + 2.8 * (np.log10(Q_c) - np.log10(Q_floor)) / span
        lc   = LineCollection(segs[active], linewidths=lw,
                              norm=Q_norm, cmap=Q_CMAP, zorder=5,
                              capstyle="round")
        lc.set_array(Q[active])
        ax.add_collection(lc)

    # — moulin positions (black dots) —————————————————————————————————————
    mx, my = moulin_positions(n_m, p["xlim"][0], p["xlim"][1],
                               p["ylim"][0], p["ylim"][1])
    ax.scatter(mx, my, s=14, c="black", zorder=9, linewidths=0,
               marker="o", label="moulins")

    # — cosmetics —————————————————————————————————————————————————————————
    ax.set_xlim(*p["xlim"])
    ax.set_ylim(*p["ylim"])
    ax.set_aspect("equal", adjustable="box")
    n_label = f"{n_m} moulin" + ("s" if n_m > 1 else "")
    ax.set_title(f"{tag}  ({n_label})", fontsize=10, fontweight="bold")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

# x-axis ticks on bottom spatial row only
x_km = np.linspace(0, 100, 6)
for ax, tag in zip(map_axes[(nrows_map - 1) * ncols: len(loaded)],
                   loaded[(nrows_map - 1) * ncols:]):
    ax.set_xticks(x_km * 1e3)
    ax.set_xticklabels([f"{v:.0f}" for v in x_km], fontsize=7)
    ax.set_xlabel("x  (km)", fontsize=8)


# ── shared colorbars for spatial panels ───────────────────────────────────────
visible = [ax for ax in map_axes if ax.get_visible()]

sm_N = plt.cm.ScalarMappable(cmap=N_CMAP, norm=N_norm)
sm_N.set_array([])
cb_N = sf_top.colorbar(sm_N, ax=visible, location="bottom",
                        shrink=0.50, aspect=40, pad=0.03)
cb_N.set_label("Effective pressure  $N$  (MPa)", fontsize=9)
cb_N.ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=False, nbins=5))

sm_Q = plt.cm.ScalarMappable(cmap=Q_CMAP, norm=Q_norm)
sm_Q.set_array([])
cb_Q = sf_top.colorbar(sm_Q, ax=visible, location="right",
                        shrink=0.60, aspect=20, pad=0.01)
cb_Q.set_label(r"Channel discharge  $|Q_c|$  (m$^3$ s$^{-1}$)", fontsize=9)
cb_Q.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())


# ═══════════════════════════════════════════════════════════════════════════════
# N(x) profile panel
# ═══════════════════════════════════════════════════════════════════════════════

ls_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

for i, tag in enumerate(loaded):
    p  = panels[tag]
    xc, Nx = width_averaged_Nx(p["N_vals"], p["x_coords"])
    nx_ax.plot(xc / 1e3, Nx / 1e6,
               color=PROFILE_COLORS[i],
               lw=1.8, ls=ls_cycle[i % len(ls_cycle)],
               label=f"{tag}  ({N_MOULINS[tag]} moulins)")

if a5_Nx is not None:
    nx_ax.plot(a5_x / 1e3, a5_Nx / 1e6,
               color="black", lw=2.0, ls="--", zorder=10,
               label="A5 (uniform reference, same total flux)")

nx_ax.set_xlabel("Along-flow distance  x  (km)", fontsize=10)
nx_ax.set_ylabel("Width-averaged  N  (MPa)", fontsize=10)
nx_ax.set_title(
    "Width-averaged effective pressure — Suite B vs A5 reference",
    fontsize=10)
nx_ax.legend(ncol=3, fontsize=8.5, loc="upper left",
             framealpha=0.85, edgecolor="0.8")
nx_ax.set_xlim(0, 100)
nx_ax.set_ylim(bottom=0)
nx_ax.grid(True, alpha=0.22)
nx_ax.xaxis.set_major_locator(mticker.MultipleLocator(20))


# ═══════════════════════════════════════════════════════════════════════════════
# Title + save
# ═══════════════════════════════════════════════════════════════════════════════

fig.suptitle(
    "SHMIP Suite B — Subglacial drainage with localised moulin input\n"
    r"(after de Fleurian et al. 2018, Fig. 3)"
    "   ·   "
    r"Background: $N$ [MPa]  ·  dashed: $\varphi$  ·  coloured lines: $|Q_c|$  ·  dots: moulins",
    fontsize=9,
)

os.makedirs(os.path.dirname(OUTFILE) or ".", exist_ok=True)
fig.savefig(OUTFILE, dpi=DPI, bbox_inches="tight")
print(f"Saved → {OUTFILE}")
