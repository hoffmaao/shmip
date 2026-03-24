#!/usr/bin/env python3
# Copyright (C) 2024 Andrew
# SPDX-License-Identifier: GPL-3.0-or-later
"""SHMIP Suite B: per-case N(x) with width min-max spread and A5 reference.

Usage: cd shmip_B && python plot_shmip.py [--cases B1 B3 B5]
Output: outputs/shmip_B.png
"""

import sys
import os
import argparse

_argv = sys.argv[:]
sys.argv = sys.argv[:1]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import firedrake as fd

CHK_DIR   = "outputs/checkpoints"
CHK_DIR_A = os.path.join("..", "shmip_A", "outputs", "checkpoints")
OUTFILE   = "outputs/shmip_B.png"

Lx        = 100e3
NX        = 141
NX_A      = 200

B_MAP     = {"B1": 1, "B2": 10, "B3": 20, "B4": 50, "B5": 100}
CASES_ALL = ["B1", "B2", "B3", "B4", "B5"]

COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

DPI = 200


def width_stats_Nx(N_vals, x_coords, nbins, lx=Lx):
    """Per-bin mean, min, max of N across the y-direction."""
    bins = np.linspace(0.0, lx, nbins + 1)
    idx  = np.digitize(x_coords, bins) - 1
    by_bin = [[] for _ in range(nbins)]
    for i, v in zip(idx, N_vals):
        if 0 <= i < nbins:
            by_bin[i].append(v)
    xc   = 0.5 * (bins[:-1] + bins[1:])
    mean = np.array([np.mean(b) if b else np.nan for b in by_bin])
    lo   = np.array([np.min(b)  if b else np.nan for b in by_bin])
    hi   = np.array([np.max(b)  if b else np.nan for b in by_bin])
    return xc, mean, lo, hi


def load_checkpoint_N(path):
    """Load N (CG1) from a Firedrake checkpoint."""
    with fd.CheckpointFile(path, "r") as f:
        mesh = f.load_mesh()
        N    = f.load_function(mesh, "N")
    x_all = mesh.coordinates.dat.data_ro[:, 0].copy()
    N_all = N.dat.data_ro.copy()
    return N_all, x_all


ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="*", default=None)
args = ap.parse_args(_argv[1:])
cases = args.cases if args.cases else CASES_ALL

os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

data = {}
for tag in cases:
    path = os.path.join(CHK_DIR, f"{tag}.h5")
    if not os.path.exists(path):
        print(f"  skip {tag}: {path} not found")
        continue
    print(f"  loading {tag} …", flush=True)
    N_all, x_all = load_checkpoint_N(path)
    xc, mean, lo, hi = width_stats_Nx(N_all, x_all, nbins=NX)
    data[tag] = (xc, mean, lo, hi)
    print(f"    N_mean max = {np.nanmax(mean)/1e6:.3f} MPa  "
          f"spread max = {np.nanmax(hi - lo)/1e6:.3f} MPa", flush=True)

if not data:
    sys.exit("No B checkpoints found — run run_shmip_B.py first.")

a5_xc, a5_mean = None, None
path_a5 = os.path.join(CHK_DIR_A, "A5.h5")
if os.path.exists(path_a5):
    print("  loading A5 reference …", flush=True)
    N_all, x_all = load_checkpoint_N(path_a5)
    xc_a5, m_a5, _, _ = width_stats_Nx(N_all, x_all, nbins=NX_A)
    a5_xc, a5_mean = xc_a5, m_a5
else:
    print(f"  A5 reference not found at {path_a5}", flush=True)

n = len(cases)
fig, axes = plt.subplots(n, 1,
                          figsize=(8, 3.0 * n),
                          sharex=True,
                          constrained_layout=True)
if n == 1:
    axes = [axes]

for idx, tag in enumerate(cases):
    ax = axes[idx]
    if tag not in data:
        ax.set_visible(False)
        continue

    color = COLORS[idx % len(COLORS)]
    xc, mean, lo, hi = data[tag]

    ax.fill_between(xc / 1e3, lo / 1e6, hi / 1e6,
                    color=color, alpha=0.18, linewidth=0,
                    label="width min-max")
    ax.plot(xc / 1e3, lo / 1e6, color=color, lw=0.8, ls="--", alpha=0.55)
    ax.plot(xc / 1e3, hi / 1e6, color=color, lw=0.8, ls="--", alpha=0.55)

    ax.plot(xc / 1e3, mean / 1e6,
            color=color, lw=2.0,
            label=f"{tag}  ({B_MAP[tag]} moulin{'s' if B_MAP[tag] > 1 else ''})")

    if a5_xc is not None:
        ax.plot(a5_xc / 1e3, a5_mean / 1e6,
                color="black", lw=1.5, ls="--", zorder=10,
                label="A5 (uniform distributed)")

    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22)
    ax.set_ylabel("N  (MPa)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)

    panel_letter = chr(ord("a") + idx)
    moulin_str = f"{B_MAP[tag]} moulin{'s' if B_MAP[tag] > 1 else ''}"
    ax.set_title(f"({panel_letter})  {tag}  —  {moulin_str}",
                 fontsize=10, loc="left", fontweight="bold")

axes[-1].set_xlabel("Along-flow distance  x  (km)", fontsize=11)

fig.suptitle(
    "SHMIP Suite B — Steady-state effective pressure  N(x)\n"
    "Solid: width-averaged; shaded band & dashed: width min-max; "
    "black dashed: A5 reference",
    fontsize=9,
)

fig.savefig(OUTFILE, dpi=DPI)
print(f"\nSaved → {OUTFILE}")
