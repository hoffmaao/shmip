#!/usr/bin/env python3
"""
Regenerate B_Nx.png from saved checkpoints.

Fixes the aliasing in the original B_Nx.png (produced by run_shmip_B.py with
nbins=200) by using nbins=141 = nx, which matches the mesh node spacing and
prevents empty bins / sawtooth oscillation.

Also updates the CSV files to the corrected bin count.

Usage:  python make_B_Nx.py [--cases B1 B2 ...]
"""
import sys
import os
import argparse

# Prevent PETSc / Firedrake from consuming command-line flags
_argv = sys.argv[:]
sys.argv = sys.argv[:1]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import firedrake as fd

# ── paths & parameters ────────────────────────────────────────────────────────
CHK_DIR = "outputs/checkpoints"
CSV_DIR = "outputs/csv"
FIG     = "outputs/B_Nx.png"

Lx = 100e3          # domain length [m]
NX = 141            # mesh nx — nbins must be ≤ nx to avoid x-aliasing

B_MAP    = {"B1": 1, "B2": 10, "B3": 20, "B4": 50, "B5": 100}
CASES_ALL = ["B1", "B2", "B3", "B4", "B5"]

# Reference Suite A case (same total volumetric flux as Suite B)
CHK_DIR_A  = os.path.join("..", "shmip_A", "outputs", "checkpoints")
NX_A       = 200    # Suite A uses nx=200

PROFILE_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]
LS_CYCLE       = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


# ── helpers ───────────────────────────────────────────────────────────────────
def width_averaged_Nx(N_vals, x_coords, nbins, lx=Lx):
    """Width-average N over x-bins (Pa → Pa).  Returns (x_centres_m, N_mean_Pa)."""
    bins  = np.linspace(0.0, lx, nbins + 1)
    idx   = np.digitize(x_coords, bins) - 1
    Nx    = np.zeros(nbins)
    count = np.zeros(nbins, dtype=int)
    for i, v in zip(idx, N_vals):
        if 0 <= i < nbins:
            Nx[i] += v
            count[i] += 1
    mask = count > 0
    Nx[mask] /= count[mask]
    return 0.5 * (bins[:-1] + bins[1:]), Nx


def load_Nx(path, nbins, lx=Lx):
    with fd.CheckpointFile(path, "r") as f:
        mesh = f.load_mesh()
        N    = f.load_function(mesh, "N")
    x, Nx = width_averaged_Nx(
        N.dat.data_ro.copy(),
        mesh.coordinates.dat.data_ro[:, 0].copy(),
        nbins=nbins, lx=lx,
    )
    return x, Nx


# ── parse args ────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="*", default=None,
                help="Subset of cases, e.g. B1 B3 B5")
args = ap.parse_args(_argv[1:])
cases = args.cases if args.cases else CASES_ALL

os.makedirs(CSV_DIR, exist_ok=True)

# ── load B cases ──────────────────────────────────────────────────────────────
curves = {}
all_x  = None

for tag in cases:
    path = os.path.join(CHK_DIR, f"{tag}.h5")
    if not os.path.exists(path):
        print(f"  skip {tag}: {path} not found")
        continue
    print(f"  loading {tag} …", flush=True)
    x, Nx = load_Nx(path, nbins=NX)
    if all_x is None:
        all_x = x
    curves[tag] = Nx
    csv = os.path.join(CSV_DIR, f"{tag}_Nx.csv")
    np.savetxt(csv, np.c_[x, Nx], delimiter=",", header="x,N", comments="")
    print(f"    saved → {csv}")

if not curves:
    sys.exit("No B checkpoints found — run run_shmip_B.py first.")

# ── optional A5 reference ─────────────────────────────────────────────────────
a5_x, a5_Nx = None, None
path_a5 = os.path.join(CHK_DIR_A, "A5.h5")
if os.path.exists(path_a5):
    print("  loading A5 reference …", flush=True)
    a5_x, a5_Nx = load_Nx(path_a5, nbins=NX_A, lx=Lx)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

for i, tag in enumerate(cases):
    if tag not in curves:
        continue
    ax.plot(all_x / 1e3, curves[tag] / 1e6,
            color=PROFILE_COLORS[i],
            lw=1.8, ls=LS_CYCLE[i % len(LS_CYCLE)],
            label=f"{tag}  ({B_MAP[tag]} moulins)")

if a5_Nx is not None:
    ax.plot(a5_x / 1e3, a5_Nx / 1e6,
            color="black", lw=2.0, ls="--", zorder=10,
            label="A5 reference (uniform, same total flux)")

ax.set_xlabel("Along-flow distance  x  (km)", fontsize=11)
ax.set_ylabel("Width-averaged effective pressure  N  (MPa)", fontsize=11)
ax.set_title("SHMIP Suite B: N(x) at steady state", fontsize=12)
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.25)
ax.legend(ncol=2, fontsize=9, framealpha=0.85)

fig.tight_layout()
fig.savefig(FIG, dpi=200)
print(f"\nSaved → {FIG}")
