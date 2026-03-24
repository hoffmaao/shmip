#!/usr/bin/env python3
"""Re-export N(x) CSVs from existing checkpoints with correct binning."""
import sys; sys.argv = sys.argv[:1]
import os, numpy as np
import firedrake as fd

Lx = 100e3
nx = 115  # must match the mesh that generated the checkpoints
CHK_DIR = "outputs/checkpoints"
CSV_DIR = "outputs/csv"

for tag in ["A1","A2","A3","A4","A5","A6"]:
    path = os.path.join(CHK_DIR, f"{tag}.h5")
    if not os.path.exists(path):
        continue
    with fd.CheckpointFile(path, "r") as chk:
        mesh = chk.load_mesh()
        N = chk.load_function(mesh, "N")
    coords = mesh.coordinates.dat.data_ro
    x = coords[:, 0]
    Nvals = N.dat.data_ro
    nbins = nx
    bins = np.linspace(0.0, Lx, nbins+1)
    idx = np.digitize(x, bins) - 1
    Nx = np.zeros(nbins); count = np.zeros(nbins, dtype=int)
    for i, val in zip(idx, Nvals):
        if 0 <= i < nbins:
            Nx[i] += val; count[i] += 1
    mask = count > 0
    Nx[mask] /= count[mask]
    xc = 0.5*(bins[:-1]+bins[1:])
    csv_path = os.path.join(CSV_DIR, f"{tag}_Nx.csv")
    np.savetxt(csv_path, np.c_[xc, Nx], delimiter=",", header="x,N", comments="")
    print(f"{tag}: N_div={Nx[mask][-1]/1e6:.3f} MPa, N_mean={Nx[mask].mean()/1e6:.3f}, neg={np.sum(Nx<0)}")
