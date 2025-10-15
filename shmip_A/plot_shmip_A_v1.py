import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import firedrake as fd
import numpy as np

from plot_cr_edges import triangulation_from_mesh, _edge_segments_and_indices, _cr_values_in_dmplex_order

chk = "outputs/checkpoints/A1.h5"
with fd.CheckpointFile(chk, "r") as f:
    mesh = f.load_mesh()
    pfo  = f.load_function(mesh, "pfo")
    S    = f.load_function(mesh, "S")
    h    = f.load_function(mesh, "h")
    N    = f.load_function(mesh, "N")

# base CG field
tri = triangulation_from_mesh(mesh)
z = N.dat.data_ro

fig, ax = plt.subplots(figsize=(12,10), dpi=300, constrained_layout=True)
im = ax.tripcolor(tri, z, shading="gouraud")
ax.triplot(tri, color="0.85", linewidth=0.3)

# CR edges colored by S
segments, _, _ = _edge_segments_and_indices(mesh)
S_vals = _cr_values_in_dmplex_order(S)
lc = LineCollection(segments, linewidths=.25, cmap="plasma")
lc.set_array(S_vals)
ax.add_collection(lc)

ax.set_aspect("equal", adjustable="box")
ax.set_title("N(CG) with S (CR) overlay")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="N")
cb2 = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.08)
cb2.set_label("S")
plt.savefig("overlay.png", dpi=200)
print("Saved overlay.png")