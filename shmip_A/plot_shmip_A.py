import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LogNorm
import firedrake as fd
import numpy as np

from plot_cr_edges import (
    triangulation_from_mesh,
    _edge_segments_and_indices,
    _cr_values_in_dmplex_order,
)

# --------- configure your 6 cases here ----------
chks = [
    "A1.h5",
    "A2.h5",
    "A3.h5",
    "A4.h5",
    "A5.h5",
    "A6.h5",
]
titles = ["A1","A2","A3","A4","A5","A6"]

# ------------------------------------------------

def load_panel_data(chkpath):
    with fd.CheckpointFile(chkpath, "r") as f:
        mesh = f.load_mesh()
        pfo  = f.load_function(mesh, "pfo")
        S    = f.load_function(mesh, "S")
    tri = triangulation_from_mesh(mesh)
    z_cg = pfo.dat.data_ro  # CG1 nodal values
    segments, _, _ = _edge_segments_and_indices(mesh)
    S_vals = _cr_values_in_dmplex_order(S)  # CR facet order to match 'segments'
    return dict(mesh=mesh, tri=tri, z=z_cg, segments=segments, S_vals=S_vals)

# Load everything and compute global ranges for consistent colorbars
panels = [load_panel_data("outputs/checkpoints/"+p) for p in chks]
# pfo range (clip tiny negatives just in case)
pfo_all = np.concatenate([np.asarray(p["z"]) for p in panels])
pfo_vmin = float(np.nanmin(pfo_all))
pfo_vmax = float(np.nanmax(pfo_all))

# S range (often very skewed; log is useful — switch to linear if you prefer)
S_all = np.concatenate([np.asarray(p["S_vals"]) for p in panels])
S_all_pos = S_all[S_all > 0]
if S_all_pos.size == 0:
    # all-zero fallback
    S_vmin, S_vmax = 1.0, 1.0
    S_norm = Normalize(vmin=S_vmin, vmax=S_vmax)
    use_log_S = False
else:
    S_vmin = np.percentile(S_all_pos, 1)    # robust floor
    S_vmax = np.percentile(S_all_pos, 99.5) # robust ceiling
    S_vmin = max(S_vmin, 1e-12)
    S_norm = LogNorm(vmin=S_vmin, vmax=S_vmax)
    use_log_S = True

pfo_norm = Normalize(vmin=pfo_vmin, vmax=pfo_vmax)

# Figure: 2x3 grid (6 panels)
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
axes = axes.ravel()

im_handles = []
lc_handles = []

for ax, panel, title in zip(axes, panels, titles):
    tri = panel["tri"]
    z   = panel["z"]
    seg = panel["segments"]
    Sv  = panel["S_vals"]

    # base CG field (pfo)
    im = ax.tripcolor(tri, z, shading="gouraud", norm=pfo_norm, cmap="viridis")
    ax.triplot(tri, color="0.85", linewidth=0.3)

    # CR edges colored by S
    lc = LineCollection(seg, linewidths=0.25, cmap="plasma", norm=S_norm)
    lc.set_array(Sv)
    ax.add_collection(lc)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

    im_handles.append(im)
    lc_handles.append(lc)

# Shared colorbars (use the first mappables, but they share norms)
cax1 = fig.add_axes([0.92, 0.55, 0.015, 0.35])  # right side, top
cb1 = fig.colorbar(im_handles[0], cax=cax1)
cb1.set_label("pfo")

cax2 = fig.add_axes([0.92, 0.08, 0.015, 0.35])  # right side, bottom
cb2 = fig.colorbar(lc_handles[0], cax=cax2)
cb2.set_label("S" + (" (log)" if use_log_S else ""))

plt.savefig("overlay_6panel.png", dpi=200)
print("Saved overlay_6panel.png")
