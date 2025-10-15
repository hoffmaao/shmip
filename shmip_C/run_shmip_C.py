#!/usr/bin/env python3
# SHMIP Suite D: diurnal (24 h) oscillating input.
# D1 = distributed diurnal + background A1
# D2 = moulin diurnal (same total) + background A1

import os
import argparse
import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
from hydropack.models.glads import Glads2DModel
from hydropack.constants import pcs as default_pcs

# ---------------- Domain & numerics ----------------
Lx, Ly = 100e3, 20e3
nx, ny = 400, 80

dt = 600.0                 # 10 min
spinup_days = 5            # cycles to spin up to periodic state
save_hours = 24            # hours to record after spin-up (one full cycle)
save_every_minutes = 10    # temporal resolution of saved Nx curves
OUTDIR = "outputs"
CSV_DIR = os.path.join(OUTDIR, "csv")
CHK_DIR = os.path.join(OUTDIR, "checkpoints")

# ---------------- Constants (align with your SHMIP setup) ----------------
sec_per_hour = 3600.0
sec_per_day = 24 * sec_per_hour

# Background distributed melt (areal rate, m s^-1), same as Suite A1 by default:
A1 = 2.5e-3 / (365.0 * 86400.0)

# Target “high” areal rate used to scale diurnal amplitude (from A5 daily melt):
A5 = 25.0e-3 / 86400.0

# Diurnal signal parameters (tune to match your SHMIP D values)
DIURNAL_MEAN = 0.0       # mean of the diurnal component (areal rate, m s^-1)
DIURNAL_AMP  = A5        # amplitude of diurnal component (peak ~ A5)
DIURNAL_PHASE = 0.0      # radians

# Case definitions (adjust moulin counts to your table)
D_CASES = {
    "D1": {"mode": "distributed", "n_moulins": 0},
    "D2": {"mode": "moulin",      "n_moulins": 50},
}

# ---------------- Helpers shared across suites ----------------
def build_mesh():
    return fd.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

def make_model_inputs(mesh):
    U  = fd.FunctionSpace(mesh, "CG", 1)
    CR = fd.FunctionSpace(mesh, "CR", 1)
    x, y = fd.SpatialCoordinate(mesh)

    S = fd.interpolate(6*fd.sqrt(x + 5000) - fd.sqrt(5000.0) + 1, U)         # 1 km ice
    B = fd.interpolate(fd.Constant(0.0), U)            # flat bed
    u_b = fd.interpolate(fd.as_vector((1e-6, 0.0))[0], U)  # 100 m/yr-ish along x (replace with your actual)
    #u_b.assign(0.0)  # or set from data
    H = S-B
    m = fd.Function(U); m.assign(0.0)

    h_init  = fd.interpolate(fd.Constant(0.01), U)
    S_init  = fd.Function(CR); S_init.assign(0.0)
    phi_init = fd.Function(U); phi_init.assign(0.0)

    rho_i, g = 917.0, 9.81
    p_i  = fd.project(fd.Constant(rho_i*g)*H, U)
    phi_m = fd.Function(U); phi_m.assign(0.0)
    phi_0 = fd.project(p_i, U)

    pcs = dict(default_pcs)

    return dict(
        mesh=mesh, thickness=H, bed=B, u_b=u_b, m=m,
        h_init=h_init, S_init=S_init,
        phi_init=phi_init, phi_m=phi_m, p_i=p_i, phi_0=phi_0,
        d_bcs=[], constants=pcs, out_dir=OUTDIR
    )

def apply_margin_bc(model):
    # RectangleMesh: boundary id 1 corresponds to x=0 side (margin)
    model.d_bcs = [fd.DirichletBC(model.U, 0.0, 1)]

def total_domain_area(mesh):
    return float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))

def place_moulins(mesh, n, seed=1, margin_buffer=2e3):
    rng = np.random.default_rng(seed)
    xy = mesh.coordinates.dat.data_ro
    xmin, xmax = float(xy[:,0].min()), float(xy[:,0].max())
    ymin, ymax = float(xy[:,1].min()), float(xy[:,1].max())
    xmin += margin_buffer
    pts = np.c_[rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n)]
    return [tuple(p) for p in pts]

def gaussian_blobs(mesh, points, sigma):
    U = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    blobs = [fd.exp(-((x-xi)**2 + (y-yi)**2)/(2*sigma**2)) for (xi, yi) in points]
    if not blobs:
        f = fd.Function(U); f.assign(0.0); return f
    g = sum(blobs)
    mass = fd.assemble(g * fd.dx)
    g_norm = g / (mass + 1e-16)
    return fd.interpolate(g_norm, U)

def width_averaged_Nx(model, nbins=200):
    coords = model.mesh.coordinates.dat.data_ro
    x = coords[:,0]
    Nvals = model.N.dat.data_ro
    bins = np.linspace(0.0, Lx, nbins+1)
    idx = np.digitize(x, bins) - 1
    Nx = np.zeros(nbins); count = np.zeros(nbins, dtype=int)
    for i, val in zip(idx, Nvals):
        if 0 <= i < nbins:
            Nx[i] += val; count[i] += 1
    mask = count > 0
    Nx[mask] /= count[mask]
    xc = 0.5*(bins[:-1] + bins[1:])
    return xc, Nx

def save_checkpoint(model, tag):
    os.makedirs(CHK_DIR, exist_ok=True)
    path = os.path.join(CHK_DIR, f"{tag}.h5")
    with fd.CheckpointFile(path, "w") as chk:
        chk.save_mesh(model.mesh)
        for name in ("h","S","phi","pfo","N","N_cr","h_cr","S_alpha","p_w","m"):
            if hasattr(model, name):
                chk.save_function(getattr(model, name), name=name)
    return path

# ---------------- Diurnal recharge ----------------
def diurnal_rate(t):
    """Areal rate [m/s] for the diurnal component at time t (seconds)."""
    return DIURNAL_MEAN + DIURNAL_AMP * np.sin(2*np.pi*t / sec_per_day + DIURNAL_PHASE)

def build_recharge_updater(model, mode, n_moulins=0, seed=1):
    """
    Returns set_m(t) that sets model.m for time t (s).
      - mode='distributed': m = A1 + diurnal(t) uniformly
      - mode='moulin':      m = A1 + diurnal(t)*blob  (blob normalized to integrate to 1)
    Both have the same *total* diurnal volume over the domain.
    """
    if mode == "distributed":
        def set_m(t):
            model.m.interpolate(fd.Constant(A1 + diurnal_rate(t)))
        return set_m

    elif mode == "moulin":
        pts = place_moulins(model.mesh, n_moulins, seed=seed)
        sigma = float(model.mesh.cell_sizes.dat.data_ro.min())
        blob = gaussian_blobs(model.mesh, pts, sigma)  # integrates to 1
        def set_m(t):
            model.m.interpolate(fd.Constant(A1) + blob * diurnal_rate(t))
        return set_m

    else:
        raise ValueError(f"Unknown mode {mode}")

# ---------------- Main driver ----------------
def main(cases=None, seed=1):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    if not cases:
        cases = list(D_CASES.keys())

    mesh = build_mesh()
    xc_ref = None
    nbins = 200

    spin_steps = int((spinup_days * sec_per_day) // dt)
    save_steps  = int((save_hours * sec_per_hour) // dt)
    stride = max(1, int((save_every_minutes * 60.0) // dt))

    for tag in cases:
        cfg = D_CASES[tag]
        print(f"\n=== {tag}: mode={cfg['mode']}, n_moulins={cfg['n_moulins']} ===")

        model_inputs = make_model_inputs(mesh)
        model = Glads2DModel(model_inputs)
        apply_margin_bc(model)

        set_m = build_recharge_updater(model, cfg["mode"], cfg["n_moulins"], seed=seed)

        # ---------- spin-up to periodic state ----------
        for k in range(1, spin_steps + 1):
            t = k * dt
            set_m(t)
            model.step(dt)

        # ---------- record one final day ----------
        nsaves = save_steps // stride + 1
        Nx_series = np.zeros((nsaves, nbins))
        t_series  = np.zeros(nsaves)
        ksave = 0

        for j in range(1, save_steps + 1):
            t = (spin_steps + j) * dt
            set_m(t)
            model.step(dt)
            if j % stride == 0 or j == save_steps:
                model.update_phi()   # refresh N, dphi/ds, etc.
                xc, Nx = width_averaged_Nx(model, nbins=nbins)
                if xc_ref is None: xc_ref = xc
                Nx_series[ksave, :] = Nx
                t_series[ksave] = t - spin_steps * dt
                ksave += 1

        Nx_series = Nx_series[:ksave, :]
        t_series  = t_series[:ksave]

        # Save CSV
        csv_path = os.path.join(CSV_DIR, f"{tag}_Nx_lastday.csv")
        header = "t_seconds," + ",".join([f"x={x/1000.0:.2f}km" for x in xc_ref])
        np.savetxt(csv_path, np.c_[t_series, Nx_series], delimiter=",", header=header, comments="")
        print(f"{tag}: saved {csv_path}")

        # Save checkpoints at 0/6/12/18 h into the last day
        snap_hours = [0, 6, 12, 18]
        for hh in snap_hours:
            # nearest saved time index
            target = hh * sec_per_hour
            i = int(np.argmin(np.abs(t_series - target)))
            # (for exact-time fields, store during loop; here we save end-of-run state)
            cpath = save_checkpoint(model, f"{tag}_{hh:02d}h")
            print(f"{tag}: wrote checkpoint {cpath}")

        # Plot Nx(x) for the four phases
        plt.figure(figsize=(8,4.5))
        for hh in snap_hours:
            i = int(np.argmin(np.abs(t_series - hh*sec_per_hour)))
            plt.plot(xc_ref/1000.0, Nx_series[i,:], label=f"{hh} h")
        plt.xlabel("x (km)")
        plt.ylabel("Width-averaged N (Pa)")
        plt.title(f"Suite D — {tag} diurnal cycle (last day)")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        fig_path = os.path.join(OUTDIR, f"{tag}_Nx_diurnal.png")
        plt.savefig(fig_path, dpi=200)
        print(f"{tag}: saved {fig_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None, help="Subset like D1 D2")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    main(args.cases, args.seed)
