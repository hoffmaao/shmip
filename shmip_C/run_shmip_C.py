#!/usr/bin/env python3
"""SHMIP Suite C: diurnal oscillating moulin recharge.

Usage: python run_shmip_C.py [--cases C1 C3] [--spinup-days 100]
Output: outputs/{checkpoints,csv,C_curves.npz}
"""

import os
import argparse
import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt

from hydropack.models.subglacialhydrology import SubglacialHydrologyModel
from hydropack.constants import ice_density, water_density, gravity

Lx, Ly = 100e3, 20e3
nx, ny  = 141, 28

dt            = 600.0
dt_ramp       = 300.0

spinup_days        = 100
save_every_minutes = 60

B5_CHECKPOINT  = "../shmip_B/outputs/checkpoints/B5.h5"
B5_MOULIN_CSV  = "../shmip_B/moulin_positions/B5_M.csv"

OUTDIR  = "outputs"
CSV_DIR = os.path.join(OUTDIR, "csv")
CHK_DIR = os.path.join(OUTDIR, "checkpoints")

sec_per_hour = 3600.0
sec_per_day  = 24.0 * sec_per_hour

A1_RATE = 7.93e-11

MOULIN_SIGMA = 1000.0

C_CASES = {
    "C1": {"ra": 0.25},
    "C2": {"ra": 0.50},
    "C3": {"ra": 1.00},
    "C4": {"ra": 2.00},
}
CASES_ALL = ["C1", "C2", "C3", "C4"]


def build_mesh():
    """Load mesh from B5 checkpoint if available, otherwise build fresh."""
    if os.path.exists(B5_CHECKPOINT):
        try:
            with fd.CheckpointFile(B5_CHECKPOINT, "r") as chk:
                mesh = chk.load_mesh()
            print(f"  Mesh loaded from {B5_CHECKPOINT}", flush=True)
            return mesh
        except Exception as e:
            print(f"  Could not load mesh from B5 ({e}); building fresh.", flush=True)
    return fd.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)


def make_model_inputs(mesh):
    """Build Firedrake fields for SubglacialHydrologyModel."""
    U   = fd.FunctionSpace(mesh, "CG", 1)
    CR  = fd.FunctionSpace(mesh, "CR", 1)
    x, y = fd.SpatialCoordinate(mesh)

    surf  = fd.interpolate(6*(fd.sqrt(x + 5000) - fd.sqrt(5000.0)) + 1, U)
    B     = fd.interpolate(fd.Constant(0.0), U)
    H     = surf - B

    u_b   = fd.interpolate(fd.Constant(1e-6), U)

    p_i   = fd.interpolate(fd.Constant(ice_density * gravity) * H, U)
    phi_m = fd.interpolate(fd.Constant(water_density * gravity) * B, U)
    phi_0 = fd.interpolate(p_i + phi_m, U)

    bc = fd.DirichletBC(U, phi_m, 1)

    h_init   = fd.interpolate(fd.Constant(0.01), U)
    S_init   = fd.interpolate(fd.Constant(0.001), CR)
    phi_init = fd.Function(U).interpolate(phi_0)

    m = fd.Function(U); m.assign(0.0)

    return dict(
        thickness = H,
        bed       = B,
        sliding_speed = u_b,
        melt_rate = m,
        h_init    = h_init,
        S_init    = S_init,
        phi_init  = phi_init,
        phi_m     = phi_m,
        p_i       = p_i,
        phi_0     = phi_0,
        dirichlet_bcs = [bc],
        englacial_void_ratio = 1e-4,
        out_dir   = OUTDIR,
    )


def load_moulin_positions(csv_path):
    """Load SHMIP moulin positions. Returns (pts, q_total)."""
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    pts     = [(float(row[1]), float(row[2])) for row in data]
    q_total = float(np.sum(data[:, 3]))
    return pts, q_total


def gaussian_blobs(mesh, points, sigma=MOULIN_SIGMA):
    """Normalised sum of Gaussians (integrates to 1 over the domain)."""
    U    = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    g    = sum(fd.exp(-((x - xi)**2 + (y - yi)**2) / (2*sigma**2))
               for (xi, yi) in points)
    mass = fd.assemble(g * fd.dx)
    return fd.interpolate(g / (mass + 1e-30), U)


def try_load_b5(model):
    """Load h, S, phi from the B5 steady-state checkpoint."""
    if not os.path.exists(B5_CHECKPOINT):
        print(f"  WARNING: {B5_CHECKPOINT} not found — starting cold.", flush=True)
        return False
    try:
        with fd.CheckpointFile(B5_CHECKPOINT, "r") as chk:
            saved_mesh = chk.load_mesh()
            for name in ("h", "S", "phi"):
                loaded = chk.load_function(saved_mesh, name)
                getattr(model, name).dat.data[:] = loaded.dat.data_ro[:]
        model.phi_prev.assign(model.phi)
        model.update_phi()
        model.update_h_cr()
        model.update_S_alpha()
        h_max = float(model.h.dat.data_ro.max())
        S_max = float(model.S.dat.data_ro.max())
        print(f"  Warm-started from B5: h_max={h_max:.4f} m, S_max={S_max:.4f} m²",
              flush=True)
        return True
    except Exception as e:
        print(f"  Could not load B5 state ({e}) — starting cold.", flush=True)
        return False


def make_set_m(model, blob, Q_mean_moulin, ra):
    """Return set_m(t) which updates model.m for model time t [s]."""
    bg = fd.Constant(A1_RATE)

    def set_m(t):
        s    = float(np.sin(2.0 * np.pi * t / sec_per_day))
        fac  = max(0.0, 1.0 - ra * s)
        model.m.interpolate(bg + fd.Constant(Q_mean_moulin * fac) * blob)

    return set_m


def width_averaged_Nx(model, nbins=None):
    coords = model.mesh.coordinates.dat.data_ro
    x      = coords[:, 0]
    Nvals  = model.N.dat.data_ro
    if nbins is None:
        nbins = nx
    bins   = np.linspace(0.0, Lx, nbins + 1)
    idx    = np.digitize(x, bins) - 1
    Nx     = np.zeros(nbins)
    count  = np.zeros(nbins, dtype=int)
    for i, val in zip(idx, Nvals):
        if 0 <= i < nbins:
            Nx[i] += val; count[i] += 1
    mask      = count > 0
    Nx[mask] /= count[mask]
    xc        = 0.5 * (bins[:-1] + bins[1:])
    return xc, Nx


def _compute_Q_in(model):
    """Total volumetric recharge [m^3/s]."""
    try:
        return max(0.0, float(fd.assemble(model.m * fd.dx)))
    except Exception:
        return 0.0


def _compute_Q_ch(model):
    """Sum of |Q_ch| over finite CR1 channel edges [m^3/s]."""
    try:
        vals   = model.Q_ch.dat.data_ro.copy()
        finite = np.isfinite(vals)
        return float(np.sum(np.abs(vals[finite]))) if finite.any() else 0.0
    except Exception:
        return 0.0


def save_checkpoint(model, tag):
    os.makedirs(CHK_DIR, exist_ok=True)
    path = os.path.join(CHK_DIR, f"{tag}.h5")
    with fd.CheckpointFile(path, "w") as chk:
        chk.save_mesh(model.mesh)
        for name in ("h", "S", "phi", "pfo", "N", "N_cr", "h_cr",
                     "S_alpha", "p_w", "m", "q_s", "q_s_mag", "Q_ch"):
            if hasattr(model, name):
                chk.save_function(getattr(model, name), name=name)
    return path


def main(cases=None):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    if not cases:
        cases = CASES_ALL

    mesh = build_mesh()

    pts, Q_mean_moulin = load_moulin_positions(B5_MOULIN_CSV)
    blob = gaussian_blobs(mesh, pts)
    print(f"  Loaded {len(pts)} moulins from B5_M.csv,"
          f" Q_mean={Q_mean_moulin:.1f} m³/s", flush=True)

    spin_steps = int(spinup_days * sec_per_day / dt)
    day_steps  = int(sec_per_day / dt)
    stride     = max(1, int(save_every_minutes * 60.0 / dt))
    print(f"  spinup_days={spinup_days}, spin_steps={spin_steps}, "
          f"day_steps={day_steps}, stride={stride}", flush=True)

    snap_hours = [0, 6, 12, 18]
    nbins      = nx
    xc_ref     = None
    all_curves = {}

    for tag in cases:
        ra = C_CASES[tag]["ra"]
        print(f"\n=== {tag}: ra={ra} ===", flush=True)

        model_inputs = make_model_inputs(mesh)
        model        = SubglacialHydrologyModel(mesh, **model_inputs)

        loaded = try_load_b5(model)
        if not loaded:
            print("  Cold start: A1 background spin-up (300 steps) …", flush=True)
            model.m.interpolate(fd.Constant(A1_RATE))
            for _ in range(300):
                model.step(dt_ramp)
            model.update_phi()
            print("  Cold start complete.", flush=True)

        set_m = make_set_m(model, blob, Q_mean_moulin, ra)

        print(f"  Ramping diurnal amplitude (0 → ra={ra}) …", flush=True)
        for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
            ra_frac = frac * ra
            set_m_ramp = make_set_m(model, blob, Q_mean_moulin, ra_frac)
            for k in range(day_steps):
                set_m_ramp(k * dt)
                model.step(dt_ramp)
            model.update_phi()
            print(f"    frac={frac:.2f} (ra_eff={ra_frac:.2f}) complete", flush=True)

        print(f"  Phase 3: {spinup_days} diurnal spin-up cycles …", flush=True)
        phi_snap = fd.Function(model.U).interpolate(model.phi)
        n_fail   = 0
        for k in range(1, spin_steps + 1):
            t_k = k * dt
            set_m(t_k)
            try:
                model.step(dt)
                if k % day_steps == 0:
                    model.update_phi()
                    phi_snap.assign(model.phi)
                    day_num = k // day_steps
                    S_max   = float(model.S.dat.data_ro.max())
                    print(f"    spin-up day {day_num}/{spinup_days},"
                          f"  S_max={S_max:.4f} m²", flush=True)
            except Exception as e:
                n_fail += 1
                if n_fail <= 30:
                    print(f"    [spin step {k}] Newton failure ({type(e).__name__}),"
                          f" skipping ({n_fail} total)", flush=True)
                    model.phi.assign(phi_snap)
                    model.update_phi()
                else:
                    print(f"    Too many Newton failures ({n_fail}), aborting spin-up.",
                          flush=True)
                    break

        model.update_phi()
        model.compute_flux_fields()
        print("  Phase 3 complete.", flush=True)

        print("  Phase 4: recording final diurnal day …", flush=True)

        max_nsaves  = day_steps // stride + 2
        Nx_series   = np.zeros((max_nsaves, nbins))
        Q_in_series = np.zeros(max_nsaves)
        Q_ch_series = np.zeros(max_nsaves)
        t_series    = np.zeros(max_nsaves)
        ksave       = 0

        model.update_phi()
        model.compute_flux_fields()
        xc, Nx = width_averaged_Nx(model, nbins=nbins)
        if xc_ref is None:
            xc_ref = xc
        Nx_series[ksave, :] = Nx
        Q_in_series[ksave]  = _compute_Q_in(model)
        Q_ch_series[ksave]  = _compute_Q_ch(model)
        t_series[ksave]     = 0.0
        ksave += 1
        cpath = save_checkpoint(model, f"{tag}_00h")
        print(f"    saved {tag}_00h → {cpath}", flush=True)

        snaps_pending = {hh: f"{tag}_{hh:02d}h" for hh in snap_hours if hh > 0}
        phi_snap.assign(model.phi)

        for j in range(1, day_steps + 1):
            t_abs    = (spin_steps + j) * dt
            t_in_day = j * dt

            set_m(t_abs)
            try:
                model.step(dt)
            except Exception as e:
                print(f"    [record step {j}] Newton failure ({type(e).__name__}),"
                      f" skipping", flush=True)
                model.phi.assign(phi_snap)
                model.update_phi()
                continue

            model.update_phi()
            phi_snap.assign(model.phi)

            if j % stride == 0:
                model.compute_flux_fields()
                xc, Nx = width_averaged_Nx(model, nbins=nbins)
                if ksave < max_nsaves:
                    Nx_series[ksave, :] = Nx
                    Q_in_series[ksave]  = _compute_Q_in(model)
                    Q_ch_series[ksave]  = _compute_Q_ch(model)
                    t_series[ksave]     = t_in_day
                    ksave += 1

            for hh in list(snaps_pending):
                if t_in_day >= hh * sec_per_hour:
                    if j % stride != 0:
                        model.compute_flux_fields()
                    cpath = save_checkpoint(model, snaps_pending[hh])
                    print(f"    saved {snaps_pending[hh]} → {cpath}", flush=True)
                    del snaps_pending[hh]

        Nx_series   = Nx_series[:ksave, :]
        Q_in_series = Q_in_series[:ksave]
        Q_ch_series = Q_ch_series[:ksave]
        t_series    = t_series[:ksave]
        all_curves[tag] = (Nx_series.copy(), t_series.copy())

        csv_path = os.path.join(CSV_DIR, f"{tag}_Nx_diurnal.csv")
        header   = "t_seconds," + ",".join([f"x={xc_ref[i]/1000:.2f}km"
                                            for i in range(nbins)])
        np.savetxt(csv_path, np.c_[t_series, Nx_series],
                   delimiter=",", header=header, comments="")
        print(f"  {tag}: N(x,t) → {csv_path}")

        q_csv_path = os.path.join(CSV_DIR, f"{tag}_Q_diurnal.csv")
        np.savetxt(q_csv_path,
                   np.c_[t_series, Q_in_series, Q_ch_series],
                   delimiter=",",
                   header="t_seconds,Q_in_m3s,Q_ch_m3s",
                   comments="")
        print(f"  {tag}: Q(t)   → {q_csv_path}")

    npz_data = {"x": xc_ref} if xc_ref is not None else {}
    for tag, (Nx_series, t_series) in all_curves.items():
        npz_data[f"{tag}_Nx"] = Nx_series
        npz_data[f"{tag}_t"]  = t_series
    if npz_data:
        np.savez(os.path.join(OUTDIR, "C_curves.npz"), **npz_data)
        print(f"Saved curves → {os.path.join(OUTDIR, 'C_curves.npz')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Subset of cases, e.g. C1 C2 C3 C4")
    ap.add_argument("--spinup-days", type=int, default=spinup_days,
                    help=f"Diurnal cycles before recording (default {spinup_days})")
    args = ap.parse_args()
    if args.spinup_days != spinup_days:
        spinup_days = args.spinup_days
    main(args.cases)
