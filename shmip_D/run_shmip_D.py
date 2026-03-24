#!/usr/bin/env python3
"""SHMIP Suite D: seasonal degree-day forcing on a rectangular ice sheet.

Usage: python run_shmip_D.py [--cases D3 D4] [--max-years 15] [--dt 900]
Output: outputs/{checkpoints,csv,D_seasonal_curves.npz}
"""

import os
import argparse
import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt

from hydropack.models.subglacialhydrology import SubglacialHydrologyModel
from hydropack.constants import ice_density, water_density, gravity

Lx, Ly = 100e3, 20e3
nx, ny  = 200, 40

dt           = 1800.0
max_years    = 30
periodic_tol = 1e-2

N_SUBSTEPS = 8
MAX_CONSEC_ABORT = 200

SEC_PER_YEAR  = 365 * 86400
steps_per_year = int(SEC_PER_YEAR / dt)
stride         = int(86400 / dt)

OUTDIR  = "outputs"
CSV_DIR = os.path.join(OUTDIR, "csv")
CHK_DIR = os.path.join(OUTDIR, "checkpoints")

A1_CHK = os.path.join("..", "shmip_A", "outputs", "checkpoints", "A1.h5")

LR          = -0.0075
DDF         = 0.01 / 86400
BASAL_RATE  = 7.93e-11
T_AMPL      = 16.0
T_BASE      = -5.0

D_MAP = {
    "D1": -4.0,
    "D2": -2.0,
    "D3":  0.0,
    "D4": +2.0,
    "D5": +4.0,
}
CASES_ALL = ["D1", "D2", "D3", "D4", "D5"]

SNAP_DAYS   = [0,     91,     182,    274   ]
SNAP_LABELS = ["Jan", "Apr",  "Jul",  "Oct" ]


def build_mesh():
    return fd.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)


def make_model_inputs(mesh):
    """Return (inputs_dict, surf_function)."""
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

    bc    = fd.DirichletBC(U, phi_m, 1)

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
    ), surf


def _air_temp(t_s, DT):
    """Seasonal air temperature [deg C] at time t_s [s]."""
    return -T_AMPL * np.cos(2 * np.pi * t_s / SEC_PER_YEAR) + T_BASE + DT


def _set_melt(model, surf, t_s, DT):
    """Update model.m with degree-day melt at time t_s [s]."""
    T_air = _air_temp(t_s, DT)
    model.m.interpolate(
        fd.max_value(
            fd.Constant(T_air) + fd.Constant(LR) * surf,
            fd.Constant(0.0)
        ) * fd.Constant(DDF) + fd.Constant(BASAL_RATE)
    )


def _compute_Q_sh(model):
    """Terminus sheet outflow [m^3/s]."""
    try:
        return max(0.0, -float(fd.assemble(model.q_s[0] * fd.ds(1))))
    except Exception:
        return 0.0


def _compute_Q_ch(model):
    """Sum of |Q_ch| over finite CR1 channel edges [m^3/s]."""
    try:
        vals = model.Q_ch.dat.data_ro.copy()
        finite = np.isfinite(vals)
        return float(np.sum(np.abs(vals[finite]))) if finite.any() else 0.0
    except Exception:
        return 0.0


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
    mask = count > 0
    Nx[mask] /= count[mask]
    return 0.5 * (bins[:-1] + bins[1:]), Nx


def save_checkpoint(model, tag):
    os.makedirs(CHK_DIR, exist_ok=True)
    path = os.path.join(CHK_DIR, f"{tag}.h5")
    with fd.CheckpointFile(path, "w") as chk:
        chk.save_mesh(model.mesh)
        for name in ("h", "S", "phi", "pfo", "N", "N_cr", "h_cr",
                     "S_alpha", "p_w", "m", "q_s", "q_s_mag", "Q_ch"):
            if hasattr(model, name):
                chk.save_function(getattr(model, name), name=name)
    # Numpy sidecar for robust reloading of CR1 fields
    npz_path = os.path.join(CHK_DIR, f"{tag}_cr.npz")
    npz_fields = {}
    for name in ("phi", "h", "S", "N_cr", "h_cr"):
        if hasattr(model, name):
            npz_fields[name] = getattr(model, name).dat.data_ro.copy()
    if npz_fields:
        np.savez(npz_path, **npz_fields)
    return path


def _load_fields(model, chk_path):
    """Load phi, h, S from checkpoint (sidecar first, HDF5 fallback)."""
    loaded = set()

    npz_path = chk_path.replace(".h5", "_cr.npz")
    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        for name in ("phi", "h", "S"):
            if name in npz:
                getattr(model, name).dat.data[:] = npz[name]
                loaded.add(name)
        if loaded:
            print(f"    [warm-start sidecar] loaded: {sorted(loaded)}", flush=True)

    remaining = [n for n in ("phi", "h", "S") if n not in loaded]
    if remaining:
        m_src = None
        try:
            with fd.CheckpointFile(chk_path, "r") as chk:
                m_src = chk.load_mesh()
                for name in remaining:
                    try:
                        fn_src = chk.load_function(m_src, name)
                        getattr(model, name).dat.data[:] = fn_src.dat.data_ro
                        loaded.add(name)
                    except Exception as exc:
                        print(f"    [warn] HDF5 load of '{name}' failed "
                              f"({type(exc).__name__}); keeping init value",
                              flush=True)
        except Exception as exc:
            print(f"    [warn] HDF5 open failed ({type(exc).__name__}); "
                  "keeping init values for: " + ", ".join(remaining), flush=True)

    model.update_phi()
    print(f"    [warm-start] fields loaded: {sorted(loaded)}", flush=True)


def _try_step(model, surf, t_s, DT, dt, phi_snap, h_snap, S_snap):
    """Advance by dt; on failure, retry with sub-steps. Returns True on success."""
    _set_melt(model, surf, t_s, DT)
    try:
        model.step(dt)
        return True
    except Exception:
        pass

    model.phi.assign(phi_snap)
    model.h.dat.data[:] = h_snap
    model.S.dat.data[:] = S_snap
    model.update_phi()

    sub_dt = dt / N_SUBSTEPS
    for sub in range(N_SUBSTEPS):
        t_sub = t_s + sub * sub_dt
        _set_melt(model, surf, t_sub, DT)
        try:
            model.step(sub_dt)
            model.update_phi()
        except Exception:
            model.phi.assign(phi_snap)
            model.h.dat.data[:] = h_snap
            model.S.dat.data[:] = S_snap
            model.update_phi()
            return False

    return True


def main(cases=None, max_yrs=max_years):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    if not cases:
        cases = CASES_ALL

    nbins  = nx
    xc_ref = None

    if os.path.exists(A1_CHK):
        print(f"Loading mesh from A1 checkpoint: {A1_CHK}", flush=True)
        with fd.CheckpointFile(A1_CHK, "r") as chk:
            mesh = chk.load_mesh()
    else:
        print("A1 checkpoint not found; building fresh mesh.", flush=True)
        mesh = build_mesh()

    for tag in cases:
        DT = D_MAP[tag]
        print(f"\n=== {tag}: DT = {DT:+.1f} °C ===", flush=True)

        model_inputs, surf = make_model_inputs(mesh)
        model = SubglacialHydrologyModel(mesh, **model_inputs)

        if os.path.exists(A1_CHK):
            print(f"  Warm-start from A1 checkpoint", flush=True)
            _load_fields(model, A1_CHK)
        else:
            print("  No warm-start; cold-starting with winter basal melt.", flush=True)
            _set_melt(model, surf, 0.0, DT)
            for _ in range(500):
                model.step(dt)
            model.update_phi()

        print(f"  Phase 1: spin-up (max {max_yrs} years, tol={periodic_tol:.0e}) …",
              flush=True)

        phi_prev_yr = fd.Function(model.U).assign(model.phi)
        N_prev_yr   = fd.Function(model.U).assign(model.N)
        phi_snap    = fd.Function(model.U).assign(model.phi)
        h_snap      = model.h.dat.data_ro.copy()
        S_snap      = model.S.dat.data_ro.copy()
        converged_year = max_yrs

        for year in range(1, max_yrs + 1):
            n_fail = 0
            for step in range(steps_per_year):
                t_s = (year - 1) * SEC_PER_YEAR + step * dt
                if _try_step(model, surf, t_s, DT, dt, phi_snap, h_snap, S_snap):
                    model.update_phi()
                    phi_snap.assign(model.phi)
                    h_snap = model.h.dat.data_ro.copy()
                    S_snap = model.S.dat.data_ro.copy()
                else:
                    n_fail += 1
                    if n_fail <= 30 or n_fail % 50 == 0:
                        print(f"    [yr{year} step{step}] Newton failure "
                              f"(incl. sub-steps), skipping  ({n_fail} total)",
                              flush=True)

            rphi = fd.norm(model.phi - phi_prev_yr) / (fd.norm(phi_prev_yr) + 1e-12)
            rN   = fd.norm(model.N   - N_prev_yr)   / (fd.norm(N_prev_yr)   + 1e-12)
            print(f"  Spin-up year {year}: rphi={rphi:.3e}  rN={rN:.3e}", flush=True)

            phi_prev_yr.assign(model.phi)
            N_prev_yr.assign(model.N)

            if rphi < periodic_tol and rN < periodic_tol:
                print(f"  Periodic state reached after {year} years.", flush=True)
                converged_year = year
                break

        print("  Phase 2: recording final year …", flush=True)
        model.update_phi()

        _h_arr = model.h.dat.data_ro.copy()
        _S_arr = model.S.dat.data_ro.copy()
        print(f"  Recording start state: "
              f"h=[{_h_arr.min():.2e}, {_h_arr.max():.2e}] m  "
              f"S=[{_S_arr.min():.2e}, {_S_arr.max():.2e}] m²  "
              f"N_min={float(model.N.dat.data_ro.min())/1e6:.3f} MPa",
              flush=True)
        H_FLOOR = 1e-4
        if _h_arr.min() < H_FLOOR:
            model.h.dat.data[:] = np.maximum(_h_arr, H_FLOOR)
            model.update_phi()
            print(f"  Applied h floor = {H_FLOOR} m", flush=True)

        max_nsaves  = 366
        Nx_series   = np.zeros((max_nsaves, nbins))
        Q_sh_series = np.zeros(max_nsaves)
        Q_ch_series = np.zeros(max_nsaves)
        t_days      = np.zeros(max_nsaves)
        ksave       = 0

        model.update_phi()
        model.compute_flux_fields()
        xc, Nx = width_averaged_Nx(model, nbins=nbins)
        if xc_ref is None:
            xc_ref = xc
        Nx_series[0]   = Nx
        Q_sh_series[0] = _compute_Q_sh(model)
        Q_ch_series[0] = _compute_Q_ch(model)
        t_days[0]      = 0.0
        ksave = 1
        cpath = save_checkpoint(model, f"{tag}_Jan")
        print(f"    saved {tag}_Jan → {cpath}", flush=True)

        snaps_pending = {day: (f"{tag}_{lbl}", day)
                         for day, lbl in zip(SNAP_DAYS[1:], SNAP_LABELS[1:])}

        phi_snap.assign(model.phi)
        h_snap = model.h.dat.data_ro.copy()
        S_snap = model.S.dat.data_ro.copy()
        n_fail      = 0
        consec_fail = 0

        for step in range(steps_per_year):
            t_s      = converged_year * SEC_PER_YEAR + step * dt
            day_frac = (step + 1) * dt / 86400.0

            if _try_step(model, surf, t_s, DT, dt, phi_snap, h_snap, S_snap):
                consec_fail = 0
                model.update_phi()
                phi_snap.assign(model.phi)
                h_snap = model.h.dat.data_ro.copy()
                S_snap = model.S.dat.data_ro.copy()
            else:
                n_fail += 1
                consec_fail += 1
                if n_fail <= 30 or n_fail % 50 == 0:
                    print(f"    [record step {step}  day {day_frac:.1f}] "
                          f"Newton failure (consec={consec_fail}, "
                          f"total={n_fail})", flush=True)
                if consec_fail >= MAX_CONSEC_ABORT:
                    print(f"  WARNING: {consec_fail} consecutive failures at "
                          f"day {day_frac:.1f}. Saving partial results and "
                          f"aborting recording.", flush=True)
                    break
                continue

            if (step + 1) % stride == 0:
                model.compute_flux_fields()
                xc, Nx = width_averaged_Nx(model, nbins=nbins)
                if ksave < max_nsaves:
                    Nx_series[ksave]   = Nx
                    Q_sh_series[ksave] = _compute_Q_sh(model)
                    Q_ch_series[ksave] = _compute_Q_ch(model)
                    t_days[ksave]      = day_frac
                    ksave += 1

            for snap_day, (snap_tag, _) in list(snaps_pending.items()):
                if day_frac >= snap_day:
                    if (step + 1) % stride != 0:
                        model.compute_flux_fields()
                    cpath = save_checkpoint(model, snap_tag)
                    print(f"    saved {snap_tag} → {cpath}", flush=True)
                    del snaps_pending[snap_day]

        Nx_series   = Nx_series[:ksave, :]
        Q_sh_series = Q_sh_series[:ksave]
        Q_ch_series = Q_ch_series[:ksave]
        t_days      = t_days[:ksave]

        model.update_phi()
        model.compute_flux_fields()
        cpath = save_checkpoint(model, f"{tag}_final")
        print(f"  {tag}: final checkpoint → {cpath}", flush=True)

        nx_csv = os.path.join(CSV_DIR, f"{tag}_Nx_seasonal.csv")
        header = "t_days," + ",".join([f"x={xc_ref[i]/1e3:.2f}km"
                                       for i in range(nbins)])
        np.savetxt(nx_csv, np.c_[t_days, Nx_series],
                   delimiter=",", header=header, comments="")
        print(f"  {tag}: N(x,t) → {nx_csv}")

        q_csv = os.path.join(CSV_DIR, f"{tag}_Q_seasonal.csv")
        np.savetxt(q_csv, np.c_[t_days, Q_sh_series, Q_ch_series],
                   delimiter=",", header="t_days,Q_sh_m3s,Q_ch_m3s", comments="")
        print(f"  {tag}: Q(t)   → {q_csv}")

    npz_data = {"x": xc_ref}
    for tag in CASES_ALL:
        nx_csv = os.path.join(CSV_DIR, f"{tag}_Nx_seasonal.csv")
        if os.path.exists(nx_csv):
            data = np.loadtxt(nx_csv, delimiter=",", skiprows=1, ndmin=2)
            npz_data[f"{tag}_t"]  = data[:, 0]
            npz_data[f"{tag}_Nx"] = data[:, 1:]
    np.savez(os.path.join(OUTDIR, "D_seasonal_curves.npz"), **npz_data)
    print(f"\nSaved combined curves → "
          f"{os.path.join(OUTDIR, 'D_seasonal_curves.npz')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Subset of cases, e.g. D3 D4 D5")
    ap.add_argument("--max-years", type=int, default=max_years,
                    help="Maximum spin-up years before recording (default 10)")
    ap.add_argument("--dt", type=float, default=dt,
                    help="Timestep in seconds (default 1800; try 900 for D5 failures)")
    args = ap.parse_args()
    if args.dt != dt:
        dt             = args.dt
        steps_per_year = int(SEC_PER_YEAR / dt)
        stride         = int(86400 / dt)
        print(f"Using dt={dt:.0f}s (steps_per_year={steps_per_year}, stride={stride})")
    main(args.cases, max_yrs=args.max_years)
