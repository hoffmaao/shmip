#!/usr/bin/env python3
"""SHMIP Suite F: seasonal degree-day forcing on valley glacier geometry.

Usage: python run_shmip_F.py [--cases F3] [--max-years 15] [--force-f0]
Output: outputs/{checkpoints,csv,F_seasonal_curves.npz}
"""

import os
import argparse
import numpy as np
import firedrake as fd
import gmsh
import tempfile
import matplotlib.pyplot as plt

from hydropack.models.subglacialhydrology import SubglacialHydrologyModel
from hydropack.constants import ice_density, water_density, gravity

Lx          = 6_000.0
EPS         = 1e-16
GAMMA_F     = 0.05
GAMMA_BENCH = 0.05

A1_RATE      = 7.93e-11
LR           = -0.0075
DDF          = 0.01 / 86400
T_AMPL       = 16.0
T_BASE       = -5.0
SEC_PER_YEAR = 365 * 86400

F_MAP     = {"F1": -6.0, "F2": -3.0, "F3": 0.0, "F4": +3.0, "F5": +6.0}
CASES_ALL = ["F1", "F2", "F3", "F4", "F5"]

dt               = 600.0
max_years        = 30
periodic_tol     = 1e-2
N_SUBSTEPS       = 4
MAX_CONSEC_ABORT = 200
steps_per_year   = int(SEC_PER_YEAR / dt)
stride           = int(86400 / dt)

rel_tol_ss   = 5e-4
max_steps_ss = 30000
dt_ss        = 14400.0
check_every  = 6

SNAP_DAYS   = [0,     91,    182,    274   ]
SNAP_LABELS = ["Jan", "Apr", "Jul",  "Oct" ]

OUTDIR  = "outputs"
CSV_DIR = os.path.join(OUTDIR, "csv")
CHK_DIR = os.path.join(OUTDIR, "checkpoints")

E1_CHK = os.path.join("..", "shmip_E", "outputs", "checkpoints", "E1.h5")


# Analytic geometry (same as Suite E with gamma = 0.05)

def surface_xy(x, y=0.0):
    """Ice-surface elevation S(x,y) [m]."""
    return 100.0*(x + 200.0)**0.25 + x/60.0 - (2.0e10)**0.25 + 1.0


def f_poly(x, gamma):
    s6 = surface_xy(6000.0)
    return ((s6 - gamma*6000.0) / (6000.0**2)) * x**2 + gamma * x


def g_inv(s):
    return 0.0 if s <= 0.0 else (s / 0.5e-6)**(1.0/3.0)


def h_fun(x, gamma):
    num = surface_xy(x) - f_poly(x, gamma)
    den = surface_xy(x) - f_poly(x, GAMMA_BENCH) + EPS
    return (-4.5*x/6000.0 + 5.0) * (num / den)


def outline_half_width(x):
    s = surface_xy(x) - f_poly(x, GAMMA_BENCH)
    return g_inv(s / (h_fun(x, GAMMA_BENCH) + EPS))


def y_bottom_py(x):
    return -outline_half_width(float(x))


def y_top_py(x):
    return +outline_half_width(float(x))


# Mesh construction

def build_valley_mesh_gmsh(
    Lx,
    y_bottom,
    y_top,
    nx_samples=81,
    hmax=60.0,
    hmin=25.0,
    x_focus=None,
    refine_halfwidth=1000.0,
):
    xs = np.linspace(0.0, Lx, int(nx_samples))
    yb = np.array([float(y_bottom(x)) for x in xs])
    yt = np.array([float(y_top(x))    for x in xs])

    gap = yt - yb
    if np.min(gap) < -1e-9:
        i = int(np.argmin(gap))
        raise ValueError(f"y_top < y_bottom at x={xs[i]:.3f} by {gap[i]:.3e} m")
    right_collapsed = abs(yt[-1] - yb[-1]) <= 1e-12

    gmsh.initialize()
    try:
        gmsh.model.add("suite_F_valley")

        p_bot = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, hmax)
                 for x, y in zip(xs, yb)]
        p_top = []
        for i, (x, y) in enumerate(zip(xs, yt)):
            if i == len(xs) - 1 and right_collapsed:
                p_top.append(p_bot[-1])
            else:
                p_top.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, hmax))

        l_bottom = gmsh.model.geo.addSpline(p_bot)
        l_top    = gmsh.model.geo.addSpline(p_top)
        l_left   = gmsh.model.geo.addLine(p_top[0], p_bot[0])

        if right_collapsed:
            cloop   = gmsh.model.geo.addCurveLoop([l_bottom, -l_top, l_left])
            l_right = None
        else:
            l_right = gmsh.model.geo.addLine(p_bot[-1], p_top[-1])
            cloop   = gmsh.model.geo.addCurveLoop([l_bottom, l_right, -l_top, l_left])

        surf_geo = gmsh.model.geo.addPlaneSurface([cloop])

        if (x_focus is not None) and (hmin is not None):
            y_mid = 0.5 * (float(y_bottom(x_focus)) + float(y_top(x_focus)))
            pref  = gmsh.model.geo.addPoint(float(x_focus), y_mid, 0.0, hmin)
            gmsh.model.geo.synchronize()
            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "NodesList", [pref])
            f_thr  = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thr, "IField",  f_dist)
            gmsh.model.mesh.field.setNumber(f_thr, "LcMin",   float(hmin))
            gmsh.model.mesh.field.setNumber(f_thr, "LcMax",   float(hmax))
            gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(f_thr, "DistMax", float(refine_halfwidth))
            gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)

        gmsh.model.geo.synchronize()

        id_x0     = gmsh.model.addPhysicalGroup(1, [l_left])
        gmsh.model.setPhysicalName(1, id_x0, "x0")
        id_bottom = gmsh.model.addPhysicalGroup(1, [l_bottom])
        gmsh.model.setPhysicalName(1, id_bottom, "bottom")
        id_top    = gmsh.model.addPhysicalGroup(1, [l_top])
        gmsh.model.setPhysicalName(1, id_top, "top")
        if l_right is not None:
            id_xL = gmsh.model.addPhysicalGroup(1, [l_right])
            gmsh.model.setPhysicalName(1, id_xL, "xL")
        else:
            id_xL = -1
        gmsh.model.addPhysicalGroup(2, [surf_geo])
        gmsh.model.setPhysicalName(2, 5, "valley")

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)

        fd_fd, fd_path = tempfile.mkstemp(suffix=".msh")
        os.close(fd_fd)
        gmsh.write(fd_path)
    finally:
        gmsh.finalize()

    mesh = fd.Mesh(fd_path)
    os.remove(fd_path)

    tags = {"x0": id_x0, "xL": id_xL, "bottom": id_bottom, "top": id_top}
    return mesh, tags


def build_mesh():
    mesh, tags = build_valley_mesh_gmsh(
        Lx=Lx,
        y_bottom=y_bottom_py,
        y_top=y_top_py,
        nx_samples=81,
        hmax=60.0,
        hmin=25.0,
    )
    build_mesh.boundary_tags = tags
    return mesh


# Firedrake field builders

def fd_surface(mesh):
    Q = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    s = 100.0*(x + 200.0)**0.25 + x/60.0 - (2.0e10)**0.25 + 1.0
    return fd.interpolate(s, Q)


def fd_bed(mesh):
    """CG1 bed elevation B(x,y) with gamma = GAMMA_F."""
    Q  = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    zs = fd_surface(mesh)

    s6    = fd.Constant(surface_xy(6000.0))
    gamma = fd.Constant(GAMMA_F)
    f     = ((s6 - gamma*6000.0) / 6000.0**2) * x**2 + gamma*x

    s_ref = (fd_surface(mesh)
             - ((s6 - GAMMA_BENCH*6000.0) / 6000.0**2) * x**2
             - GAMMA_BENCH * x)
    h_fd  = (-4.5*x/6000.0 + 5.0) * ((zs - f) / fd.max_value(s_ref, fd.Constant(1.0)))

    g  = 0.5e-6 * fd.sqrt(y**2)**3
    zb = f + g * h_fd
    return fd.interpolate(zb, Q)


def make_model_inputs(mesh):
    """Return (inputs_dict, surf_CG1)."""
    Q   = fd.FunctionSpace(mesh, "CG", 1)
    CR  = fd.FunctionSpace(mesh, "CR", 1)

    zs = fd_surface(mesh)
    zb = fd_bed(mesh)
    H  = fd.interpolate(fd.max_value(zs - zb, fd.Constant(1.0)), Q)

    u_b   = fd.Function(Q).interpolate(fd.Constant(1.0e-6))
    p_i   = fd.Function(Q).interpolate(fd.Constant(ice_density   * gravity) * H)
    phi_m = fd.Function(Q).interpolate(fd.Constant(water_density * gravity) * zb)
    phi_0 = fd.Function(Q).interpolate(p_i + phi_m)

    h_init   = fd.interpolate(fd.Constant(0.01),  Q)
    S_init   = fd.interpolate(fd.Constant(1e-3),  CR)
    phi_init = fd.Function(Q).interpolate(phi_0)

    bc = fd.DirichletBC(Q, phi_m, [1])

    return dict(
        thickness = H,
        bed       = zb,
        sliding_speed = u_b,
        melt_rate = fd.interpolate(fd.Constant(0.0), Q),
        h_init    = h_init,
        S_init    = S_init,
        phi_init  = phi_init,
        dirichlet_bcs = [bc],
        phi_m     = phi_m,
        p_i       = p_i,
        phi_0     = phi_0,
        englacial_void_ratio = 1e-4,
        out_dir   = OUTDIR,
    ), zs


# Seasonal forcing

def _air_temp(t_s, DT):
    return -T_AMPL * np.cos(2.0 * np.pi * t_s / SEC_PER_YEAR) + T_BASE + DT


def _set_melt(model, surf, t_s, DT):
    """Update model.m with degree-day melt at time t_s [s]."""
    T_air = _air_temp(t_s, DT)
    model.m.interpolate(
        fd.max_value(
            fd.Constant(T_air) + fd.Constant(LR) * surf,
            fd.Constant(0.0)
        ) * fd.Constant(DDF) + fd.Constant(A1_RATE)
    )


# Diagnostics

def _compute_Q_sh(model):
    try:
        return max(0.0, -float(fd.assemble(model.q_s[0] * fd.ds(1))))
    except Exception:
        return 0.0


def _compute_Q_ch(model):
    try:
        vals   = model.Q_ch.dat.data_ro.copy()
        finite = np.isfinite(vals)
        return float(np.sum(np.abs(vals[finite]))) if finite.any() else 0.0
    except Exception:
        return 0.0


def width_averaged_Nx(model, nbins=120, band_half=100.0):
    """Area-weighted N(x) averaged over a centre strip."""
    N0  = fd.Function(fd.FunctionSpace(model.mesh, "DG", 0)).project(model.N)
    N0v = N0.dat.data_ro

    VdgV   = fd.VectorFunctionSpace(model.mesh, "DG", 0)
    coords = fd.Function(VdgV).interpolate(fd.SpatialCoordinate(model.mesh))
    xc, yc = coords.dat.data_ro[:, 0], coords.dat.data_ro[:, 1]

    Vdg = fd.FunctionSpace(model.mesh, "DG", 0)
    A   = fd.Function(Vdg).interpolate(fd.CellVolume(model.mesh)).dat.data_ro

    mask = np.abs(yc) <= band_half
    bins = np.linspace(0.0, Lx, nbins + 1)
    ix   = np.clip(np.searchsorted(bins, xc[mask], side="right") - 1, 0, nbins - 1)

    Nsum = np.zeros(nbins)
    Asum = np.zeros(nbins)
    np.add.at(Nsum, ix, N0v[mask] * A[mask])
    np.add.at(Asum, ix, A[mask])
    Nx   = np.divide(Nsum, np.where(Asum > 0, Asum, 1.0))
    xmid = 0.5 * (bins[:-1] + bins[1:])
    return xmid, Nx


# Checkpoint I/O

def save_checkpoint(model, tag):
    os.makedirs(CHK_DIR, exist_ok=True)
    path = os.path.join(CHK_DIR, f"{tag}.h5")
    with fd.CheckpointFile(path, "w") as chk:
        chk.save_mesh(model.mesh)
        for name in ("h", "S", "phi", "pfo", "N", "N_cr", "h_cr",
                     "S_alpha", "p_w", "m", "q_s", "q_s_mag", "Q_ch"):
            if hasattr(model, name):
                chk.save_function(getattr(model, name), name=name)
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


# Steady-state advance (for F0)

def advance_to_steady(model, dt, *, rel_tol=5e-4, max_steps=30000,
                      check_every=6):
    """Step until phi, N, h, and S all converge."""
    phi_prev = fd.Function(model.U).assign(model.phi)
    N_prev   = fd.Function(model.U).assign(model.N)
    h_prev   = fd.Function(model.U).assign(model.h)
    S_prev   = fd.Function(model.CR).assign(model.S)

    current_dt = dt
    consec_fail = 0

    for k in range(1, max_steps + 1):
        phi_snap = model.phi.dat.data_ro.copy()
        h_snap   = model.h.dat.data_ro.copy()
        S_snap   = model.S.dat.data_ro.copy()

        try:
            model.step(current_dt)
            consec_fail = 0
        except Exception as exc:
            model.phi.dat.data[:] = phi_snap
            model.h.dat.data[:] = h_snap
            model.S.dat.data[:] = S_snap
            model.update_phi()
            consec_fail += 1
            if current_dt > dt / 8.0:
                current_dt = current_dt / 2.0
                print(f"  [SS] step {k} failed ({type(exc).__name__}), "
                      f"halving dt → {current_dt:.0f}s", flush=True)
            else:
                print(f"  [SS] step {k} failed at min dt={current_dt:.0f}s; "
                      "skipping.", flush=True)
            if consec_fail > 50:
                print("  WARNING: too many consecutive failures, giving up.",
                      flush=True)
                return k
            continue

        if k % check_every == 0:
            model.update_phi()
            rphi = float(fd.norm(model.phi - phi_prev) / (fd.norm(model.phi) + 1e-30))
            rN   = float(fd.norm(model.N   - N_prev)   / (fd.norm(model.N)   + 1e-30))
            rh   = float(fd.norm(model.h   - h_prev)   / (fd.norm(model.h)   + 1e-30))
            rS   = float(fd.norm(model.S   - S_prev)   / (fd.norm(model.S)   + 1e-30))
            sim_days = k * current_dt / 86400
            print(f"  iter {k:6d} ({sim_days:.0f}d): rphi={rphi:.3e}  rN={rN:.3e}"
                  f"  rh={rh:.3e}  rS={rS:.3e}  dt={current_dt:.0f}s",
                  flush=True)
            phi_prev.assign(model.phi)
            N_prev.assign(model.N)
            h_prev.assign(model.h)
            S_prev.assign(model.S)
            if rphi < rel_tol and rN < rel_tol and rh < rel_tol and rS < rel_tol:
                return k

    print("  WARNING: hit max_steps without steady convergence.", flush=True)
    return max_steps


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


# Main driver

def main(cases=None, max_yrs=max_years, force_f0=False):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    if not cases:
        cases = CASES_ALL

    print("Building valley mesh …", flush=True)
    mesh = build_mesh()
    print(f"  mesh: {mesh.num_vertices()} vertices, "
          f"{mesh.num_cells()} cells", flush=True)

    nbins  = 120
    xc_ref = None

    f0_chk = os.path.join(CHK_DIR, "F0.h5")

    if force_f0 or not os.path.exists(f0_chk):
        print("\n=== F0: winter steady state (m = A1_RATE) ===", flush=True)

        model_inputs, surf = make_model_inputs(mesh)
        model = SubglacialHydrologyModel(mesh, **model_inputs)
        model.phi.dat.data[:] = model_inputs["phi_0"].dat.data_ro
        model.update_phi()

        # Load h from E1 if available; reset phi and S for safe winter spin-up
        if os.path.exists(E1_CHK):
            print(f"  Warm-start h from E1: {E1_CHK}", flush=True)
            _load_fields(model, E1_CHK)
            model.S.dat.data[:] = 1e-3
            model.phi.dat.data[:] = model_inputs["phi_0"].dat.data_ro
            model.update_phi()
        else:
            print("  Cold-start from phi_0 (no E1 checkpoint found).", flush=True)

        model.m.interpolate(fd.Constant(A1_RATE))
        model.compute_flux_fields()

        print(f"  Advancing to steady state "
              f"(max {max_steps_ss} steps, tol {rel_tol_ss:.0e}) …", flush=True)
        iters = advance_to_steady(model, dt_ss, rel_tol=rel_tol_ss,
                                  max_steps=max_steps_ss,
                                  check_every=check_every)
        model.update_phi()
        model.compute_flux_fields()

        cpath = save_checkpoint(model, "F0")
        print(f"  F0 converged in {iters} iters → {cpath}", flush=True)
        xc_ref_f0, Nx_f0 = width_averaged_Nx(model, nbins=nbins)
        print(f"  F0: N_max = {Nx_f0.max()/1e6:.3f} MPa  "
              f"N_min = {Nx_f0.min()/1e6:.3f} MPa", flush=True)
    else:
        print(f"\n=== F0: checkpoint found, skipping ({f0_chk}) ===", flush=True)

    all_curves = {}

    for tag in cases:
        DT = F_MAP[tag]
        print(f"\n=== {tag}: DT = {DT:+.1f} °C ===", flush=True)

        model_inputs, surf = make_model_inputs(mesh)
        model = SubglacialHydrologyModel(mesh, **model_inputs)
        model.phi.dat.data[:] = model_inputs["phi_0"].dat.data_ro
        model.update_phi()

        print(f"  Warm-start from F0 ({f0_chk})", flush=True)
        _load_fields(model, f0_chk)

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

        snaps_pending = {day: f"{tag}_{lbl}"
                         for day, lbl in zip(SNAP_DAYS[1:], SNAP_LABELS[1:])}

        phi_snap.assign(model.phi)
        h_snap      = model.h.dat.data_ro.copy()
        S_snap      = model.S.dat.data_ro.copy()
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

            for snap_day, snap_tag in list(snaps_pending.items()):
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

        if xc_ref is None:
            xc_ref = np.linspace(0, Lx, nbins)

        nx_csv = os.path.join(CSV_DIR, f"{tag}_Nx_seasonal.csv")
        header = "t_days," + ",".join([f"x={xc_ref[i]/1e3:.3f}km"
                                       for i in range(nbins)])
        np.savetxt(nx_csv, np.c_[t_days, Nx_series],
                   delimiter=",", header=header, comments="")
        print(f"  {tag}: N(x,t) → {nx_csv}")

        q_csv = os.path.join(CSV_DIR, f"{tag}_Q_seasonal.csv")
        np.savetxt(q_csv, np.c_[t_days, Q_sh_series, Q_ch_series],
                   delimiter=",", header="t_days,Q_sh_m3s,Q_ch_m3s", comments="")
        print(f"  {tag}: Q(t)   → {q_csv}")

        all_curves[tag] = (t_days, Nx_series)

    if xc_ref is not None:
        npz_data = {"x": xc_ref}
        for tag in CASES_ALL:
            csv_p = os.path.join(CSV_DIR, f"{tag}_Nx_seasonal.csv")
            if os.path.exists(csv_p):
                data = np.loadtxt(csv_p, delimiter=",", skiprows=1, ndmin=2)
                npz_data[f"{tag}_t"]  = data[:, 0]
                npz_data[f"{tag}_Nx"] = data[:, 1:]
        np.savez(os.path.join(OUTDIR, "F_seasonal_curves.npz"), **npz_data)
        print(f"\nSaved combined curves → "
              f"{os.path.join(OUTDIR, 'F_seasonal_curves.npz')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Subset of cases, e.g. F3 F4 F5 (F0 computed automatically)")
    ap.add_argument("--max-years", type=int, default=max_years,
                    help=f"Maximum spin-up years before recording (default {max_years})")
    ap.add_argument("--force-f0", action="store_true",
                    help="Recompute F0 even if checkpoint already exists")
    args = ap.parse_args()
    main(args.cases, max_yrs=args.max_years, force_f0=args.force_f0)
