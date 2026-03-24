#!/usr/bin/env python3
"""SHMIP Suite E: mountain-glacier (valley) geometry with varying bed slope.

Usage: python run_shmip_E.py [--cases E3+] [--max-steps 30000]
Output: outputs/{checkpoints,csv,E_Nx.png,E_curves.npz}
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

Lx = 6_000.0
EPS = 1e-16

M_E = 2.0 * 5.79e-7

E_GAMMA = {"E1": 0.05, "E2": 0.0, "E3": -0.1, "E4": -0.5, "E5": -0.7}

GAMMA_BENCH = 0.05

dt          = 600.0
rel_tol     = 5e-4
max_steps   = 750000
check_every = 144

OUTDIR   = "outputs"
CSV_DIR  = os.path.join(OUTDIR, "csv")
CHK_DIR  = os.path.join(OUTDIR, "checkpoints")
PLOT_PNG = os.path.join(OUTDIR, "E_Nx.png")

CASES_ALL = ["E1", "E2", "E3", "E4", "E5"]


# Analytic geometry

def surface_xy(x, y=0.0):
    """Ice-surface elevation S(x,y) [m]."""
    return 100.0*(x + 200.0)**0.25 + x/60.0 - (2.0e10)**0.25 + 1.0


def f_poly(x, gamma):
    """Along-flow bed polynomial f(x,gamma)."""
    s6 = surface_xy(6000.0)
    return ((s6 - gamma*6000.0) / (6000.0**2)) * x**2 + gamma * x


def g_y(y):
    return 0.5e-6 * abs(y)**3


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
        gmsh.model.add("suite_E_valley")

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
        gmsh.model.setPhysicalName(2, 5 if l_right is None else 5, "valley")

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
        x_focus=None,
        refine_halfwidth=1000.0,
    )
    build_mesh.boundary_tags = tags
    return mesh


# Firedrake field builders

def fd_surface(mesh):
    Q = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    s = 100.0*(x + 200.0)**0.25 + x/60.0 - (2.0e10)**0.25 + 1.0
    return fd.interpolate(s, Q)


def fd_bed(mesh, gamma):
    """B(x,y;gamma) on CG1."""
    Q  = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    zs = fd_surface(mesh)

    s6 = fd.Constant(surface_xy(6000.0))
    f  = ((s6 - gamma*6000.0)/6000.0**2) * x**2 + gamma*x

    # Clamp denominator to avoid division by ~0 near the snout
    s_ref = fd_surface(mesh) - ((s6 - GAMMA_BENCH*6000.0)/6000.0**2) * x**2 - GAMMA_BENCH*x
    h     = (-4.5*x/6000.0 + 5.0) * ((zs - f) / fd.max_value(s_ref, fd.Constant(1.0)))

    g  = 0.5e-6 * fd.sqrt(y**2)**3
    zb = f + g*h
    return fd.interpolate(zb, Q)


def make_model_inputs(mesh, gamma):
    """Build Firedrake fields for SubglacialHydrologyModel."""
    Q  = fd.FunctionSpace(mesh, "CG", 1)
    V  = fd.VectorFunctionSpace(mesh, "CG", 1)
    CR = fd.FunctionSpace(mesh, "CR", 1)

    zs = fd_surface(mesh)
    zb = fd_bed(mesh, gamma)
    H  = fd.interpolate(fd.max_value(zs - zb, fd.Constant(1.0)), Q)

    u_b   = fd.Function(Q).interpolate(
                fd.sqrt(fd.inner(fd.as_vector((1.0e-6, 0.0)),
                                 fd.as_vector((1.0e-6, 0.0)))))

    p_i   = fd.Function(Q).interpolate(fd.Constant(ice_density   * gravity) * H)
    phi_m = fd.Function(Q).interpolate(fd.Constant(water_density * gravity) * zb)
    phi_0 = fd.Function(Q).interpolate(p_i + phi_m)

    h_init   = fd.interpolate(fd.Constant(0.01), Q)
    S_init   = fd.interpolate(fd.Constant(1e-3), CR)
    phi_init = fd.Function(Q).interpolate(phi_0)

    bc = fd.DirichletBC(Q, phi_m, [1])

    return {
        "thickness": H,
        "bed":       zb,
        "sliding_speed": u_b,
        "melt_rate": fd.interpolate(fd.Constant(0.0), Q),
        "h_init":    h_init,
        "S_init":    S_init,
        "phi_init":  phi_init,
        "dirichlet_bcs": [bc],
        "phi_m":     phi_m,
        "p_i":       p_i,
        "phi_0":     phi_0,
        "englacial_void_ratio": 0.0,
        "out_dir":   OUTDIR,
    }


def save_diagnostic_plots(mesh, inputs, tag="E1"):
    """Save mesh and field sanity-check plots."""
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"diagnostic_{tag}.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.set_aspect("equal")
    fd.triplot(mesh, axes=ax)
    ax.set_title("Mesh")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[0, 1]
    c = fd.tripcolor(inputs["thickness"], axes=ax, cmap="Blues")
    fig.colorbar(c, ax=ax, fraction=0.03, pad=0.04)
    ax.set_aspect("equal")
    ax.set_title("Ice thickness H (m)")

    ax = axes[1, 0]
    c = fd.tripcolor(inputs["bed"], axes=ax, cmap="terrain")
    fig.colorbar(c, ax=ax, fraction=0.03, pad=0.04)
    ax.set_aspect("equal")
    ax.set_title("Bed elevation B (m)")

    ax = axes[1, 1]
    zs = fd_surface(mesh)
    c = fd.tripcolor(zs, axes=ax, cmap="YlOrBr")
    fig.colorbar(c, ax=ax, fraction=0.03, pad=0.04)
    ax.set_aspect("equal")
    ax.set_title("Surface elevation S (m)")

    fig.suptitle(f"Suite E — {tag} geometry diagnostics", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved diagnostic → {path}")


def advance_to_steady(model, dt, *, rel_tol=5e-4, max_steps=30000,
                      check_every=6):
    """Step until phi, N, h, and S all converge."""
    phi_prev = fd.Function(model.U).assign(model.phi)
    N_prev   = fd.Function(model.U).assign(model.N)
    h_prev   = fd.Function(model.U).assign(model.h)
    S_prev   = fd.Function(model.CR).assign(model.S)

    n_failures = 0
    for k in range(1, max_steps + 1):
        try:
            model.step(dt)
        except Exception as e:
            n_failures += 1
            if n_failures <= 50:
                model.phi.assign(phi_prev)
                model.update_phi()
                continue
            else:
                print(f"  Too many Newton failures ({n_failures}), aborting.", flush=True)
                return k
        if k % check_every == 0:
            model.update_phi()
            rphi = float(fd.norm(model.phi - phi_prev) / (fd.norm(model.phi) + 1e-30))
            rN   = float(fd.norm(model.N   - N_prev)   / (fd.norm(model.N)   + 1e-30))
            rh   = float(fd.norm(model.h   - h_prev)   / (fd.norm(model.h)   + 1e-30))
            rS   = float(fd.norm(model.S   - S_prev)   / (fd.norm(model.S)   + 1e-30))
            sim_days = k * dt / 86400
            print(f"  iter {k:5d} ({sim_days:.0f}d): rphi={rphi:.3e}  rN={rN:.3e}  rh={rh:.3e}  rS={rS:.3e}",
                  flush=True)
            phi_prev.assign(model.phi)
            N_prev.assign(model.N)
            h_prev.assign(model.h)
            S_prev.assign(model.S)
            if rphi < rel_tol and rN < rel_tol and rh < rel_tol and rS < rel_tol:
                return k
    print("  WARNING: hit max_steps without steady convergence.", flush=True)
    return max_steps


def width_averaged_Nx(model, nbins=120, band_half=100.0):
    """Area-weighted N(x) averaged within a centre strip."""
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


def save_checkpoint(model, tag):
    os.makedirs(CHK_DIR, exist_ok=True)
    fname = os.path.join(CHK_DIR, f"{tag}.h5")
    with fd.CheckpointFile(fname, "w") as chk:
        chk.save_mesh(model.mesh)
        for name in ("h", "S", "phi", "pfo", "N", "N_cr", "h_cr",
                     "S_alpha", "p_w", "q_s", "q_s_mag", "Q_ch"):
            if hasattr(model, name):
                chk.save_function(getattr(model, name), name=name)
    npz_path = os.path.join(CHK_DIR, f"{tag}_cr.npz")
    cr_fields = {}
    for name in ("S", "N_cr", "h_cr"):
        if hasattr(model, name):
            cr_fields[name] = getattr(model, name).dat.data_ro.copy()
    if cr_fields:
        np.savez(npz_path, **cr_fields)
    return fname


def _load_fields(model, chk_path):
    """Warm-start model phi/h/S from a previous checkpoint."""
    with fd.CheckpointFile(chk_path, "r") as chk:
        m_src   = chk.load_mesh()
        phi_src = chk.load_function(m_src, "phi")
        h_src   = chk.load_function(m_src, "h")
    model.phi.dat.data[:] = phi_src.dat.data_ro
    model.h.dat.data[:]   = h_src.dat.data_ro

    # Prefer numpy sidecar for S (avoids CR1 projection failure)
    npz_path = chk_path.replace(".h5", "_cr.npz")
    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        if "S" in npz:
            model.S.dat.data[:] = npz["S"]
    else:
        try:
            with fd.CheckpointFile(chk_path, "r") as chk:
                m_src = chk.load_mesh()
                S_src = chk.load_function(m_src, "S")
            model.S.dat.data[:] = S_src.dat.data_ro
        except Exception as exc:
            print(f"  [warn] S warm-start failed ({type(exc).__name__}); "
                  f"S kept at init", flush=True)

    model.update_phi()


def expand_cases(sel):
    if not sel:
        return CASES_ALL
    out = []
    for s in sel:
        s = s.upper()
        if s.endswith("+") and s[:-1] in CASES_ALL:
            i = CASES_ALL.index(s[:-1])
            out.extend(CASES_ALL[i:])
        elif "-" in s:
            a, b = s.split("-", 1)
            if a in CASES_ALL and b in CASES_ALL:
                lo, hi = CASES_ALL.index(a), CASES_ALL.index(b)
                out.extend(CASES_ALL[min(lo, hi):max(lo, hi) + 1])
        elif s in CASES_ALL:
            out.append(s)
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def main(cases=None, max_steps_arg=max_steps):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    if not cases:
        cases = CASES_ALL

    mesh   = build_mesh()
    all_x  = None
    curves = {}

    for ci, tag in enumerate(cases):
        gamma = E_GAMMA[tag]
        print(f"\n=== Suite E: {tag} (gamma = {gamma:+.2f}),  m = {M_E:.3e} m/s ===",
              flush=True)

        model_inputs = make_model_inputs(mesh, gamma)
        model        = SubglacialHydrologyModel(mesh, **model_inputs)

        # Initialise phi from phi_0 (N=0) for the valley geometry
        model.phi.dat.data[:] = model_inputs["phi_0"].dat.data_ro
        model.update_phi()

        if ci == 0:
            save_diagnostic_plots(mesh, model_inputs, tag=tag)

        prev_idx = CASES_ALL.index(tag) - 1
        prev_chk = (os.path.join(CHK_DIR, f"{CASES_ALL[prev_idx]}.h5")
                    if prev_idx >= 0 else None)

        if prev_chk and os.path.exists(prev_chk):
            print(f"  warm-start from {prev_chk}", flush=True)
            _load_fields(model, prev_chk)
            model.m.interpolate(fd.Constant(M_E))
            print("  warm-start settling (300 steps at dt_ramp)...", flush=True)
            for _ in range(300):
                try:
                    model.step(300.0)
                except Exception:
                    pass
            model.update_phi()
        else:
            print("  cold-start with gradual recharge ramp …", flush=True)
            dt_ramp = 300.0
            for frac in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                         0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                         0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]:
                model.m.interpolate(fd.Constant(frac * M_E))
                for _ in range(300):
                    model.step(dt_ramp)
                model.update_phi()
                print(f"  ramp {int(frac*100):3d}%:  "
                      f"N_max = {model.N.dat.data_ro.max()/1e6:.3f} MPa", flush=True)

        model.compute_flux_fields()

        iters = advance_to_steady(model, dt, rel_tol=rel_tol,
                                  max_steps=max_steps_arg,
                                  check_every=check_every)
        print(f"  {tag}: converged in {iters} iters", flush=True)

        model.update_phi()
        model.compute_flux_fields()

        cpath = save_checkpoint(model, tag)
        print(f"  {tag}: checkpoint → {cpath}", flush=True)

        x, Nx = width_averaged_Nx(model, nbins=120, band_half=100.0)
        if all_x is None:
            all_x = x
        curves[tag] = Nx
        np.savetxt(os.path.join(CSV_DIR, f"{tag}_Nx.csv"),
                   np.c_[x, Nx], delimiter=",", header="x,N", comments="")
        print(f"  {tag}: N_max = {Nx.max()/1e6:.3f} MPa  "
              f"N_min = {Nx.min()/1e6:.3f} MPa", flush=True)

    if curves:
        colors = {"E1": "#1b7837", "E2": "#4393c3", "E3": "#f46d43",
                  "E4": "#d73027", "E5": "#7b2d8b"}
        plt.figure(figsize=(8.5, 4.8))
        for t in CASES_ALL:
            if t in curves:
                plt.plot(all_x / 1e3, curves[t] / 1e6,
                         color=colors.get(t, "k"), lw=1.8,
                         label=f"{t} (gamma={E_GAMMA[t]:+.2f})")
        plt.xlabel("x  (km)", fontsize=10)
        plt.ylabel(r"Width-averaged $\bar{N}$  (MPa)", fontsize=10)
        plt.title("SHMIP Suite E — centre-band effective pressure N(x)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(PLOT_PNG, dpi=200)
        plt.close()
        print(f"\nSaved overlay → {PLOT_PNG}")

    if all_x is not None:
        np.savez(os.path.join(OUTDIR, "E_curves.npz"), x=all_x, **curves)
        print(f"Saved curves  → {os.path.join(OUTDIR, 'E_curves.npz')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*",
                    help="Subset: E3, E3 E4 E5, E3+, or E2-E5")
    ap.add_argument("--max-steps", type=int, default=max_steps,
                    help=f"Max steady-state iterations (default {max_steps})")
    args = ap.parse_args()
    main(expand_cases(args.cases), max_steps_arg=args.max_steps)
