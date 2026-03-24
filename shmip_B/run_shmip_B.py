"""SHMIP Suite B: steady-state moulin recharge on a rectangular ice sheet.

Usage: python run_shmip_B.py [--cases B1 B3 B5]
Output: outputs/{checkpoints,csv,B_Nx.png,B_curves.npz}
"""

import os
import argparse
import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt

from hydropack.models.subglacialhydrology import SubglacialHydrologyModel
from hydropack.constants import ice_density, water_density, gravity

Lx, Ly = 100e3, 20e3
nx, ny = 141, 28

dt          = 14400
max_steps   = 30000
rel_tol     = 5e-4
check_every = 6

OUTDIR  = "outputs"
CSV_DIR = os.path.join(OUTDIR, "csv")
CHK_DIR = os.path.join(OUTDIR, "checkpoints")
FIG     = os.path.join(OUTDIR, "B_Nx.png")

A1_RATE = 7.93e-11
A5_RATE = 4.5e-8

B_MAP = {
    "B1":   1,
    "B2":  10,
    "B3":  20,
    "B4":  50,
    "B5": 100,
}
CASES_ALL = ["B1", "B2", "B3", "B4", "B5"]

MOULIN_SIGMA = 1000.0
MOULIN_DIR   = "moulin_positions"


def build_mesh():
    return fd.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)


def make_model_inputs(mesh):
    """Build Firedrake fields for SubglacialHydrologyModel."""
    U   = fd.FunctionSpace(mesh, "CG", 1)
    CR  = fd.FunctionSpace(mesh, "CR", 1)
    x, y = fd.SpatialCoordinate(mesh)

    surf    = fd.interpolate(6*(fd.sqrt(x + 5000) - fd.sqrt(5000.0)) + 1, U)
    B       = fd.interpolate(fd.Constant(0.0), U)
    H       = surf - B

    u_b     = fd.interpolate(fd.Constant(1e-6), U)

    p_i     = fd.interpolate(fd.Constant(ice_density * gravity) * H, U)
    phi_m   = fd.interpolate(fd.Constant(water_density * gravity) * B, U)
    phi_0   = fd.interpolate(p_i + phi_m, U)

    bc      = fd.DirichletBC(U, phi_m, 1)

    h_init   = fd.interpolate(fd.Constant(0.01), U)
    S_init   = fd.interpolate(fd.Constant(0.001), CR)
    phi_init = fd.Function(U).interpolate(phi_0)

    m        = fd.Function(U); m.assign(0.0)

    return dict(
        thickness= H,
        bed      = B,
        sliding_speed = u_b,
        melt_rate= m,
        h_init   = h_init,
        S_init   = S_init,
        phi_init = phi_init,
        phi_m    = phi_m,
        p_i      = p_i,
        phi_0    = phi_0,
        dirichlet_bcs = [bc],
        englacial_void_ratio = 1e-4,
        out_dir  = OUTDIR,
    )


def total_domain_area(mesh):
    return float(fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh)))


def load_moulin_csv(tag):
    """Load SHMIP moulin positions. Returns (pts, q_total)."""
    path = os.path.join(MOULIN_DIR, f"{tag}_M.csv")
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    pts     = [(float(row[1]), float(row[2])) for row in data]
    q_total = float(np.sum(data[:, 3]))
    return pts, q_total


def gaussian_blobs(mesh, points, sigma):
    """Normalised sum of Gaussians (integrates to 1 over the domain)."""
    U    = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    if not points:
        f = fd.Function(U); f.assign(0.0); return f
    g    = sum(fd.exp(-((x - xi)**2 + (y - yi)**2) / (2 * sigma**2))
               for (xi, yi) in points)
    mass = fd.assemble(g * fd.dx)
    return fd.interpolate(g / (mass + 1e-30), U)


def build_recharge_field(mesh, tag):
    """Return CG1 total recharge: A1 background + moulin Gaussians."""
    U    = fd.FunctionSpace(mesh, "CG", 1)

    pts, Q_moulin = load_moulin_csv(tag)
    print(f"  Loaded {len(pts)} moulin(s) from {tag}_M.csv,"
          f" Q_total={Q_moulin:.1f} m³/s", flush=True)

    blob     = gaussian_blobs(mesh, pts, MOULIN_SIGMA)
    m_moulin = fd.Function(U).interpolate(fd.Constant(Q_moulin) * blob)
    m_full   = fd.Function(U).interpolate(fd.Constant(A1_RATE) + m_moulin)
    return m_full


def advance_to_steady(model, dt, *, rel_tol=5e-4, max_steps=30000,
                      check_every=6):
    """Step until phi, N, h, and S all converge."""
    phi_prev = fd.Function(model.U).interpolate(model.phi)
    N_prev   = fd.Function(model.U).interpolate(model.N)
    h_prev   = fd.Function(model.U).interpolate(model.h)
    S_prev   = fd.Function(model.CR).interpolate(model.S)

    n_failures = 0
    for k in range(1, max_steps + 1):
        try:
            model.step(dt)
        except Exception as e:
            n_failures += 1
            if n_failures <= 20:
                print(f"  [step {k}] Newton failure ({type(e).__name__}), skipping step "
                      f"({n_failures} total)", flush=True)
                model.phi.assign(phi_prev)
                model.update_phi()
                continue
            else:
                print(f"  Too many Newton failures ({n_failures}), aborting.", flush=True)
                return k

        if k % check_every == 0:
            model.update_phi()

            rphi = fd.norm(model.phi - phi_prev) / (fd.norm(model.phi) + 1e-30)
            rN   = fd.norm(model.N   - N_prev)   / (fd.norm(model.N)   + 1e-30)
            rh   = fd.norm(model.h   - h_prev)   / (fd.norm(model.h)   + 1e-30)
            rS   = fd.norm(model.S   - S_prev)   / (fd.norm(model.S)   + 1e-30)

            sim_days = k * dt / 86400
            print(f"iter {k} ({sim_days:.0f}d): rphi={rphi:.3e}, rN={rN:.3e}, rh={rh:.3e}, rS={rS:.3e}",
                  flush=True)

            phi_prev.assign(model.phi)
            N_prev.assign(model.N)
            h_prev.assign(model.h)
            S_prev.assign(model.S)

            if rphi < rel_tol and rN < rel_tol and rh < rel_tol and rS < rel_tol:
                return k

    print("WARNING: hit max_steps without steady convergence.")
    return max_steps


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
    mask   = count > 0
    Nx[mask] /= count[mask]
    xc     = 0.5 * (bins[:-1] + bins[1:])
    return xc, Nx


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

    mesh  = build_mesh()
    all_x = None
    curves = {}

    for tag in cases:
        n_moulins = B_MAP[tag]
        print(f"\n=== {tag}: {n_moulins} moulin(s) ===", flush=True)

        model_inputs = make_model_inputs(mesh)
        model        = SubglacialHydrologyModel(mesh, **model_inputs)

        m_full = build_recharge_field(mesh, tag)

        dt_ramp = 60 * 5
        m_bg = fd.Function(model.U).interpolate(fd.Constant(A1_RATE))
        model.m.assign(m_bg)
        for _ in range(300):
            model.step(dt_ramp)
        model.update_phi()
        print("  Phase 1 (A1 background) complete", flush=True)

        m_moulin = fd.Function(model.U).interpolate(m_full - m_bg)
        for frac in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
            model.m.interpolate(fd.Constant(A1_RATE) + fd.Constant(frac) * m_moulin)
            for _ in range(200):
                model.step(dt_ramp)
            model.update_phi()
            print(f"  Ramp frac={frac:.2f} complete", flush=True)

        model.m.assign(m_full)
        model.compute_flux_fields()

        iters = advance_to_steady(model, dt, rel_tol=rel_tol,
                                  max_steps=max_steps,
                                  check_every=check_every)
        print(f"{tag}: converged in {iters} checks.", flush=True)

        model.update_phi()
        model.compute_flux_fields()

        cpath = save_checkpoint(model, tag)
        print(f"{tag}: checkpoint → {cpath}")

        x, Nx = width_averaged_Nx(model)
        if all_x is None:
            all_x = x
        csv = os.path.join(CSV_DIR, f"{tag}_Nx.csv")
        np.savetxt(csv, np.c_[x, Nx], delimiter=",", header="x,N", comments="")
        print(f"{tag}: N(x) → {csv}")
        curves[tag] = Nx

    plt.figure(figsize=(8, 4.5))
    for tag in cases:
        plt.plot(all_x / 1000.0, curves[tag], label=f"{tag} ({B_MAP[tag]} moulins)")
    plt.xlabel("x  (km)")
    plt.ylabel("Width-averaged effective pressure  N  (Pa)")
    plt.title("SHMIP Suite B: N(x) at steady state")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG, dpi=200)
    print(f"Saved figure → {FIG}")

    np.savez(os.path.join(OUTDIR, "B_curves.npz"), x=all_x, **curves)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Subset of cases, e.g. B1 B3 B5")
    args = ap.parse_args()
    main(args.cases)
