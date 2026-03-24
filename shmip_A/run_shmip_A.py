"""SHMIP Suite A: steady-state distributed recharge on a rectangular ice sheet.

Usage: python run_shmip_A.py [--cases A3 A4 A5] [--plot-ribbons]
Output: outputs/{checkpoints,csv,A_Nx.png,A_curves.npz}
"""

import os
import math
import argparse
import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
from hydropack.models.subglacialhydrology import SubglacialHydrologyModel
from hydropack.constants import ice_density, water_density, gravity

Lx = 100e3
Ly = 20e3
nx = 115
ny = 23

A_RATES = {
    "A1": 7.93e-11,
    "A2": 1.59e-9,
    "A3": 5.79e-9,
    "A4": 2.5e-8,
    "A5": 4.5e-8,
    "A6": 5.79e-7,
}

dt = 14400
max_steps = 30000
rel_tol = 5e-4
check_every = 6

OUTDIR = "outputs"
PLOT_PNG = os.path.join(OUTDIR, "A_Nx.png")
CSV_DIR = os.path.join(OUTDIR, "csv")
CHK_DIR = os.path.join(OUTDIR, "checkpoints")


def build_mesh():
    return fd.RectangleMesh(nx, ny, Lx, Ly)


CASES_ALL = ["A1", "A2", "A3", "A4", "A5", "A6"]

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
                i, j = CASES_ALL.index(a), CASES_ALL.index(b)
                lo, hi = min(i, j), max(i, j)
                out.extend(CASES_ALL[lo:hi+1])
        elif s in CASES_ALL:
            out.append(s)
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def make_model_inputs(mesh):
    """Build Firedrake fields for SubglacialHydrologyModel."""
    Q = fd.FunctionSpace(mesh, "CG", 1)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    CR = fd.FunctionSpace(mesh, "CR", 1)

    x, y = fd.SpatialCoordinate(mesh)

    S = fd.interpolate(6*(fd.sqrt(x + 5000) - fd.sqrt(5000.0)) + 1, Q)
    B = fd.interpolate(fd.Constant(0.0), Q)
    u = fd.interpolate(fd.as_vector((1e-6, 0.0)), V)
    u_b = fd.Function(Q).interpolate(fd.sqrt(fd.inner(u,u)))

    H = S-B

    m = fd.Function(Q).interpolate(fd.Constant(0.0))

    p_i = fd.Function(Q).interpolate(fd.Constant(ice_density * gravity) * H)
    phi_m = fd.Function(Q).interpolate(fd.Constant(water_density * gravity) * B)
    phi_0 = fd.Function(Q).interpolate(p_i + phi_m)
    bc = fd.DirichletBC(Q, phi_m, 1)

    h_init   = fd.interpolate(fd.Constant(0.0001), Q)
    S_init   = fd.interpolate(fd.Constant(0.001), CR)
    phi_init = fd.Function(Q).interpolate(phi_0)

    return {
        "thickness": H,
        "bed": B,
        "sliding_speed": u_b,
        "melt_rate": m,
        "h_init": h_init,
        "S_init": S_init,
        "phi_init": phi_init,
        "phi_m": phi_m,
        "p_i": p_i,
        "phi_0": phi_0,
        "dirichlet_bcs": [bc],
        "englacial_void_ratio": 0.0,
        "sheet_conductivity": 0.005,
        "out_dir": OUTDIR,
    }

def set_distributed_recharge(model, rate_m_per_s):
    model.m.interpolate(fd.Constant(rate_m_per_s))

def advance_to_steady(model, dt, *, rel_tol=1e-3, max_steps=5000, check_every=25):
    """Step forward until phi, N, h, and S all converge."""
    phi_prev = fd.Function(model.U).interpolate(model.phi)
    N_prev   = fd.Function(model.U).interpolate(model.N)
    h_prev   = fd.Function(model.U).interpolate(model.h)
    S_prev   = fd.Function(model.CR).interpolate(model.S)

    for k in range(1, max_steps + 1):
        model.step(dt)

        if k % check_every == 0:
            model.update_phi()

            rphi = fd.norm(model.phi - phi_prev) / (fd.norm(model.phi) + 1e-30)
            rN   = fd.norm(model.N   - N_prev  ) / (fd.norm(model.N  ) + 1e-30)
            rh   = fd.norm(model.h   - h_prev  ) / (fd.norm(model.h  ) + 1e-30)
            rS   = fd.norm(model.S   - S_prev  ) / (fd.norm(model.S  ) + 1e-30)

            sim_days = k * dt / 86400
            print(f"iter {k} ({sim_days:.0f}d): rphi={rphi:.3e}, rN={rN:.3e}, rh={rh:.3e}, rS={rS:.3e}")

            phi_prev.assign(model.phi)
            N_prev.assign(model.N)
            h_prev.assign(model.h)
            S_prev.assign(model.S)

            if rphi < rel_tol and rN < rel_tol and rh < rel_tol and rS < rel_tol:
                return k

    print("WARNING: hit max_steps without steady convergence.")
    return max_steps

def width_averaged_Nx(model, nbins=None):
    """Bin-average N over the y-direction. Returns (x_centers, N_mean, mask)."""
    V = model.U
    coords = model.mesh.coordinates.dat.data_ro
    x = coords[:, 0]
    Nvals = model.N.dat.data_ro

    if nbins is None:
        nbins = nx
    bins = np.linspace(0.0, Lx, nbins+1)
    idx  = np.digitize(x, bins) - 1
    Nx = np.zeros(nbins)
    count = np.zeros(nbins, dtype=int)
    for i, val in zip(idx, Nvals):
        if 0 <= i < nbins:
            Nx[i] += val
            count[i] += 1
    mask = count > 0
    Nx[mask] /= count[mask]
    xc = 0.5 * (bins[:-1] + bins[1:])
    return xc, Nx, mask

def make_ribbon_plot(cases=None, outfile="shmip_ribbons.png"):
    """Generate ribbon plot from saved checkpoints via shmip_summary_3d.py."""
    import subprocess, sys
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shmip_summary_3d.py")
    if cases is None:
        cases = CASES_ALL
    checkpoints, labels = [], []
    for tag in cases:
        p = os.path.join(CHK_DIR, f"{tag}.h5")
        if os.path.exists(p):
            checkpoints.append(p)
            labels.append(tag)
        else:
            print(f"  (skipping {tag}: no checkpoint at {p})")
    if not checkpoints:
        print("No checkpoints found — run the simulation first.")
        return
    cmd = ([sys.executable, script]
           + checkpoints
           + ["--labels"] + labels
           + ["--outfile", outfile])
    print(f"Generating ribbon plot → {outfile}")
    subprocess.run(cmd, check=True)


def save_checkpoint(model, tag):
    os.makedirs(CHK_DIR, exist_ok=True)
    fname = os.path.join(CHK_DIR, f"{tag}.h5")
    with fd.CheckpointFile(fname, "w") as chk:
        chk.save_mesh(model.mesh)
        for name in ("h", "S", "phi", "pfo", "N", "N_cr", "h_cr", "S_alpha", "p_w", "q_s", "q_s_mag", "Q_ch"):
            if hasattr(model, name):
                chk.save_function(getattr(model, name), name=name)
    return fname


def try_load_prev_state(prev_tag, mesh):
    """Load phi/h/S from a saved checkpoint to warm-start the next case."""
    fname = os.path.join(CHK_DIR, f"{prev_tag}.h5")
    if not os.path.exists(fname):
        return None
    try:
        Q  = fd.FunctionSpace(mesh, "CG", 1)
        CR = fd.FunctionSpace(mesh, "CR", 1)
        with fd.CheckpointFile(fname, "r") as chk:
            saved_mesh = chk.load_mesh()
            phi_saved = chk.load_function(saved_mesh, name="phi")
            h_saved   = chk.load_function(saved_mesh, name="h")
            S_saved   = chk.load_function(saved_mesh, name="S")
        phi_out = fd.Function(Q);  phi_out.dat.data[:] = phi_saved.dat.data_ro[:]
        h_out   = fd.Function(Q);  h_out.dat.data[:]   = h_saved.dat.data_ro[:]
        S_out   = fd.Function(CR); S_out.dat.data[:]   = S_saved.dat.data_ro[:]
        print(f"  loaded warm start from checkpoint {fname}")
        return {"phi": phi_out, "h": h_out, "S": S_out}
    except Exception as e:
        print(f"  Warning: could not load checkpoint {fname}: {e}")
        return None

def main(cases=None, plot_ribbons=False):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    if plot_ribbons and not cases:
        make_ribbon_plot(outfile="shmip_ribbons.png")
        return

    if cases is None or len(cases) == 0:
        cases = list(A_RATES.keys())

    mesh = build_mesh()
    all_x = None
    curves = {}

    prev_state = None

    if cases and cases[0] != "A1":
        pred_idx = CASES_ALL.index(cases[0]) - 1
        if pred_idx >= 0:
            prev_state = try_load_prev_state(CASES_ALL[pred_idx], mesh)

    for tag in cases:
        print(f"\n=== Running {tag} ===")
        rate = A_RATES[tag]

        model_inputs = make_model_inputs(mesh)
        model = SubglacialHydrologyModel(mesh, **model_inputs)

        if prev_state is not None:
            model.phi.assign(prev_state["phi"])
            model.phi_prev.assign(prev_state["phi"])
            model.h.assign(prev_state["h"])
            model.S.assign(prev_state["S"])
            model.update_phi()
            model.update_h_cr()
            model.update_S_alpha()
            print(f"  warm-started from previous case")

        for frac in [0.1, 0.25, 0.5, 1.0]:
            set_distributed_recharge(model, frac * rate)
            for _ in range(50):
                model.step(dt)
            model.update_phi()

        model.compute_flux_fields()

        set_distributed_recharge(model, rate)
        iters = advance_to_steady(model, dt, rel_tol=rel_tol, max_steps=max_steps, check_every=check_every)
        print(f"Converged in {iters} checks.")

        model.update_phi()
        model.compute_flux_fields()

        prev_state = {
            "phi": fd.Function(model.U).assign(model.phi),
            "h":   fd.Function(model.U).assign(model.h),
            "S":   fd.Function(model.CR).assign(model.S),
        }

        cpath = save_checkpoint(model, tag)
        print(f"{tag}: wrote checkpoint → {cpath}")

        x, Nx, mask = width_averaged_Nx(model)
        if all_x is None:
            all_x = x
        csv_path = os.path.join(CSV_DIR, f"{tag}_Nx.csv")
        np.savetxt(csv_path, np.c_[x, Nx], delimiter=",", header="x,N", comments="")
        print(f"{tag}: saved Nx curve → {csv_path}")
        curves[tag] = Nx

    plt.figure(figsize=(8, 4.5))
    for tag in cases:
        plt.plot(all_x/1000.0, curves[tag], label=tag)
    plt.xlabel("x (km)")
    plt.ylabel("Width-averaged effective pressure N (Pa)")
    plt.title("SHMIP Suite A: N(x) at steady state")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=200)
    print(f"Saved figure → {PLOT_PNG}")

    np.savez(os.path.join(OUTDIR, "A_curves.npz"), x=all_x, **curves)

    if plot_ribbons:
        make_ribbon_plot(outfile="shmip_ribbons.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="Subset like A3, A3 A4 A5, A3+, or A2-A5")
    ap.add_argument("--plot-ribbons", action="store_true",
                    help="Generate shmip_ribbons.png from saved checkpoints "
                         "(skips simulation if --cases is not given)")
    args = ap.parse_args()
    main(expand_cases(args.cases), plot_ribbons=args.plot_ribbons)
