import os
import math
import argparse
import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
from hydropack.models.glads import Glads2DModel
from hydropack.constants import pcs as default_pcs

pcs = dict(default_pcs)

# ---------- experiment config ----------
# Domain (ice-sheet rectangle used by SHMIP)
Lx = 100e3  # 100 km
Ly = 20e3   # 20 km
nx = 200
ny = 40

# Cases A1..A6, distributed recharge rates.
# Values here are examples; set to your chosen SHMIP rates (m/s).
# e.g. 2.5 mm/a, 10 mm/d, ..., 50 mm/d converted to m/s
A_RATES = {
    "A1": 7.93e-11,   # 2.5 mm/a -> m/s
    "A2": 1.59e-9,             # 2 mm/d  -> m/s
    "A3": 5.79e-9,             # etc.
    "A4": 2.5e-8,
    "A5": 4.5e-8,
    "A6": 5.79e-7,
}

# Timestep + steady criteria
dt = 60*30/8             # 30 minutes
max_steps = 20000       # safety
rel_tol = 5e-4          # steady if relative change below this for N and phi
check_every = 50        # assess steady state every N steps

# output
OUTDIR = "outputs"
PLOT_PNG = os.path.join(OUTDIR, "A_Nx.png")
CSV_DIR = os.path.join(OUTDIR, "csv")
CHK_DIR = os.path.join(OUTDIR, "checkpoints")


# ---------- helpers ----------
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
    # dedupe, preserve order
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def make_model_inputs(mesh):
    """
    Replace this stub with your existing builder that returns the
    'model_inputs' dict hydropack expects. It MUST include:
      mesh, thickness (H), bed (B), u_b, m, h_init, phi_init, phi_m, p_i, phi_0, d_bcs, S_init, constants
    Here we create minimal placeholders you should wire to your real fields.
    """
    Q = fd.FunctionSpace(mesh, "CG", 1)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    CR = fd.FunctionSpace(mesh, "CR", 1)
    
    x, y = fd.SpatialCoordinate(mesh)

    # --- YOU: replace with real initial/forcings/topo ---
    S = fd.interpolate(6*(fd.sqrt(x + 5000) - fd.sqrt(5000.0)) + 1, Q)         # 1 km ice
    B = fd.interpolate(fd.Constant(0.0), Q)            # flat bed
    u = fd.interpolate(fd.as_vector((1e-6, 0.0)), V)  # 100 m/yr-ish along x (replace with your actual)
    u_b = fd.Function(Q).interpolate(fd.sqrt(fd.inner(u,u)))

    #u_b.assign(0.0)  # or set from data
    H = S-B

    # distributed melt (we overwrite per A-case later)
    m = fd.Function(Q).interpolate(fd.Constant(0.0))

    # Initial conditions
    h_init = fd.interpolate(fd.Constant(0.0001), Q)      # 1 cm sheet
    S_init = fd.interpolate(fd.Constant(0.001), CR)       # no channels initially
    phi_init = fd.interpolate(fd.Constant(0.0001), Q)     # to be solved
    # potentials/pressures
    p_i = fd.Function(Q).interpolate(fd.Constant(pcs['rho_ice'] * pcs['g']) * H)
    phi_m = fd.Function(Q).interpolate(fd.Constant(pcs['rho_water'] * pcs['g']) * B)
    phi_0 = fd.Function(Q).interpolate(p_i + phi_m)
    bc = fd.DirichletBC(Q,phi_m,1)


    # Dirichlet BCs set later on model; just pass empty for init
    d_bcs = [bc]



    model_inputs = {
        "mesh": mesh,
        "h_init": h_init,
        "S_init": S_init,
        "thickness": H,
        "bed": B,
        "u_b": u_b,
        "m": m,
        "phi_prev": phi_init,
        "phi_init": phi_init,
        "d_bcs": d_bcs,
        "phi_m": phi_m,
        "p_i": p_i,
        "phi_0": phi_0,
        "constants": pcs,
        "out_dir": OUTDIR,
    }
    return model_inputs

def set_distributed_recharge(model, rate_m_per_s):
    # model.m is a CG function; set uniform rate
    model.m.interpolate(fd.Constant(rate_m_per_s))

def advance_to_steady(model, dt, *, rel_tol=1e-3, max_steps=5000, check_every=25, check_h=False):
    """
    March forward the *coupled* system until steady state:
      ‖phi - phi_prev‖ / (‖phi‖ + eps) < rel_tol  AND
      ‖N   - N_prev  ‖ / (‖N  ‖ + eps) < rel_tol
    (optionally also check h). Evaluated every 'check_every' steps.
    """
    phi_prev = fd.Function(model.U).interpolate(model.phi)
    N_prev   = fd.Function(model.U).interpolate(model.N)
    if check_h:
        h_prev   = fd.Function(model.U).interpolate(model.h)

    for k in range(1, max_steps + 1):
        # advance the *coupled* model (phi + sheet/channel ODEs)
        model.step(dt)


        if k % check_every == 0:
            # phi_solver.step already updates derived fields,
            # but this ensures N_cr, dphi_ds_cr, pfo are consistent
            model.update_phi()

            rphi = fd.norm(model.phi - phi_prev) / (fd.norm(model.phi))
            rN   = fd.norm(model.N   - N_prev  ) / (fd.norm(model.N  ))

            if check_h:
                rh = fd.norm(model.h - h_prev) / (fd.norm(model.h))
                print(f"iter {k}: rphi={rphi:.3e}, rN={rN:.3e}, rh={rh:.3e}")
            else:
                print(f"iter {k}: rphi={rphi:.3e}, rN={rN:.3e}")

            # refresh snapshots
            phi_prev.assign(model.phi)
            N_prev.assign(model.N)
            if check_h:
                h_prev.assign(model.h)

            # convergence test
            if (rphi < rel_tol) and (rN < rel_tol) and (not check_h or rh < rel_tol):
                return k

    print("WARNING: hit max_steps without steady convergence.")
    return max_steps

def width_averaged_Nx(model, nbins=200):
    """
    Compute width-averaged effective pressure N(x):
      For each x-bin, average N over all vertices in that bin.
    Returns (x_centers, N_mean)
    """
    V = model.U
    coords = model.mesh.coordinates.dat.data_ro
    x = coords[:, 0]
    # sample N at dofs (CG1 → vertex dofs)
    Nvals = model.N.dat.data_ro

    bins = np.linspace(0.0, Lx, nbins+1)
    idx  = np.digitize(x, bins) - 1  # 0..nbins-1
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

def save_checkpoint(model, tag):
    os.makedirs(CHK_DIR, exist_ok=True)
    fname = os.path.join(CHK_DIR, f"{tag}.h5")
    with fd.CheckpointFile(fname, "w") as chk:
        chk.save_mesh(model.mesh)
        for name in ("h", "S", "phi", "pfo", "N", "N_cr", "h_cr", "S_alpha", "p_w", "q_s", "q_s_mag", "Q_ch"):
            if hasattr(model, name):
                chk.save_function(getattr(model, name), name=name)
    return fname

def main(cases=None):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    # pick subset if requested
    if cases is None or len(cases) == 0:
        cases = list(A_RATES.keys())

    mesh = build_mesh()
    all_x = None
    curves = {}

    for tag in cases:
        print(f"\n=== Running {tag} ===")
        rate = A_RATES[tag]

        # fresh model per case
        model_inputs = make_model_inputs(mesh)
        model = Glads2DModel(model_inputs)

        for f in np.arange(0,1,100):
            set_distributed_recharge(model, f*rate)
            for _ in range(30):
                model.step(min(300.0, dt))  # small settling
                model.update_phi()

        model.compute_flux_fields()


        set_distributed_recharge(model, rate)
        iters = advance_to_steady(model, dt, rel_tol=rel_tol, max_steps=max_steps, check_every=25, check_h=True)
        print(f"Converged in {iters} checks.")

        # final derived updates
        model.update_phi()
        model.compute_flux_fields()

        # save checkpoint
        cpath = save_checkpoint(model, tag)
        print(f"{tag}: wrote checkpoint → {cpath}")

        # diagnostics: width-averaged N(x)
        x, Nx, mask = width_averaged_Nx(model)
        if all_x is None:
            all_x = x
        # save curve to CSV
        csv_path = os.path.join(CSV_DIR, f"{tag}_Nx.csv")
        np.savetxt(csv_path, np.c_[x, Nx], delimiter=",", header="x,N", comments="")
        print(f"{tag}: saved Nx curve → {csv_path}")
        curves[tag] = Nx

    # plot all curves together
    plt.figure(figsize=(8, 4.5))
    for tag in cases:
        plt.plot(all_x/1000.0, curves[tag], label=tag)  # x in km
    plt.xlabel("x (km)")
    plt.ylabel("Width-averaged effective pressure N (Pa)")
    plt.title("SHMIP Suite A: N(x) at steady state")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=200)
    print(f"Saved figure → {PLOT_PNG}")

    # also save NPZ bundle
    np.savez(os.path.join(OUTDIR, "A_curves.npz"), x=all_x, **curves)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="Subset like A3, A3 A4 A5, A3+, or A2-A5")
    args = ap.parse_args()
    main(expand_cases(args.cases))