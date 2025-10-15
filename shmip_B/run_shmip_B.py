import os
import argparse
import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt

from hydropack.models.glads import Glads2DModel
from hydropack.constants import pcs as default_pcs

# ---------------- domain & numerics ----------------
Lx, Ly = 100e3, 20e3
nx, ny = 141, 28

dt = 60*30/10            # 10 min
max_steps = 20000
rel_tol = 5e-4
check_every = 50

OUTDIR = "outputs"
CSV_DIR = os.path.join(OUTDIR, "csv")
CHK_DIR = os.path.join(OUTDIR, "checkpoints")
FIG = os.path.join(OUTDIR, "B_Nx.png")

# ---- Suite A reference rates (m/s) used to define B forcing ----
A1 = 7.93e-11
A5 = 4.5e-8         # target total, as in your Suite A setup

# Default B mapping (# of moulins). Adjust to match the SHMIP table you’re following.
B_MAP = {
    "B1": 1,
    "B2": 10,
    "B3": 20,
    "B4": 50,
    "B5": 100,
}

# ---------------- helpers ----------------
def build_mesh():
    return fd.RectangleMesh(nx, ny, Lx, Ly, quadrilateral=False)

def make_model_inputs(mesh):
    U  = fd.FunctionSpace(mesh, "CG", 1)
    CR = fd.FunctionSpace(mesh, "CR", 1)
    x, y = fd.SpatialCoordinate(mesh)

    S = fd.interpolate(6*(fd.sqrt(x + 5000) - fd.sqrt(5000.0)) + 1, U)           # 1 km ice
    B = fd.interpolate(fd.Constant(0.0), U)            # flat bed
    u_b = fd.interpolate(fd.as_vector((1e-6, 0.0))[0], U)  # 100 m/yr-ish along x (replace with your actual)
    #u_b.assign(0.0)  # or set from data
    H = S-B

    # recharge field (we’ll set it every step)
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
    # RectangleMesh: x=0 side is boundary id 1
    model.d_bcs = [fd.DirichletBC(model.U, 0.0, 1)]

def total_domain_area(mesh):
    one = fd.Constant(1.0)
    return float(fd.assemble(one*fd.dx(domain=mesh)))

def gaussian_blobs(mesh, points, sigma):
    """Return a CG1 function that is a sum of Gaussians centered at points; normalized to integrate to 1."""
    U = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    blobs = [fd.exp(-((x-xi)**2 + (y-yi)**2)/(2*sigma**2)) for (xi, yi) in points]
    if not blobs:
        f = fd.Function(U); f.assign(0.0); return f
    g = sum(blobs)
    mass = fd.assemble(g*fd.dx)
    g_norm = g / (mass + 1e-16)
    return fd.interpolate(g_norm, U)

def place_moulins(mesh, n, seed=1, margin_buffer=2e3):
    """Uniform random interior moulin locations (simple; replace with your SHMIP grid if you like)."""
    rng = np.random.default_rng(seed)
    # bounds
    xy = mesh.coordinates.dat.data_ro
    xmin, xmax = float(xy[:,0].min()), float(xy[:,0].max())
    ymin, ymax = float(xy[:,1].min()), float(xy[:,1].max())
    xmin += margin_buffer
    pts = np.c_[rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n)]
    return [tuple(p) for p in pts]

def build_moulin_recharge(model, n_moulins, Q_total, seed=1):
    """
    Returns a callable set_recharge() that sets model.m each step to:
      m = A1 (distributed background) + moulin blobs summing to Q_total / H_ref
    Units: if your model expects water thickness rate [m/s], we convert volumetric discharge to areal by dividing by domain area.
    """
    A = fd.FunctionSpace(model.mesh, "CG", 1)
    # background A1 (distributed)
    bg_rate = A1
    area = total_domain_area(model.mesh)

    # place moulins and build normalized blob function
    pts = place_moulins(model.mesh, n_moulins, seed=seed)
    sigma = float(model.mesh.cell_sizes.dat.data_ro.min())  # ~ cell size
    blob = gaussian_blobs(model.mesh, pts, sigma)

    # Convert Q_total [m^3/s] to an *areal* recharge [m/s] by spreading the volume over the normalized blob.
    # Because blob integrates to 1 over area, Q_total/area is the mean rate; multiplying by blob gives spatial pattern.
    q_areal = (Q_total / area) * blob

    def set_recharge():
        model.m.interpolate(fd.Constant(bg_rate) + q_areal)

    return set_recharge

def advance_to_steady(model):
    eps = 1e-12
    phi_prev = fd.Function(model.U).assign(model.phi)
    N_prev   = fd.Function(model.U).assign(model.N)
    for k in range(1, max_steps+1):
        model.step(dt)
        if k % check_every == 0:
            model.update_phi()
            rphi = fd.norm(model.phi - phi_prev) / (fd.norm(model.phi) + eps)
            rN   = fd.norm(model.N   - N_prev)   / (fd.norm(model.N)   + eps)
            print(f"iter {k}: rphi={rphi:.3e}, rN={rN:.3e}")
            phi_prev.assign(model.phi)
            N_prev.assign(model.N)
            if (rphi < rel_tol) and (rN < rel_tol):
                return k
    print("WARNING: hit max_steps without steady convergence.")
    return max_steps

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
        for name in ("h","S","phi","pfo","N","N_cr","h_cr","S_alpha","p_w","m", "q_s", "q_s_mag", "Q_ch"):
            if hasattr(model, name):
                chk.save_function(getattr(model, name), name=name)
    return path

# ---------------- main driver ----------------
def main(cases=None, seed=1):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    if cases is None or len(cases) == 0:
        cases = list(B_MAP.keys())

    mesh = build_mesh()
    # Total volumetric discharge used for moulins: match A5 total (rate × area)
    Q_total = A5 * total_domain_area(mesh)

    all_x = None
    curves = {}

    for tag in cases:
        n_moulins = B_MAP[tag]
        print(f"\n=== {tag}: {n_moulins} moulin(s) ===")

        model_inputs = make_model_inputs(mesh)
        model = Glads2DModel(model_inputs)
        apply_margin_bc(model)

        set_recharge = build_moulin_recharge(model, n_moulins, Q_total, seed=seed)

        # march to steady
        k = 0
        for k in range(1, max_steps+1):
            set_recharge()      # keep recharge applied
            model.step(dt)
            if k % check_every == 0:
                model.update_phi()
                # steady check
                rphi = fd.norm(model.phi - model.phi_prev) / (fd.norm(model.phi) + 1e-12)
                rN   = 0.0  # model.update_phi() already refreshed N; use phi-only or add N like in A
                print(f"iter {k}: rphi={rphi:.3e}")
                if rphi < rel_tol:
                    break
        print(f"{tag}: steady after {k} iterations")

        model.update_phi()

        # checkpoint + Nx curve
        cpath = save_checkpoint(model, tag)
        print(f"{tag}: wrote {cpath}")

        x, Nx = width_averaged_Nx(model)
        if all_x is None: all_x = x
        csv = os.path.join(CSV_DIR, f"{tag}_Nx.csv")
        np.savetxt(csv, np.c_[x, Nx], delimiter=",", header="x,N", comments="")
        print(f"{tag}: saved {csv}")
        curves[tag] = Nx

    # plot
    plt.figure(figsize=(8,4.5))
    for tag in cases:
        plt.plot(all_x/1000.0, curves[tag], label=f"{tag} ({B_MAP[tag]})")
    plt.xlabel("x (km)")
    plt.ylabel("Width-averaged effective pressure N (Pa)")
    plt.title("SHMIP Suite B: N(x) at steady state")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG, dpi=200)
    print(f"Saved figure → {FIG}")

    np.savez(os.path.join(OUTDIR, "B_curves.npz"), x=all_x, **curves)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=None, help="Subset like B1 B5")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    main(args.cases, args.seed)
