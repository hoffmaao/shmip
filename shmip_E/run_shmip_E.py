import os
import argparse
import numpy as np
import firedrake as fd
import gmsh
import tempfile
import matplotlib.pyplot as plt
from hydropack.models.glads import Glads2DModel
from hydropack.constants import pcs as default_pcs

# ----------------------------- Suite-E constants -----------------------------
Lx = 6_000.0        # 6 km along-flow domain (per SHMIP Fig. 1b/eqs.)
EPS = 1e-16

# A6 = 5.79e-7 m/s  -> Suite E uses 2×A6
M_E = 7.93e-11#2.0 * 5.79e-7                        # JoG text (≈ 100 mm d^-1)
# Table 4: bed parameter γ per run
E_GAMMA = {"E1": 0.05, "E2": 0.0, "E3": -0.1, "E4": -0.5, "E5": -0.7}

# Reference parameter for outline, per website text
GAMMA_BENCH = 0.05

# Time stepping / convergence
dt          = 5.0 
rel_tol     = 5e-4
max_steps   = 20000
check_every = 50

# Output folders
OUTDIR   = "outputs_E"
CSV_DIR  = os.path.join(OUTDIR, "csv")
CHK_DIR  = os.path.join(OUTDIR, "checkpoints")
PLOT_PNG = os.path.join(OUTDIR, "E_Nx.png")

# ----------------------------- SHMIP analytic defs ---------------------------
def surface_xy(x: float, y: float) -> float:
    """surface(x,y) in meters, x,y in meters (SHMIP definition)."""
    return 100.0*(x + 200.0)**0.25 + x/60.0 - (2.0e10)**0.25 + 1.0

def f_poly(x: float, gamma: float) -> float:
    """f(x,γ) using surface(6000,0) and x in meters."""
    s6 = surface_xy(6000.0, 0.0)
    return ((s6 - gamma*6000.0) / (6000.0**2)) * (x**2) + gamma * x

def g_y(y: float) -> float:
    """g(y) = 0.5e-6 * |y|^3."""
    return 0.5e-6 * abs(y)**3

def g_inv(s: float) -> float:
    """Inverse of g: y = (s / 0.5e-6)^(1/3), sign preserved (outline uses s>=0)."""
    if s <= 0.0:
        return 0.0
    return (s / 0.5e-6)**(1.0/3.0)

def h_fun(x: float, gamma: float) -> float:
    """h(x,γ) per SHMIP."""
    num = surface_xy(x, 0.0) - f_poly(x, gamma)
    den = surface_xy(x, 0.0) - f_poly(x, GAMMA_BENCH) + EPS
    return (-4.5*x/6000.0 + 5.0) * (num / den)

def outline_half_width(x: float) -> float:
    """yo(x) = g^{-1}((surface - f(x,γ_bench)) / (h(x,γ_bench)+eps))."""
    s = surface_xy(x, 0.0) - f_poly(x, GAMMA_BENCH)
    return g_inv(s / (h_fun(x, GAMMA_BENCH) + EPS))

def y_bottom_py(x: float) -> float:
    return -outline_half_width(x)

def y_top_py(x: float) -> float:
    return +outline_half_width(x)

# ----------------------------- mesh generation -------------------------------
def build_valley_mesh_gmsh(
    Lx: float,
    y_bottom,               # callable: float -> float
    y_top,                  # callable: float -> float
    nx_samples: int = 41,
    hmax: float = 400.0,
    hmin: float | None = 300.0,
    x_focus: float | None = 3000.0,
    refine_halfwidth: float = 1000.0,
):

    xs = np.linspace(0.0, Lx, int(nx_samples))
    yb = np.array([float(y_bottom(float(x))) for x in xs])
    yt = np.array([float(y_top(float(x)))    for x in xs])

    # allow zero width only at the terminus; forbid top<bottom anywhere
    gap = yt - yb
    tol = 1e-12
    if np.min(gap) < -1e-9:
        i = int(np.argmin(gap))
        raise ValueError(f"y_top<y_bottom at x={xs[i]:.3f} by {gap[i]:.3e} m")
    right_collapsed = abs(yt[-1] - yb[-1]) <= tol

    gmsh.initialize()
    try:
        gmsh.model.add("suite_E_valley")

        # --- points
        p_bot = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, hmax) for x, y in zip(xs, yb)]

        p_top = []
        for i, (x, y) in enumerate(zip(xs, yt)):
            if i == len(xs) - 1 and right_collapsed:
                # REUSE the bottom terminus point so curves share the same tag
                p_top.append(p_bot[-1])
            else:
                p_top.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, hmax))

        # --- curves
        l_bottom = gmsh.model.geo.addSpline(p_bot)            # 0 -> Lx along bottom
        l_top    = gmsh.model.geo.addSpline(p_top)            # 0 -> Lx along top
        l_left   = gmsh.model.geo.addLine(p_top[0], p_bot[0]) # top-left -> bottom-left

        if right_collapsed:
            # Close without a right wall; endpoints coincide by construction
            cloop = gmsh.model.geo.addCurveLoop([l_bottom, -l_top, l_left])
            l_right = None
        else:
            l_right = gmsh.model.geo.addLine(p_bot[-1], p_top[-1])  # bottom-right -> top-right
            cloop   = gmsh.model.geo.addCurveLoop([l_bottom, l_right, -l_top, l_left])

        surf = gmsh.model.geo.addPlaneSurface([cloop])

        # optional local refinement near x_focus on centreline
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

        # physical groups (facet tags)
        id_x0     = gmsh.model.addPhysicalGroup(1, [l_left]);    gmsh.model.setPhysicalName(1, id_x0, "x0")
        id_bottom = gmsh.model.addPhysicalGroup(1, [l_bottom]);  gmsh.model.setPhysicalName(1, id_bottom, "bottom")
        id_top    = gmsh.model.addPhysicalGroup(1, [l_top]);     gmsh.model.setPhysicalName(1, id_top, "top")
        if l_right is not None:
            id_xL = gmsh.model.addPhysicalGroup(1, [l_right]);   gmsh.model.setPhysicalName(1, id_xL, "xL")
        else:
            id_xL = -1
        id_dom   = gmsh.model.addPhysicalGroup(2, [surf]);       gmsh.model.setPhysicalName(2, id_dom, "valley")

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)

        fd_fd, fd_path = tempfile.mkstemp(suffix=".msh")
        os.close(fd_fd)
        gmsh.write(fd_path)
    finally:
        gmsh.finalize()

    mesh = fd.Mesh(fd_path)
    os.remove(fd_path)

    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    fd.triplot(mesh, axes=axes)
    axes.legend(loc="upper right");
    fig.savefig('mesh.png')




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

# ----------------------------- FD fields for bed/surface ----------------------
def fd_surface(mesh):
    Q = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)
    s = 100.0*(x + 200.0)**0.25 + x/60.0 - (2.0e10)**0.25 + 1.0
    return fd.interpolate(s, Q)

def fd_bed(mesh, gamma: float):
    """zb(x,y;γ) = f(x,γ) + g(y)*h(x,γ)"""
    Q = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)

    s6 = fd.Constant(surface_xy(6000.0, 0.0))
    f  = ((s6 - gamma*6000.0)/6000.0**2) * x**2 + gamma*x

    # h(x,γ)
    num = fd_surface(mesh) - f
    s_b = fd_surface(mesh) - ((s6 - GAMMA_BENCH*6000.0)/6000.0**2) * x**2 - GAMMA_BENCH*x
    h   = (-4.5*x/6000.0 + 5.0) * (num / (s_b + fd.Constant(EPS)))

    g = 0.5e-6 * fd.sqrt((y)**2)**3
    zb = f + g*h
    return fd.interpolate(zb, Q)

# ----------------------------- model inputs -----------------------------------
def make_model_inputs(mesh, gamma):
    Q = fd.FunctionSpace(mesh, "CG", 1)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    CR = fd.FunctionSpace(mesh, "CR", 1)

    zs = fd_surface(mesh)
    zb = fd_bed(mesh, gamma)
    H  = fd.interpolate(fd.max_value(zs - zb, fd.Constant(1.0)), Q)   # keep H ≥ 1 m

    fig, ax = plt.subplots()
    colors_beta = fd.tripcolor(H, axes=ax)
    fig.colorbar(colors_beta, ax=ax, fraction=0.012, pad=0.04);
    fig.savefig('thickness.png')

    fig, ax = plt.subplots()
    colors_beta = fd.tripcolor(zb, axes=ax)
    fig.colorbar(colors_beta, ax=ax, fraction=0.012, pad=0.04);
    fig.savefig('bed.png')

    fig, ax = plt.subplots()
    colors_beta = fd.tripcolor(zs, axes=ax)
    fig.colorbar(colors_beta, ax=ax, fraction=0.012, pad=0.04);
    fig.savefig('surface.png')

    # Basal speed (Table 3)
    u = fd.interpolate(fd.as_vector((1.0e-6, 0.0)), V)
    u_b  = fd.Function(Q).interpolate(fd.sqrt(fd.inner(u, u)))

    # Initial conditions
    h_init = fd.interpolate(fd.Constant(0.01), Q)      # 1 cm sheet
    S_init = fd.interpolate(fd.Constant(0.000001), CR)       # no channels initially
    phi_init = fd.interpolate(fd.Constant(0.000001), Q)      # 1 cm sheet

    # Potentials/pressures
    pcs = dict(default_pcs)
    #pcs["ev"] = 1.0e-3    # Suite E,F englacial void fraction (Table 3)
    p_i   = fd.Function(Q).interpolate(fd.Constant(pcs["rho_ice"]  * pcs["g"]) * H)
    phi_m = fd.Function(Q).interpolate(fd.Constant(pcs["rho_water"]* pcs["g"]) * zb)
    phi_0 = fd.Function(Q).interpolate(p_i + phi_m)


    # Suite E: no Dirichlet by default (optional outlet patch is commented in mesh builder)
    bcs = fd.DirichletBC(Q,phi_m,[1])

    return {
        "mesh": mesh,
        "thickness": H,
        "bed": zb,
        "u_b": u_b,
        "m": fd.interpolate(fd.Constant(0.0), Q),  # set to M_E later
        "h_init": h_init,
        "S_init": S_init,
        "phi_prev": phi_init,
        "phi_init": phi_init,
        "d_bcs": bcs,
        "phi_m": phi_m,
        "p_i": p_i,
        "phi_0": phi_0,
        "constants": pcs,
        "out_dir": OUTDIR,
    }

# ----------------------------- stepping utilities -----------------------------
def advance_to_steady(model, dt, *, rel_tol=1e-3, max_steps=5000, check_every=25, check_h=False):
    phi_prev = fd.Function(model.U).interpolate(model.phi)
    N_prev   = fd.Function(model.U).interpolate(model.N)
    if check_h:
        h_prev = fd.Function(model.U).interpolate(model.h)

    for k in range(1, max_steps+1):
        model.step(dt)
        if k % check_every == 0:
            model.update_phi()
            rphi = float(fd.norm(model.phi - phi_prev) / (fd.norm(model.phi) + 1e-30))
            rN   = float(fd.norm(model.N   - N_prev  ) / (fd.norm(model.N  ) + 1e-30))
            if check_h:
                rh = float(fd.norm(model.h - h_prev) / (fd.norm(model.h) + 1e-30))
                print(f"iter {k}: rphi={rphi:.3e}, rN={rN:.3e}, rh={rh:.3e}")
            else:
                print(f"iter {k}: rphi={rphi:.3e}, rN={rN:.3e}")
            phi_prev.assign(model.phi); N_prev.assign(model.N)
            if check_h: h_prev.assign(model.h)
            if (rphi < rel_tol) and (rN < rel_tol) and (not check_h or rh < rel_tol):
                return k
    print("WARNING: hit max_steps without steady convergence.")
    return max_steps

# ----------------------------- width-averaged N(x) ----------------------------
def cell_areas(mesh):
    Vdg = fd.FunctionSpace(mesh, "DG", 0)
    A   = fd.Function(Vdg).interpolate(fd.CellVolume(mesh))
    return A.dat.data_ro.copy()

def cell_centroids(mesh):
    VdgV = fd.VectorFunctionSpace(mesh, "DG", 0)
    X    = fd.Function(VdgV).interpolate(fd.SpatialCoordinate(mesh)).dat.data_ro
    return X[:,0].copy(), X[:,1].copy()

def width_averaged_Nx(model, nbins=120, band_half=100.0):
    """Area-weighted average of N(x) in a 200 m centre band."""
    N0 = fd.Function(fd.FunctionSpace(model.mesh, "DG", 0)).project(model.N).dat.data_ro
    xc, yc = cell_centroids(model.mesh)
    A      = cell_areas(model.mesh)

    mask   = (np.abs(yc) <= band_half)
    bins   = np.linspace(0.0, Lx, nbins+1)
    ix     = np.clip(np.searchsorted(bins, xc[mask], side="right") - 1, 0, nbins-1)

    Nsum = np.zeros(nbins); Asum = np.zeros(nbins)
    np.add.at(Nsum, ix, N0[mask]*A[mask]); np.add.at(Asum, ix, A[mask])
    Nx = np.divide(Nsum, np.where(Asum>0, Asum, 1.0))
    xmid = 0.5*(bins[:-1] + bins[1:])
    return xmid, Nx

# ----------------------------- I/O -------------------------------------------
def save_checkpoint(model, tag):
    os.makedirs(CHK_DIR, exist_ok=True)
    fname = os.path.join(CHK_DIR, f"{tag}.h5")
    with fd.CheckpointFile(fname, "w") as chk:
        chk.save_mesh(model.mesh)
        for name in ("h","S","phi","pfo","N","N_cr","h_cr","S_alpha","p_w","q_s","q_s_mag","Q_ch"):
            if hasattr(model, name):
                chk.save_function(getattr(model, name), name=name)
    return fname

# ----------------------------- case expansion --------------------------------
CASES_ALL = ["E1","E2","E3","E4","E5"]

def expand_cases(sel):
    if not sel: return CASES_ALL
    out = []
    for s in sel:
        s = s.upper()
        if s.endswith("+") and s[:-1] in CASES_ALL:
            i = CASES_ALL.index(s[:-1]); out.extend(CASES_ALL[i:])
        elif "-" in s:
            a,b = s.split("-",1)
            if a in CASES_ALL and b in CASES_ALL:
                i,j = CASES_ALL.index(a), CASES_ALL.index(b)
                lo,hi = min(i,j), max(i,j); out.extend(CASES_ALL[lo:hi+1])
        elif s in CASES_ALL:
            out.append(s)
    seen=set(); return [c for c in out if not (c in seen or seen.add(c))]

# ----------------------------- main ------------------------------------------
def main(cases=None):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    if not cases:
        cases = CASES_ALL

    mesh = build_mesh()
    all_x = None
    curves = {}

    for tag in cases:
        gamma = E_GAMMA[tag]
        print(f"\n=== Suite E: {tag} (γ={gamma}), m = {M_E:.3e} m/s uniform ===")

        # fresh model per case
        model_inputs = make_model_inputs(mesh, gamma)
        model = Glads2DModel(model_inputs)

        # Uniform recharge (Suite E)
        model.m.interpolate(fd.Constant(M_E))

        # Gentle spin-up
        for f in np.arange(0,1,100):
            for _ in range(30):
                model.step(min(1.0, dt))
                model.update_phi()

        model.compute_flux_fields()

        # If your model exposes these:
        if hasattr(model, "compute_flux_fields"):
            model.compute_flux_fields()

        # Advance to steady
        iters = advance_to_steady(model, dt, rel_tol=rel_tol,
                                  max_steps=max_steps, check_every=check_every, check_h=True)
        print(f"{tag}: steady checks = {iters}")

        # final derived updates
        model.update_phi()
        model.compute_flux_fields()

        # save checkpoints
        cpath = save_checkpoint(model, tag)
        print(f"{tag}: wrote checkpoint → {cpath}")

        # Width-averaged N(x) over 200 m centre band
        x, Nx = width_averaged_Nx(model, nbins=120, band_half=100.0)
        if all_x is None: all_x = x
        curves[tag] = Nx
        np.savetxt(os.path.join(CSV_DIR, f"{tag}_Nx.csv"), np.c_[x, Nx],
                   delimiter=",", header="x,N", comments="")

    # Quick overlay
    plt.figure(figsize=(8.2, 4.6))
    for tag in cases:
        plt.plot(all_x/1000.0, curves[tag]/1e6, label=tag)
    plt.xlabel("x (km)"); plt.ylabel(r"Width-averaged $\bar N$ (MPa)")
    plt.title("SHMIP Suite E (centre 200 m band): $\\bar N(x)$")
    plt.grid(True, alpha=0.3); plt.legend(ncol=3, fontsize=9)
    plt.tight_layout(); plt.savefig(PLOT_PNG, dpi=200)
    print(f"Saved figure → {PLOT_PNG}")

    np.savez(os.path.join(OUTDIR, "E_curves.npz"), x=all_x, **curves)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="Subset like E3, E3 E4 E5, E3+, or E2-E5")
    args = ap.parse_args()
    main(expand_cases(args.cases))
