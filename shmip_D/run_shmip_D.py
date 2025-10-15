# hydropack/solvers/hs_solver.py
# Implicit RK (Irksome) solver for the GLADS sheet/channel ODEs.
# h in U (CG1):  dh/dt = u_b*(h_r - h)/l_r - A*h*N^3
# S in CR (CR1): dS/dt = mask * ( |Q*phi_s| + |l_c*q*phi_s| )/(rho_i L) - A*S*N_cr^3
#   where Q = -k_c S^alpha |phi_s + eps|^delta * phi_s,
#         q = -k   h_cr^alpha |phi_s + eps|^delta * phi_s
#
# This mirrors the FEniCS (Downs) GLADS RHS with a smooth |.| regularization.

import firedrake as fd
from irksome import Dt, TimeStepper, RadauIIA


class HSSolver:
    def __init__(self, model):
        self.m = model

        # --- constants (accept floats or fd.Constant) ---
        def C(x, default=None):
            if x is None: x = default
            return x if isinstance(x, fd.Constant) else fd.Constant(x)

        pcs = model.pcs
        self.A       = C(pcs.get("A"),        1.0e-24)
        self.l_r     = C(pcs.get("l_r"),      10.0)
        self.h_r     = C(pcs.get("h_r"),      0.01)
        self.rho_i   = C(pcs.get("rho_ice", pcs.get("rho_i")), 910.0)
        self.L       = C(pcs.get("L"),        3.34e5)
        self.k       = C(pcs.get("k"),        1.0)
        self.k_c     = C(pcs.get("k_c"),      1.0)
        self.l_c     = C(pcs.get("l_c"),      1.0)
        self.alpha   = C(pcs.get("alpha"),    5.0/4.0)
        self.delta   = C(pcs.get("delta"),    1.0/2.0)
        self.phi_reg = C(pcs.get("phi_reg"),  1e-15)  # for |.| smoothing

        # optional edge mask (CR): 1=allow opening, 0=clamped
        if hasattr(model, "mask") and isinstance(model.mask, fd.Function) and model.mask.function_space() == model.CR:
            self.mask_cr = model.mask
        else:
            self.mask_cr = fd.Function(model.CR, name="mask_cr").assign(1.0)

        # time handles for Irksome
        self._t  = fd.Constant(float(model.t))
        self._dt = fd.Constant(0.0)

        # lower bounds for SNESVI
        self._h_lb = fd.Function(model.U).assign(0.0)
        self._S_lb = fd.Function(model.CR).assign(0.0)

        # build steppers with a dummy dt; we’ll set dt each call to step()
        self._stepper_h = self._build_h_stepper()
        self._stepper_S = self._build_S_stepper()

    # ---- helpers -------------------------------------------------------------

    def _abs_reg(self, x):
        # smooth |x| = sqrt(x^2 + eps^2)
        return fd.sqrt(x*x + self.phi_reg*self.phi_reg)

    # h RHS on U: u_b*(h_r - h)/l_r - A*h*N^3
    def _rhs_h(self, h):
        m = self.m
        return m.u_b * (self.h_r - h) / self.l_r - self.A * h * (m.N**3)

    # S RHS on CR: mask*Xi/(rho_i*L) - A*S*N_cr^3
    def _rhs_S(self, S):
        m = self.m
        phi_s = m.dphi_ds_cr     # (CR) derivative of potential along edges
        hmid  = m.h_cr           # (CR) sheet thickness at edge midpoints
        Nmid  = m.N_cr           # (CR) effective pressure at edge midpoints

        # |phi_s|^delta
        abs_phi = self._abs_reg(phi_s)**self.delta

        # channel and sheet fluxes
        Q = - self.k_c * (S**self.alpha)   * abs_phi * phi_s
        q = - self.k   * (hmid**self.alpha)* abs_phi * phi_s

        # dissipation term Xi = |Q*phi_s| + |l_c*q*phi_s|
        Xi = self._abs_reg(Q*phi_s) + self._abs_reg(self.l_c*q*phi_s)

        v_open  = self.mask_cr * Xi / (self.rho_i * self.L)
        v_close = self.A * S * (Nmid**3)
        return v_open - v_close

    # ---- Irksome steppers ----------------------------------------------------

    def _build_h_stepper(self):
        m = self.m
        v = fd.TestFunction(m.U)
        h = m.h
        Fh = fd.inner(Dt(h), v) * fd.dx - fd.inner(self._rhs_h(h), v) * fd.dx
        ts = TimeStepper(
            Fh, RadauIIA(1), self._t, self._dt, h,
            solver_parameters={
                "snes_type": "vinewtonrsls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-9,
                "snes_atol": 1e-12,
                "snes_max_it": 50,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "mat_type": "aij",
            },
        )
        ts.stage_bounds = (self._h_lb, None)   # h >= 0
        return ts

    def _build_S_stepper(self):
        m = self.m
        w = fd.TestFunction(m.CR)
        S = m.S
        FS = fd.inner(Dt(S), w) * fd.dx - fd.inner(self._rhs_S(S), w) * fd.dx
        ts = TimeStepper(
            FS, RadauIIA(1), self._t, self._dt, S,
            solver_parameters={
                "snes_type": "vinewtonrsls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-9,
                "snes_atol": 1e-12,
                "snes_max_it": 50,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "mat_type": "aij",
            },
        )
        ts.stage_bounds = (self._S_lb, None)   # S >= 0
        return ts

    # ---- advance one macro step ---------------------------------------------

    def step(self, dt):
        """Advance h then S by dt with phi fixed (GLADS splitting)."""
        m = self.m

        # set Irksome time knobs
        self._dt.assign(float(dt))
        self._t.assign(float(m.t))

        # --- advance h ---
        self._stepper_h.advance()
        # update edge quantities that depend on h
        m.update_h_cr()

        # --- advance S ---
        # make sure CR drivers for S are current (phi is fixed during GLADS split)
        m.update_N_cr()
        m.update_dphi_ds_cr()
        self._stepper_S.advance()

        # derived fields and time
        m.update_S_alpha()
        m.t = float(self._t)
