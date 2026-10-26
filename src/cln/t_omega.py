"""Cauer ladder network (CLN) of a conductor by the T-Omega method, with expansion point s0.

Field problem (quasi-static, conductor only, T = H in the conductor):

    find T :  a(T, N) + (s - s0) m(T, N) = <drive, N>
    a(u, v) = (1/sigma curl u, curl v) + s0 (mu u, v)
    m(u, v) = (mu u, v)

The impedance is expanded around s = s0 as the Cauer continued fraction

    Z(s) = R0 + 1/( 1/((s-s0) L1) + 1/( R2 + 1/( 1/((s-s0) L3) + ... )))

Recursion (one formula for every s0 and both boundary conditions):

    T0       : a(T0, N) = <drive, N>                     (unit voltage)
    R_{2n}   = 1 / a(T_{2n}, T_{2n})
    H_{2n+1} = H_{2n-1} + R_{2n} T_{2n},   L_{2n+1} = m(H_{2n+1}, H_{2n+1})
    T_{2n+2} : a(T_{2n+2}, N) = -m(H_{2n+1}, N)/L_{2n+1} + a(T_{2n}, N)

The last term makes T_{2n+2} = T_{2n} + T' (accumulation). For s0 = 0 the
operator a is singular (gradients); the right-hand side is made consistent by
the Omega gauge  (mu grad Omega, grad psi) = -(mu T, grad psi).
"""
import math
from numbers import Integral
from ngsolve import (HCurl, H1, BilinearForm, LinearForm, GridFunction, Integrate,
                     CF, Cross, curl, grad, BND, specialcf, Projector, TaskManager, BitArray)
from ._integration import electromagnetic_measures


class TOmegaCLN:
    """CLN of a straight conductor segment of length ``length``.

    Parameters
    ----------
    mesh : ngsolve.Mesh   conductor-only mesh
    sigma, mu : float     conductivity [S/m], permeability [H/m]
    radius, length : float  used for the current-drive boundary value and per-length scaling
    s0 : float            expansion point [1/s] (real, >= 0)
    bc : 'Neumann' (voltage drive, E_t = 1 V/m on the lateral surface)
         'Dirichlet' (current drive, n x T = I/(2 pi a), I = 1 A)
    lateral : name of the lateral boundary
    order : FE order
    solver : 'sparsesolv' (zero-shift ICCG, default) or 'direct'.
             Positive s0 requires explicit 'direct' (unregularized sparse Cholesky).
             At zero shift, 'direct' uses
             Cholesky of a + eps m with refinement, eps = eps_rel/(mu sigma a^2).
             ICCG never switches to direct refinement. An unconverged original-
             system residual raises; choose solver='direct' explicitly to use
             direct refinement. solve_diagnostics records the selected backend.
    integration_order : common quadrature order for solves and norms, default 2*order+6
    """

    def __init__(self, mesh, sigma, mu, radius, length, s0=0.0, bc="Neumann",
                 lateral="conductorBND", order=3, solver="sparsesolv",
                 tol=1e-10, maxiter=20, eps_rel=1e-3, verbose=False, integration_order=None):
        if bc not in ("Neumann", "Dirichlet"):
            raise ValueError("bc must be 'Neumann' or 'Dirichlet'")
        for label, value in (("sigma", sigma), ("mu", mu), ("radius", radius),
                             ("length", length), ("tol", tol), ("eps_rel", eps_rel)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        if not math.isfinite(s0) or s0 < 0:
            raise ValueError("s0 must be real, finite and nonnegative")
        if solver not in ("sparsesolv", "direct"):
            raise ValueError("solver must be 'sparsesolv' or 'direct'")
        if s0 > 0 and solver != "direct":
            raise ValueError("s0 > 0 requires explicit solver='direct'; positive-shift sparsesolv is not supported")
        for label, value in (("maxiter", maxiter), ("order", order)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.mesh, self.sigma, self.mu = mesh, sigma, mu
        self.radius, self.length = radius, length
        self.s0, self.bc, self.lateral, self.order = float(s0), bc, lateral, order
        self.tol, self.maxiter, self.verbose = tol, maxiter, verbose
        self.eps_rel = eps_rel
        self._iccg_tolerance = None
        self.iterations = []
        self.solve_diagnostics = []
        self._dx, self._ds, self.integration_order = electromagnetic_measures(order, integration_order)

        kw = dict(dirichlet=lateral) if bc == "Dirichlet" else {}
        self.fes = HCurl(mesh, order=order, nograds=(self.s0 == 0), type1=True, **kw)
        T, N = self.fes.TnT()
        self.a = BilinearForm(self.fes)
        self.a += (1/sigma*curl(T)*curl(N) + self.s0*mu*T*N)*self._dx
        with TaskManager():
            self.a.Assemble()

        self.solver = solver
        if self.s0 > 0:
            self._inv = self.a.mat.Inverse(self.fes.FreeDofs(), "sparsecholesky")
        elif solver == "sparsesolv":
            # radia-ngsolve SparseSolv: shifted ICCG on the singular (consistent) system
            from radia.sparsesolv_ngsolve import SparseSolvSolver
            # Keep a two-order margin to the outer true-residual target and
            # preserve the historical 1e-12 inner target at default tol=1e-10.
            self._iccg_tolerance = min(self.tol * 1e-2, 1e-12)
            if self._iccg_tolerance <= 0:
                raise ValueError("tol is too small for a positive ICCG working tolerance")
            self._inv = SparseSolvSolver(self.a.mat, method="ICCG", freedofs=self.fes.FreeDofs(),
                                         tol=self._iccg_tolerance, maxiter=3000, shift=1.0)
            self._inv.auto_shift = True
            self._inv.diagonal_scaling = True
        else:
            # a is singular on the (lowest-order) gradients. Factorize a + eps*m and use
            # iterative refinement on a itself: for a consistent rhs this converges to the
            # solution whose gradient part is m-orthogonal to the kernel (= gauged solution).
            self._inv = self._regularized_inverse()
        if self.s0 == 0:
            # Omega must be free wherever T is free:
            #   Neumann  -> no Dirichlet (only the constant is undetermined)
            #   Dirichlet-> Omega = 0 on the lateral surface (n x T is fixed there)
            kwg = dict(dirichlet=lateral) if bc == "Dirichlet" else {}
            self.fesG = H1(mesh, order=order, **kwg)
            u, v = self.fesG.TnT()
            g = BilinearForm(self.fesG)
            g += mu*grad(u)*grad(v)*self._dx
            gauge_free = BitArray(self.fesG.FreeDofs())
            if bc == "Neumann":
                # Fix one scalar potential value, removing only the constant.
                # A mass regularization also perturbs nonconstant gauge modes.
                gauge_free[0] = False
            with TaskManager():
                g.Assemble()
            self._ginv = g.mat.Inverse(gauge_free, "sparsecholesky")

    # ------------------------------------------------------------------ helpers
    def _regularized_inverse(self):
        self._eps = self.eps_rel/(self.mu*self.sigma*self.radius**2)
        T, N = self.fes.TnT()
        reg = BilinearForm(self.fes)
        reg += (1/self.sigma*curl(T)*curl(N) + self._eps*self.mu*T*N)*self._dx
        with TaskManager():
            reg.Assemble()
        return reg.mat.Inverse(self.fes.FreeDofs(), "sparsecholesky")

    def _solve(self, f, gf):
        r = f.vec.CreateVector()
        r.data = f.vec - self.a.mat*gf.vec
        r.data = Projector(self.fes.FreeDofs(), True)*r
        x = gf.vec.CreateVector()
        x.data = self._inv*r
        P = Projector(self.fes.FreeDofs(), True)
        res = r.CreateVector()
        r0 = max(r.Norm(), 1e-300)
        iccg_iterations = self._inv.last_result.iterations if self.s0 == 0 and self.solver == "sparsesolv" else 0
        if self.s0 == 0:
            for k in range(min(self.maxiter, 2) if self.solver == "sparsesolv" else self.maxiter):
                res.data = r - self.a.mat*x
                res.data = P*res
                rel = res.Norm()/r0
                if rel < self.tol:
                    break
                x.data += self._inv*res
                if self.solver == "sparsesolv":
                    iccg_iterations += self._inv.last_result.iterations
        # Check the actual final iterate, including after the last refinement.
        # SparseSolv/direct operator application alone is not convergence proof.
        res.data = P*(r - self.a.mat*x)
        rel = res.Norm()/r0
        backend = "sparsecholesky" if self.s0 > 0 else self.solver
        self.solve_diagnostics.append({"relative_residual": rel, "tolerance": self.tol,
                                       "iccg_tolerance": self._iccg_tolerance,
                                       "s0": self.s0, "requested_solver": self.solver, "backend": backend,
                                       "converged": math.isfinite(rel) and rel <= self.tol})
        if self.s0 == 0 and self.solver == "direct":
            self.iterations.append((k, rel))
        elif self.s0 == 0:
            self.iterations.append(iccg_iterations)
        if not math.isfinite(rel) or rel > self.tol:
            raise RuntimeError(f"CLN field solve did not converge: relative residual {rel:.3e} > {self.tol:.3e}")
        gf.vec.data += x
        return gf

    def _gauge(self, T):
        # Coefficient expressions reference their GridFunction operands. Own a
        # coefficient snapshot before retaining T or T+grad(Omega); never Set()
        # the combined expression into another space (that changes the gauge).
        snapshot = GridFunction(T.space)
        snapshot.vec.data = T.vec
        T = snapshot
        if self.s0 > 0:
            return T
        u, v = self.fesG.TnT()
        f = LinearForm(self.fesG)
        f += -self.mu*grad(v)*T*self._dx
        with TaskManager():
            f.Assemble()
        om = GridFunction(self.fesG)
        om.vec.data = self._ginv*f.vec
        # Preserve the gradient exactly as a coefficient expression; interpolating
        # into a different HCurl space can reintroduce a gradient residual.
        return T + grad(om)

    def _stage_fields(self, solved):
        """Detach returned H/J expressions from mutable solver work vectors."""
        current_source = GridFunction(solved.space)
        current_source.vec.data = solved.vec
        return self._gauge(solved), curl(current_source)

    def Pj(self, J):
        return Integrate(1/self.sigma*J*J*self._dx, self.mesh)/self.length

    def Wm(self, H):
        return Integrate(self.mu*H*H*self._dx, self.mesh)/self.length

    # ------------------------------------------------------------------ stages
    def initial(self):
        T, N = self.fes.TnT()
        gf = GridFunction(self.fes)
        f = LinearForm(self.fes)
        n = specialcf.normal(3)
        if self.bc == "Dirichlet":
            Hin = 1/(2*math.pi*self.radius)
            # Dual interpolation preserves the HCurl boundary moments. The
            # default local projection biases the full order-2 space's current.
            # NGSolve 6.2.2607 gives nonfinite values with dual+nograds=True;
            # retain the reduced-space projection at exactly zero shift.
            gf.Set(Hin*Cross(CF((0, 0, 1)), n), BND,
                   definedon=self.mesh.Boundaries(self.lateral), dual=self.s0 > 0)
        else:
            f += -(Cross(CF((0, 0, 1)), N.Trace())*n)*self._ds(self.lateral)
        with TaskManager():
            f.Assemble()
        gf = self._solve(f, gf)
        T0, J0 = self._stage_fields(gf)
        if self.bc == "Dirichlet":                 # I = 1 A  ->  scale to unit voltage
            R = self.Pj(J0) + self.s0*self.Wm(T0)
            T0, J0 = T0*(1/R), J0*(1/R)
        return T0, J0

    def step(self, Tprev, Jprev, H, L):
        T, N = self.fes.TnT()
        f = LinearForm(self.fes)
        f += -self.mu*H/L*N*self._dx
        f += self.s0*self.mu*Tprev*N*self._dx      # a(T_{n-2}, N): accumulation
        f += 1/self.sigma*Jprev*curl(N)*self._dx
        with TaskManager():
            f.Assemble()
        gf = self._solve(f, GridFunction(self.fes))
        return self._stage_fields(gf)

    def run(self, nmax=8, *, keep_modes=False):
        """Return (R, L, Pj_ratio) as dicts keyed by stage index (per unit length)."""
        if isinstance(nmax, bool) or not isinstance(nmax, Integral) or nmax < 0:
            raise ValueError("nmax must be a nonnegative integer")
        R, L, ratio = {}, {}, {}
        self.field_modes = []
        self.current_modes = []
        T, J = self.initial()
        H = None
        for n in range(0, nmax + 1, 2):
            if n > 0:
                T, J = self.step(T, J, H, L[n-1])
            pj, sl = self.Pj(J), self.s0*self.Wm(T)
            if not math.isfinite(pj+sl) or pj+sl <= 0 or pj < 0 or sl < 0:
                raise ValueError("invalid stage energy; recurrence exhausted or numerical failure")
            R[n], ratio[n] = 1/(pj + sl), pj/(pj + sl)
            if self.verbose:
                print(f"R{n} = {R[n]:.6e}")
            if n + 1 > nmax:
                break
            H = T*R[n] if H is None else H + T*R[n]
            if keep_modes:
                self.field_modes.append(H)
                current = R[n]*J if not self.current_modes else self.current_modes[-1]+R[n]*J
                self.current_modes.append(current)
            L[n+1] = self.Wm(H)
            if not math.isfinite(L[n+1]) or L[n+1] <= 0:
                raise ValueError("invalid stage inductance; recurrence exhausted or numerical failure")
            if self.verbose:
                print(f"L{n+1} = {L[n+1]:.6e}")
        return R, L, ratio

    def field_model(self, nstages=3, *, audit_tolerance=1e-7):
        """Certified inductance-terminated admittance with retained H-field modes.

        States are shunt-inductor currents, H=sum(H_odd[i]*x[i]). A resistance
        termination eliminates a storage mode and is NOT this field model.
        Positive s0 changes the basis, not the physical mu-weighted energy.
        """
        if isinstance(nstages, bool) or not isinstance(nstages, Integral) or nstages < 1:
            raise ValueError("nstages must be a positive integer")
        from .magnetic import MagneticFieldCLN
        r, ell, _ = self.run(2*nstages-1, keep_modes=True)
        return MagneticFieldCLN(self, r, ell, audit_tolerance=audit_tolerance)
