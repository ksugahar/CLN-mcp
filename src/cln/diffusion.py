"""Physical CLN for scalar diffusion-type PDEs (current/flux-source driven RC ladder).

    -div(a grad u) + b du/dt = f(r) F(t)      in Omega,   u = 0 on the Dirichlet part

Heat conduction is a = k [W/(m K)], b = rho c [J/(m^3 K)], u = T [K], f = q_s
(normalized so that int f dOmega = 1, i.e. F(t) is the injected power [W]).
Other instances: mass diffusion (a = D, b = 1), groundwater flow (a = hydraulic
conductivity, b = specific storage), 1/mu-sigma magnetic diffusion (a = 1/mu, b = sigma).

Algorithm (Tanimoto, Sugahara, Takahashi, Matsuo, IEEJ SA-25-014, 2025):

    T^_{-1} = 0,  q^_0 = f
    for n = 0, 1, ...
        solve  -div(a grad Tt_2n) = q^_2n
        R_2n   = int a |grad Tt_2n|^2            ("thermal resistance" [K/W])
        T^_2n  = Tt_2n / R_2n,   T^_{2n+1} = T^_{2n-1} + T^_2n
        C_2n+1 = int b (T^_{2n+1})^2             ("thermal capacity" [J/K])
        q^_{2n+2} = q^_2n - b T^_{2n+1} / C_2n+1

Ladder (current source F into node 1): R_2n from node 2n+1 to the reference,
C_2n+1 in series between node 2n+1 and node 2n+3. Port quantity is the
average u_port = int f u dOmega, conjugate to F in the quadratic-storage balance, and

    Z(s) = u_port/F = 1/(1/R0 + 1/(1/(s C1) + 1/(1/R2 + 1/(1/(s C3) + ...)))).

For a 2-D extrusion of total depth `length`, R is divided by the depth and C
is multiplied by it. A 3-D mesh already includes its volume: use length=1.
For heat the quadratic storage is J K, not actual heat content in J.
energy_model() retains all capacitors and provides the field/circuit certificate;
the legacy impedance below is resistance-terminated unless explicitly changed.
"""
import math
from numbers import Integral

from ngsolve import (H1, BilinearForm, LinearForm, GridFunction, Integrate, grad, dx,
                     TaskManager, CF, ET, IntegrationRule)


class DiffusionCLN:
    def __init__(self, mesh, a, b, source, dirichlet=".*", order=3, length=1.0,
                 integration_order=None):
        """a, b: CoefficientFunction or float. source: CoefficientFunction f(r) (any scale;
        it is normalized to unit integral). dirichlet: boundary regex with u = 0."""
        if not math.isfinite(length) or length <= 0:
            raise ValueError("length must be positive and finite")
        self.integration_order = 2*order + 6 if integration_order is None else integration_order
        if (isinstance(self.integration_order, bool) or
                not isinstance(self.integration_order, Integral) or self.integration_order < 1):
            raise ValueError("integration_order must be a positive integer")
        types = {1: (ET.SEGM,), 2: (ET.TRIG, ET.QUAD),
                 3: (ET.TET, ET.PRISM, ET.PYRAMID, ET.HEX)}[mesh.dim]
        # The same quadrature metric must define solves, normalization and
        # certificates; mismatched rules destroy the discrete energy identity.
        self._dx = dx(intrules={et: IntegrationRule(et, order=int(self.integration_order)) for et in types})
        self.mesh = mesh
        source = CF(source)
        total = self._integrate(source)
        if not math.isfinite(total) or total == 0:
            raise ValueError("source must have a finite nonzero integral")
        self.mesh, self.a, self.b, self.length = mesh, a, b, length
        self.fes = H1(mesh, order=order, dirichlet=dirichlet)
        if all(self.fes.FreeDofs()):
            raise ValueError("a coercive Dirichlet boundary is required; pure Neumann null modes are not implemented")
        u, v = self.fes.TnT()
        k = BilinearForm(self.fes, symmetric=True)
        k += a*grad(u)*grad(v)*self._dx
        with TaskManager():
            k.Assemble()
        self._inv = k.mat.Inverse(self.fes.FreeDofs(), "sparsecholesky")
        self.f = source*(1.0/total)

    def _integrate(self, coefficient):
        return Integrate(coefficient*self._dx, self.mesh)

    def _poisson(self, q):
        v = self.fes.TestFunction()
        rhs = LinearForm(self.fes)
        rhs += q*v*self._dx
        with TaskManager():
            rhs.Assemble()
        g = GridFunction(self.fes)
        g.vec.data = self._inv*rhs.vec
        return g

    def run(self, nstages=6, keep_modes=False):
        """Return dict(R=[R0, R2, ...], C=[C1, C3, ...]) scaled to the extrusion depth.

        Optional T_modes are odd, capacity-orthogonal modes (capacitor-drop
        coordinates). T_even_modes reconstruct fields from node coordinates.
        Both retain the unit-depth normalization.
        """
        _stage_count(nstages)
        R, C, modes, even_modes = [], [], [], []
        q = self.f
        That_prev = None
        for _ in range(nstages):
            tt = self._poisson(q)
            r2n = self._integrate(self.a*grad(tt)*grad(tt))
            if not math.isfinite(r2n) or r2n <= 0:
                raise ValueError("nonpositive or nonfinite stage resistance; recurrence exhausted or invalid coefficients")
            th = GridFunction(self.fes)
            th.vec.data = (1.0/r2n)*tt.vec
            if keep_modes:
                even = GridFunction(self.fes)
                even.vec.data = th.vec
                even_modes.append(even)
            if That_prev is not None:
                th.vec.data += That_prev.vec
            c = self._integrate(self.b*th*th)
            if not math.isfinite(c) or c <= 0:
                raise ValueError("nonpositive or nonfinite stage capacity; recurrence exhausted or invalid coefficients")
            R.append(r2n/self.length)
            C.append(c*self.length)
            if keep_modes:
                modes.append(th)
            q = q - self.b*th*(1.0/c)
            # snapshot: q references `th`, which is not mutated afterwards
            That_prev = th
        out = {"R": R, "C": C}
        if keep_modes:
            out["T_modes"] = modes
            out["T_even_modes"] = even_modes
        return out

    def energy_model(self, nstages=6, audit_tolerance=1e-8):
        """Derive and certify the full capacitor-terminated RC field model.

        Uses native FE integrals, never a dense full-order matrix. Reconstruct
        temperature with model.reconstruct(node_temperatures); capacitor drops
        multiply odd modes. For heat the quadratic storage is J K, and the
        separate linear_storage_content in model.balance is reconstructed heat
        content in J. Material a,b must be positive, real scalar coefficients.
        These continuum material preconditions are supplied by the caller and
        are not validated throughout the mesh; the certificate checks only the
        retained subspace, not global material positivity.
        """
        import numpy as np
        from .energy import _certified_prefix
        out = self.run(nstages, keep_modes=True)
        E, H = out["T_even_modes"], out["T_modes"]
        d = self.length
        kg = np.array([[d*self._integrate(self.a*grad(e)*grad(f))
                        for f in E] for e in E])
        mg = np.array([[d*self._integrate(self.b*e*f) for f in E] for e in E])
        mh = np.array([[d*self._integrate(self.b*e*f) for f in H] for e in H])
        port = np.array([self._integrate(self.f*e) for e in E])
        weights = np.array([d*self._integrate(self.b*e) for e in E])
        model = _certified_prefix(out["R"], out["C"], kg, mg, port, mh,
                                  storage_weights=weights, audit_tolerance=audit_tolerance)
        n = int(model.R.size)
        model.field_modes = E[:n]
        model.flux_modes = [-self.a*r*grad(e) for r, e in zip(model.R, E[:n])]
        from .initial import project_metric

        def project(field):
            if not isinstance(field, GridFunction) or field.space != self.fes:
                raise ValueError("initial_field must be a GridFunction in this model's FE space")
            coefficients = field.vec.FV().NumPy()
            if np.iscomplexobj(coefficients) or not np.all(np.isfinite(coefficients)):
                raise ValueError("initial_field must be finite and real")
            fixed = ~np.array(list(self.fes.FreeDofs()), dtype=bool)
            if np.any(coefficients[fixed] != 0):
                raise ValueError("initial_field must satisfy homogeneous essential boundary conditions")
            mass = lambda u, v: d*self._integrate(self.b*u*v)
            moments = np.array([mass(e, field) for e in model.field_modes])
            result = project_metric(mg[:n, :n], moments, mass(field, field),
                                    lambda v: mass(field-model.reconstruct(v), field-model.reconstruct(v)),
                                    circuit_metric=model.M, tolerance=audit_tolerance)
            result.diagnostics["full_linear_storage_content"] = d*self._integrate(self.b*field)
            result.diagnostics["retained_linear_storage_content"] = float(model.storage_weights @ result.x0)
            return result

        model._initial_projector = project
        model.metadata = {"backend": "NGSolve_integrals", "requested_stages": nstages,
                          "effective_stages": n,
                          "stop_reason": "orthogonality_limit" if n < nstages else "requested_stages",
                          "extrusion_depth": d, "integration_order": int(self.integration_order),
                          "source_normalization": "unit_continuum_integral"}
        return model


def _stage_count(n):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("nstages must be a positive integer")
    return int(n)


def rc_ladder_impedance(R, C, s, nstages=None, termination="resistance"):
    """Z(s) of the current-source RC ladder.

    Legacy termination="resistance" omits the final capacitor. Use
    termination="capacitance" for the full energy/Galerkin model.
    """
    n = _stage_count(len(R) if nstages is None else nstages)
    if termination not in ("resistance", "capacitance"):
        raise ValueError("termination must be 'resistance' or 'capacitance'")
    nc = n if termination == "capacitance" else n - 1
    if n > len(R) or len(C) < nc:
        raise ValueError("not enough resistance/capacity elements for the requested termination")
    z = R[n - 1]
    if termination == "capacitance":
        z = z/(1 + s*C[n-1]*z)
    for i in range(n - 2, -1, -1):
        sc = s*C[i]
        z = 1/(1/R[i] + sc/(1 + sc*z))
    return z
