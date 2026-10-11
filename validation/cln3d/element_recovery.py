"""Checked local recovery adapters for the unchanged zero-shift A-T recursion.

The trace system stays global. NGSolve owns static condensation and local
harmonic extension. A separate curl-only element sweep can replace each
potential before the next CLN stage; failure is explicit, never a fallback.
"""
import time

import ngsolve as ng
import numpy as np
import scipy.sparse as sp

from ._iccg import _project_out, gradient_kernel
from .wire_pencils import factor


class RecoveryRejected(RuntimeError):
    def __init__(self, info):
        self.info = info
        super().__init__(f'One-pass CLN curl recovery incompatible: {info}')


class Condensed:
    def __init__(self, space, weight, bonus, condense=True):
        begin = time.perf_counter()
        self.space = space
        self.condense = condense
        self.free = np.array(list(space.FreeDofs()), dtype=bool)
        self.trace = np.array(list(space.FreeDofs(coupling=condense)), dtype=bool)
        u, v = space.TnT()
        self.form = ng.BilinearForm(space, symmetric=True, condense=condense)
        self.form += weight*ng.curl(u)*ng.curl(v)*ng.dx(bonus_intorder=bonus)
        self.form.Assemble()
        self.g, kfree = gradient_kernel(space)
        self.g = self.g[:, kfree]
        s = sp.csr_matrix(self.form.mat.CSR(), shape=(space.ndof,space.ndof)).copy()[self.trace][:, self.trace]
        gt = self.g[self.trace]
        gt = gt[:, np.asarray(gt.getnnz(axis=0)).ravel() > 0]
        defect = float(sp.linalg.norm(s@gt)/max(sp.linalg.norm(s)*sp.linalg.norm(gt), 1e-300))
        if not np.isfinite(defect) or defect > 1e-10:
            raise RuntimeError(f'Invalid condensed gradient kernel: {defect}')
        gg = gt@gt.T
        scale = float(abs(s.diagonal()).max()/gg.diagonal().max())
        self.solve_trace = factor(s+scale*gg)
        self.setup_seconds = time.perf_counter()-begin
        self.metadata = dict(full_free=int(self.free.sum()), trace_free=int(self.trace.sum()),
                             internal=int((self.free & ~self.trace).sum()),
                             gradient_defect=defect, mass_perturbation=0,condensation_enabled=condense)

    def solve(self, original, rhs):
        begin = time.perf_counter()
        b = rhs.FV().NumPy().copy()
        projected = _project_out(b[self.free], self.g[self.free])
        fraction = float(np.linalg.norm(projected-b[self.free])/np.linalg.norm(b[self.free]))
        if not np.isfinite(fraction) or fraction > 1e-8:
            raise RuntimeError(f'Incompatible condensed CLN load: {fraction}')
        r = rhs.CreateVector()
        r[:] = 0
        r.FV().NumPy()[self.free] = projected
        condensed = r.CreateVector()
        condensed.data = r
        if self.condense:
            condensed.data += self.form.harmonic_extension_trans*r
        gf = ng.GridFunction(self.space)
        gf.vec[:] = 0
        gf.vec.FV().NumPy()[self.trace] = self.solve_trace(condensed.FV().NumPy()[self.trace])
        trace_seconds = time.perf_counter()-begin
        begin = time.perf_counter()
        if self.condense:
            gf.vec.data += self.form.harmonic_extension*gf.vec
            gf.vec.data += self.form.inner_solve*r
        local_seconds = time.perf_counter()-begin
        residual = rhs.CreateVector()
        residual.data = original.mat*gf.vec-rhs
        defect = float(np.linalg.norm(residual.FV().NumPy()[self.free])/np.linalg.norm(b[self.free]))
        if not np.isfinite(defect) or defect > 1e-6:
            raise RuntimeError(f'Original A-T equation residual: {defect}')
        return gf, dict(solver='NGSolve sparsecholesky'+(' condensed + element harmonic recovery' if self.condense else ' full system control'),
                        original_rhs_residual=defect, explicit_residual=defect,
                        rhs_kernel_fraction=fraction, trace_seconds=trace_seconds,
                        local_seconds=local_seconds, **self.metadata)


class CurlSweep:
    """Full HCurl curl inverse, fixing previously visited shared dofs.

    This is the high-order prototype applied to actual curved CLN modes.
    No target potential coefficients enter the local right hand sides.
    """
    def __init__(self, mesh, order, bonus, boundary):
        self.mesh = mesh
        self.bonus = bonus
        self.space = ng.HCurl(mesh, order=order, nograds=False, dirichlet=boundary)
        self.free = np.array(list(self.space.FreeDofs()), dtype=bool)
        u, v = self.space.TnT()
        self.test = v
        bfi = ng.SymbolicBFI(ng.curl(u)*ng.curl(v), bonus_intorder=bonus)
        self.elements = []
        for el in self.space.Elements():
            ids = np.array(self.space.GetDofNrs(el), dtype=int)
            k = np.array(bfi.CalcElementMatrix(el.GetFE(), el.GetTrafo()))
            w, q = np.linalg.eigh((k+k.T)/2)
            keep = w > w.max()*1e-11
            self.elements.append((el, ids, np.sqrt(w[keep])[:, None]*q[:, keep].T,
                                  q[:, keep].T/np.sqrt(w[keep])[:, None]))

    def recover(self, field):
        begin = time.perf_counter()
        lfi = ng.SymbolicLFI(ng.curl(self.test)*field, bonus_intorder=self.bonus)
        value = np.zeros(self.space.ndof)
        fixed = ~self.free.copy()
        norm2, residual2 = 0., 0.
        for el, ids, c, dual in self.elements:
            load = np.array(lfi.CalcElementVector(el.GetFE(), el.GetTrafo()))
            target = dual@load
            free = ~fixed[ids]
            rhs = target-c@value[ids]
            if free.any():
                value[ids[free]] += np.linalg.lstsq(c[:, free], rhs, rcond=1e-11)[0]
            fixed[ids] = True
            norm2 += float(target@target)
        # A later element can only assign previously unset dofs. Check all
        # elements again and also integrate the physical vector-field error.
        for el, ids, c, dual in self.elements:
            load = np.array(lfi.CalcElementVector(el.GetFE(), el.GetTrafo()))
            error = c@value[ids]-dual@load
            residual2 += float(error@error)
        gf = ng.GridFunction(self.space)
        gf.vec.FV().NumPy()[:] = value
        measure = ng.dx(bonus_intorder=self.bonus)
        error = ng.curl(gf)-field
        field_error = float(ng.Integrate(error*error*measure, self.mesh)/
                            ng.Integrate(field*field*measure, self.mesh))**.5
        info = dict(curl_fit_residual=(residual2/norm2)**.5, relative_curl_field_error=field_error,
                    recovery_seconds=time.perf_counter()-begin, recovery_ndof=self.space.ndof)
        if not np.isfinite(field_error) or max(field_error, info['curl_fit_residual']) > 1e-8:
            raise RecoveryRejected(info)
        return gf, info
