"""Compatible V/Phi Poisson steps with only scalar global unknowns.

The Biro potential equations use a hybrid-mixed discretization: discontinuous
HDiv flux and cell scalar are eliminated locally, leaving scalar facet traces.
This differs from the paper's nodal H1 realization and ensures that the flux
entering high-order local curl recovery is in the discrete div-free space.
Scope: simply connected, constant-material internal wire, zero expansion point.
Caller owns TaskManager. No global A-T vector solve or vector fallback.
"""
import time

import ngsolve as ng
import numpy as np
import scipy.sparse as sp

from .wire_pencils import factor


class ScalarPoisson:
    def __init__(self,mesh,order,magnetic,bonus=8):
        if order<2:
            raise ValueError('This compatible scalar route requires HCurl order >= 2')
        begin=time.perf_counter()
        self.mesh,self.bonus=mesh,bonus
        flux=ng.Discontinuous(ng.HDiv(mesh,order=order-1))
        cell=ng.L2(mesh,order=order-2)
        trace=ng.FacetFESpace(mesh,order=order-1,dirichlet='' if magnetic else '.*')
        self.space=flux*cell*trace
        q,u,uhat=self.space.TrialFunction()
        v,w,what=self.space.TestFunction()
        self.test=v
        normal=ng.specialcf.normal(3)
        self.form=ng.BilinearForm(self.space,condense=True,symmetric=True)
        self.form += (q*v-u*ng.div(v)-w*ng.div(q))*ng.dx(bonus_intorder=bonus)
        self.form += (uhat*v*normal+what*q*normal)*ng.dx(element_boundary=True,bonus_intorder=bonus)
        self.form.Assemble()
        # Independent matrix-free application of the uncondensed equations.
        self.original=ng.BilinearForm(self.space,nonassemble=True)
        self.original += (q*v-u*ng.div(v)-w*ng.div(q))*ng.dx(bonus_intorder=bonus)
        self.original += (uhat*v*normal+what*q*normal)*ng.dx(element_boundary=True,bonus_intorder=bonus)
        self.original.Assemble()
        self.original_free=np.array(list(self.space.FreeDofs()),dtype=bool)
        self.free=np.array(list(self.space.FreeDofs(coupling=True)),dtype=bool)
        if self.free[:flux.ndof+cell.ndof].any():
            raise RuntimeError('Hybrid condensation retained a global vector/cell unknown')
        if magnetic:
            self.free[np.flatnonzero(self.free)[0]]=False  # one scalar gauge, no mass shift
        s=sp.csr_matrix(self.form.mat.CSR(),shape=(self.space.ndof,self.space.ndof))
        self.solve_trace=factor(-s[self.free][:,self.free])
        self.setup_seconds=time.perf_counter()-begin
        self.metadata=dict(global_scalar_dofs=int(self.free.sum()),eliminated_cell_dofs=flux.ndof+cell.ndof,
                           fixed_scalar_dofs=trace.ndof-int(self.free.sum()),
                           global_factor_shape=[int(self.free.sum())]*2,
                           global_vector_solves=0,global_unknown='scalar facet potential',
                           role='Phi' if magnetic else 'V',mass_perturbation=0)

    def solve(self,particular):
        start=time.perf_counter()
        rhs=ng.LinearForm(self.space)
        rhs += particular*self.test*ng.dx(bonus_intorder=self.bonus)
        rhs.Assemble()
        condensed=rhs.vec.CreateVector()
        condensed.data=rhs.vec+self.form.harmonic_extension_trans*rhs.vec
        gf=ng.GridFunction(self.space)
        gf.vec[:]=0
        gf.vec.FV().NumPy()[self.free]=self.solve_trace(-condensed.FV().NumPy()[self.free])
        residual=rhs.vec.CreateVector()
        residual.data=self.form.mat*gf.vec-condensed
        b=condensed.FV().NumPy()[self.free]
        defect=float(np.linalg.norm(residual.FV().NumPy()[self.free])/np.linalg.norm(b))
        gf.vec.data += self.form.harmonic_extension*gf.vec
        gf.vec.data += self.form.inner_solve*rhs.vec
        residual.data=self.original.mat*gf.vec-rhs.vec
        original_defect=float(np.linalg.norm(residual.FV().NumPy()[self.original_free])/
                              np.linalg.norm(rhs.vec.FV().NumPy()[self.original_free]))
        flux=gf.components[0]
        measure=ng.dx(bonus_intorder=self.bonus)
        divnorm=float(ng.Integrate(ng.div(flux)**2*measure,self.mesh))**.5
        norm=float(ng.Integrate(flux*flux*measure,self.mesh))**.5
        relative_div=divnorm*.01/norm
        if not np.isfinite(original_defect) or max(defect,original_defect)>1e-6 or relative_div>1e-8:
            raise RuntimeError(f'Scalar Poisson residual/divergence failed: {defect}, {original_defect}, {relative_div}')
        return gf,dict(**self.metadata,scalar_trace_residual=defect,relative_divergence=relative_div,
                       original_scalar_residual=original_defect,
                       scalar_seconds=time.perf_counter()-start)
