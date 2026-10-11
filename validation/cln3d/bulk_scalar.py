"""Original compatible scalar equations, paired native divergence/flux integration."""
import time
import ngsolve as ng
import numpy as np
from .scalar_potentials import ScalarPoisson
from .bulk_curl import integrals


class BulkScalar(ScalarPoisson):
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
        values=integrals(self.mesh,(ng.div(flux)**2,flux*flux),self.bonus)
        if not np.isfinite(values).all() or values[0]<0 or values[1]<=0:
            raise RuntimeError('Invalid scalar flux norms')
        divnorm,norm=np.sqrt(values)
        relative_div=divnorm*.01/norm
        if not np.isfinite(original_defect) or max(defect,original_defect)>1e-6 or relative_div>1e-8:
            raise RuntimeError(f'Scalar Poisson residual/divergence failed: {defect}, {original_defect}, {relative_div}')
        return gf,dict(**self.metadata,scalar_trace_residual=defect,relative_divergence=relative_div,
                       original_scalar_residual=original_defect,
                       scalar_seconds=time.perf_counter()-start)
