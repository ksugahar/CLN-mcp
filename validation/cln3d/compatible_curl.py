"""Iterative element-by-element curl inverse with shared-dof corrections.

Unlike the archived greedy sweep, every local projection may correct shared
coefficients. No assembled global curl system or target potential is used.
The final physical curl gate is mandatory. This is not a one-pass claim.
"""
import time

import ngsolve as ng
import numpy as np

from .element_recovery import CurlSweep


class CompatibleSweep(CurlSweep):
    def __init__(self, mesh, order, bonus=8, boundary='', sequence='forward'):
        super().__init__(mesh,order,bonus,boundary)
        self.sequence=sequence
        self.blocks=[]
        for el,ids,c,dual in self.elements:
            free=self.free[ids]
            # The right inverse projects onto the local affine constraint
            # set with homogeneous boundary dofs excluded, not frozen traces.
            cf=c[:,free]
            if cf.shape[1]:
                u,s,vt=np.linalg.svd(cf,full_matrices=False)
                keep=s>s.max()*1e-11
                inverse=(vt[keep].T/s[keep])@u[:,keep].T
            else:
                inverse=np.zeros((0,len(c)))
            self.blocks.append((el,ids,free,c,dual,inverse))

    def recover(self,field,tolerance=1e-10,max_sweeps=4000):
        if not 0 < tolerance <= 1e-8 or max_sweeps < 1:
            raise ValueError('Positive sweep limit and tolerance <= 1e-8 required')
        begin=time.perf_counter()
        lfi=ng.SymbolicLFI(ng.curl(self.test)*field,bonus_intorder=self.bonus)
        targets=[]
        local_defects=[]
        for el,ids,free,c,dual,inverse in self.blocks:
            load=np.array(lfi.CalcElementVector(el.GetFE(),el.GetTrafo()))
            targets.append(dual@load)
            local_defects.append(np.linalg.norm(c[:,free]@inverse@targets[-1]-targets[-1]))
        denominator=np.sqrt(sum(float(v@v) for v in targets))
        if not np.isfinite(denominator) or denominator == 0:
            raise ValueError('A finite nonzero compatible curl field is required')
        local_defect=float(np.linalg.norm(local_defects)/denominator)
        if local_defect>1e-8:
            raise RuntimeError(f'Locally incompatible curl or boundary constraints: {local_defect}')
        value=np.zeros(self.space.ndof)
        order=np.arange(len(self.blocks))
        if self.sequence=='reverse':order=order[::-1]
        elif self.sequence=='shuffled':order=np.random.default_rng(19).permutation(order)
        elif self.sequence!='forward':raise ValueError('Unsupported element sequence')
        history=[]
        for sweep in range(1,max_sweeps+1):
            for j in order:
                el,ids,free,c,dual,inverse=self.blocks[j]
                value[ids[free]]+=inverse@(targets[j]-c@value[ids])
            if sweep==1 or sweep%10==0 or sweep==max_sweeps:
                residual=np.sqrt(sum(float((c@value[ids]-target)@(c@value[ids]-target))
                    for (_,ids,_,c,_,_),target in zip(self.blocks,targets)))/denominator
                history.append([sweep,float(residual)])
                if residual<tolerance:break
        gf=ng.GridFunction(self.space)
        gf.vec.FV().NumPy()[:]=value
        measure=ng.dx(bonus_intorder=self.bonus)
        error=ng.curl(gf)-field
        field_error=float(ng.Integrate(error*error*measure,self.mesh)/ng.Integrate(field*field*measure,self.mesh))**.5
        info=dict(sweeps=sweep,curl_fit_residual=float(residual),relative_curl_field_error=field_error,
                  recovery_seconds=time.perf_counter()-begin,recovery_ndof=self.space.ndof,
                  sequence=self.sequence,history=history,global_matrix_assembled=False,
                  shared_dofs_corrected=True, requested_tolerance=tolerance,
                  max_sweeps=max_sweeps, converged=bool(residual<tolerance))
        info['local_representability_defect']=local_defect
        info['boundary_dof_max']=float(np.max(abs(value[~self.free]),initial=0))
        if not np.isfinite(field_error) or residual>=tolerance or field_error>1e-8:
            exc=RuntimeError(f'Iterative element curl recovery rejected: {info}')
            exc.info=info
            raise exc
        return gf,info
