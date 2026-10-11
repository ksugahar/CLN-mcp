"""Colored element projections: disjoint free coefficients commute in a batch.

NGSolve retains orientation and curved element operators. Only the execution
order changes; no global curl matrix or vector factorization is introduced.
"""
import time
import ngsolve as ng
import numpy as np
from .compatible_curl import CompatibleSweep


class BatchedSweep(CompatibleSweep):
    def __init__(self,*args,**kwargs):
        start=time.perf_counter()
        super().__init__(*args,**kwargs)
        colors=[]
        occupied=[]
        for j,(_,ids,free,c,dual,inverse) in enumerate(self.blocks):
            support=set(ids[free].tolist())
            color=next((k for k,s in enumerate(occupied) if not support&s),len(colors))
            if color==len(colors):
                colors.append([]);occupied.append(set())
            colors[color].append(j);occupied[color].update(support)
        self.batches=[]
        for color in colors:
            groups={}
            for j in color:
                _,ids,free,c,dual,inverse=self.blocks[j]
                groups.setdefault((c.shape,inverse.shape),[]).append(j)
            for indices in groups.values():
                ids=np.stack([self.blocks[j][1] for j in indices])
                active=np.stack([self.blocks[j][1][self.blocks[j][2]] for j in indices])
                assert len(np.unique(active))==active.size
                self.batches.append((np.array(indices),ids,active,
                    np.stack([self.blocks[j][3] for j in indices]),
                    np.stack([self.blocks[j][5] for j in indices])))
        self.colors=len(colors)
        self.construction_seconds=time.perf_counter()-start

    def recover(self,field,tolerance=1e-10,max_sweeps=4000):
        if not 0<tolerance<=1e-8 or max_sweeps<1:
            raise ValueError('Positive sweep limit and tolerance <= 1e-8 required')
        begin=time.perf_counter()
        lfi=ng.SymbolicLFI(ng.curl(self.test)*field,bonus_intorder=self.bonus)
        targets=[];defects=[]
        for el,ids,free,c,dual,inverse in self.blocks:
            target=dual@np.array(lfi.CalcElementVector(el.GetFE(),el.GetTrafo()))
            targets.append(target)
            defects.append(np.linalg.norm(c[:,free]@inverse@target-target))
        denominator=np.sqrt(sum(float(v@v) for v in targets))
        if not np.isfinite(denominator) or denominator==0:
            raise ValueError('A finite nonzero compatible curl field is required')
        local_defect=float(np.linalg.norm(defects)/denominator)
        if local_defect>1e-8:
            raise RuntimeError(f'Locally incompatible curl or boundary constraints: {local_defect}')
        batches=[(ids,active,c,inverse,np.stack([targets[j] for j in indices]))
                 for indices,ids,active,c,inverse in self.batches]
        target_seconds=time.perf_counter()-begin
        value=np.zeros(self.space.ndof);history=[];check_seconds=0.
        start=time.perf_counter()
        for sweep in range(1,max_sweeps+1):
            for ids,active,c,inverse,target in batches:
                defect=target-np.einsum('bij,bj->bi',c,value[ids],optimize=False)
                value[active]+=np.einsum('bij,bj->bi',inverse,defect,optimize=False)
            if sweep==1 or sweep%10==0 or sweep==max_sweeps:
                check=time.perf_counter()
                squared=0.
                for ids,active,c,inverse,target in batches:
                    defect=np.einsum('bij,bj->bi',c,value[ids],optimize=False)-target
                    squared+=float(np.sum(defect*defect))
                residual=float(np.sqrt(squared)/denominator)
                history.append([sweep,residual])
                check_seconds+=time.perf_counter()-check
                if residual<tolerance:break
        iteration_seconds=time.perf_counter()-start-check_seconds
        check=time.perf_counter()
        gf=ng.GridFunction(self.space);gf.vec.FV().NumPy()[:]=value
        measure=ng.dx(bonus_intorder=self.bonus);error=ng.curl(gf)-field
        field_error=float(ng.Integrate(error*error*measure,self.mesh)/ng.Integrate(field*field*measure,self.mesh))**.5
        field_check_seconds=time.perf_counter()-check
        info=dict(sweeps=sweep,curl_fit_residual=residual,relative_curl_field_error=field_error,
            recovery_seconds=time.perf_counter()-begin,recovery_ndof=self.space.ndof,
            sequence='colored',history=history,global_matrix_assembled=False,
            shared_dofs_corrected=True,requested_tolerance=tolerance,max_sweeps=max_sweeps,
            converged=bool(residual<tolerance),local_representability_defect=local_defect,
            boundary_dof_max=float(np.max(abs(value[~self.free]),initial=0)),
            target_seconds=target_seconds,iteration_seconds=iteration_seconds,
            curl_check_seconds=check_seconds,field_check_seconds=field_check_seconds,
            colors=self.colors,batches=len(batches),construction_seconds=self.construction_seconds)
        if not np.isfinite(field_error) or residual>=tolerance or field_error>1e-8:
            exc=RuntimeError(f'Batched element curl recovery rejected: {info}');exc.info=info;raise exc
        return gf,info
