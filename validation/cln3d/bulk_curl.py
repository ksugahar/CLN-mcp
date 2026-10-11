"""NGSolve bulk element loads and paired physical checks for colored curls."""
import time
import ngsolve as ng
import numpy as np
from .batched_curl import BatchedSweep


def integrals(mesh,functions,bonus=8):
    if bonus!=8:
        raise ValueError('Bulk integral rule validated only for quadrature bonus 8')
    return np.array(ng.Integrate(ng.CoefficientFunction(tuple(functions)).Compile(),
                                mesh,order=13),dtype=float).reshape(-1)


def numpy_storage(value):
    roots={}
    def visit(x):
        if isinstance(x,np.ndarray):
            while isinstance(x.base,np.ndarray):x=x.base
            roots[id(x)]=x.nbytes
        elif isinstance(x,(list,tuple)):
            for item in x:visit(item)
        elif isinstance(x,dict):
            for item in x.values():visit(item)
    visit(value)
    return sum(roots.values())


class BulkSweep(BatchedSweep):
    def __init__(self,*args,**kwargs):
        start=time.perf_counter()
        super().__init__(*args,**kwargs)
        if self.bonus!=8:raise ValueError('Quadrature bonus 8 required')
        self.broken=ng.Discontinuous(self.space)
        self.packed=[]
        for indices,ids,active,c,inverse in self.batches:
            broken_ids=np.stack([self.broken.GetDofNrs(self.blocks[j][0]) for j in indices])
            dual=np.stack([self.blocks[j][4] for j in indices])
            cf=np.stack([self.blocks[j][3][:,self.blocks[j][2]] for j in indices])
            self.packed.append((indices,ids,active,c,inverse,broken_ids,dual,cf))
        # Retain one packed representation; archived per-element copies are
        # only construction inputs, not needed by the bulk execution route.
        self.elements=[]
        self.blocks=[]
        self.batches=[]
        self.construction_seconds=time.perf_counter()-start
        self.local_numpy_bytes=numpy_storage(vars(self))

    def targets(self,field):
        load=ng.LinearForm(self.broken)
        load+=ng.curl(self.broken.TestFunction())*field*ng.dx(bonus_intorder=self.bonus)
        load.Assemble()
        data=load.vec.FV().NumPy()
        batches=[];norm2=0.;defect2=0.
        for _,ids,active,c,inverse,broken_ids,dual,cf in self.packed:
            target=np.einsum('bij,bj->bi',dual,data[broken_ids],optimize=False)
            correction=np.einsum('bij,bj->bi',inverse,target,optimize=False)
            defect=np.einsum('bij,bj->bi',cf,correction,optimize=False)-target
            defect2+=float(np.sum(defect*defect));norm2+=float(np.sum(target*target))
            batches.append((ids,active,c,inverse,target))
        denominator=np.sqrt(norm2)
        if not np.isfinite(denominator) or denominator==0:
            raise ValueError('Finite nonzero curl target required')
        local_defect=float(np.sqrt(defect2)/denominator)
        if not np.isfinite(local_defect) or local_defect>1e-8:
            raise RuntimeError(f'Locally incompatible curl or boundary constraints: {local_defect}')
        return batches,denominator,local_defect

    def iterate(self,batches,denominator,tolerance,max_sweeps):
        value=np.zeros(self.space.ndof);history=[];check_seconds=0.
        start=time.perf_counter()
        for sweep in range(1,max_sweeps+1):
            for ids,active,c,inverse,target in batches:
                defect=target-np.einsum('bij,bj->bi',c,value[ids],optimize=False)
                value[active]+=np.einsum('bij,bj->bi',inverse,defect,optimize=False)
            if sweep==1 or sweep%10==0 or sweep==max_sweeps:
                check=time.perf_counter();squared=0.
                for ids,active,c,inverse,target in batches:
                    defect=np.einsum('bij,bj->bi',c,value[ids],optimize=False)-target
                    squared+=float(np.sum(defect*defect))
                residual=float(np.sqrt(squared)/denominator)
                history.append([sweep,residual]);check_seconds+=time.perf_counter()-check
                if residual<tolerance:break
        return value,dict(sweeps=sweep,curl_fit_residual=residual,history=history,
            iteration_seconds=time.perf_counter()-start-check_seconds,curl_check_seconds=check_seconds)

    def recover(self,field,tolerance=1e-10,max_sweeps=4000):
        if not 0<tolerance<=1e-8 or max_sweeps<1:raise ValueError('Invalid recovery limits')
        begin=time.perf_counter()
        batches,denominator,local_defect=self.targets(field)
        target_seconds=time.perf_counter()-begin
        value,info=self.iterate(batches,denominator,tolerance,max_sweeps)
        start=time.perf_counter()
        gf=ng.GridFunction(self.space);gf.vec.FV().NumPy()[:]=value
        error=ng.curl(gf)-field
        norms=integrals(self.mesh,(error*error,field*field),self.bonus)
        if not np.isfinite(norms).all() or norms[0]<0 or norms[1]<=0:
            raise RuntimeError('Invalid integrated curl norms')
        field_error=float(np.sqrt(norms[0]/norms[1]))
        field_check_seconds=time.perf_counter()-start
        info.update(relative_curl_field_error=field_error,recovery_seconds=time.perf_counter()-begin,
            recovery_ndof=self.space.ndof,sequence='colored',global_matrix_assembled=False,
            shared_dofs_corrected=True,requested_tolerance=tolerance,max_sweeps=max_sweeps,
            converged=bool(info['curl_fit_residual']<tolerance),local_representability_defect=local_defect,
            boundary_dof_max=float(np.max(abs(value[~self.free]),initial=0)),target_seconds=target_seconds,
            field_check_seconds=field_check_seconds,colors=self.colors,batches=len(batches),
            construction_seconds=self.construction_seconds,local_numpy_bytes=self.local_numpy_bytes,
            broken_load_dofs=self.broken.ndof,load_assembly='NGSolve discontinuous HCurl LinearForm',
            physical_integral_order=13,backend='cpu',unpacked_operators_released=True)
        if not np.isfinite(field_error) or not info['converged'] or field_error>1e-8:
            exc=RuntimeError(f'Bulk recovery rejected: {info}');exc.info=info;raise exc
        return gf,info
