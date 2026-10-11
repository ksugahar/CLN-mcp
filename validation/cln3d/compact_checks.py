"""Reuse assembled native load and validation operators within one CLN solve.

No mesh/operator cache survives a solve. Quadrature, free-dof masks and
all physical checks are retained. Explicit vector matrices serve checks and
loads only, never a global vector solve. Their setup and storage are counted.
"""
import time
import hashlib
import ngsolve as ng
import numpy as np
from .bulk_scalar import BulkScalar
from .bulk_curl import BulkSweep


def linear_form_rule(space,bonus):
    # LFI uses 2*test FE order + bonus. BFI defaults also depend on trial
    # order and derivative orders; explicitly match the archived LFI rule.
    orders={el.GetFE().order for el in space.Elements()}
    types={el.type for el in space.Elements()}
    if len(orders)!=1 or types!={ng.ET.TET}:raise ValueError('Uniform tetrahedral space required')
    return {ng.ET.TET:ng.IntegrationRule(ng.ET.TET,order=2*orders.pop()+bonus)}


class OperatorPool:
    """Solve-owned operators; reuse only after checking native element layouts."""
    def __init__(self):
        self.forms={}
        self.timings={}
        self.signatures={}
    def signature(self,space):
        identity=id(space)
        if identity not in self.signatures:
            digest=hashlib.sha256()
            for el in space.Elements():
                dofs=np.asarray(space.GetDofNrs(el),dtype=np.int64)
                digest.update(np.asarray([len(dofs)],dtype=np.int64).tobytes())
                digest.update(dofs.tobytes())
            self.signatures[identity]=(space, (space.ndof,digest.hexdigest()))
        return self.signatures[identity][1]
    def get(self,key,spaces,build):
        signature=tuple(self.signature(space) for space in spaces)
        if key in self.forms:
            previous,form=self.forms[key]
            if previous!=signature:raise RuntimeError('Shared operator layout mismatch: '+key)
            return form
        start=time.perf_counter()
        form=build()
        self.forms[key]=(signature,form)
        self.timings[key]=time.perf_counter()-start
        return form
    def storage(self):
        return {key:operator_storage(form) for key,(_,form) in self.forms.items()}


class CachedScalar(BulkScalar):
    def __init__(self,*args,**kwargs):
        self.pool=kwargs.pop("pool")
        super().__init__(*args,**kwargs)
        self.solution=ng.GridFunction(self.space)
        flux=self.solution.components[0]
        self.norms=ng.CoefficientFunction((ng.div(flux)**2,flux*flux)).Compile()
        q,u,uhat=self.space.TrialFunction()
        v,w,what=self.space.TestFunction()
        normal=ng.specialcf.normal(3)
        def build_original():
            form=ng.BilinearForm(self.space)
            form+=(q*v-u*ng.div(v)-w*ng.div(q))*ng.dx(bonus_intorder=self.bonus)
            form+=(uhat*v*normal+what*q*normal)*ng.dx(element_boundary=True,bonus_intorder=self.bonus)
            form.Assemble()
            return form
        self.original=self.pool.get('original-scalar',(self.space,),build_original)
        self.load=None

    def prepare(self,potential_space):
        self.load=ng.LinearForm(self.space)
        flux=self.space.components[0]
        self.flux_load=ng.LinearForm(flux)
        def build_load():
            form=ng.BilinearForm(trialspace=potential_space,testspace=flux)
            form+=potential_space.TrialFunction()*flux.TestFunction()*ng.dx(intrules=linear_form_rule(self.space,self.bonus))
            form.Assemble()
            return form
        self.load_operator=self.pool.get('scalar-load',(potential_space,flux),build_load)
        self.initial=ng.LinearForm(self.space)
        self.initial+=(ng.CoefficientFunction((0,0,1))*self.test).Compile()*ng.dx(bonus_intorder=self.bonus)

    def solve(self,potential,scale=1):
        start=time.perf_counter()
        if potential is None:rhs=self.initial
        else:
            rhs=self.load
        if potential is None:rhs.Assemble()
        else:
            self.flux_load.vec.data=scale*(self.load_operator.mat*potential.vec)
            rhs.vec[:]=0
            rhs.vec.FV().NumPy()[:self.space.components[0].ndof]=self.flux_load.vec.FV().NumPy()
        assembly=time.perf_counter()-start
        condensed=rhs.vec.CreateVector()
        condensed.data=rhs.vec+self.form.harmonic_extension_trans*rhs.vec
        gf=self.solution
        gf.vec[:]=0
        gf.vec.FV().NumPy()[self.free]=self.solve_trace(-condensed.FV().NumPy()[self.free])
        residual=rhs.vec.CreateVector()
        residual.data=self.form.mat*gf.vec-condensed
        b=condensed.FV().NumPy()[self.free]
        defect=float(np.linalg.norm(residual.FV().NumPy()[self.free])/np.linalg.norm(b))
        gf.vec.data+=self.form.harmonic_extension*gf.vec
        gf.vec.data+=self.form.inner_solve*rhs.vec
        solve_end=time.perf_counter()
        residual.data=self.original.mat*gf.vec-rhs.vec
        original_defect=float(np.linalg.norm(residual.FV().NumPy()[self.original_free])/
                              np.linalg.norm(rhs.vec.FV().NumPy()[self.original_free]))
        values=np.asarray(ng.Integrate(self.norms,self.mesh,order=13),dtype=float)
        if not np.isfinite(values).all() or values[0]<0 or values[1]<=0:
            raise RuntimeError('Invalid scalar flux norms')
        relative_div=float(np.sqrt(values[0]/values[1])*.01)
        if not np.isfinite([defect,original_defect,relative_div]).all() or max(defect,original_defect)>1e-6 or relative_div>1e-8:
            raise RuntimeError('Cached scalar residual/divergence failed')
        return gf,dict(**self.metadata,scalar_trace_residual=defect,relative_divergence=relative_div,
            original_scalar_residual=original_defect,scalar_seconds=time.perf_counter()-start,
            scalar_load_seconds=assembly,scalar_linear_seconds=solve_end-start-assembly,
            scalar_check_seconds=time.perf_counter()-solve_end)


class CachedSweep(BulkSweep):
    def prepare(self,flux_space,pool):
        self.target_field=ng.GridFunction(flux_space)
        self.check_potential=ng.GridFunction(self.space)
        self.load=ng.LinearForm(self.broken)
        def build_load():
            form=ng.BilinearForm(trialspace=flux_space,testspace=self.broken)
            form+=ng.curl(self.broken.TestFunction())*flux_space.TrialFunction()*ng.dx(intrules=linear_form_rule(self.broken,self.bonus))
            form.Assemble()
            return form
        self.load_operator=pool.get('recovery-load',(flux_space,self.broken),build_load)
        error=ng.curl(self.check_potential)-self.target_field
        self.norms=ng.CoefficientFunction((error*error,self.target_field*self.target_field)).Compile()

    def targets(self,field):
        self.target_field.vec.data=field.vec
        self.load.vec.data=self.load_operator.mat*field.vec
        data=self.load.vec.FV().NumPy()
        batches=[];norm2=0.;defect2=0.
        for _,ids,active,c,inverse,broken_ids,dual,cf in self.packed:
            target=np.einsum('bij,bj->bi',dual,data[broken_ids],optimize=False)
            correction=np.einsum('bij,bj->bi',inverse,target,optimize=False)
            defect=np.einsum('bij,bj->bi',cf,correction,optimize=False)-target
            defect2+=float(np.sum(defect*defect));norm2+=float(np.sum(target*target))
            batches.append((ids,active,c,inverse,target))
        denominator=float(np.sqrt(norm2))
        if not np.isfinite(denominator) or denominator<=0:raise RuntimeError('Invalid curl target norm')
        local_defect=float(np.sqrt(defect2)/denominator)
        if not np.isfinite([denominator,local_defect]).all() or denominator<=0 or local_defect>1e-8:
            raise RuntimeError('Cached curl target incompatible')
        return batches,denominator,local_defect

    def recover(self,field,tolerance=1e-10,max_sweeps=4000):
        if not 0<tolerance<=1e-8 or max_sweeps<1:raise ValueError('Invalid recovery limits')
        begin=time.perf_counter()
        batches,denominator,local_defect=self.targets(field)
        target_seconds=time.perf_counter()-begin
        value,info=self.iterate(batches,denominator,tolerance,max_sweeps)
        start=time.perf_counter()
        # A fresh result owns each mode; only the check buffer is reused.
        gf=ng.GridFunction(self.space);gf.vec.FV().NumPy()[:]=value
        self.check_potential.vec.data=gf.vec
        norms=np.asarray(ng.Integrate(self.norms,self.mesh,order=13),dtype=float)
        if not np.isfinite(norms).all() or norms[0]<0 or norms[1]<=0:raise RuntimeError('Invalid curl norms')
        field_error=float(np.sqrt(norms[0]/norms[1]))
        info.update(relative_curl_field_error=field_error,recovery_seconds=time.perf_counter()-begin,
            recovery_ndof=self.space.ndof,sequence='colored',global_matrix_assembled=False,
            shared_dofs_corrected=True,requested_tolerance=tolerance,max_sweeps=max_sweeps,
            converged=bool(info['curl_fit_residual']<tolerance),local_representability_defect=local_defect,
            boundary_dof_max=float(np.max(abs(value[~self.free]),initial=0)),target_seconds=target_seconds,
            field_check_seconds=time.perf_counter()-start,colors=self.colors,batches=len(batches),
            construction_seconds=self.construction_seconds,local_numpy_bytes=self.local_numpy_bytes,
            broken_load_dofs=self.broken.ndof,load_assembly='NGSolve cached rectangular curl-load operator',
            physical_integral_order=13,backend='cpu',unpacked_operators_released=True)
        if not np.isfinite(field_error) or not info['converged'] or field_error>1e-8:
            raise RuntimeError('Cached curl recovery rejected')
        return gf,info


class PhysicalCheck:
    def __init__(self,mesh,order,flux_space,potential_space,magnetic,bonus,pool):
        self.mesh=mesh
        fes=ng.HCurl(mesh,order=order,nograds=True,dirichlet='.*' if magnetic else '')
        self.mask=np.array(list(fes.FreeDofs()),dtype=bool)
        self.ndof=fes.ndof
        self.space=fes*fes*fes
        v0,v1,v2=self.space.TestFunction()
        self.form=ng.LinearForm(self.space)
        self.form.Assemble()
        trial=flux_space.TrialFunction();test=fes.TestFunction()
        rule=linear_form_rule(self.space,bonus)
        def build_curl_load():
            form=ng.BilinearForm(trialspace=flux_space,testspace=fes)
            form+=trial*ng.curl(test)*ng.dx(intrules=rule)
            form.Assemble()
            return form
        self.curl_load=pool.get("physical-curl_load",(flux_space,fes),build_curl_load)
        def build_force_load():
            form=ng.BilinearForm(trialspace=flux_space,testspace=fes)
            form+=trial*test*ng.dx(intrules=rule)
            form.Assemble()
            return form
        self.force_load=pool.get("physical-force_load",(flux_space,fes),build_force_load)
        def build_curl_check():
            form=ng.BilinearForm(trialspace=potential_space,testspace=fes)
            form+=ng.curl(potential_space.TrialFunction())*ng.curl(test)*ng.dx(intrules=rule)
            form.Assemble()
            return form
        self.curl_check=pool.get("physical-curl_check",(potential_space,fes),build_curl_check)
        self.boundary=ng.LinearForm(fes)
        self.boundary+=-ng.Cross(ng.CoefficientFunction((0,0,1)),test.Trace())*ng.specialcf.normal(3)*ng.ds('conductorBND',bonus_intorder=bonus)
        self.boundary.Assemble()
        self.work=[self.boundary.vec.CreateVector() for _ in range(3)]

    def check(self,increment,potential,previous,scale,forcing,forcing_scale):
        start=time.perf_counter()
        expected,exact,recovered=self.work
        expected.data=self.boundary.vec if forcing is None else forcing_scale*(self.force_load.mat*forcing.vec)
        exact.data=self.curl_load.mat*increment.vec
        recovered.data=self.curl_check.mat*potential.vec
        if previous is not None:recovered.data-=self.curl_load.mat*previous.vec
        recovered.data*=1/scale
        raw=self.form.vec.FV().NumPy().reshape(3,self.ndof)
        for row,vec in zip(raw,self.work):row[:]=vec.FV().NumPy()
        values=self.form.vec.FV().NumPy().reshape(3,self.ndof)[:,self.mask]
        denominator=np.linalg.norm(values[0])
        if not np.isfinite(denominator) or denominator<=0:raise RuntimeError('Invalid physical load norm')
        errors=[float(np.linalg.norm(values[i]-values[0])/denominator) for i in (1,2)]
        if not np.isfinite(errors).all() or max(errors)>1e-6:raise RuntimeError('Cached original physical equation failed')
        return dict(original_physical_residual=errors[0],recovered_original_residual=errors[1],
                    physical_check_seconds=time.perf_counter()-start,
                    physical_check_assembly='NGSolve cached rectangular validation operators',
                    validation_global_vector_operator=True)


def operator_storage(*forms):
    """Logical CSR array bytes, excluding native graphs/allocator overhead."""
    return sum(sum(np.asarray(a).nbytes for a in form.mat.CSR()) for form in forms)
