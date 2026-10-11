"""Complete V/Phi CLN with compatible hybrid scalar solves and local curls.

No global A-T vector equation is assembled or solved on this route. HCurl
test spaces serve only residual checks, and element matrices serve curl
recovery. Homogeneous constant-material simply connected internal-wire case.
"""
import time

import ngsolve as ng
import numpy as np

from .CLN_AT import SIGMA,MU
from .bulk_curl import BulkSweep,integrals
from .bulk_scalar import BulkScalar


def run(mesh,order,stages=3,bonus=8,recovery_class=BulkSweep):
    if stages<1:
        raise ValueError('At least one CLN stage required')
    begin=time.perf_counter()
    scalar={role:BulkScalar(mesh,order,role=='A',bonus) for role in ('T','A')}
    local={role:recovery_class(mesh,order,bonus,'.*' if role=='A' else '') for role in ('T','A')}
    tests={role:ng.HCurl(mesh,order=order,nograds=True,dirichlet='.*' if role=='A' else '') for role in ('T','A')}
    compound={role:fes*fes*fes for role,fes in tests.items()}
    masks={role:np.array(list(fes.FreeDofs()),dtype=bool) for role,fes in tests.items()}
    setup=time.perf_counter()-begin
    measure=ng.dx(bonus_intorder=bonus)
    normal=ng.specialcf.normal(3)
    solves,solve_times,fields,coefficients,owners=[],[],[],[],[]

    def advance(particular,role,scale,previous,forcing,label):
        start=time.perf_counter()
        g,info=scalar[role].solve(particular)

        increment=scale*g.components[0]
        total=ng.GridFunction(scalar[role].space.components[0])
        total.vec.data=scale*g.components[0].vec
        if previous is not None:total.vec.data+=previous.vec
        potential,recovery=local[role].recover(total)
        owners.append(potential)
        info.update(recovery)
        check_start=time.perf_counter()
        fes=tests[role]
        v0,v1,v2=compound[role].TestFunction()
        recovered=ng.curl(potential) if previous is None else ng.curl(potential)-previous
        form=ng.LinearForm(compound[role])
        integrand=(increment/scale)*ng.curl(v1)+(recovered/scale)*ng.curl(v2)
        if forcing is None:
            form+=-ng.Cross(ng.CoefficientFunction((0,0,1)),v0.Trace())*normal*ng.ds('conductorBND',bonus_intorder=bonus)
        else:
            integrand+=forcing*v0
        form+=integrand*measure
        form.Assemble()
        values=form.vec.FV().NumPy().reshape(3,fes.ndof)[:,masks[role]]
        denominator=np.linalg.norm(values[0])
        defect=float(np.linalg.norm(values[1]-values[0])/denominator)
        recovered_defect=float(np.linalg.norm(values[2]-values[0])/denominator)
        info['physical_check_seconds']=time.perf_counter()-check_start
        info['physical_check_assembly']='NGSolve compound HCurl LinearForm'
        if not np.isfinite(recovered_defect) or max(defect,recovered_defect)>1e-6:
            raise RuntimeError(f'Original physical equation failed at {label}: {defect}, {recovered_defect}')
        info.update(label=label,original_physical_residual=defect,recovered_original_residual=recovered_defect)
        solves.append(info)
        solve_times.append(time.perf_counter()-start)
        return total,potential

    J,T=advance(ng.CoefficientFunction((0,0,1)),'T',SIGMA,None,None,'T stage 0')
    fields.append(J/SIGMA)
    R=1/float(ng.Integrate(J*J/SIGMA*measure,mesh))
    coefficients.append(R)
    B=None
    for j in range(stages):
        # H = R*T - grad(Phi), with zero normal B on the whole boundary.
        B,A=advance(R*T,'A',MU,B,R*J,f'A stage {j+1}')
        L=float(ng.Integrate(B*B/MU*measure,mesh))
        coefficients.append(L)
        fields.append(B)
        # dE = -A/L - grad(V), with zero tangential dE on the boundary.
        J,T=advance(-A/L,'T',SIGMA,J,-B/L,f'T stage {j+1}')
        R=1/float(ng.Integrate(J*J/SIGMA*measure,mesh))
        coefficients.append(R)
        fields.append(J/SIGMA)
    elapsed=time.perf_counter()-begin
    points=[(r*np.cos(theta),r*np.sin(theta),z) for z in (.0025,.005,.0075)
            for theta in (0.,np.pi/4) for r in np.linspace(0,.0095,13)]
    record=dict(route='scalar-bulk',status='pass',seconds=elapsed,setup_seconds=setup,
                coefficients=coefficients,solves=solves,solve_seconds=solve_times,
                global_vector_solves=0,field_accumulation='original discontinuous HDiv coefficients',
                local_numpy_bytes_by_role={k:v.local_numpy_bytes for k,v in local.items()},
                global_scalar_dofs_by_role={k:v.metadata['global_scalar_dofs'] for k,v in scalar.items()},
                mode_norm2=[float(ng.Integrate(f*f*measure,mesh)) for f in fields],
                sample_points_m=points,mode_samples=[[list(f(mesh(*p))) for p in points] for f in fields])
    return record,fields


