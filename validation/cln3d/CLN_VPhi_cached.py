"""Complete V/Phi CLN with compatible hybrid scalar solves and local curls.

No global A-T vector equation is solved. Assembled rectangular load/check
operators are reused, including a curl-curl validation operator; no vector
factorization or inverse enters recovery. Element matrices serve curl recovery. Homogeneous constant-material simply connected internal-wire case.
"""
import time

import ngsolve as ng
import numpy as np

from .CLN_AT import SIGMA,MU
from .cached_checks import CachedScalar, CachedSweep, PhysicalCheck, operator_storage


def run(mesh,order,stages=3,bonus=8,recovery_class=CachedSweep):
    if stages<1:
        raise ValueError('At least one CLN stage required')
    begin=time.perf_counter()
    scalar={role:CachedScalar(mesh,order,role=='A',bonus) for role in ('T','A')}
    local={role:recovery_class(mesh,order,bonus,'.*' if role=='A' else '') for role in ('T','A')}
    for role in ('T','A'):
        scalar[role].prepare(local['A' if role=='T' else 'T'].space)
        local[role].prepare(scalar[role].space.components[0])
    checks={role:PhysicalCheck(mesh,order,scalar[role].space.components[0],local[role].space,role=='A',bonus) for role in ('T','A')}
    setup=time.perf_counter()-begin
    measure=ng.dx(bonus_intorder=bonus)
    solves,solve_times,fields,coefficients,owners=[],[],[],[],[]

    def advance(particular,particular_scale,role,scale,previous,forcing,forcing_scale,label):
        start=time.perf_counter()
        g,info=scalar[role].solve(particular,particular_scale)

        total=ng.GridFunction(scalar[role].space.components[0])
        total.vec.data=scale*g.components[0].vec
        if previous is not None:total.vec.data+=previous.vec
        potential,recovery=local[role].recover(total)
        owners.append(potential)
        info.update(recovery)
        info.update(checks[role].check(g.components[0],potential,previous,scale,forcing,forcing_scale))
        info['label']=label
        solves.append(info)
        solve_times.append(time.perf_counter()-start)
        return total,potential

    J,T=advance(None,1,'T',SIGMA,None,None,1,'T stage 0')
    fields.append(J/SIGMA)
    R=1/float(ng.Integrate(J*J/SIGMA*measure,mesh))
    coefficients.append(R)
    B=None
    for j in range(stages):
        # H = R*T - grad(Phi), with zero normal B on the whole boundary.
        B,A=advance(T,R,'A',MU,B,J,R,f'A stage {j+1}')
        L=float(ng.Integrate(B*B/MU*measure,mesh))
        coefficients.append(L)
        fields.append(B)
        # dE = -A/L - grad(V), with zero tangential dE on the boundary.
        J,T=advance(A,-1/L,'T',SIGMA,J,B,-1/L,f'T stage {j+1}')
        R=1/float(ng.Integrate(J*J/SIGMA*measure,mesh))
        coefficients.append(R)
        fields.append(J/SIGMA)
    elapsed=time.perf_counter()-begin
    points=[(r*np.cos(theta),r*np.sin(theta),z) for z in (.0025,.005,.0075)
            for theta in (0.,np.pi/4) for r in np.linspace(0,.0095,13)]
    record=dict(route='scalar-cached',status='pass',seconds=elapsed,setup_seconds=setup,
                coefficients=coefficients,solves=solves,solve_seconds=solve_times,
                cached_operator_csr_bytes_by_role={role:operator_storage(scalar[role].original,scalar[role].load_operator,local[role].load_operator,checks[role].curl_load,checks[role].force_load,checks[role].curl_check) for role in ('T','A')},
                cache_scope='one full recursion; no reuse across repeats',
                global_vector_solves=0,field_accumulation='original discontinuous HDiv coefficients',
                local_numpy_bytes_by_role={k:v.local_numpy_bytes for k,v in local.items()},
                global_scalar_dofs_by_role={k:v.metadata['global_scalar_dofs'] for k,v in scalar.items()},
                mode_norm2=[float(ng.Integrate(f*f*measure,mesh)) for f in fields],
                sample_points_m=points,mode_samples=[[list(f(mesh(*p))) for p in points] for f in fields])
    return record,fields


