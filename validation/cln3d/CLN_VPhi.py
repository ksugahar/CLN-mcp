"""Complete V/Phi CLN with compatible hybrid scalar solves and local curls.

No global A-T vector equation is assembled or solved on this route. HCurl
test spaces serve only residual checks, and element matrices serve curl
recovery. Homogeneous constant-material simply connected internal-wire case.
"""
import time

import ngsolve as ng
import numpy as np

from .CLN_AT import SIGMA,MU
from .compatible_curl import CompatibleSweep
from .scalar_potentials import ScalarPoisson


def run(mesh,order,stages=3,bonus=8):
    if stages<1:
        raise ValueError('At least one CLN stage required')
    begin=time.perf_counter()
    scalar={role:ScalarPoisson(mesh,order,role=='A',bonus) for role in ('T','A')}
    local={role:CompatibleSweep(mesh,order,bonus,'.*' if role=='A' else '') for role in ('T','A')}
    tests={role:ng.HCurl(mesh,order=order,nograds=True,dirichlet='.*' if role=='A' else '') for role in ('T','A')}
    setup=time.perf_counter()-begin
    measure=ng.dx(bonus_intorder=bonus)
    normal=ng.specialcf.normal(3)
    solves,solve_times,fields,coefficients,owners=[],[],[],[],[]

    def advance(particular,role,scale,previous,forcing,label):
        start=time.perf_counter()
        g,info=scalar[role].solve(particular)
        owners.append(g)
        increment=scale*g.components[0]
        total=increment if previous is None else previous+increment
        potential,recovery=local[role].recover(total)
        owners.append(potential)
        info.update(recovery)
        fes=tests[role]
        _,v=fes.TnT()
        rhs=ng.LinearForm(fes)
        if forcing is None:
            rhs += -ng.Cross(ng.CoefficientFunction((0,0,1)),v.Trace())*normal*ng.ds('conductorBND',bonus_intorder=bonus)
        else:
            rhs += forcing*v*measure
        rhs.Assemble()
        res=ng.LinearForm(fes)
        res += (increment/scale)*ng.curl(v)*measure
        res.Assemble()
        free=np.array(list(fes.FreeDofs()),dtype=bool)
        denominator=np.linalg.norm(rhs.vec.FV().NumPy()[free])
        defect=float(np.linalg.norm((res.vec.FV().NumPy()-rhs.vec.FV().NumPy())[free])/denominator)
        recovered=ng.curl(potential) if previous is None else ng.curl(potential)-previous
        res=ng.LinearForm(fes)
        res += (recovered/scale)*ng.curl(v)*measure
        res.Assemble()
        recovered_defect=float(np.linalg.norm((res.vec.FV().NumPy()-rhs.vec.FV().NumPy())[free])/denominator)
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
    record=dict(route='scalar-VPhi',status='pass',seconds=elapsed,setup_seconds=setup,
                coefficients=coefficients,solves=solves,solve_seconds=solve_times,
                global_vector_solves=0,mode_norm2=[float(ng.Integrate(f*f*measure,mesh)) for f in fields],
                sample_points_m=points,mode_samples=[[list(f(mesh(*p))) for p in points] for f in fields])
    return record,fields


if __name__=='__main__':
    import argparse
    import json
    from pathlib import Path
    from .CLN_AT import make_mesh
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maxh',type=float,default=.003)
    parser.add_argument('--order',type=int,default=3)
    parser.add_argument('--stages',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ng.SetNumThreads(4)
    mesh=make_mesh(args.maxh,3)
    with ng.TaskManager():
        record,_=run(mesh,args.order,args.stages)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf8')
