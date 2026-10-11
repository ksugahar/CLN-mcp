"""Independent native checks of broken loads and unchanged quadrature."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import ngsolve as ng
import numpy as np
from threadpoolctl import threadpool_limits
import bulk_cln_study as study
from cln3d.bulk_curl import BulkSweep,integrals
from cln3d.batched_curl import BatchedSweep


def run(output):
    ng.SetNumThreads(4)
    hashes=study.hashes()
    hashes['validation/bulk_controls.py']=hashlib.sha256(Path(__file__).read_text(encoding='utf-8-sig').encode()).hexdigest()
    result=dict(schema=1,complete=False,host=socket.gethostname(),source_sha256_lf=hashes,cases=[])
    with threadpool_limits(limits=1,user_api='blas'),ng.TaskManager():
        for curve in (1,3):
            mesh=study.baseline.CLN_AT.make_mesh(.003,curve)
            for p in (2,3,4):
                for boundary in ('','.*'):
                    old=BatchedSweep(mesh,p,8,boundary)
                    bulk=BulkSweep(mesh,p,8,boundary)
                    lift=ng.GridFunction(old.space)
                    lift.Set(ng.CoefficientFunction((ng.y*ng.z,ng.x*ng.z,ng.x*ng.y+ng.x)),definedon=mesh.Materials('.*'))
                    lift.vec.FV().NumPy()[~old.free]=0
                    field=ng.curl(lift)
                    batches,_,defect=bulk.targets(field)
                    lfi=ng.SymbolicLFI(ng.curl(old.test)*field,bonus_intorder=8)
                    difference=norm=0.
                    for packed,batch in zip(bulk.packed,batches):
                        indices=packed[0]
                        expected=[]
                        for j in indices:
                            el,ids,free,c,dual,inverse=old.blocks[j]
                            expected.append(dual@np.array(lfi.CalcElementVector(el.GetFE(),el.GetTrafo())))
                        expected=np.array(expected)
                        difference+=float(np.sum((expected-batch[-1])**2));norm+=float(np.sum(expected**2))
                    functions=(field*field,ng.exp(173*ng.x+91*ng.y+77*ng.z))
                    reference=np.array([ng.Integrate(f*ng.dx(bonus_intorder=8),mesh) for f in functions])
                    paired=integrals(mesh,functions)
                    compound=old.space*old.space*old.space
                    v0,v1,v2=compound.TestFunction()
                    force=ng.CoefficientFunction((1,2,3))
                    form=ng.LinearForm(compound)
                    form+=(field*ng.curl(v0)+force*v1+(1+ng.x)*field*ng.curl(v2))*ng.dx(bonus_intorder=8)
                    form.Assemble()
                    v=old.space.TestFunction()
                    separate=[]
                    for term in (field*ng.curl(v),force*v,(1+ng.x)*field*ng.curl(v)):
                        lf=ng.LinearForm(old.space);lf+=term*ng.dx(bonus_intorder=8);lf.Assemble()
                        separate.append(lf.vec.FV().NumPy().copy())
                    separate=np.array(separate)
                    compound_error=float(np.linalg.norm(form.vec.FV().NumPy().reshape(separate.shape)-separate)/np.linalg.norm(separate))
                    load_error=float(np.sqrt(difference/norm))
                    integral_error=float(np.max(abs(paired/reference-1)))
                    assert load_error<1e-12 and integral_error<1e-12 and compound_error<1e-12 and defect<1e-8
                    result['cases'].append(dict(curve=curve,order=p,boundary=boundary,load_relative_error=load_error,integral_relative_error=integral_error,compound_relative_error=compound_error,local_defect=defect))
                    print(json.dumps(result['cases'][-1]),flush=True)
    result['complete']=True
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
