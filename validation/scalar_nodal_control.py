"""Nodal V/Phi control: a checked scalar solve does not ensure curl-compatible flux."""
import argparse
import hashlib
import json
from pathlib import Path

import ngsolve as ng
import numpy as np
import scipy.sparse as sp

from cln3d import CLN_AT
from cln3d.compatible_curl import CompatibleSweep
from cln3d.scalar_potentials import ScalarPoisson
from cln3d.wire_pencils import factor
from scalar_cln_study import hashes

ROOT=Path(__file__).resolve().parents[1]


def run(output):
    ng.SetNumThreads(4)
    import socket
    identity=hashes()
    name='validation/scalar_nodal_control.py'
    identity[name]=hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()
    data=dict(schema=1,complete=False,host=socket.gethostname(),source_sha256_lf=identity,cases=[])
    for curve in (1,3):
        mesh=CLN_AT.make_mesh(.003,curve)
        with ng.TaskManager():
            for p in (2,3):
                electric=ScalarPoisson(mesh,p,False)
                j,_=electric.solve(ng.CoefficientFunction((0,0,1)))
                J=CLN_AT.SIGMA*j.components[0]
                R=1/float(ng.Integrate(J*J/CLN_AT.SIGMA*ng.dx(bonus_intorder=8),mesh))
                T,_=CompatibleSweep(mesh,p,8,'').recover(J)
                h1=ng.H1(mesh,order=p+1)
                u,v=h1.TnT()
                a=ng.BilinearForm(h1,symmetric=True)
                a += ng.grad(u)*ng.grad(v)*ng.dx(bonus_intorder=8)
                f=ng.LinearForm(h1)
                f += R*T*ng.grad(v)*ng.dx(bonus_intorder=8)
                a.Assemble();f.Assemble()
                free=np.array(list(h1.FreeDofs()),dtype=bool)
                free[0]=False
                mat=sp.csr_matrix(a.mat.CSR(),shape=(h1.ndof,h1.ndof))
                phi=ng.GridFunction(h1)
                phi.vec.FV().NumPy()[free]=factor(mat[free][:,free])(f.vec.FV().NumPy()[free])
                residual=f.vec.CreateVector()
                residual.data=a.mat*phi.vec-f.vec
                defect=float(np.linalg.norm(residual.FV().NumPy())/np.linalg.norm(f.vec.FV().NumPy()))
                if defect>1e-6:raise RuntimeError('Nodal scalar reference failed')
                B=CLN_AT.MU*(R*T-ng.grad(phi))
                engine=CompatibleSweep(mesh,p,8,'.*')
                lfi=ng.SymbolicLFI(ng.curl(engine.test)*B,bonus_intorder=8)
                projected=0.
                for el,ids,c,dual in engine.elements:
                    load=np.array(lfi.CalcElementVector(el.GetFE(),el.GetTrafo()))
                    target=dual@load
                    projected+=float(target@target)
                norm2=float(ng.Integrate(B*B*ng.dx(bonus_intorder=8),mesh))
                lower_bound=max(1-projected/norm2,0)**.5
                try:engine.recover(B,max_sweeps=40)
                except RuntimeError as exc:
                    failure=dict(reason=str(exc),diagnostics=getattr(exc,'info',{}))
                else:raise AssertionError('Nodal target unexpectedly passed the strict local recovery gate')
                hybrid=ScalarPoisson(mesh,p,True)
                g,hybrid_info=hybrid.solve(R*T)
                delta=B-CLN_AT.MU*g.components[0]
                difference=float(ng.Integrate(delta*delta*ng.dx(bonus_intorder=8),mesh)/norm2)**.5
                data['cases'].append(dict(curve=curve,order=p,scalar_residual=defect,
                    element_projection_lower_bound=lower_bound,nodal_hybrid_field_difference=difference,
                    strict_recovery_failure=failure,hybrid_scalar_residual=hybrid_info['original_scalar_residual']))
                print(json.dumps(data['cases'][-1]),flush=True)
    assert identity==hashes()|{name:hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()}
    data['complete']=True
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.output)
