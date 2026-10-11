"""Separate curvature, order, equation weighting and sweep compatibility."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import ngsolve as ng
import numpy as np

from cln3d import CLN_AT
from cln3d.compatible_curl import CompatibleSweep
from compatible_recovery_study import hashes

ROOT=Path(__file__).resolve().parents[1]


def greedy(engine,field,sequence,equation):
    lfi=ng.SymbolicLFI(ng.curl(engine.test)*field,bonus_intorder=8)
    blocks=[]
    for el,ids,c,dual in engine.elements:
        load=np.array(lfi.CalcElementVector(el.GetFE(),el.GetTrafo()))
        blocks.append((ids,c,dual@load,load))
    ordering=np.arange(len(blocks))
    if sequence=='reverse':ordering=ordering[::-1]
    if sequence=='shuffled':ordering=np.random.default_rng(19).permutation(ordering)
    value=np.zeros(engine.space.ndof)
    fixed=~engine.free.copy()
    first_conflict=None
    max_local=0.
    for visited,j in enumerate(ordering):
        ids,c,target,load=blocks[j]
        free=~fixed[ids]
        matrix,rhs=(c.T@c,load) if equation=='K' else (c,target)
        remainder=rhs-matrix@value[ids]
        if free.any():value[ids[free]]+=np.linalg.lstsq(matrix[:,free],remainder,rcond=1e-11)[0]
        defect=float(np.linalg.norm(c@value[ids]-target)/max(np.linalg.norm(target),1e-300))
        max_local=max(max_local,defect)
        if first_conflict is None and defect>1e-8:
            first_conflict=dict(visited=visited,element=int(j),unknown_dofs=int(free.sum()),local_dofs=len(ids),relative_local_curl_error=defect)
        fixed[ids]=True
    error2=sum(float((c@value[ids]-target)@(c@value[ids]-target)) for ids,c,target,_ in blocks)
    norm2=sum(float(target@target) for _,_,target,_ in blocks)
    gf=ng.GridFunction(engine.space)
    gf.vec.FV().NumPy()[:]=value
    delta=ng.curl(gf)-field
    physical=float(ng.Integrate(delta*delta*ng.dx(bonus_intorder=8),engine.mesh)/ng.Integrate(field*field*ng.dx(bonus_intorder=8),engine.mesh))**.5
    return dict(sequence=sequence,equation=equation,curl_fit_residual=(error2/norm2)**.5,
                relative_curl_field_error=physical,first_conflict=first_conflict,max_local_error=max_local)


def run(output):
    ng.SetNumThreads(4)
    import socket
    identities=hashes()
    name='validation/curl_sweep_diagnosis.py'
    identities[name]=hashlib.sha256((ROOT/name).read_text(encoding='utf8').encode()).hexdigest()
    data=dict(schema=1,complete=False,host=socket.gethostname(),source_sha256_lf=identities,cases=[])
    for curve in (1,3):
        mesh=CLN_AT.make_mesh(.003,curve)
        connectivity=[[int(v.nr) for v in el.vertices] for el in mesh.Elements(ng.VOL)]
        topology=hashlib.sha256(json.dumps(connectivity).encode()).hexdigest()
        for p in (1,2,3):
            targets=[]
            original=CLN_AT.solve_iccg
            def collect(a,f,fes,**kw):
                gf,info=original(a,f,fes,**kw)
                if not targets:targets.append(gf)
                return gf,info
            with ng.TaskManager(),patch.object(CLN_AT,'make_mesh',return_value=mesh),patch.object(CLN_AT,'solve_iccg',collect),contextlib.redirect_stdout(io.StringIO()):
                CLN_AT.run(.003,p,1,curve,bonus_intorder=8)
                field=ng.curl(targets[0])
                engine=CompatibleSweep(mesh,p,8,'')
                case=dict(curve=curve,order=p,ne=mesh.ne,topology_sha256=topology,greedy=[],iterative=[])
                counts=np.zeros(engine.space.ndof,dtype=int)
                for _,ids,_,_ in engine.elements:np.add.at(counts,ids,1)
                case['shared_dofs']=int((counts>1).sum())
                for sequence in ('forward','reverse','shuffled'):
                    for equation in ('K','sqrtK'):
                        case['greedy'].append(greedy(engine,field,sequence,equation))
                    engine.sequence=sequence
                    _,info=engine.recover(field)
                    case['iterative'].append(info)
                try:engine.recover(field,max_sweeps=1)
                except RuntimeError as exc:case['capped_failure']=exc.info
                else:raise AssertionError('A capped nonconverged recovery was silently accepted')
                try:engine.recover(ng.CoefficientFunction((ng.x,ng.y,ng.z)),max_sweeps=40)
                except RuntimeError as exc:case['incompatible_failure']=getattr(exc,'info',dict(reason=str(exc)))
                else:raise AssertionError('A field with nonzero divergence was accepted as a curl')
                data['cases'].append(case)
            print(json.dumps(dict(curve=curve,order=p,greedy_max=max(x['curl_fit_residual'] for x in case['greedy']),iterative_max=max(x['relative_curl_field_error'] for x in case['iterative']))),flush=True)
            output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf8')
    assert identities==hashes()|{name:hashlib.sha256((ROOT/name).read_text(encoding='utf8').encode()).hexdigest()}
    data['complete']=True
    output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    run(args.output)
