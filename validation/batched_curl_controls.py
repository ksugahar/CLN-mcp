"""Native rejection controls and separate untimed-protocol recovery profile."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import socket
import time
from unittest.mock import patch
import ngsolve as ng
import numpy as np
from threadpoolctl import threadpool_limits
from cln3d.CLN_AT import make_mesh
from cln3d.compatible_curl import CompatibleSweep
from cln3d.batched_curl import BatchedSweep
from cln3d.CLN_VPhi_fast import run as fast_run
from cln3d import CLN_VPhi,CLN_AT
from cln3d.element_recovery import Condensed

ROOT=Path(__file__).resolve().parents[1]


def run(output):
    ng.SetNumThreads(4)
    result=dict(schema=1,complete=False,host=socket.gethostname(),cases=[],profiles=[],batch_equivalence=[],setup_profiles=[])
    names=['validation/batched_curl_controls.py','validation/cln3d/batched_curl.py',
           'validation/cln3d/compatible_curl.py','validation/cln3d/element_recovery.py','validation/cln3d/CLN_AT.py',
           'validation/cln3d/CLN_VPhi_fast.py']
    result['source_sha256_lf']={n:hashlib.sha256((ROOT/n).read_text(encoding='utf-8-sig').encode()).hexdigest() for n in names}
    with threadpool_limits(limits=1,user_api='blas'),ng.TaskManager():
        mesh=make_mesh(.003,3)
        for p in (2,3,4):
            engine=BatchedSweep(mesh,p)
            lift=ng.GridFunction(engine.space)
            lift.Set(ng.CoefficientFunction((-ng.y/2,ng.x/2,0)))
            field=ng.curl(lift)
            random=np.random.default_rng(19)
            sequential=random.normal(size=engine.space.ndof);sequential[~engine.free]=0
            batched=sequential.copy()
            for indices,ids,active,c,inverse in engine.batches:
                targets=random.normal(size=c.shape[:2])
                for k,j in enumerate(indices):
                    _,ei,free,ec,_,pinv=engine.blocks[j]
                    sequential[ei[free]]+=pinv@(targets[k]-ec@sequential[ei])
                batched[active]+=np.einsum('bij,bj->bi',inverse,
                    targets-np.einsum('bij,bj->bi',c,batched[ids],optimize=False),optimize=False)
            difference=float(np.linalg.norm(batched-sequential)/np.linalg.norm(sequential))
            assert difference<1e-12 and np.max(abs(batched[~engine.free]),initial=0)==0
            result['batch_equivalence'].append(dict(order=p,relative_difference=difference))
            for name,target,cap in [('capped',field,1),('incompatible',ng.CoefficientFunction(((ng.x/.01)**6,0,0)),200)]:
                try:engine.recover(target,max_sweeps=cap)
                except RuntimeError as exc:
                    result['cases'].append(dict(order=p,control=name,rejected=True,reason=str(exc),info=getattr(exc,'info',{})))
                else:raise AssertionError(f'Control accepted: {p} {name}')
        for cls in (CompatibleSweep,BatchedSweep):
            engine=cls(mesh,3)
            lift=ng.GridFunction(engine.space)
            lift.Set(ng.CoefficientFunction((-ng.y/2,ng.x/2,0)))
            field=ng.curl(lift)
            profiler=cProfile.Profile()
            profiler.enable();_,info=engine.recover(field);profiler.disable()
            stats=pstats.Stats(profiler)
            rows=[]
            for (filename,line,name),(primitive,calls,self_time,cumulative,callers) in stats.stats.items():
                rows.append(dict(function=name,file=Path(filename).name,line=line,calls=calls,
                                 primitive_calls=primitive,self_seconds=self_time,cumulative_seconds=cumulative))
            result['profiles'].append(dict(route=cls.__name__,info=info,
                functions=sorted(rows,key=lambda x:x['self_seconds'],reverse=True)))
        result['fast_entrypoint'],_=fast_run(mesh,2,3)
        scalar_class=CLN_VPhi.ScalarPoisson
        for recovery_class in (CompatibleSweep,BatchedSweep):
            construction=[]
            def factory(cls,kind):
                def construct(*args,**kwargs):
                    start=time.perf_counter();obj=cls(*args,**kwargs)
                    magnetic=args[2] if kind=='scalar' else bool(args[3])
                    construction.append(dict(kind=kind,role='A' if magnetic else 'T',seconds=time.perf_counter()-start))
                    return obj
                return construct
            with patch.object(CLN_VPhi,'ScalarPoisson',factory(scalar_class,'scalar')), \
                 patch.object(CLN_VPhi,'CompatibleSweep',factory(recovery_class,'recovery')), \
                 patch.object(CLN_AT,'solve_iccg',side_effect=AssertionError('Global A-T fallback')), \
                 patch.object(Condensed,'__init__',side_effect=AssertionError('Global vector fallback')):
                record,_=CLN_VPhi.run(mesh,3,3,8)
            result['setup_profiles'].append(dict(route=recovery_class.__name__,construction=construction,record=record))
    result['complete']=True
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
