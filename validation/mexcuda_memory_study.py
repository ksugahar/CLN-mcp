"""Non-timing fresh-process CPU/MATLAB-process memory comparison for full CLN."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import matlab.engine
import ngsolve as ng
import psutil
from threadpoolctl import threadpool_limits
import mexcuda_cln_verify as verify
from cln3d.mex_curl import MexSweep
from bulk_memory_study import snapshot


def hashes():
    result=verify.hashes()
    root=Path(__file__).resolve().parents[1]
    for name in ('validation/mexcuda_memory_study.py','validation/bulk_memory_study.py'):
        result[name]=hashlib.sha256((root/name).read_text().encode()).hexdigest()
    return result


def worker(directory,output,route,h,p):
    ng.SetNumThreads(4);engine=None;matlab_process=None
    result=dict(route=route,maxh=h,order=p,stages=3,import_baseline=snapshot())
    done=threading.Event();joint=[0];sampler=None
    def observe():
        total=psutil.Process().memory_info().rss
        if matlab_process is not None:total+=matlab_process.memory_info().rss
        joint[0]=max(joint[0],int(total))
    def sample():
        while not done.is_set():observe();done.wait(.02)
    try:
        if route=='scalar-mexcuda':
            engine=matlab.engine.start_matlab('-nodesktop -nosplash')
            engine.addpath(str(directory.resolve()),nargout=0);engine.maxNumCompThreads(1.,nargout=0)
            engine.eval('cln_device=gpuDevice;',nargout=0)
            matlab_process=psutil.Process(int(engine.eval("feature('getpid')")))
            result['matlab_baseline']={k:int(v) for k,v in matlab_process.memory_info()._asdict().items() if k in ('rss','private','peak_wset','peak_pagefile')}
            result['matlab']=engine.version();result['device']=engine.eval('cln_device.Name')
            MexSweep.engine=engine;MexSweep.device_name=result['device'];MexSweep.matlab_version=result['matlab']
        with threadpool_limits(limits=1,user_api='blas'):
            mesh=verify.study.baseline.CLN_AT.make_mesh(h,3)
            result['mesh_sha256']=verify.study.baseline.mesh_identity(mesh)
            result['ne']=mesh.ne;result['pre_solve']=snapshot()
            observe();result['joint_pre_solve_rss']=joint[0]
            sampler=threading.Thread(target=sample,daemon=True);sampler.start()
            with ng.TaskManager():
                if route=='scalar-mexcuda':row,fields=verify.study.bulk_execute(mesh,p,MexSweep)
                elif route=='scalar-bulk':row,fields=verify.study.bulk_execute(mesh,p)
                else:row,fields=verify.study.baseline.execute(mesh,h,p,3,route)
            row['route']=route;result['record']=verify.without_timings(row)
            result['post_solve']=snapshot();observe()
            done.set();sampler.join();result['joint_sampled_peak_rss']=joint[0]
            if matlab_process is not None:
                result['matlab_post_solve']={k:int(v) for k,v in matlab_process.memory_info()._asdict().items() if k in ('rss','private','peak_wset','peak_pagefile')}
            del fields;gc.collect();result['post_gc']=snapshot()
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    finally:
        done.set()
        if sampler is not None:sampler.join()
        MexSweep.engine=None
        if engine is not None:engine.quit()


def run(directory,output,workers):
    workers.mkdir(parents=True,exist_ok=True)
    result=dict(schema=1,complete=False,host=socket.gethostname(),source_sha256_lf=hashes(),
        protocol='Memory/accuracy only; fresh process per route/cohort; native process peaks plus 20 ms sampled simultaneous Python+owned MATLAB RSS; no performance timings',cases=[])
    for h,p in ((.003,2),(.003,3),(.002,3),(.003,4)):
        for route in ('original','condensed','scalar-bulk','scalar-mexcuda'):
            path=workers/f'p{p}-h{h:g}-{route}.json'
            command=[sys.executable,__file__,'--worker','--directory',str(directory),'--output',str(path),'--route',route,'--order',str(p),'--maxh',str(h)]
            with path.with_suffix('.log').open('w',encoding='utf8') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            row=json.loads(path.read_text());result['cases'].append(row)
            output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
            print(json.dumps(dict(order=p,maxh=h,route=route,joint_peak_mib=row['joint_sampled_peak_rss']/2**20)),flush=True)
    assert hashes()==result['source_sha256_lf']
    result['complete']=True;output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=Path,default=Path('mex-memory-workers'))
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--route');parser.add_argument('--order',type=int);parser.add_argument('--maxh',type=float)
    args=parser.parse_args()
    if args.worker:worker(args.directory,args.output,args.route,args.maxh,args.order)
    else:run(args.directory,args.output,args.workers)
