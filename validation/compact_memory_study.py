"""Isolated-process peak working-set/private-memory audit of complete CLN."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import socket
import subprocess
import sys
import ngsolve as ng
import psutil
from threadpoolctl import threadpool_limits
import compact_cln_study as study

ROOT=Path(__file__).resolve().parents[1]


def hashes():
    result=study.hashes()
    for name in ('validation/compact_memory_study.py',):
        result[name]=hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()
    return result


def snapshot():
    info=psutil.Process().memory_info()._asdict()
    if 'peak_wset' not in info or 'peak_pagefile' not in info:
        raise RuntimeError('This memory lane requires Windows native peak counters')
    return {k:int(info[k]) for k in ('rss','private','peak_wset','peak_pagefile')}


def worker(output,route,h,p):
    ng.SetNumThreads(4)
    with threadpool_limits(limits=1,user_api='blas'):
        gc.collect();before_imports=snapshot()
        mesh=study.baseline.CLN_AT.make_mesh(h,3)
        before=snapshot()
        with ng.TaskManager():
            if route=='scalar-cached':record,fields=study.bulk_execute(mesh,p,cached=True)
            elif route=='scalar-compact':record,fields=study.bulk_execute(mesh,p,recovery_class=study.CompactSweep,compact=True)
            else:record,fields=study.baseline.execute(mesh,h,p,3,route)
        after=snapshot()
        cache=record.get('local_numpy_bytes_by_role',{})
        result=dict(route=route,order=p,maxh=h,ne=mesh.ne,stages=3,mesh_sha256=study.baseline.mesh_identity(mesh),
            import_baseline=before_imports,pre_solve=before,post_solve=after,
            local_numpy_bytes_by_role=cache,record=record)
        del fields;gc.collect();result['post_gc']=snapshot()
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


def run(output,directory):
    directory.mkdir(parents=True,exist_ok=True)
    result=dict(schema=1,complete=False,source_sha256_lf=hashes(),
        host=socket.gethostname(),python=platform.python_version(),ngsolve=ng.__version__,radia=importlib.metadata.version("radia"),threads=4,blas_threads=1,
        protocol='One fresh process per cohort/route; same imports and mesh; native lifetime peak working set and private commit; not allocator estimates or timing medians',cases=[])
    routes=['original','condensed','scalar-cached','scalar-compact']
    for h,p in ((.003,2),(.003,3),(.002,3),(.003,4)):
        for route in routes:
            path=directory/f'p{p}-h{h:g}-{route}.json'
            command=[sys.executable,__file__,'--worker','--route',route,'--maxh',str(h),'--order',str(p),'--output',str(path)]
            with path.with_suffix('.log').open('w',encoding='utf8') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            row=json.loads(path.read_text(encoding='utf8'))
            result['cases'].append(row)
            output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
            print(json.dumps(dict(order=p,maxh=h,route=route,peak_mib=row['post_solve']['peak_wset']/2**20)),flush=True)
    result['complete']=True
    assert hashes()==result['source_sha256_lf']
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--directory',type=Path,default=Path('memory-workers'))
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--route',choices=['original','condensed','scalar-cached','scalar-compact'])
    parser.add_argument('--order',type=int)
    parser.add_argument('--maxh',type=float)
    args=parser.parse_args()
    if args.worker:worker(args.output,args.route,args.maxh,args.order)
    else:run(args.output,args.directory)
