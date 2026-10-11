"""Fresh-process whole-method memory on 100; timing comes from saved mdx data."""
import argparse,gc,hashlib,json,platform,socket,subprocess,sys,threading,time
from pathlib import Path
import importlib.metadata
import numpy as np
import ngsolve as ng
import psutil
import matlab.engine
from threadpoolctl import threadpool_limits
import compact_cln_study as study
from mexcuda_cln_verify import without_timings
from compact_memory_study import snapshot
from cln3d.mex_curl import MexSweep
ROOT=Path(__file__).resolve().parents[1]
ROUTES=('original','condensed','scalar-cached','scalar-compact','scalar-cached-mexcuda','scalar-compact-mexcuda')

def hashes():
    result=study.hashes()
    for name in ('validation/whole_memory100.py','validation/compact_memory_study.py','validation/mexcuda_cln_verify.py'):
        result[name]=hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()
    return result

class Family:
    def __init__(self):
        self.root=psutil.Process();self.owned={};self.peaks={};self.samples=0
        self.joint_rss=0;self.joint_private=0;self.errors=[];self.stop=threading.Event()
        self.thread=threading.Thread(target=self.loop)
    def observe(self):
        rows=[self.root]+self.root.children(recursive=True)
        for p in rows:
            try:self.owned[(p.pid,p.create_time())]=p
            except psutil.NoSuchProcess:pass
        rss=private=0
        for key,p in list(self.owned.items()):
            try:
                if p.create_time()!=key[1]:continue
                info=p.memory_info()._asdict()
                rss+=info['rss'];private+=info['private']
                entry=self.peaks.setdefault(key,dict(pid=p.pid,creation=key[1],name=p.name(),peak_wset=0,peak_pagefile=0))
                for k in ('peak_wset','peak_pagefile'):entry[k]=max(entry[k],int(info[k]))
            except psutil.NoSuchProcess:pass
        self.samples+=1;self.joint_rss=max(self.joint_rss,rss);self.joint_private=max(self.joint_private,private)
    def loop(self):
        while not self.stop.is_set():
            try:self.observe()
            except Exception as e:self.errors.append(type(e).__name__+': '+str(e))
            self.stop.wait(.02)
    def finish(self):
        self.stop.set();self.thread.join();self.observe()
        return dict(sample_interval_seconds=.02,samples=self.samples,joint_sampled_peak_rss=self.joint_rss,
            joint_sampled_peak_private=self.joint_private,observed_native_peaks=list(self.peaks.values()),errors=self.errors,
            observed_native_peak_sum_rss=sum(x['peak_wset'] for x in self.peaks.values()),
            observed_native_peak_sum_private=sum(x['peak_pagefile'] for x in self.peaks.values()))

def worker(directory,output,route,h,p):
    ng.SetNumThreads(4);engine=None;family=Family();family.thread.start()
    result=dict(route=route,maxh=h,order=p,stages=3,import_baseline=snapshot())
    try:
        if route.endswith('mexcuda'):
            engine=matlab.engine.start_matlab('-nodesktop -nosplash')
            engine.addpath(str(directory.resolve()),nargout=0);engine.maxNumCompThreads(1.,nargout=0)
            engine.eval('cln_device=gpuDevice;',nargout=0)
            result.update(matlab=engine.version(),device=engine.eval('cln_device.Name'),matlab_pid=int(engine.eval("feature('getpid')")))
            MexSweep.engine=engine;MexSweep.device_name=result['device'];MexSweep.matlab_version=result['matlab']
        with threadpool_limits(limits=1,user_api='blas'):
            mesh=study.baseline.CLN_AT.make_mesh(h,3)
            result.update(mesh_sha256=study.baseline.mesh_identity(mesh),ne=mesh.ne,pre_solve=snapshot())
            with ng.TaskManager():
                if route=='scalar-cached':record,fields=study.bulk_execute(mesh,p,cached=True)
                elif route=='scalar-compact':record,fields=study.bulk_execute(mesh,p,study.CompactSweep,compact=True)
                elif route=='scalar-cached-mexcuda':record,fields=study.bulk_execute(mesh,p,study.CachedMexSweep,cached=True)
                elif route=='scalar-compact-mexcuda':record,fields=study.bulk_execute(mesh,p,study.CompactMexSweep,compact=True)
                else:record,fields=study.baseline.execute(mesh,h,p,3,route)
            record['route']=route;result['record']=without_timings(record)
            result['post_solve']=snapshot();family.observe()
            if engine is not None:
                result['known_device_operator_bytes']=sum(next(s['device_operator_bytes'] for s in record['solves'] if s['label'].startswith(role)) for role in ('A','T'))
            del fields;gc.collect();result['post_gc']=snapshot()
    finally:
        MexSweep.engine=None
        if engine is not None:
            pid=result.get('matlab_pid');engine.quit()
            if pid:
                try:psutil.Process(pid).wait(timeout=90)
                except psutil.NoSuchProcess:pass
        result['family']=family.finish()
    if result['family']['errors']:raise RuntimeError('Memory sampling failed')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')

def run(directory,output,workers):
    workers.mkdir(parents=True,exist_ok=True)
    mdx=ROOT/'docs/data/compact_cln_timing.json'
    result=dict(schema=1,complete=False,host=socket.gethostname(),host_role='100',python=platform.python_version(),
        ngsolve=ng.__version__,numpy=np.__version__,radia=importlib.metadata.version('radia'),threads=4,blas_threads=1,
        source_sha256_lf=hashes(),mdx_timing_sha256=hashlib.sha256(mdx.read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256((directory/'cln_projection_mex.mexw64').read_bytes()).hexdigest(),
        protocol='Memory only; one fresh Python process and owned MATLAB session per route/cohort. Sample owned process-family RSS/private every 20 ms from before Engine startup through actual MATLAB exit. Native lifetime peaks retained separately. Shared pages can be double-counted. Known MEX device operator allocation is not total VRAM peak. No 100-host timing claims.',cases=[])
    for h,p in ((.003,2),(.003,3),(.002,3),(.003,4)):
        for route in ROUTES:
            path=workers/f'p{p}-h{h:g}-{route}.json'
            with path.with_suffix('.log').open('w',encoding='utf8') as log:
                subprocess.run([sys.executable,__file__,'--worker','--directory',str(directory),'--output',str(path),'--route',route,'--order',str(p),'--maxh',str(h)],stdout=log,stderr=subprocess.STDOUT,check=True)
            row=json.loads(path.read_text(encoding='utf8'));result['cases'].append(row)
            output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
            print(json.dumps(dict(order=p,maxh=h,route=route,joint_peak_mib=row['family']['joint_sampled_peak_rss']/2**20)),flush=True)
    assert result['source_sha256_lf']==hashes()
    result['complete']=True;output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=Path,default=Path('memory-workers'));parser.add_argument('--worker',action='store_true')
    parser.add_argument('--route',choices=ROUTES);parser.add_argument('--order',type=int);parser.add_argument('--maxh',type=float)
    args=parser.parse_args()
    if args.worker:worker(args.directory,args.output,args.route,args.maxh,args.order)
    else:run(args.directory,args.output,args.workers)
