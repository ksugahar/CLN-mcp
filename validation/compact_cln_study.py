"""Matched full V/Phi-versus-A-T coefficient, vector-field, residual and cost audit."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import socket
import time
import matlab.engine
from pathlib import Path
from unittest.mock import patch

import ngsolve as ng
import numpy as np
from threadpoolctl import threadpool_info,threadpool_limits

import element_cln_study as baseline
from cln3d.batched_curl import BatchedSweep
from cln3d.mex_curl import MexSweep
from cln3d import CLN_VPhi_bulk, CLN_VPhi_cached, CLN_VPhi_compact
from cln3d.cached_checks import CachedSweep
from cln3d.compact_checks import CachedSweep as CompactSweep

class CompactMexSweep(MexSweep,CompactSweep):
    pass

class CachedMexSweep(MexSweep,CachedSweep):
    pass
from cln3d import CLN_VPhi
from cln3d.element_recovery import Condensed

ROOT=Path(__file__).resolve().parents[1]


def hashes():
    data=baseline.hashes()
    for name in ('validation/bulk_cln_study.py','validation/mexcuda_cln_study.py','validation/cln3d/mex_curl.py','validation/cln3d/mexcuda/colored_projection.cu','validation/build_mexcuda.py','validation/cln3d/bulk_curl.py','validation/cln3d/bulk_scalar.py','validation/cln3d/CLN_VPhi_bulk.py','validation/scalar_cln_study.py','validation/batched_cln_study.py','validation/cln3d/batched_curl.py','validation/cln3d/scalar_potentials.py',
                 'validation/cln3d/CLN_VPhi.py','validation/cln3d/compatible_curl.py'):
        data[name]=hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()
    for name in ('validation/cached_cln_study.py','validation/cln3d/cached_checks.py','validation/cln3d/CLN_VPhi_cached.py'):
        data[name]=hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()
    for name in ('validation/compact_cln_study.py','validation/cln3d/compact_checks.py','validation/cln3d/CLN_VPhi_compact.py'):
        data[name]=hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()
    return data


def scalar_execute(mesh,order,fast=False):
    # These guards stay active throughout setup and every recursion stage.
    # The reference routes are executed separately, outside these guards.
    with patch.object(baseline.CLN_AT,'solve_iccg',side_effect=AssertionError('Global A-T solve called by scalar route')), \
         patch.object(Condensed,'__init__',side_effect=AssertionError('Vector trace fallback called by scalar route')), \
         contextlib.redirect_stdout(io.StringIO()), \
         patch.object(CLN_VPhi,'CompatibleSweep',BatchedSweep if fast else CLN_VPhi.CompatibleSweep):
        return CLN_VPhi.run(mesh,order,3,8)


def bulk_execute(mesh,order,recovery_class=None,cached=False,compact=False):
    with patch.object(baseline.CLN_AT,'solve_iccg',side_effect=AssertionError('Global A-T fallback forbidden')), \
         patch.object(Condensed,'__init__',side_effect=AssertionError('Global vector trace fallback forbidden')), \
         contextlib.redirect_stdout(io.StringIO()):
        kwargs={} if recovery_class is None else {'recovery_class':recovery_class}
        return (CLN_VPhi_compact if compact else CLN_VPhi_cached if cached else CLN_VPhi_bulk).run(mesh,order,3,8,**kwargs)


ENGINE_RUNTIME={}


def run(output,quick=False,directory=None):
    if directory is None:raise ValueError('Explicit built MEX directory required')
    binary=directory/'cln_projection_mex.mexw64'
    if not binary.is_file():raise FileNotFoundError('Build CUDA MEX before benchmarking')
    start=time.perf_counter()
    engine=matlab.engine.start_matlab('-nodesktop -nosplash')
    ENGINE_RUNTIME['start_seconds']=time.perf_counter()-start
    try:
        engine.addpath(str(directory.resolve()),nargout=0)
        engine.maxNumCompThreads(1.,nargout=0)
        ENGINE_RUNTIME['version']=engine.version()
        ENGINE_RUNTIME['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        start=time.perf_counter();engine.eval('cln_device=gpuDevice;',nargout=0)
        ENGINE_RUNTIME['device_start_seconds']=time.perf_counter()-start
        ENGINE_RUNTIME['device']=engine.eval('cln_device.Name')
        ENGINE_RUNTIME['compute_capability']=engine.eval('cln_device.ComputeCapability')
        MexSweep.engine=engine;MexSweep.device_name=ENGINE_RUNTIME['device'];MexSweep.matlab_version=ENGINE_RUNTIME['version']
        ng.SetNumThreads(4)
        with threadpool_limits(limits=1,user_api='blas'):
            _run(output,quick)
    finally:
        MexSweep.engine=None
        engine.quit()


def _run(output,quick=False):
    result=dict(schema=1,complete=False,source_sha256_lf=hashes(),matlab_engine=dict(ENGINE_RUNTIME),
        runtime=dict(host=socket.gethostname(),python=platform.python_version(),ngsolve=ng.__version__,
                     numpy=np.__version__,radia=importlib.metadata.version('radia'),threads=4,
                     blas_pools=[{k:v for k,v in pool.items() if k!='filepath'} for pool in threadpool_info()]),
        scope='Compatible hybrid-mixed discretization of Biro V/Phi Poisson equations; zero-shift internal wire; only scalar trace global solves; local curl inverse',
        timing_protocol='First observed plus three warmed complete runs per route, alternating order; setup, scalar/local recovery and all residual checks included; mesh and cross-comparison excluded',cohorts=[])
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for h,p in ([(.003,2)] if quick else [(.003,2),(.003,3),(.002,3),(.003,4)]):
        mesh=baseline.CLN_AT.make_mesh(h,3)
        cohort=dict(maxh=h,order=p,stages=3,mesh_sha256=baseline.mesh_identity(mesh),ne=mesh.ne,repeats=[],comparisons=[])
        result['cohorts'].append(cohort)
        with ng.TaskManager():
            for repeat in range(1 if quick else 4):
                paths=['original','condensed','scalar-compact-mexcuda','scalar-compact','scalar-cached','scalar-cached-mexcuda']
                if repeat%2:paths.reverse()
                records,fields={},{}
                for route in paths:
                    if route in ('scalar-cached','scalar-cached-mexcuda'):
                        record,field=bulk_execute(mesh,p,recovery_class=CachedMexSweep if route.endswith('mexcuda') else CachedSweep,cached=True)
                        record['route']=route
                    elif route=='scalar-compact':
                        record,field=bulk_execute(mesh,p,recovery_class=CompactSweep,compact=True)
                    elif route=='scalar-compact-mexcuda':
                        record,field=bulk_execute(mesh,p,recovery_class=CompactMexSweep,compact=True)
                        record['route']=route
                    else:record,field=baseline.execute(mesh,h,p,3,route)
                    record['repeat']=repeat
                    cohort['repeats'].append(record)
                    records[route],fields[route]=record,field
                    print(json.dumps(dict(order=p,maxh=h,repeat=repeat,route=route,status=record['status'],seconds=record['seconds'])),flush=True)
                    save()
                    if record['status']!='pass':raise RuntimeError('Reference failed')
                for route in ('condensed','scalar-compact-mexcuda','scalar-compact','scalar-cached','scalar-cached-mexcuda'):
                    volume=[]
                    for f,g in zip(fields['original'],fields[route]):
                        delta=f-g
                        volume.append(float(ng.Integrate(delta*delta*ng.dx(bonus_intorder=8),mesh)/ng.Integrate(f*f*ng.dx(bonus_intorder=8),mesh))**.5)
                    error=np.array(records[route]['coefficients'])/records['original']['coefficients']-1
                    cohort['comparisons'].append(dict(route=route,repeat=repeat,relative_coefficients=error.tolist(),volume_relative_field_errors=volume))
                    if max(abs(error))>1e-6 or max(volume)>1e-6:raise RuntimeError('Scalar/AT parity failed')
                save()
    assert result['source_sha256_lf']==hashes()
    result['complete']=True
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    run(args.output,args.quick,args.directory)

