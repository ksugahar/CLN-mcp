"""Measure fresh MATLAB Engine and CUDA-context initialization, including configuration."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time
import matlab.engine
from mexcuda_host_timing import cpu_times,foreign_jobs,gpu


def run(directory,output):
    root=Path(__file__).resolve().parents[1]
    before=cpu_times();time.sleep(3);after=cpu_times()
    delta=[b-a for a,b in zip(before,after)]
    busy=1-delta[0]/(delta[1]+delta[2]);pre=gpu();jobs=foreign_jobs()
    if busy>.05 or pre['utilization_percent'] or jobs:raise RuntimeError('Initialization host is not idle')
    result=dict(schema=1,complete=False,binary_sha256=hashlib.sha256((directory/'cln_projection_mex.mexw64').read_bytes()).hexdigest(),
        preflight_cpu_busy_fraction=busy,preflight_gpu=pre,foreign_compute_job_samples=[jobs],
        source_sha256_lf={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest()
            for p in (Path(__file__),root/'validation/mexcuda_host_timing.py')},
        protocol='Four sequential fresh Engine processes and CUDA contexts; installed MEX, filesystem and driver caches may be warm; Python imports excluded',runs=[])
    for repeat in range(4):
        begin=time.perf_counter();engine=matlab.engine.start_matlab('-nodesktop -nosplash')
        start=time.perf_counter();row=dict(repeat=repeat,engine_start_seconds=start-begin)
        try:
            engine.addpath(str(directory.resolve()),nargout=0)
            engine.maxNumCompThreads(1.,nargout=0)
            row['matlab']=engine.version()
            row['matlab_pid']=int(engine.feature('getpid'))
            now=time.perf_counter();row['configuration_seconds']=now-start
            engine.eval('cln_device=gpuDevice;',nargout=0)
            row['device_start_seconds']=time.perf_counter()-now
            row['device']=engine.eval('cln_device.Name')
            row['compute_capability']=engine.eval('cln_device.ComputeCapability')
            row['initialization_seconds']=time.perf_counter()-begin
        finally:
            # quit() acknowledges asynchronously. Wait on this exact process
            # handle before starting another fresh session; PID reuse is safe.
            kernel=ctypes.windll.kernel32
            kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
            kernel.OpenProcess.restype=ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
            kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            handle=kernel.OpenProcess(0x1000|0x100000,False,int(engine.feature('getpid')))
            if not handle:raise ctypes.WinError()
            try:
                start=time.perf_counter();engine.quit()
                row['quit_call_seconds']=time.perf_counter()-start
                if kernel.WaitForSingleObject(handle,30000)!=0:raise RuntimeError('Owned MATLAB did not exit within 30 seconds')
                row['shutdown_seconds']=time.perf_counter()-start
            finally:kernel.CloseHandle(handle)
        result['runs'].append(row)
        result['foreign_compute_job_samples'].append(foreign_jobs())
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    result['complete']=not any(result['foreign_compute_job_samples'])
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    if not result['complete']:raise RuntimeError('Foreign compute overlapped initialization study')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.output)
