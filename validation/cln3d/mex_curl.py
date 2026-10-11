"""Caller-owned MATLAB Engine and checked mexcuda local curl projections."""
import time
import weakref
import matlab
import numpy as np
from .bulk_curl import BulkSweep


class MexSweep(BulkSweep):
    # Set only inside the sequential experiment's caller-owned engine region.
    engine=None
    device_name=None
    matlab_version=None

    def __init__(self,*args,**kwargs):
        if self.engine is None:raise RuntimeError('Caller-owned MATLAB Engine required')
        begin=time.perf_counter()
        super().__init__(*args,**kwargs)
        ids=[];active=[];curl=[];inverse=[];metadata=[]
        offsets=[0]*5
        for _,i,a,c,v,_,_,_ in self.packed:
            count,rank,local=c.shape;free=a.shape[1]
            metadata.extend([count,rank,local,free,*offsets])
            ids.append(i.ravel());active.append(a.ravel());curl.append(c.ravel());inverse.append(v.ravel())
            for j,n in enumerate((i.size,a.size,c.size,v.size,count*rank)):offsets[j]+=n
        transfer=time.perf_counter()
        handle,nbytes=self.engine.feval('cln_projection_mex','setup',float(self.space.ndof),
            matlab.double(np.asarray(metadata,dtype=float)),matlab.int64(np.concatenate(ids)),
            matlab.int64(np.concatenate(active)),matlab.double(np.concatenate(curl)),
            matlab.double(np.concatenate(inverse)),nargout=2)
        handle=matlab.uint64([int(np.asarray(handle).item())])
        self.handle=handle;self.device_operator_bytes=int(nbytes)
        self.mex_setup_transfer_seconds=time.perf_counter()-transfer
        self.finalizer=weakref.finalize(self,self.engine.feval,'cln_projection_mex','destroy',handle,nargout=0)
        self.construction_seconds=time.perf_counter()-begin

    def close(self):
        self.finalizer()

    def iterate(self,batches,denominator,tolerance,max_sweeps):
        begin=time.perf_counter()
        target=matlab.double(np.concatenate([b[-1].ravel() for b in batches]))
        value,history,timing=self.engine.feval('cln_projection_mex','solve',self.handle,target,
            float(denominator),float(tolerance),float(max_sweeps),nargout=3)
        value=np.asarray(value,dtype=float).reshape(-1)
        history=np.asarray(history,dtype=float).tolist()
        timing=np.asarray(timing,dtype=float).reshape(-1)
        elapsed=time.perf_counter()-begin
        if len(history)==0 or not np.isfinite(value).all():raise RuntimeError('Invalid CUDA MEX result')
        # Complete bridge overhead is included, never hidden as FE work.
        info=dict(sweeps=int(history[-1][0]),curl_fit_residual=float(history[-1][1]),history=history,
            iteration_seconds=elapsed-float(timing[2]),curl_check_seconds=float(timing[2]),
            mex_internal_projection_seconds=float(timing[1]),mex_call_seconds=elapsed,
            matlab_bridge_seconds=max(0.,elapsed-float(sum(timing))),
            device_upload_seconds=float(timing[0]),device_download_seconds=float(timing[3]),
            device_operator_bytes=self.device_operator_bytes,device=self.device_name,
            matlab=self.matlab_version,mex_setup_transfer_seconds=self.mex_setup_transfer_seconds,
            projection_backend='mexcuda FP64 fused element kernels',gpu_dtype='float64')
        return value,info

    def recover(self,*args,**kwargs):
        value,info=super().recover(*args,**kwargs)
        info['backend']='mexcuda-local/cpu-fe'
        return value,info
