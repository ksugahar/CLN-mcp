"""Explicit CUDA FP64 local projections; FE assembly/checks remain on CPU.

Optional CuPy backend. Missing CUDA is an error, never host substitution.
Each block owns one element; disjoint free supports avoid atomic updates.
"""
import time
import numpy as np
from .bulk_curl import BulkSweep


KERNEL=r'''
extern "C" __global__ void project(
    double* value, const long long* ids, const long long* active,
    const double* c, const double* inverse, const double* target,
    int local, int rank, int free_count)
{
    int e=blockIdx.x, t=threadIdx.x;
    extern __shared__ double residual[];
    if(t<rank) {
        double v=target[e*rank+t];
        for(int j=0;j<local;j++) v-=c[(e*rank+t)*local+j]*value[ids[e*local+j]];
        residual[t]=v;
    }
    __syncthreads();
    if(t<free_count) {
        double v=0;
        for(int j=0;j<rank;j++) v+=inverse[(e*free_count+t)*rank+j]*residual[j];
        value[active[e*free_count+t]]+=v;
    }
}
extern "C" __global__ void fit(
    const double* value, const long long* ids, const double* c,
    const double* target, double* squared, int local, int rank)
{
    int e=blockIdx.x, t=threadIdx.x;
    extern __shared__ double residual[];
    if(t<rank) {
        double v=-target[e*rank+t];
        for(int j=0;j<local;j++) v+=c[(e*rank+t)*local+j]*value[ids[e*local+j]];
        residual[t]=v*v;
    }
    __syncthreads();
    if(t==0) {
        double v=0;
        for(int j=0;j<rank;j++) v+=residual[j];
        squared[e]=v;
    }
}
'''


class GPUSweep(BulkSweep):
    def __init__(self,*args,**kwargs):
        start=time.perf_counter()
        super().__init__(*args,**kwargs)
        import cupy as cp
        self.cp=cp
        properties=cp.cuda.runtime.getDeviceProperties(0)
        self.device_name=properties['name'].decode()
        self.module=cp.RawModule(code=KERNEL,options=('--std=c++11',))
        self.project=self.module.get_function('project');self.fit=self.module.get_function('fit')
        self.device=[]
        for _,ids,active,c,inverse,_,_,_ in self.packed:
            self.device.append((cp.asarray(ids,dtype=cp.int64),cp.asarray(active,dtype=cp.int64),
                                cp.asarray(c),cp.asarray(inverse)))
        cp.cuda.get_current_stream().synchronize()
        self.device_operator_bytes=sum(a.nbytes for group in self.device for a in group)
        self.construction_seconds=time.perf_counter()-start

    def recover(self,*args,**kwargs):
        gf,info=super().recover(*args,**kwargs)
        info['backend']='cuda-local/cpu-fe'
        return gf,info

    def iterate(self,batches,denominator,tolerance,max_sweeps):
        cp=self.cp;start=time.perf_counter()
        value=cp.zeros(self.space.ndof,dtype=cp.float64)
        targets=[cp.asarray(b[-1]) for b in batches]
        squares=[cp.empty(t.shape[0],dtype=cp.float64) for t in targets]
        cp.cuda.get_current_stream().synchronize()
        upload=time.perf_counter()-start;history=[];check_seconds=0.
        for sweep in range(1,max_sweeps+1):
            for (ids,active,c,inverse),target in zip(self.device,targets):
                count,rank,local=c.shape;free_count=active.shape[1]
                threads=((max(rank,free_count)+31)//32)*32
                self.project((count,),(threads,),
                    (value,ids,active,c,inverse,target,np.int32(local),np.int32(rank),np.int32(free_count)),
                    shared_mem=rank*8)
            if sweep==1 or sweep%10==0 or sweep==max_sweeps:
                cp.cuda.get_current_stream().synchronize();check=time.perf_counter()
                squared=0.
                for (ids,active,c,inverse),target,square in zip(self.device,targets,squares):
                    count,rank,local=c.shape;threads=((rank+31)//32)*32
                    self.fit((count,),(threads,),(value,ids,c,target,square,np.int32(local),np.int32(rank)),shared_mem=rank*8)
                    squared+=float(cp.sum(square).get())
                residual=float(np.sqrt(squared)/denominator)
                history.append([sweep,residual]);check_seconds+=time.perf_counter()-check
                if residual<tolerance:break
        cp.cuda.get_current_stream().synchronize();download=time.perf_counter()
        host=cp.asnumpy(value);download=time.perf_counter()-download
        return host,dict(sweeps=sweep,curl_fit_residual=residual,history=history,
            iteration_seconds=time.perf_counter()-start-check_seconds,curl_check_seconds=check_seconds,
            device_upload_seconds=upload,device_download_seconds=download,
            device_operator_bytes=self.device_operator_bytes,device=self.device_name,
            cuda_runtime=int(cp.cuda.runtime.runtimeGetVersion()),cupy=cp.__version__,
            projection_backend='CUDA FP64 fused element kernels',gpu_dtype='float64')
