// Generic colored local right-inverse projections, FP64, zero-based indices.
// NGSolve owns all FE operators and constraints; this kernel owns no physics.
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
void checked(cudaError_t code) { if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }

__global__ void project(double* value,const int64_t* ids,const int64_t* active,
                        const double* c,const double* inverse,const double* target,
                        int local,int rank,int free_count) {
    int e=blockIdx.x,t=threadIdx.x;
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
__global__ void fit(const double* value,const int64_t* ids,const double* c,
                    const double* target,double* squared,int local,int rank) {
    int e=blockIdx.x,t=threadIdx.x;
    extern __shared__ double residual[];
    if(t<rank) {
        double v=-target[e*rank+t];
        for(int j=0;j<local;j++) v+=c[(e*rank+t)*local+j]*value[ids[e*local+j]];
        residual[t]=v*v;
    }
    __syncthreads();
    if(t==0) { double v=0;for(int j=0;j<rank;j++)v+=residual[j];squared[e]=v; }
}
__global__ void reduce(const double* input,double* output,int count) {
    __shared__ double sums[256];
    int t=threadIdx.x;double v=0;
    for(int i=t;i<count;i+=256)v+=input[i];
    sums[t]=v;__syncthreads();
    for(int stride=128;stride;stride/=2) {
        if(t<stride)sums[t]+=sums[t+stride];__syncthreads();
    }
    if(t==0)*output=sums[0];
}

struct Buffer {
    void* data=nullptr;size_t bytes=0;
    ~Buffer() { if(data)cudaFree(data); }
    void allocate(size_t n) { require(!data && n>0,"Invalid CUDA allocation");checked(cudaMalloc(&data,n));bytes=n; }
    void upload(const void* source) { checked(cudaMemcpy(data,source,bytes,cudaMemcpyHostToDevice)); }
    template<class T>T* as() { return static_cast<T*>(data); }
};
struct Group { int count,rank,local,free_count;size_t ids,active,c,inverse,target,squared; };
struct State {
    int ndof=0,targets=0,elements=0;
    std::vector<Group> groups;
    Buffer ids,active,c,inverse,value,target,squared,total;
    size_t bytes() { return ids.bytes+active.bytes+c.bytes+inverse.bytes+value.bytes+target.bytes+squared.bytes+total.bytes; }
};
std::unordered_map<uint64_t,std::unique_ptr<State>> registry;
uint64_t next_handle=uint64_t(Clock::now().time_since_epoch().count())|uint64_t(1);
bool registered=false;
void cleanup() { registry.clear(); }
size_t count(const mxArray* a,mxClassID type) {
    require(mxGetClassID(a)==type && !mxIsComplex(a) && !mxIsSparse(a),"Invalid dense input type");
    return mxGetNumberOfElements(a);
}
double scalar(const mxArray* a) {
    require(count(a,mxDOUBLE_CLASS)==1,"Scalar double required");
    double v=mxGetScalar(a);require(std::isfinite(v),"Finite scalar required");return v;
}
size_t integer(double v) {
    require(std::isfinite(v) && v>=0 && v<=2147483647. && v==std::floor(v),"Invalid dimension/index");return static_cast<size_t>(v);
}
void range(size_t offset,size_t n,size_t available) { require(offset<=available && n<=available-offset,"Operator buffer out of bounds"); }
uint64_t handle(const mxArray* a) {
    require(count(a,mxUINT64_CLASS)==1,"Checked uint64 handle required");
    auto h=*static_cast<const uint64_t*>(mxGetData(a));
    require(registry.count(h)==1,"Stale, foreign or destroyed projection handle");return h;
}

void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]) {
    try {
        require(nrhs>=1 && mxIsChar(prhs[0]),"Command required");
        char* raw=mxArrayToUTF8String(prhs[0]);require(raw!=nullptr,"Invalid command");
        std::string command(raw);mxFree(raw);
        if(command=="setup") {
            require(nrhs==7 && nlhs==2,"setup(ndof,metadata,ids,active,C,inverse) requires two outputs");
            mxInitGPU();
            auto s=std::make_unique<State>();s->ndof=static_cast<int>(integer(scalar(prhs[1])));
            require(s->ndof>0,"Positive coefficient count required");
            size_t nm=count(prhs[2],mxDOUBLE_CLASS);require(nm>0 && nm%9==0,"Invalid group metadata");
            size_t ni=count(prhs[3],mxINT64_CLASS),na=count(prhs[4],mxINT64_CLASS);
            size_t nc=count(prhs[5],mxDOUBLE_CLASS),nv=count(prhs[6],mxDOUBLE_CLASS);
            const double* m=mxGetPr(prhs[2]);
            const auto* ids=static_cast<const int64_t*>(mxGetData(prhs[3]));
            const auto* active=static_cast<const int64_t*>(mxGetData(prhs[4]));
            for(size_t j=0;j<nm;j+=9) {
                size_t v[9];for(int k=0;k<9;k++)v[k]=integer(m[j+k]);
                require(v[0]>0 && v[1]>0 && v[2]>0 && v[1]<=1024 && v[2]<=1024 && v[3]<=1024,"Unsupported local dimensions");
                size_t n=v[0],rank=v[1],local=v[2],nf=v[3];
                require(n*rank*local<=2147483647ULL && n*nf*rank<=2147483647ULL,"CUDA local index overflow");
                range(v[4],n*local,ni);range(v[5],n*nf,na);
                range(v[6],n*rank*local,nc);range(v[7],n*nf*rank,nv);
                require(v[8]==static_cast<size_t>(s->targets),"Noncontiguous targets");
                require(n*rank<=2147483647ULL-s->targets && n<=2147483647ULL-s->elements,"Target size overflow");
                std::unordered_set<int64_t> writable;
                for(size_t e=0;e<n;e++) {
                    std::unordered_set<int64_t> support;
                    for(size_t k=0;k<local;k++) {
                        int64_t id=ids[v[4]+e*local+k];require(id>=0 && id<s->ndof,"Invalid global index");support.insert(id);
                    }
                    for(size_t k=0;k<nf;k++) {
                        int64_t id=active[v[5]+e*nf+k];
                        require(support.count(id)==1 && writable.insert(id).second,"Invalid or overlapping writable support");
                    }
                }
                for(size_t e=0;e<n;e++) {
                    std::unordered_set<int64_t> own;
                    for(size_t k=0;k<nf;k++)own.insert(active[v[5]+e*nf+k]);
                    for(size_t k=0;k<local;k++) {
                        int64_t id=ids[v[4]+e*local+k];
                        require(!writable.count(id) || own.count(id),"Cross-element read/write collision");
                    }
                }
                s->groups.push_back({int(n),int(rank),int(local),int(nf),v[4],v[5],v[6],v[7],v[8],size_t(s->elements)});
                s->targets+=int(n*rank);s->elements+=int(n);
            }
            for(size_t j=0;j<nc;j++)require(std::isfinite(mxGetPr(prhs[5])[j]),"Nonfinite curl operator");
            for(size_t j=0;j<nv;j++)require(std::isfinite(mxGetPr(prhs[6])[j]),"Nonfinite inverse operator");
            s->ids.allocate(ni*sizeof(int64_t));s->ids.upload(ids);
            s->active.allocate(na*sizeof(int64_t));s->active.upload(active);
            s->c.allocate(nc*sizeof(double));s->c.upload(mxGetPr(prhs[5]));
            s->inverse.allocate(nv*sizeof(double));s->inverse.upload(mxGetPr(prhs[6]));
            s->value.allocate(size_t(s->ndof)*8);s->target.allocate(size_t(s->targets)*8);
            s->squared.allocate(size_t(s->elements)*8);s->total.allocate(8);
            checked(cudaDeviceSynchronize());
            require(next_handle!=0,"Handle generation exhausted");uint64_t h=next_handle++;
            plhs[0]=mxCreateNumericMatrix(1,1,mxUINT64_CLASS,mxREAL);*static_cast<uint64_t*>(mxGetData(plhs[0]))=h;
            plhs[1]=mxCreateDoubleScalar(double(s->bytes()));
            if(!registered) { mexAtExit(cleanup);registered=true; }
            if(registry.empty())mexLock();registry.emplace(h,std::move(s));return;
        }
        if(command=="destroy") {
            require(nrhs==2 && nlhs==0,"destroy(handle) returns nothing");
            registry.erase(handle(prhs[1]));if(registry.empty())mexUnlock();return;
        }
        require(command=="solve" && nrhs==6 && nlhs==3,"solve(handle,target,denominator,tolerance,max_sweeps) requires three outputs");
        State& s=*registry.at(handle(prhs[1]));
        require(count(prhs[2],mxDOUBLE_CLASS)==size_t(s.targets),"Wrong target size");
        double denominator=scalar(prhs[3]),tolerance=scalar(prhs[4]);
        int limit=int(integer(scalar(prhs[5])));
        require(denominator>0 && tolerance>0 && tolerance<=1e-8 && limit>0,"Invalid convergence limits");
        for(int j=0;j<s.targets;j++)require(std::isfinite(mxGetPr(prhs[2])[j]),"Nonfinite target");
        auto begin=Clock::now();s.target.upload(mxGetPr(prhs[2]));
        checked(cudaMemset(s.value.data,0,s.value.bytes));checked(cudaDeviceSynchronize());
        double upload=seconds(begin),checks=0.;begin=Clock::now();
        std::vector<double> iterations,residuals;
        for(int sweep=1;sweep<=limit;sweep++) {
            for(auto g:s.groups) {
                int threads=((std::max(g.rank,g.free_count)+31)/32)*32;
                project<<<g.count,threads,g.rank*8>>>(s.value.as<double>(),s.ids.as<int64_t>()+g.ids,s.active.as<int64_t>()+g.active,
                    s.c.as<double>()+g.c,s.inverse.as<double>()+g.inverse,s.target.as<double>()+g.target,g.local,g.rank,g.free_count);
                checked(cudaGetLastError());
            }
            if(sweep==1 || sweep%10==0 || sweep==limit) {
                checked(cudaDeviceSynchronize());auto check=Clock::now();
                for(auto g:s.groups) {
                    int threads=((g.rank+31)/32)*32;
                    fit<<<g.count,threads,g.rank*8>>>(s.value.as<double>(),s.ids.as<int64_t>()+g.ids,s.c.as<double>()+g.c,
                        s.target.as<double>()+g.target,s.squared.as<double>()+g.squared,g.local,g.rank);
                    checked(cudaGetLastError());
                }
                reduce<<<1,256>>>(s.squared.as<double>(),s.total.as<double>(),s.elements);checked(cudaGetLastError());
                double sum=0;checked(cudaMemcpy(&sum,s.total.data,8,cudaMemcpyDeviceToHost));
                double residual=std::sqrt(sum)/denominator;require(std::isfinite(residual),"Nonfinite CUDA residual");
                iterations.push_back(sweep);residuals.push_back(residual);checks+=seconds(check);
                if(residual<tolerance)break;
            }
        }
        checked(cudaDeviceSynchronize());double iteration=seconds(begin)-checks;
        begin=Clock::now();plhs[0]=mxCreateDoubleMatrix(s.ndof,1,mxREAL);
        checked(cudaMemcpy(mxGetPr(plhs[0]),s.value.data,s.value.bytes,cudaMemcpyDeviceToHost));double download=seconds(begin);
        plhs[1]=mxCreateDoubleMatrix(iterations.size(),2,mxREAL);
        std::copy(iterations.begin(),iterations.end(),mxGetPr(plhs[1]));
        std::copy(residuals.begin(),residuals.end(),mxGetPr(plhs[1])+iterations.size());
        plhs[2]=mxCreateDoubleMatrix(1,4,mxREAL);
        double* timing=mxGetPr(plhs[2]);timing[0]=upload;timing[1]=iteration;timing[2]=checks;timing[3]=download;
    } catch(const std::exception& e) { mexErrMsgIdAndTxt("CLN:CUDA:Projection","%s",e.what()); }
}
