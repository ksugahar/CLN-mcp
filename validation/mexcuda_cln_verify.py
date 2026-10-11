"""Non-timing full-CLN CUDA MEX accuracy validation on an authorized test host."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import matlab.engine
import ngsolve as ng
import numpy as np
from threadpoolctl import threadpool_limits
import mexcuda_cln_study as study
from cln3d.mex_curl import MexSweep


def without_timings(value):
    if isinstance(value,dict):
        return {k:without_timings(v) for k,v in value.items() if k!='seconds' and not k.endswith('_seconds')}
    if isinstance(value,list):return [without_timings(v) for v in value]
    return value


def hashes():
    result=study.hashes()
    result['validation/mexcuda_cln_verify.py']=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest()
    return result


def run(directory,output,quick=False):
    ng.SetNumThreads(4)
    result=dict(schema=1,complete=False,host=socket.gethostname(),python=platform.python_version(),
        ngsolve=ng.__version__,numpy=np.__version__,threads=4,blas_threads=1,
        protocol='Accuracy validation only; one complete run per route/cohort; no performance timings',
        source_sha256_lf=hashes(),cohorts=[])
    engine=matlab.engine.start_matlab('-nodesktop -nosplash')
    try:
        engine.addpath(str(directory.resolve()),nargout=0);engine.maxNumCompThreads(1.,nargout=0)
        engine.eval('cln_device=gpuDevice;',nargout=0)
        result['matlab']=engine.version();result['device']=engine.eval('cln_device.Name')
        result['binary_sha256']=hashlib.sha256((directory/'cln_projection_mex.mexw64').read_bytes()).hexdigest()
        MexSweep.engine=engine;MexSweep.device_name=result['device'];MexSweep.matlab_version=result['matlab']
        with threadpool_limits(limits=1,user_api='blas'):
            for h,p in ([(.003,2)] if quick else [(.003,2),(.003,3),(.002,3),(.003,4)]):
                mesh=study.baseline.CLN_AT.make_mesh(h,3)
                cohort=dict(maxh=h,order=p,ne=mesh.ne,stages=3,mesh_sha256=study.baseline.mesh_identity(mesh),records=[],comparisons=[])
                result['cohorts'].append(cohort)
                records,fields={},{}
                with ng.TaskManager():
                    for route in ('original','condensed','scalar-bulk','scalar-mexcuda'):
                        if route=='scalar-mexcuda':row,field=study.bulk_execute(mesh,p,MexSweep)
                        elif route=='scalar-bulk':row,field=study.bulk_execute(mesh,p)
                        else:row,field=study.baseline.execute(mesh,h,p,3,route)
                        row['route']=route
                        assert row['status']=='pass'
                        records[route],fields[route]=row,field
                        cohort['records'].append(without_timings(row))
                        print(json.dumps(dict(order=p,maxh=h,route=route,status=row['status'])),flush=True)
                    for route in ('condensed','scalar-bulk','scalar-mexcuda'):
                        volume=[]
                        for f,g in zip(fields['original'],fields[route]):
                            delta=f-g
                            volume.append(float(ng.Integrate(delta*delta*ng.dx(bonus_intorder=8),mesh)/ng.Integrate(f*f*ng.dx(bonus_intorder=8),mesh))**.5)
                        error=np.array(records[route]['coefficients'])/records['original']['coefficients']-1
                        assert max(abs(error))<1e-6 and max(volume)<1e-6
                        cohort['comparisons'].append(dict(route=route,relative_coefficients=error.tolist(),volume_relative_field_errors=volume))
                output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
        assert hashes()==result['source_sha256_lf']
        result['complete']=True
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    finally:
        MexSweep.engine=None;engine.quit()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();run(args.directory,args.output,args.quick)
