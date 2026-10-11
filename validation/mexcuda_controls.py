"""Native CUDA MEX parity, input validation and checked-handle lifecycle."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import matlab
import matlab.engine
import numpy as np


def run(directory,output):
    engine=matlab.engine.start_matlab('-nodesktop -nosplash')
    root=Path(__file__).resolve().parents[1]
    hashes={name:hashlib.sha256((root/name).read_text().encode()).hexdigest() for name in
        ('validation/mexcuda_controls.py','validation/build_mexcuda.py','validation/cln3d/mexcuda/colored_projection.cu')}
    result=dict(schema=1,complete=False,host=socket.gethostname(),cases=[],source_sha256_lf=hashes)
    handle=None
    try:
        engine.addpath(str(directory.resolve()),nargout=0)
        def call(command,*args,nargout=0):return engine.feval('cln_projection_mex',command,*args,nargout=nargout)
        metadata=matlab.double([2,1,2,1,0,0,0,0,0])
        ids=matlab.int64([0,1,2,3]);active=matlab.int64([0,2])
        c=matlab.double([1,0,1,0]);inverse=matlab.double([1,1])
        def setup(m=metadata,i=ids,a=active):
            value=call('setup',4.,m,i,a,c,inverse,nargout=2)[0]
            return matlab.uint64([int(np.asarray(value).item())])
        def reject(name,func):
            try:func()
            except matlab.engine.MatlabExecutionError as e:
                result['cases'].append(dict(control=name,rejected=True,reason=str(e).splitlines()[-1]))
            else:raise AssertionError('Invalid MEX input accepted: '+name)
        handle=setup()
        value,history,timing=call('solve',handle,matlab.double([2,3]),1.,1e-10,10.,nargout=3)
        assert np.array_equal(np.asarray(value).ravel(),[2,0,3,0])
        assert np.asarray(history)[-1,1]==0 and np.all(np.asarray(timing)>=0)
        result['parity']=dict(value=np.asarray(value).ravel().tolist(),history=np.asarray(history).tolist())
        reject('foreign_handle',lambda:call('solve',matlab.uint64([0]),matlab.double([2,3]),1.,1e-10,10.,nargout=3))
        reject('wrong_handle_type',lambda:call('destroy',1.))
        reject('wrong_target_size',lambda:call('solve',handle,matlab.double([2]),1.,1e-10,10.,nargout=3))
        reject('nonfinite_target',lambda:call('solve',handle,matlab.double([float('nan'),3]),1.,1e-10,10.,nargout=3))
        reject('overlapping_writes',lambda:setup(a=matlab.int64([0,0])))
        reject('cross_element_read_write',lambda:setup(i=matlab.int64([0,1,2,0])))
        reject('index_overflow',lambda:setup(m=matlab.double([2147483647,1024,1024,1,0,0,0,0,0])))
        previous=handle;call('destroy',handle);handle=None
        reject('double_destroy',lambda:call('destroy',previous))
        handle=setup()
        assert np.asarray(handle).item()!=np.asarray(previous).item()
        reject('stale_handle_after_recreate',lambda:call('destroy',previous))
        call('destroy',handle);handle=None
        # Failed calls must leave the MEX usable and no registered object alive.
        for _ in range(3):
            handle=setup();call('destroy',handle);handle=None
        engine.eval('clear cln_projection_mex;',nargout=0)
        handle=setup()
        assert np.asarray(handle).item()!=np.asarray(previous).item()
        reject('stale_handle_after_unload',lambda:call('destroy',previous))
        call('destroy',handle);handle=None
        result['complete']=True
    finally:
        try:
            if handle is not None:engine.feval('cln_projection_mex','destroy',handle,nargout=0)
        finally:
            engine.quit()
            output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.output)
