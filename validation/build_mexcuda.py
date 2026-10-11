"""Build the requested standalone CUDA MEX with the official MATLAB Engine."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import socket
import time
import matlab.engine


def run(directory):
    directory.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parent/'cln3d/mexcuda/colored_projection.cu'
    log=io.StringIO();start=time.perf_counter()
    engine=matlab.engine.start_matlab('-nodesktop -nosplash')
    result=dict(host=socket.gethostname(),engine_start_seconds=time.perf_counter()-start,
        source_sha256_lf=hashlib.sha256(source.read_text().encode()).hexdigest())
    try:
        result['matlab']=engine.version()
        result['parallel_license']=bool(engine.license('test','Distrib_Computing_Toolbox'))
        assert result['parallel_license'],'Parallel Computing Toolbox required'
        engine.eval("d=gpuDevice; fprintf('%s, CC %s\\n',d.Name,d.ComputeCapability);",nargout=0,stdout=log,stderr=log)
        result['device']=engine.eval('d.Name')
        result['compute_capability']=engine.eval('d.ComputeCapability')
        engine.cd(str(directory.resolve()),nargout=0)
        start=time.perf_counter()
        engine.mexcuda('-v',str(source),'-output','cln_projection_mex','-outdir',str(directory.resolve()),nargout=0,stdout=log,stderr=log)
        result['build_seconds']=time.perf_counter()-start
        binary=directory/'cln_projection_mex.mexw64'
        result['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        result['complete']=True
    finally:
        engine.quit()
        (directory/'build.log').write_text(log.getvalue(),encoding='utf8')
        (directory/'build.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    run(parser.parse_args().directory)
