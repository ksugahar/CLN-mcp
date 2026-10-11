"""Record the actual native FE and CUDA device capabilities without fallback."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import cupy as cp
import ngsolve as ng
import ngsolve.gpu as gpu


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    props=cp.cuda.runtime.getDeviceProperties(0)
    probe=float(cp.arange(3,dtype=cp.float64).sum().get())
    assert probe==3.
    args.output.write_text(json.dumps(dict(schema=1,host=socket.gethostname(),python=platform.python_version(),
        platform=platform.platform(),ngsolve=ng.__version__,ngsolve_gpu_backend=gpu.backend,
        ngsolve_gpu_device=str(gpu.GetGPUDevice()),cupy=cp.__version__,
        cuda_runtime=int(cp.cuda.runtime.runtimeGetVersion()),cuda_driver=int(cp.cuda.runtime.driverGetVersion()),
        device=props['name'].decode(),device_total_bytes=int(props['totalGlobalMem']),
        compute_capability=[int(props['major']),int(props['minor'])],fp64_sum_probe=probe,
        dependencies={name:importlib.metadata.version(name) for name in ('numpy','radia','psutil','cupy-cuda12x')},
        source_sha256_lf={'validation/bulk_environment.py':hashlib.sha256(Path(__file__).read_text(encoding='utf-8-sig').encode()).hexdigest()}),
        indent=2,allow_nan=False)+'\n',encoding='utf8')
