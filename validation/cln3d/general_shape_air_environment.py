"""Record the public reproduction runtime without host names or local paths."""
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np,scipy,ngsolve as ng,netgen,mpmath

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=dict(complete=True,source_sha256={'validation/cln3d/general_shape_air_environment.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,ngsolve=ng.__version__,netgen=netgen.__version__,mpmath=mpmath.__version__,platform=platform.system(),architecture=platform.machine(),declared_ngsolve_threads=4,float64_epsilon=float(np.finfo(np.float64).eps),longdouble_epsilon=float(np.finfo(np.longdouble).eps)),
        scope='Runtime used for native enclosure reproduction; SuperLU as provided by SciPy. Platform-dependent longdouble precision is explicit.')
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PASS recorded reproduction environment')

if __name__=='__main__':main()
