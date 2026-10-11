"""Scalar V/Phi CLN using colored local curl projections.

The execution adapter keeps the archived equation implementation unchanged.
It temporarily selects the recovery engine; do not call concurrently with
the reference implementation in the same Python process.
"""
from unittest.mock import patch
from . import CLN_VPhi,CLN_AT
from .batched_curl import BatchedSweep
from .element_recovery import Condensed


def run(mesh,order,stages=3,bonus=8):
    with patch.object(CLN_VPhi,'CompatibleSweep',BatchedSweep), \
         patch.object(CLN_AT,'solve_iccg',side_effect=AssertionError('Global A-T fallback forbidden')), \
         patch.object(Condensed,'__init__',side_effect=AssertionError('Global vector trace fallback forbidden')):
        record,fields=CLN_VPhi.run(mesh,order,stages,bonus)
    record['route']='scalar-batched'
    return record,fields


if __name__=='__main__':
    import argparse
    import json
    from pathlib import Path
    import ngsolve as ng
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maxh',type=float,default=.003)
    parser.add_argument('--order',type=int,default=3)
    parser.add_argument('--stages',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ng.SetNumThreads(4)
    mesh=CLN_AT.make_mesh(args.maxh,3)
    with threadpool_limits(limits=1,user_api='blas'),ng.TaskManager():
        record,_=run(mesh,args.order,args.stages)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf8')
