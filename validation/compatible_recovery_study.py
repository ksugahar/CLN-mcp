"""Matched full CLN audit of iterative, conforming element curl recovery.

Uses the previously checked CLN runner, replacing only its recovery adapter.
The global physical trace solve is retained; no global recovery matrix is built.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import ngsolve as ng
import numpy as np

import element_cln_study as baseline
from cln3d.compatible_curl import CompatibleSweep

ROOT=Path(__file__).resolve().parents[1]


def hashes():
    result=baseline.hashes()
    for name in ('validation/compatible_recovery_study.py', 'validation/cln3d/compatible_curl.py'):
        result[name]=hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()
    return result


def run(output,quick=False):
    ng.SetNumThreads(4)
    import socket,platform,importlib.metadata
    result=dict(schema=1,complete=False,source_sha256_lf=hashes(),
        runtime=dict(host=socket.gethostname(),python=platform.python_version(),ngsolve=ng.__version__,
                     numpy=np.__version__,radia=importlib.metadata.version('radia'),threads=4),
        timing_protocol='First observed plus three warmed complete runs; alternating routes; mesh and cross-comparison excluded; assembly, setup, recovery and checks included',
        scope='Zero-shift three-stage internal-wire A-T; physical condensed trace solve retained; iterative local curl inverse feeds every following CLN stage',cohorts=[])
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for h,p in ([(.003,3)] if quick else [(.003,2),(.003,3),(.002,3),(.003,4)]):
        mesh=baseline.CLN_AT.make_mesh(h,3)
        cohort=dict(maxh=h,order=p,stages=3,mesh_sha256=baseline.mesh_identity(mesh),ne=mesh.ne,repeats=[],comparisons=[])
        result['cohorts'].append(cohort)
        with ng.TaskManager():
            for repeat in range(1 if quick else 4):
                paths=['original','condensed','compatible-sweep']
                if repeat%2:paths.reverse()
                records,fields={},{}
                for route in paths:
                    with patch.object(baseline,'CurlSweep',CompatibleSweep):
                        record,field=baseline.execute(mesh,h,p,3,'condensed+sweep' if route=='compatible-sweep' else route)
                    record.update(route=route,repeat=repeat)
                    cohort['repeats'].append(record)
                    records[route],fields[route]=record,field
                    print(json.dumps(dict(order=p,maxh=h,repeat=repeat,route=route,status=record['status'],seconds=record['seconds'],reason=record.get('reason'))),flush=True)
                    save()
                    if record['status']!='pass':raise RuntimeError(record)
                for route in ('condensed','compatible-sweep'):
                    volume=[]
                    for f,g in zip(fields['original'],fields[route]):
                        delta=f-g
                        volume.append(float(ng.Integrate(delta*delta*ng.dx(bonus_intorder=8),mesh)/ng.Integrate(f*f*ng.dx(bonus_intorder=8),mesh))**.5)
                    error=np.array(records[route]['coefficients'])/records['original']['coefficients']-1
                    cohort['comparisons'].append(dict(route=route,repeat=repeat,relative_coefficients=error.tolist(),volume_relative_field_errors=volume))
                    if max(abs(error))>1e-6 or max(volume)>1e-6:raise RuntimeError('Full CLN parity failed')
                save()
    assert result['source_sha256_lf']==hashes()
    result['complete']=True
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    run(args.output,args.quick)
