"""Matched 3D real-shift oracle using full FE Taylor solves, not field recursion."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
from pathlib import Path

import mpmath as mp
import ngsolve as ng
import numpy as np

from cln3d import CLN_AT
from cln3d.wire_pencils import Pencils
from formulation_ranking import mesh_identity

ROOT=Path(__file__).resolve().parents[1]


def reciprocal(c):
    out=[1/c[0]]
    for k in range(1,len(c)):
        out.append(-mp.fsum(c[j]*out[k-j] for j in range(1,k+1))/c[0])
    return out


def cauer(c,count=5):
    out=[]
    for k in range(count):
        out.append(c[0] if k%2==0 else 1/c[0])
        if k+1<count:c=reciprocal(c[1:])
    return out


def both(y,z0):
    y=list(map(mp.mpf,y));out={'Type2':cauer(reciprocal(y))}
    if z0>0:
        g=[z0*y[0]]+[z0*y[k]+y[k-1] for k in range(1,len(y))]
        out['Type1']=[1/v for v in cauer(g)]
    return out


def exact(z):
    if z==0:return mp.mpf(1)
    x=mp.sqrt(z)
    return x*mp.besseli(0,x)/(2*mp.besseli(1,x))


def identities():
    paths=[Path(__file__),ROOT/'validation/formulation_ranking.py',
           ROOT/'validation/cln3d/wire_pencils.py',ROOT/'validation/cln3d/_iccg.py',
           ROOT/'validation/cln3d/CLN_AT.py']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest()
            for p in paths}


def run(output,quick=False):
    ng.SetNumThreads(4);mp.mp.dps=80;sources=identities()
    result=dict(schema_version=1,scope='Independent full 3D FE response/Taylor oracle; five elements; zero and positive real shifts',
                source_sha256_lf=sources,runtime=dict(host=socket.gethostname(),python=platform.python_version(),
                    ngsolve=ng.__version__,radia=importlib.metadata.version('radia'),mpmath=mp.__version__),
                shift_definition='z0=tau*s0; positive case s0=2*pi*1e5; same conductor as zero-shift field study',
                precision_digits=80,meshes=[],cases=[])
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    specs=[(.003,1)] if quick else [(.003,1),(.003,2),(.003,3),(.0015,3)]
    for h,order in specs:
        mesh=CLN_AT.make_mesh(h,3);identity=mesh_identity(mesh)
        with ng.TaskManager():pencils=Pencils(mesh,order)
        result['meshes'].append(dict(maxh=h,order=order,mesh_sha256=identity,ne=mesh.ne,**pencils.metadata))
        for z0 in [mp.mpf(0),8*mp.pi**2]:
            oracle_y=mp.taylor(lambda z:1/exact(z),z0,6)
            oracle=both(oracle_y,z0)
            for family in ['A-Phi','T-Omega','A-T']:
                begin=len(pencils.diagnostics)
                with ng.TaskManager():y=pencils.admittance_moments(family,float(z0))
                el=both(y,z0)
                for kind,values in el.items():
                    signed=[float(v/r-1) for v,r in zip(values,oracle[kind])]
                    row=dict(maxh=h,order=order,mesh_sha256=identity,formulation=family,type=kind,z0=float(z0),
                             normalized_admittance_moments=y,normalized_elements=[float(v) for v in values],
                             exact_normalized_elements=[float(v) for v in oracle[kind]],signed_relative_errors=signed,
                             max_first3_error=max(abs(v) for v in signed[:3]),max_first5_error=max(abs(v) for v in signed),
                             positive_elements=all(v>0 for v in values),
                             solve_diagnostics=pencils.diagnostics[begin:])
                    result['cases'].append(row);save()
                    print(json.dumps({k:row[k] for k in ['maxh','order','formulation','type','z0','max_first3_error','max_first5_error']}),flush=True)
    assert sources==identities(),'Source changed during execution'
    result['complete']=True
    result['passed']=all(c['positive_elements'] and all(d['true_residual']<1e-8 for d in c['solve_diagnostics'])
                         for c in result['cases'])
    save()
    if not result['passed']:raise SystemExit('shifted pencil gate failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();run(args.output,args.quick)
