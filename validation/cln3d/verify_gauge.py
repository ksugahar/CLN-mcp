"""Check solver robustness under gradient-penalty weights 0.1, 1 and 10.
Run from a writable output directory.
"""
import argparse, hashlib, platform, json, importlib
from pathlib import Path
import numpy as np
import ngsolve
from ngsolve import TaskManager
ngsolve.SetNumThreads(4)
import _iccg
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
args=parser.parse_args()
original=_iccg.solve_iccg
out={}
for name in ['CLN_APhi','CLN_T_Omega','CLN_AT']:
    m=importlib.import_module(name)
    rows=[]
    for weight in [0.1,1.0,10.0]:
        def weighted(*a, **kw):
            return original(*a,**kw,gauge_weight=weight)
        m.solve_iccg=weighted
        with TaskManager():
            data=m.run(maxh=.003,stages=1)
        rows.append(dict(weight=weight,coefficients=[data['R_all'][0]['R'],data['Ln'][0],data['R_all'][1]['R']],solves=data['solves'],field_profiles=data['field_profiles']))
    cs=np.array([r['coefficients'] for r in rows])
    err=float(np.max(abs(cs/cs[1]-1)))
    assert err < 1e-7, (name,err)
    field_changes={}
    for field in ['E0','B1','E2']:
        ref=np.asarray(rows[1]['field_profiles'][field])
        change=max(float(np.linalg.norm(np.asarray(row['field_profiles'][field])-ref)/np.linalg.norm(ref)) for row in rows)
        assert change < 1e-7, (name,field,change)
        field_changes[field]=change
    out[name]=dict(runs=rows,max_relative_gauge_change=err,max_relative_field_change=field_changes)
paths=[Path(__file__),Path(_iccg.__file__),*(Path(importlib.import_module(name).__file__) for name in out)]
metadata=dict(scope='Penalty-weight solver robustness on projected loads; not an independent gauge choice or kernel-completeness proof',
              runtime=dict(python=platform.python_version(),ngsolve=ngsolve.__version__),
              source_sha256_lf={p.name:hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest() for p in paths})
Path(args.output).write_text(json.dumps(dict(out,provenance=metadata),indent=2),encoding='utf-8')
print('All gauge checks passed', {k:v['max_relative_gauge_change'] for k,v in out.items()})
