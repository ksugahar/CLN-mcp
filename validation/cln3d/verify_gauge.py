"""Check physical circuit sensitivity to gauge weights 0.1, 1 and 10.
Run from a writable output directory.
"""
import contextlib, json, importlib
from pathlib import Path
import numpy as np
from ngsolve import TaskManager
import _iccg
root=Path.cwd()
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
        if weight==1:
            (root/(name+'_h003.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
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
(root/'gauge_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('All gauge checks passed', {k:v['max_relative_gauge_change'] for k,v in out.items()})
