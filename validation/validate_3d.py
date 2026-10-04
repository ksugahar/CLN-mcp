"""Reproduce one-stage 3D round-wire CLN and check mesh convergence.

Run with a Python environment containing NGSolve and the public Radia package:
    python validation/validate_3d.py --output three_dimensional.json
Stored evidence is displayed by docs/06_3d_round_wire.ipynb.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np

import ngsolve
from cln3d import CLN_APhi, CLN_T_Omega, CLN_AT


def coefficients(d):
    return [d['R_all'][0]['R'], d['Ln'][0], d['R_all'][1]['R']]


def validate():
    modules=[CLN_APhi, CLN_T_Omega, CLN_AT]
    cases=[]
    for m in modules:
        pair=[]
        for maxh in [.003, .001]:
            with ngsolve.TaskManager():
                d=m.run(maxh=maxh,order=1,stages=1)
            vals=coefficients(d)
            theory=[d['R_all'][0]['R_theory'],d['L_theory'][0],d['R_all'][1]['R_theory']]
            errs=[abs(v/t-1) for v,t in zip(vals,theory)]
            assert all(v>0 for v in vals)
            for info in d['solves']:
                assert info['converged']
                assert info['explicit_residual'] <= 1e-7
                assert info['original_rhs_residual'] <= 1.001e-4
            profiles=d['field_profiles']
            rho=np.asarray(profiles['rho'])
            zeros=np.zeros_like(rho)
            R0=theory[0]; mu=d['params']['mu']; sig=d['params']['sigma']; radius=d['params']['r']
            exacts={'E0':np.column_stack([zeros,zeros,np.ones_like(rho)]),
                    'B1':np.column_stack([zeros,mu*R0*sig*radius*rho/2,zeros]),
                    'E2':np.column_stack([zeros,zeros,2*rho**2-1])}
            field_errors={key:float(np.linalg.norm(np.asarray(profiles[key])-oracle)/np.linalg.norm(oracle))
                          for key,oracle in exacts.items()}
            assert all(np.isfinite(list(field_errors.values())))
            row={'formulation':d['formulation'],'maxh':maxh,'coefficients':vals,
                 'theory':theory,'relative_errors':errs,'field_relative_rms':field_errors,'result':d}
            cases.append(row); pair.append(row)
        assert all(a>2.5*b for a,b in zip(pair[0]['relative_errors'],pair[1]['relative_errors']))
        assert max(pair[1]['relative_errors'])<.05
        print('field errors', m.__name__, [v['field_relative_rms'] for v in pair])
        assert max(pair[1]['field_relative_rms'].values()) < .1
        assert pair[1]['field_relative_rms']['E2'] < pair[0]['field_relative_rms']['E2']
    root=Path(__file__).parent
    sources={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [root/'validate_3d.py',*(root/'cln3d'/name for name in
                       ['CLN_APhi.py','CLN_T_Omega.py','CLN_AT.py','_iccg.py'])]}
    return {'scope':'3D cylinder, internal energy, s0=0, order=1, R0/L1/R2',
            'review':'Codex self-review; independent Claude review pending',
            'runtime':{'python':platform.python_version(),'ngsolve':ngsolve.__version__},
            'source_sha256':sources,'cases':cases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    data=validate()
    Path(args.output).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('PASS: six 3D runs, explicit residuals, positive energies and mesh improvement')


if __name__=='__main__':
    main()
