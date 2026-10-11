"""Matched-mesh 3D round-wire formulation study, with explicit metric windows.

Reuses the published physical field recursions without changing them. The
mesh factory is patched only to supply one caller-owned mesh to every route;
the optional A-T controls change one HCurl order at a time, explicitly.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import socket
import time
from pathlib import Path
from unittest.mock import patch

import mpmath as mp
import ngsolve as ng
import numpy as np

from cln3d import CLN_APhi, CLN_AT, CLN_T_Omega

ROOT = Path(__file__).resolve().parents[1]
MODULES = [CLN_APhi, CLN_T_Omega, CLN_AT]


def hashes():
    paths = [Path(__file__), *(ROOT/'validation/cln3d'/name for name in
             ['CLN_APhi.py', 'CLN_T_Omega.py', 'CLN_AT.py', '_iccg.py'])]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_text(encoding='utf-8-sig').encode()).hexdigest() for p in paths}


def mesh_identity(mesh):
    geometry = dict(points=[list(p.p) for p in mesh.ngmesh.Points()],
                    tetrahedra=[[v.nr for v in e.vertices] for e in mesh.ngmesh.Elements3D()])
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def interleave(rs, ls):
    return [v for k,r in enumerate(rs) for v in ([r,ls[k]] if k<len(ls) else [r])]


def ladder(el, s):
    out = complex(el[-1])
    for k in range(len(el)-2, -1, -1):
        out = el[k]+out if k%2==0 else s*el[k]*out/(s*el[k]+out)
    return out


def reference(frequency):
    # Normalized coefficients in the baseline are per metre / cylinder length.
    with mp.workdps(60):
        radius=mp.mpf('.01'); length=mp.mpf('.01'); sigma=mp.mpf('1e6')
        mu=4*mp.pi*mp.mpf('1e-7'); tau=mu*sigma*radius**2
        r0=1/(mp.pi*radius**2*sigma*length)
        def z(f):
            x=mp.sqrt(2j*mp.pi*float(f)*tau)
            return complex(r0*x*mp.besseli(0,x)/(2*mp.besseli(1,x)))
        el=interleave([float((2*k+1)*r0) for k in range(4)],
                      [float(mu/(8*mp.pi*(k+1)*length)) for k in range(3)])
        return el, [z(f) for f in frequency]


def execute(module, mesh, order, h, stages, bonus, control, oracle, frequencies, exact_z):
    base_hcurl=ng.HCurl
    def hcurl(*args, **kwargs):
        # A has prescribed tangential trace; T does not in this wire problem.
        which='A' if kwargs.get('dirichlet') else 'T'
        if module is CLN_AT and control == 'enrich_'+which:
            kwargs['order'] += 1
        return base_hcurl(*args, **kwargs)
    row=dict(formulation=module.run.__module__.split('.')[-1], order=order,
             maxh=h, stages=stages, bonus_intorder=bonus, control=control,
             vector_orders=dict(A=order+(control=='enrich_A'), T=order+(control=='enrich_T'))
             if module is CLN_AT else dict(vector=order, scalar=order+1))
    start=time.perf_counter()
    log=io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            with patch.object(module, 'make_mesh', return_value=mesh), patch.object(ng,'HCurl',hcurl):
                with ng.TaskManager():
                    d=module.run(maxh=h, order=order, stages=stages, curve=3,
                                 tol=1e-10, bonus_intorder=bonus)
            values=interleave(d['Rn'],d['Ln'])
            signed=np.asarray(values)/np.asarray(oracle[:len(values)])-1
            assert np.all(np.isfinite(signed)) and min(values)>0
            residual=max(max(v['explicit_residual'],v['original_rhs_residual']) for v in d['solves'])
            if residual>1e-6:
                raise RuntimeError(f'Original-system residual {residual} exceeds 1e-6')
            windows={str(count):float(max(abs(signed[:count]))) for count in (3,5,7) if count<=len(values)}
            terminal={str(count):[float(abs(ladder(values[:count],2j*np.pi*f)/z-1))
                                  for f,z in zip(frequencies,exact_z)]
                      for count in (3,5,7) if count<=len(values)}
            # Ideal finite ladders separate analytic truncation from total FE+reduction error.
            row.update(status='pass', coefficients=values, signed_relative_errors=signed.tolist(),
                       max_absolute_relative_error=windows, max_original_residual=residual,
                       terminal_total_relative_error=terminal, mesh=d['mesh'], solves=d['solves'])
            if stages==1:
                rho=np.asarray(d['field_profiles']['rho']); zeros=np.zeros_like(rho)
                mu=d['params']['mu'];sig=d['params']['sigma'];radius=d['params']['r']
                targets=dict(E0=np.column_stack([zeros,zeros,np.ones_like(rho)]),
                             B1=np.column_stack([zeros,mu*oracle[0]*sig*radius*rho/2,zeros]),
                             E2=np.column_stack([zeros,zeros,2*rho**2-1]))
                row['field_profiles']=d['field_profiles']
                row['field_relative_rms']={name:float(np.linalg.norm(np.asarray(d['field_profiles'][name])-value)
                                                    /np.linalg.norm(value)) for name,value in targets.items()}
        except RuntimeError as exc:
            row.update(status='solver_failure', error=str(exc))
    row['seconds']=time.perf_counter()-start
    # Raw console logs remain operational artifacts, not numerical evidence.
    print(json.dumps({k:row[k] for k in ['formulation','order','maxh','stages','control','status']}
                     | {'errors':row.get('max_absolute_relative_error'), 'seconds':row['seconds']}),flush=True)
    if row['status']!='pass':
        print(log.getvalue(),flush=True)
    return row


def run(output):
    ng.SetNumThreads(4)
    sources=hashes();freq=np.logspace(2,6,81).tolist();oracle,exact_z=reference(freq)
    result=dict(schema_version=1, scope='3D internal round wire; zero-shift Type2; matched meshes, p and metric windows',
                parameters=dict(radius_m=.01,length_m=.01,sigma_s_per_m=1e6,mu_h_per_m=4e-7*np.pi,
                                s0=0,curve_order=3,threads=4,tolerance=1e-10),
                boundary='n x A=0 on entire surface; voltage drive E_z=1; conductor internal energy only',
                normalization='baseline coefficients per metre divided by length; terminal coefficients multiply length^2',
                runtime=dict(host=socket.gethostname(),python=platform.python_version(),ngsolve=ng.__version__,
                             radia=importlib.metadata.version('radia'),numpy=np.__version__,mpmath=mp.__version__),
                source_sha256_lf=sources, oracle_coefficients=oracle, frequency_hz=freq,
                exact_impedance=[[z.real,z.imag] for z in exact_z],
                analytic_truncation_relative_error={str(n):[float(abs(ladder(oracle[:n],2j*np.pi*f)/z-1))
                                                           for f,z in zip(freq,exact_z)] for n in (3,5,7)},
                meshes=[],cases=[])
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for h in (.003,.002,.0015):
        mesh=CLN_AT.make_mesh(h,3);identity=mesh_identity(mesh)
        result['meshes'].append(dict(maxh=h,mesh_sha256=identity,ne=mesh.ne,nv=mesh.nv))
        for order in (1,2,3):
            for module in MODULES:
                row=execute(module,mesh,order,h,3,8,'baseline',oracle,freq,exact_z)
                row['mesh_sha256']=identity;result['cases'].append(row);save()
            if h==.003 and order in (1,2):
                for control in ('enrich_A','enrich_T'):
                    row=execute(CLN_AT,mesh,order,h,3,8,control,oracle,freq,exact_z)
                    row['mesh_sha256']=identity;result['cases'].append(row);save()
        # Profile comparisons use the original E0/B1/E2 sampling procedure.
        if h in (.003,.0015):
            for module in MODULES:
                row=execute(module,mesh,1,h,1,8,'profile',oracle,freq,exact_z)
                row['mesh_sha256']=identity;result['cases'].append(row);save()
    # A separate quadrature control keeps failures visible and checks their cause.
    mesh=CLN_AT.make_mesh(.003,3);identity=mesh_identity(mesh)
    for module in MODULES:
        row=execute(module,mesh,1,.003,3,4,'quadrature4',oracle,freq,exact_z)
        row['mesh_sha256']=identity;result['cases'].append(row);save()
    assert sources==hashes(),'sources changed during run'
    result['complete']=True
    result['passed']=all(c['status']=='pass' for c in result['cases'] if c['control']!='quadrature4')
    save()
    if not result['passed']:
        raise SystemExit('Matched study contains unexpected failed solves; inspect recorded cases')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
