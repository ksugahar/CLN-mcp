"""Full zero-shift A-T CLN with local recovery; matched coefficient/field/timing audit."""
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

import ngsolve as ng
import numpy as np

from cln3d import CLN_AT
from cln3d.element_recovery import Condensed, CurlSweep
from formulation_ranking import mesh_identity

ROOT = Path(__file__).resolve().parents[1]


def hashes():
    names = ['validation/element_cln_study.py', 'validation/cln3d/element_recovery.py',
             'validation/cln3d/CLN_AT.py', 'validation/cln3d/_iccg.py',
             'validation/cln3d/wire_pencils.py', 'validation/formulation_ranking.py']
    return {n:hashlib.sha256((ROOT/n).read_text(encoding='utf-8-sig').encode()).hexdigest() for n in names}


def execute(mesh, maxh, order, stages, route):
    captures = []
    engines, sweeps = {}, {}
    original_solver = CLN_AT.solve_iccg
    timings = []
    setup = 0.
    active_label = None
    def solver(a, f, fes, **kwargs):
        nonlocal setup, active_label
        begin = time.perf_counter()
        label = kwargs['label']
        active_label = label
        role = label[0]
        if route == 'original':
            gf, info = original_solver(a, f, fes, **kwargs)
        else:
            if role not in engines:
                engines[role] = Condensed(fes, 1/CLN_AT.SIGMA if role=='T' else 1/CLN_AT.MU, 8,
                                          condense=route!='full-cholesky')
                setup += engines[role].setup_seconds
            gf, info = engines[role].solve(a, f.vec)
            info['label'] = label
            if route == 'condensed+sweep':
                if role not in sweeps:
                    start = time.perf_counter()
                    sweeps[role] = CurlSweep(mesh, order, 8, 'in|out|conductorBND' if role=='A' else '')
                    setup += time.perf_counter()-start
                gf, recovered = sweeps[role].recover(ng.curl(gf))
                info.update(recovered)
                # Check the recovered physical field in the ORIGINAL space.
                _, v = fes.TnT()
                residual = ng.LinearForm(fes)
                weight = 1/CLN_AT.SIGMA if role=='T' else 1/CLN_AT.MU
                residual += weight*ng.curl(v)*ng.curl(gf)*ng.dx(bonus_intorder=8)
                residual.Assemble()
                mask = np.array(list(fes.FreeDofs()), dtype=bool)
                defect = float(np.linalg.norm((residual.vec.FV().NumPy()-f.vec.FV().NumPy())[mask])/
                               np.linalg.norm(f.vec.FV().NumPy()[mask]))
                info['recovered_original_residual'] = defect
                if not np.isfinite(defect) or defect > 1e-6:
                    raise RuntimeError(f'Recovered original equation failed: {defect}')
        timings.append(time.perf_counter()-begin)
        captures.append((role, gf))
        return gf, info
    begin = time.perf_counter()
    with patch.object(CLN_AT, 'make_mesh', return_value=mesh), patch.object(CLN_AT, 'solve_iccg', solver), patch.object(CLN_AT, 'print_solve', lambda info: None), contextlib.redirect_stdout(io.StringIO()):
        try:
            data = CLN_AT.run(maxh=maxh, order=order, stages=stages, curve=3, tol=1e-10, bonus_intorder=8)
        except RuntimeError as exc:
            return dict(route=route, status='rejected', reason=str(exc), completed_solves=len(captures),
                        failed_stage=active_label, failure_diagnostics=getattr(exc,'info',{}),
                        seconds=time.perf_counter()-begin, setup_seconds=setup), None
    elapsed = time.perf_counter()-begin
    fields = [ng.curl(captures[0][1])/CLN_AT.SIGMA]
    electric = ng.curl(captures[0][1])
    magnetic = None
    for j in range(stages):
        magnetic = ng.curl(captures[2*j+1][1]) if magnetic is None else magnetic+ng.curl(captures[2*j+1][1])
        electric = electric+ng.curl(captures[2*j+2][1])
        fields += [magnetic, electric/CLN_AT.SIGMA]
    measure = ng.dx(bonus_intorder=8)
    coefficients = [v for pair in zip(data['Rn'],data['Ln']) for v in pair]+[data['Rn'][-1]]
    record = dict(route=route,status='pass',seconds=elapsed,setup_seconds=setup,
                  solve_seconds=timings,coefficients=coefficients,solves=data['solves'],
                  mode_norm2=[float(ng.Integrate(f*f*measure,mesh)) for f in fields])
    # Physical vector components; no potential comparisons or magnitude-only norm.
    points = [(r*np.cos(theta),r*np.sin(theta),z) for z in (.0025,.005,.0075)
              for theta in (0.,np.pi/4) for r in np.linspace(0,.0095,13)]
    record['sample_points_m'] = points
    record['mode_samples'] = [[list(f(mesh(*p))) for p in points] for f in fields]
    return record, fields


def run(output, quick=False):
    ng.SetNumThreads(4)
    identity = hashes()
    result = dict(schema=1, source_sha256_lf=identity,
                  scope='Internal round wire, zero shift, identical A-T spaces and recursion; trace coupling stays global',
                  runtime=dict(host=socket.gethostname(),python=platform.python_version(),ngsolve=ng.__version__,
                               numpy=np.__version__,radia=importlib.metadata.version('radia'),threads=4),
                  timing_protocol='First observed run and three subsequent complete runs per route in one process; alternating route order; setup and verification included; no mesh generation or comparison/postprocessing time; not process cold-start latency',
                  cohorts=[], complete=False)
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    specs = [(.003,2)] if quick else [(.003,2),(.003,3),(.002,3),(.003,4)]
    for h,p in specs:
        mesh = CLN_AT.make_mesh(h,3)
        cohort = dict(maxh=h,order=p,stages=1 if quick else 3,mesh_sha256=mesh_identity(mesh),
                      ne=mesh.ne,repeats=[],comparisons=[])
        result['cohorts'].append(cohort)
        with ng.TaskManager():
            for repeat in range(1 if quick else 4):
                paths = ('original','full-cholesky','condensed','condensed+sweep')
                if repeat%2: paths=paths[::-1]
                records,fields = {},{}
                for route in paths:
                    record,field = execute(mesh,h,p,cohort['stages'],route)
                    record['repeat'] = repeat
                    cohort['repeats'].append(record)
                    records[route],fields[route] = record,field
                    print(json.dumps({k:record[k] for k in ['route','status','seconds']}|dict(order=p,maxh=h,repeat=repeat,reason=record.get('reason'))),flush=True)
                    save()
                if records['original']['status']!='pass':
                    raise RuntimeError('Original A-T reference failed; no accepted comparison')
                for route in paths:
                    if route=='original' or records[route]['status']!='pass':continue
                    c0=np.array(records['original']['coefficients'])
                    c1=np.array(records[route]['coefficients'])
                    volume=[]
                    for f,g in zip(fields['original'],fields[route]):
                        delta=f-g
                        volume.append(float(ng.Integrate(delta*delta*ng.dx(bonus_intorder=8),mesh)/
                                            ng.Integrate(f*f*ng.dx(bonus_intorder=8),mesh))**.5)
                    row=dict(route=route,repeat=repeat,relative_coefficients=(c1/c0-1).tolist(),
                             volume_relative_field_errors=volume)
                    cohort['comparisons'].append(row)
                    if max(abs(c1/c0-1))>1e-5 or max(volume)>1e-5:
                        raise RuntimeError(f'CLN parity failed: {row}')
                save()
    assert identity==hashes(), 'Source changed during run'
    result['complete']=True
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    run(args.output,args.quick)
