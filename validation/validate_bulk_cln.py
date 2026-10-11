"""Portable full-CLN bulk, actual CUDA, quadrature and memory evidence gate."""
import copy
import json
import math
from pathlib import Path
import numpy as np
from validate_scalar_cln import sources,verify

ROOT=Path(__file__).resolve().parents[1]


def check(cpu,gpu,memory,controls):
    original=json.loads((ROOT/'docs/data/scalar_cln_study.json').read_text())
    nodal=json.loads((ROOT/'docs/data/scalar_nodal_control.json').read_text())
    for data,other,count in ((cpu,'scalar-batched',16),(gpu,'scalar-gpu',18)):
        assert data['complete'] is True
        sources(data,count)
        for scalar in ('scalar-bulk',other):
            view=copy.deepcopy(data)
            view['source_sha256_lf']={n:data['source_sha256_lf'][n] for n in original['source_sha256_lf']}
            for c in view['cohorts']:
                c['repeats']=[r for r in c['repeats'] if r['route'] in ('original','condensed',scalar)]
                c['comparisons']=[r for r in c['comparisons'] if r['route'] in ('condensed',scalar)]
                for r in c['repeats']+c['comparisons']:
                    if r['route']==scalar:r['route']='scalar-VPhi'
            verify(view,nodal)
        for c in data['cohorts']:
            assert len(c['repeats'])==16 and len(c['comparisons'])==12
            for r in c['repeats']:
                if r['route'] not in ('scalar-bulk','scalar-gpu'):continue
                assert r['field_accumulation']=='original discontinuous HDiv coefficients'
                assert set(r['local_numpy_bytes_by_role'])=={'A','T'}
                for s in r['solves']:
                    assert s['load_assembly']=='NGSolve discontinuous HCurl LinearForm'
                    assert s['physical_check_assembly']=='NGSolve compound HCurl LinearForm'
                    assert s['physical_integral_order']==13 and s['unpacked_operators_released'] is True
                    parts=[s[k] for k in ('target_seconds','iteration_seconds','curl_check_seconds','field_check_seconds')]
                    assert all(math.isfinite(v) and v>=0 for v in parts)
                    assert sum(parts)<=s['recovery_seconds']*1.00001
                    if r['route']=='scalar-gpu':
                        assert s['backend']=='cuda-local/cpu-fe' and s['gpu_dtype']=='float64'
                        assert s['projection_backend']=='CUDA FP64 fused element kernels'
                        assert s['device'] and s['device_operator_bytes']>0 and s['cuda_runtime']>0
                        assert s['device_upload_seconds']>=0 and s['device_download_seconds']>=0
                    else:assert s['backend']=='cpu'
    assert controls['complete'] is memory['complete'] is True
    sources(controls,17);sources(memory,18)
    assert len(controls['cases'])==12
    assert {(r['curve'],r['order'],r['boundary']) for r in controls['cases']}=={(c,p,b) for c in (1,3) for p in (2,3,4) for b in ('','.*')}
    for r in controls['cases']:
        assert 0<=r['load_relative_error']<1e-12
        assert 0<=r['compound_relative_error']<1e-12
        assert 0<=r['integral_relative_error']<1e-12 and 0<=r['local_defect']<1e-8
    assert len(memory['cases'])==20
    assert {(r['maxh'],r['order'],r['route']) for r in memory['cases']}=={(h,p,r) for h,p in ((.003,2),(.003,3),(.002,3),(.003,4)) for r in ('original','condensed','scalar-batched','scalar-bulk','scalar-gpu')}
    for case in memory['cases']:
        assert case['record']['status']=='pass'
        for key in ('import_baseline','pre_solve','post_solve','post_gc'):
            s=case[key]
            assert s['peak_wset']>=s['rss']>0 and s['peak_pagefile']>=s['private']>0
        c=next(c for c in cpu['cohorts'] if (c['maxh'],c['order'])==(case['maxh'],case['order']))
        ref=next(r for r in c['repeats'] if r['route']=='original')
        row=case['record']
        assert max(abs(np.asarray(row['coefficients'])/ref['coefficients']-1))<1e-6
        delta=np.asarray(row['mode_samples'])-ref['mode_samples']
        assert max(np.linalg.norm(delta,axis=(1,2))/np.linalg.norm(ref['mode_samples'],axis=(1,2)))<1e-6


if __name__=='__main__':
    environment=json.loads((ROOT/'docs/data/bulk_environment.json').read_text())
    sources(environment,1)
    assert environment['ngsolve_gpu_backend']=='host'
    assert environment['fp64_sum_probe']==3. and environment['device_total_bytes']>0
    data=[json.loads((ROOT/'docs/data'/name).read_text()) for name in ('bulk_cln_study.json','gpu_cln_study.json','bulk_memory_study.json','bulk_controls.json')]
    for c in data[1]['cohorts']:
        for r in c['repeats']:
            if r['route']=='scalar-gpu':
                assert all(s['device']==environment['device'] and s['cupy']==environment['cupy'] and s['cuda_runtime']==environment['cuda_runtime'] for s in r['solves'])
    check(*data)
    for kind in ('quadrature','gpu_fallback','memory','coverage'):
        bad=copy.deepcopy(data)
        if kind=='quadrature':
            row=next(r for r in bad[0]['cohorts'][0]['repeats'] if r['route']=='scalar-bulk')
            row['solves'][0]['physical_integral_order']=8
        elif kind=='gpu_fallback':
            row=next(r for r in bad[1]['cohorts'][0]['repeats'] if r['route']=='scalar-gpu')
            row['solves'][0]['backend']='host'
        elif kind=='memory':bad[2]['cases'][0]['post_solve']['peak_wset']=0
        else:bad[3]['cases'].pop()
        try:check(*bad)
        except AssertionError:pass
        else:raise AssertionError(f'Corruption accepted: {kind}')
    print('PASS bulk/CUDA full CLN, isolated memory, unchanged quadrature and corruption gates')
