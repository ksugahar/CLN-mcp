"""Portable parity, timing accounting and rejection gates for colored recovery."""
import copy
import json
import math
from pathlib import Path
from validate_scalar_cln import sources,verify

ROOT=Path(__file__).resolve().parents[1]


def check(data,controls):
    assert data['complete'] is controls['complete'] is True
    sources(data,12);sources(controls,6)
    original=json.loads((ROOT/'docs/data/scalar_cln_study.json').read_text())
    nodal=json.loads((ROOT/'docs/data/scalar_nodal_control.json').read_text())
    # Reuse the entire existing scalar/AT parity gate on each scalar route.
    for scalar in ('scalar-VPhi','scalar-batched'):
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
            if r['route']!='scalar-batched':continue
            for s in r['solves']:
                assert s['sequence']=='colored' and s['batches']>=s['colors']>0
                parts=[s[k] for k in ('target_seconds','iteration_seconds','curl_check_seconds','field_check_seconds')]
                assert all(math.isfinite(v) and v>=0 for v in parts)
                assert sum(parts)<=s['recovery_seconds']*1.00001
    assert len(controls['cases'])==6
    assert {(r['order'],r['control']) for r in controls['cases']}=={(p,c) for p in (2,3,4) for c in ('capped','incompatible')}
    for r in controls['cases']:
        assert r['rejected'] is True and r['reason']
        if r['control']=='capped':assert r['info']['converged'] is False and r['info']['max_sweeps']==1
        else:assert r['info']['relative_curl_field_error']>1e-8
    assert {r['route'] for r in controls['profiles']}=={'CompatibleSweep','BatchedSweep'}
    assert all(r['functions'] and r['info']['converged'] for r in controls['profiles'])
    assert {r['order'] for r in controls['batch_equivalence']}=={2,3,4}
    assert all(0<=r['relative_difference']<1e-12 for r in controls['batch_equivalence'])
    assert controls['fast_entrypoint']['route']=='scalar-batched'
    assert controls['fast_entrypoint']['global_vector_solves']==0
    assert len(controls['fast_entrypoint']['solves'])==7
    assert all(s['recovered_original_residual']<1e-6 for s in controls['fast_entrypoint']['solves'])
    assert {r['route'] for r in controls['setup_profiles']}=={'CompatibleSweep','BatchedSweep'}
    for p in controls['setup_profiles']:
        assert {(s['kind'],s['role']) for s in p['construction']}=={(k,r) for k in ('scalar','recovery') for r in ('T','A')}
        assert len(p['construction'])==4 and all(s['seconds']>0 for s in p['construction'])
        assert sum(s['seconds'] for s in p['construction'])<=p['record']['setup_seconds']
        assert all(s['recovered_original_residual']<1e-6 for s in p['record']['solves'])


if __name__=='__main__':
    data=json.loads((ROOT/'docs/data/batched_cln_study.json').read_text())
    controls=json.loads((ROOT/'docs/data/batched_curl_controls.json').read_text())
    check(data,controls)
    for name in ('accuracy','fallback','timing','coverage','rejection'):
        bad,ctrl=copy.deepcopy(data),copy.deepcopy(controls)
        row=next(r for r in bad['cohorts'][0]['repeats'] if r['route']=='scalar-batched')
        if name=='accuracy':row['solves'][0]['curl_fit_residual']=1e-5
        elif name=='fallback':row['solves'][0]['global_vector_solves']=1
        elif name=='timing':row['solves'][0]['iteration_seconds']=-1
        elif name=='coverage':bad['cohorts'].pop()
        else:ctrl['cases'][0]['rejected']=False
        try:check(bad,ctrl)
        except AssertionError:pass
        else:raise AssertionError(f'Corruption accepted: {name}')
    print('PASS colored recovery: 64 full runs, 112 fast steps, six native rejection controls, five corruptions')
