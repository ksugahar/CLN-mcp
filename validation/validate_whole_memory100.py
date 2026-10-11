"""Portable gate: 100 memory evidence; saved mdx timings stay unchanged."""
import copy,hashlib,json,math
from pathlib import Path
import numpy as np
from validate_scalar_cln import sources
ROOT=Path(__file__).resolve().parents[1]
ROUTES=('original','condensed','scalar-cached','scalar-compact','scalar-cached-mexcuda','scalar-compact-mexcuda')

def check(data,mdx):
    assert data['complete'] is True and data['host_role']=='100' and data['host']=='T7920'
    sources(data,29)
    # Raw host-local JSON byte identity; Git normalizes CRLF on Linux checkout.
    text=(ROOT/'docs/data/compact_cln_timing.json').read_text(encoding='utf8')
    assert data['mdx_timing_sha256'] in {hashlib.sha256(t.encode()).hexdigest() for t in (text,text.replace('\n','\r\n'))}
    assert data['binary_sha256']==mdx['matlab_engine']['binary_sha256']
    expected={(c['maxh'],c['order'],route) for c in mdx['cohorts'] for route in ROUTES}
    assert len(data['cases'])==24 and {(r['maxh'],r['order'],r['route']) for r in data['cases']}==expected
    for case in data['cases']:
        cohort=next(c for c in mdx['cohorts'] if (c['maxh'],c['order'])==(case['maxh'],case['order']))
        assert case['mesh_sha256']==cohort['mesh_sha256'] and case['stages']==3
        row=case['record'];assert row['status']=='pass' and row['route']==case['route']
        reference=next(r for r in cohort['repeats'] if r['route']=='original')
        a,b=np.array(row['coefficients']),np.array(reference['coefficients'])
        assert a.shape==b.shape==(7,) and np.isfinite(a).all() and max(abs(a/b-1))<=1e-6
        assert row['sample_points_m']==reference['sample_points_m']
        a,b=np.array(row['mode_samples']),np.array(reference['mode_samples'])
        assert a.shape==b.shape==(7,78,3) and np.isfinite(a).all()
        error=np.linalg.norm((a-b).reshape(7,-1),axis=1)/np.linalg.norm(b.reshape(7,-1),axis=1)
        assert max(error)<=1e-6
        assert max(abs(np.array(row['mode_norm2'])/reference['mode_norm2']-1))<=1e-6
        def no_time(value):
            if isinstance(value,dict):
                assert not any(k=='seconds' or k.endswith('_seconds') for k in value)
                for x in value.values():no_time(x)
            elif isinstance(value,list):
                for x in value:no_time(x)
        no_time(row)
        assert len(row['solves'])==7
        for s in row['solves']:
            if case['route'] in ('original','condensed'):
                assert math.isfinite(s['original_rhs_residual']) and 0<=s['original_rhs_residual']<=1e-6
            else:
                assert s['global_vector_solves']==0 and s['boundary_dof_max']==0
                for key in ('scalar_trace_residual','original_scalar_residual','original_physical_residual','recovered_original_residual'):
                    assert math.isfinite(s[key]) and 0<=s[key]<=1e-6
                for key in ('relative_divergence','relative_curl_field_error'):
                    assert math.isfinite(s[key]) and 0<=s[key]<=1e-8
                assert 0<=s['curl_fit_residual']<1e-10
        family=case['family']
        assert family['samples']>1 and not family['errors'] and family['sample_interval_seconds']==.02
        for key in ('joint_sampled_peak_rss','joint_sampled_peak_private','observed_native_peak_sum_rss','observed_native_peak_sum_private'):
            assert isinstance(family[key],int) and family[key]>0
        for moment in ('import_baseline','pre_solve','post_solve','post_gc'):
            v=case[moment];assert v['peak_wset']>=v['rss']>0 and v['peak_pagefile']>=v['private']>0
        assert family['observed_native_peak_sum_rss']==sum(x['peak_wset'] for x in family['observed_native_peaks'])
        assert family['observed_native_peak_sum_private']==sum(x['peak_pagefile'] for x in family['observed_native_peaks'])
        if case['route'].endswith('mexcuda'):
            assert case['known_device_operator_bytes']>0 and case['matlab'] and case['device']
            assert any(x['pid']==case['matlab_pid'] and x['name'].lower()=='matlab.exe' for x in family['observed_native_peaks'])
        else:assert 'matlab_pid' not in case

if __name__=='__main__':
    data=json.loads((ROOT/'docs/data/whole_memory100.json').read_text())
    mdx=json.loads((ROOT/'docs/data/compact_cln_timing.json').read_text())
    check(data,mdx)
    for kind in ('residual','field','memory','owner','time','source'):
        bad=copy.deepcopy(data)
        row=next(x for x in bad['cases'] if x['route']=='scalar-compact-mexcuda')
        if kind=='residual':row['record']['solves'][0]['original_scalar_residual']=1e-3
        elif kind=='field':row['record']['mode_samples'][0][0][0]+=1
        elif kind=='memory':row['family']['joint_sampled_peak_rss']=0
        elif kind=='owner':row['matlab_pid']=-1
        elif kind=='time':row['record']['seconds']=1
        else:bad['mdx_timing_sha256']='bad'
        try:check(bad,mdx)
        except AssertionError:pass
        else:raise AssertionError('Corruption accepted: '+kind)
    print('PASS 100 whole-memory audit, saved mdx identity, full-stage checks and six corruptions')
