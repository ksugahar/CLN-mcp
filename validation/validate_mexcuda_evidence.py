"""Check built CUDA MEX and lifecycle evidence without claiming full CLN timing."""
import hashlib
import copy
import json
from pathlib import Path
import numpy as np
from validate_scalar_cln import sources

ROOT=Path(__file__).resolve().parents[1]


def accuracy(data,build):
    assert data['complete'] is True and data['schema']==1
    sources(data,21)
    assert data['binary_sha256']==build['binary_sha256'] and data['device']==build['device']
    assert {(c['maxh'],c['order']) for c in data['cohorts']}=={(.003,2),(.003,3),(.002,3),(.003,4)}
    def no_times(value):
        if isinstance(value,dict):
            assert all(k!='seconds' and not k.endswith('_seconds') for k in value)
            for v in value.values():no_times(v)
        elif isinstance(value,list):
            for v in value:no_times(v)
    no_times(data)
    for c in data['cohorts']:
        assert len(c['records'])==4 and len(c['comparisons'])==3
        assert {r['route'] for r in c['records']}=={'original','condensed','scalar-bulk','scalar-mexcuda'}
        reference=next(r for r in c['records'] if r['route']=='original')
        assert c['stages']==3 and c['ne']>0 and len(c['mesh_sha256'])==64
        for r in c['records']:
            assert r['status']=='pass' and len(r['solves'])==7
            assert np.asarray(r['mode_samples']).shape==(7,78,3)
            energy=np.asarray(r['mode_norm2']);coeff=np.asarray(r['coefficients'])
            assert energy.shape==coeff.shape==(7,) and np.all(energy>0) and np.all(coeff>0)
            reconstructed=np.where(np.arange(7)%2==0,1/(1e6*energy),energy/(4e-7*np.pi))
            assert max(abs(reconstructed/coeff-1))<1e-10
            for s in r['solves']:
                if r['route'] in ('original','condensed'):
                    assert 0<=s['original_rhs_residual']<1e-6
                    continue
                assert s['global_vector_solves']==r['global_vector_solves']==0
                assert s['global_unknown']=='scalar facet potential' and s['mass_perturbation']==0
                assert s['global_factor_shape']==[s['global_scalar_dofs']]*2
                assert s['global_matrix_assembled'] is False and s['boundary_dof_max']==0
                assert s['converged'] is True and 0<=s['curl_fit_residual']<s['requested_tolerance']==1e-10
                for key in ('original_scalar_residual','scalar_trace_residual','original_physical_residual','recovered_original_residual'):
                    assert 0<=s[key]<1e-6
                assert 0<=s['relative_divergence']<1e-8 and 0<=s['relative_curl_field_error']<1e-8
                assert s['physical_integral_order']==13
                if r['route']=='scalar-mexcuda':
                    assert s['backend']=='mexcuda-local/cpu-fe' and s['gpu_dtype']=='float64'
                    assert s['projection_backend']=='mexcuda FP64 fused element kernels'
                    assert s['device']==data['device'] and s['device_operator_bytes']>0
            if r['route']!='original':
                error=np.asarray(r['coefficients'])/reference['coefficients']-1
                comparison=next(v for v in c['comparisons'] if v['route']==r['route'])
                assert np.allclose(error,comparison['relative_coefficients'],rtol=1e-8,atol=1e-14)
                assert max(abs(error))<1e-6
                assert len(comparison['volume_relative_field_errors'])==7
                assert all(0<=v<1e-6 for v in comparison['volume_relative_field_errors'])
                assert r['sample_points_m']==reference['sample_points_m']
                delta=np.asarray(r['mode_samples'])-reference['mode_samples']
                assert max(np.linalg.norm(delta,axis=(1,2))/np.linalg.norm(reference['mode_samples'],axis=(1,2)))<1e-6


def memory_check(memory,verified):
    assert memory['complete'] is True
    sources(memory,23)
    assert len(memory['cases'])==16
    assert {(r['maxh'],r['order'],r['route']) for r in memory['cases']}=={(h,p,r) for h,p in ((.003,2),(.003,3),(.002,3),(.003,4)) for r in ('original','condensed','scalar-bulk','scalar-mexcuda')}
    for r in memory['cases']:
        c=next(c for c in verified['cohorts'] if (c['maxh'],c['order'])==(r['maxh'],r['order']))
        assert r['mesh_sha256']==c['mesh_sha256']
        reference=next(v for v in c['records'] if v['route']=='original')
        row=r['record'];assert row['status']=='pass'
        assert 'seconds' not in row and 'solve_seconds' not in row
        assert max(abs(np.asarray(row['coefficients'])/reference['coefficients']-1))<1e-6
        delta=np.asarray(row['mode_samples'])-reference['mode_samples']
        assert max(np.linalg.norm(delta,axis=(1,2))/np.linalg.norm(reference['mode_samples'],axis=(1,2)))<1e-6
        for key in ('import_baseline','pre_solve','post_solve','post_gc'):
            s=r[key];assert s['peak_wset']>=s['rss']>0 and s['peak_pagefile']>=s['private']>0
        assert r['joint_sampled_peak_rss']>=r['joint_pre_solve_rss']>=r['pre_solve']['rss']
        if r['route']=='scalar-mexcuda':
            assert r['device']==verified['device']
            for key in ('matlab_baseline','matlab_post_solve'):
                s=r[key];assert s['peak_wset']>=s['rss']>0 and s['peak_pagefile']>=s['private']>0
            assert r['joint_sampled_peak_rss']>=r['matlab_baseline']['rss']
            for s in row['solves']:
                assert s['backend']=='mexcuda-local/cpu-fe' and s['converged'] is True
                assert 0<=s['recovered_original_residual']<1e-6


if __name__=='__main__':
    build=json.loads((ROOT/'docs/data/mexcuda_build.json').read_text())
    controls=json.loads((ROOT/'docs/data/mexcuda_controls.json').read_text())
    assert build['complete'] is controls['complete'] is True
    source=ROOT/'validation/cln3d/mexcuda/colored_projection.cu'
    assert build['source_sha256_lf']==hashlib.sha256(source.read_text().encode()).hexdigest()
    assert len(build['binary_sha256'])==64 and build['parallel_license'] is True
    sources(controls,3)
    assert {r['control'] for r in controls['cases']}=={'foreign_handle','wrong_handle_type','wrong_target_size','nonfinite_target','overlapping_writes','cross_element_read_write','index_overflow','double_destroy','stale_handle_after_recreate','stale_handle_after_unload'}
    assert all(r['rejected'] is True and r['reason'] for r in controls['cases'])
    assert controls['parity']['value']==[2.,0.,3.,0.] and controls['parity']['history']==[[1.,0.]]
    data=json.loads((ROOT/'docs/data/mexcuda_cln_verify.json').read_text())
    accuracy(data,build)
    memory=json.loads((ROOT/'docs/data/mexcuda_memory_study.json').read_text())
    memory_check(memory,data)
    for kind in ('host_fallback','physical_residual','field','performance_claim'):
        bad=copy.deepcopy(data)
        row=next(r for r in bad['cohorts'][0]['records'] if r['route']=='scalar-mexcuda')
        if kind=='host_fallback':row['solves'][0]['backend']='host'
        elif kind=='physical_residual':row['solves'][0]['recovered_original_residual']=1e-3
        elif kind=='field':row['mode_samples'][0][0][0]+=1
        else:row['seconds']=1.
        try:accuracy(bad,build)
        except AssertionError:pass
        else:raise AssertionError(f'Corruption accepted: {kind}')
    print('PASS mexcuda build, full-CLN accuracy, ten native rejection/lifecycle controls and four corruptions; formal timing remains separate')
