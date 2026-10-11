"""Portable gate for a complete scalar-global CLN replacement, not postprocessing."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sources(data,count):
    assert len(data['source_sha256_lf'])==count
    for name,digest in data['source_sha256_lf'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name


def verify(data,control,*,same_host=True):
    assert data['schema']==control['schema']==1
    assert data['complete'] is control['complete'] is True
    sources(data,10);sources(control,11)
    assert data['runtime']['threads']==4
    if same_host:assert data['runtime']['host']==control['host']
    pools=[p for p in data['runtime']['blas_pools'] if p['user_api']=='blas']
    assert pools and all(p['num_threads']==1 for p in pools)
    assert len(control['cases'])==4
    assert {(r['curve'],r['order']) for r in control['cases']}=={(c,p) for c in (1,3) for p in (2,3)}
    for row in control['cases']:
        assert 0<=row['scalar_residual']<1e-6 and 0<=row['hybrid_scalar_residual']<1e-6
        assert 1e-8<row['element_projection_lower_bound']<1
        assert 1e-8<row['nodal_hybrid_field_difference']<1
        assert row['strict_recovery_failure']['reason']
    assert len(data['cohorts'])==4
    assert {(c['maxh'],c['order']) for c in data['cohorts']}=={(.003,2),(.003,3),(.002,3),(.003,4)}
    for cohort in data['cohorts']:
        rows=cohort['repeats']
        assert cohort['stages']==3 and cohort['ne']>0 and len(cohort['mesh_sha256'])==64
        assert len(rows)==12 and {(r['route'],r['repeat']) for r in rows}=={(s,k) for s in ('original','condensed','scalar-VPhi') for k in range(4)}
        for r in rows:
            assert r['status']=='pass' and math.isfinite(r['seconds']) and r['seconds']>0
            assert 0<=r['setup_seconds']<=r['seconds']
            c=np.asarray(r['coefficients']);energies=np.asarray(r['mode_norm2'])
            assert c.shape==energies.shape==(7,) and np.all(c>0) and np.all(energies>0)
            reconstructed=np.where(np.arange(7)%2==0,1/(1e6*energies),energies/(4e-7*np.pi))
            assert max(abs(reconstructed/c-1))<1e-10
            assert np.asarray(r['mode_samples']).shape==(7,78,3) and np.isfinite(r['mode_samples']).all()
            assert len(r['solve_seconds'])==len(r['solves'])==7
            assert all(t>0 and math.isfinite(t) for t in r['solve_seconds'])
            assert sum(r['solve_seconds'])<=r['seconds']*1.000001
            for j,s in enumerate(r['solves']):
                if r['route']!='scalar-VPhi':
                    assert 0<=s['original_rhs_residual']<1e-6
                    continue
                assert r['global_vector_solves']==s['global_vector_solves']==0
                assert s['global_unknown']=='scalar facet potential' and s['role']==('V' if j%2==0 else 'Phi')
                assert s['global_factor_shape']==[s['global_scalar_dofs']]*2 and s['global_scalar_dofs']>0
                assert s['eliminated_cell_dofs']>0 and s['fixed_scalar_dofs']>=1 and s['mass_perturbation']==0
                for name in ('scalar_trace_residual','original_scalar_residual','original_physical_residual','recovered_original_residual'):
                    assert 0<=s[name]<1e-6,name
                assert 0<=s['relative_divergence']<1e-8
                assert s['global_matrix_assembled'] is False and s['shared_dofs_corrected'] is True
                assert s['converged'] is True and 1<s['sweeps']<=s['max_sweeps']
                assert 0<=s['curl_fit_residual']<s['requested_tolerance']==1e-10
                assert 0<=s['relative_curl_field_error']<1e-8 and s['boundary_dof_max']==0
                assert s['history'][-1]==[s['sweeps'],s['curl_fit_residual']]
        assert len(cohort['comparisons'])==8
        assert {(c['route'],c['repeat']) for c in cohort['comparisons']}=={(s,k) for s in ('condensed','scalar-VPhi') for k in range(4)}
        for c in cohort['comparisons']:
            a=next(r for r in rows if r['route']==c['route'] and r['repeat']==c['repeat'])
            b=next(r for r in rows if r['route']=='original' and r['repeat']==c['repeat'])
            difference=np.asarray(a['coefficients'])/b['coefficients']-1
            assert np.allclose(difference,c['relative_coefficients'],rtol=1e-8,atol=1e-14)
            assert max(abs(difference))<1e-6
            errors=np.asarray(c['volume_relative_field_errors'])
            assert errors.shape==(7,) and np.all((errors>=0)&(errors<1e-6))
            assert a['sample_points_m']==b['sample_points_m'] and len(a['sample_points_m'])==78
            delta=np.asarray(a['mode_samples'])-b['mode_samples']
            assert max(np.linalg.norm(delta,axis=(1,2))/np.linalg.norm(b['mode_samples'],axis=(1,2)))<1e-6


if __name__=='__main__':
    data=json.loads((ROOT/'docs/data/scalar_cln_study.json').read_text())
    control=json.loads((ROOT/'docs/data/scalar_nodal_control.json').read_text())
    verify(data,control)
    for name in ('vector_fallback','vector_factor','scalar_residual','physical_residual','coefficient','field','coverage','nodal_control'):
        bad,ctrl=copy.deepcopy(data),copy.deepcopy(control)
        row=next(r for r in bad['cohorts'][0]['repeats'] if r['route']=='scalar-VPhi')
        if name=='vector_fallback':row['solves'][0]['global_vector_solves']=1
        elif name=='vector_factor':row['solves'][0]['global_factor_shape'][0]+=1
        elif name=='scalar_residual':row['solves'][0]['original_scalar_residual']=1e-3
        elif name=='physical_residual':row['solves'][1]['original_physical_residual']=1e-3
        elif name=='coefficient':row['coefficients'][2]*=1.01
        elif name=='field':row['mode_samples'][0][0][2]+=1
        elif name=='coverage':bad['cohorts'].pop()
        else:ctrl['cases'][0]['strict_recovery_failure']['reason']=''
        try:verify(bad,ctrl)
        except AssertionError:pass
        else:raise AssertionError(f'Corruption accepted: {name}')
    print('PASS scalar-global complete CLN, physical parity, nodal incompatibility and eight corruptions')
