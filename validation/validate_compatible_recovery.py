"""Portable gates for compatible recovery, full CLN parity and causal controls."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sources(data):
    for name,digest in data['source_sha256_lf'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name


def recovery(info):
    assert info['converged'] is True and info['shared_dofs_corrected'] is True
    assert info['global_matrix_assembled'] is False and info['boundary_dof_max']==0
    assert info['requested_tolerance']==1e-10 and 1<info['sweeps']<=4000
    assert 0<=info['curl_fit_residual']<1e-10
    assert 0<=info['relative_curl_field_error']<1e-8
    assert 0<=info['local_representability_defect']<1e-8
    assert 0<info['recovery_seconds'] and math.isfinite(info['recovery_seconds'])
    history=info['history']
    assert history[0][0]==1 and history[-1][0]==info['sweeps']
    assert history[-1][1]==info['curl_fit_residual'] and history[0][1]>1e-8
    assert all(math.isfinite(x) and x>=0 for _,x in history)


def verify(data,diagnosis):
    assert data['schema']==diagnosis['schema']==1
    assert data['complete'] is diagnosis['complete'] is True
    sources(data);sources(diagnosis)
    assert len(data['source_sha256_lf'])==8 and len(diagnosis['source_sha256_lf'])==9
    assert data['runtime']['threads']==4 and data['runtime']['host']==diagnosis['host']
    assert len(diagnosis['cases'])==6
    assert {(c['curve'],c['order']) for c in diagnosis['cases']}=={(g,p) for g in (1,3) for p in (1,2,3)}
    assert len({c['topology_sha256'] for c in diagnosis['cases']})==1
    for c in diagnosis['cases']:
        assert c['shared_dofs']>0 and c['ne']>0
        assert len(c['greedy'])==6 and len(c['iterative'])==3
        assert {(r['sequence'],r['equation']) for r in c['greedy']}=={(s,e) for s in ('forward','reverse','shuffled') for e in ('K','sqrtK')}
        for row in c['greedy']:
            assert row['first_conflict'] and row['first_conflict']['relative_local_curl_error']>1e-8
            assert 1e-8<row['relative_curl_field_error']<100
            assert np.isclose(row['curl_fit_residual'],row['relative_curl_field_error'],rtol=1e-6,atol=1e-10)
        assert {r['sequence'] for r in c['iterative']}=={'forward','reverse','shuffled'}
        for row in c['iterative']:recovery(row)
        failure=c['capped_failure']
        assert failure['converged'] is False and failure['sweeps']==failure['max_sweeps']==1
        assert failure['curl_fit_residual']>failure['requested_tolerance']
        incompatible=c['incompatible_failure']
        assert incompatible.get('reason') or (incompatible['converged'] is False or incompatible['relative_curl_field_error']>1e-8)
    assert len(data['cohorts'])==4
    assert {(c['maxh'],c['order']) for c in data['cohorts']}=={(.003,2),(.003,3),(.002,3),(.003,4)}
    for cohort in data['cohorts']:
        runs=cohort['repeats']
        assert cohort['stages']==3 and len(runs)==12 and len(cohort['mesh_sha256'])==64
        assert {(r['route'],r['repeat']) for r in runs}=={(p,k) for p in ('original','condensed','compatible-sweep') for k in range(4)}
        for r in runs:
            assert r['status']=='pass' and math.isfinite(r['seconds']) and r['seconds']>0
            assert 0<=r['setup_seconds']<=r['seconds']
            c=np.asarray(r['coefficients']);energy=np.asarray(r['mode_norm2'])
            assert c.shape==energy.shape==(7,) and np.all(c>0) and np.all(energy>0)
            reconstructed=np.where(np.arange(7)%2==0,1/(1e6*energy),energy/(4e-7*np.pi))
            assert np.max(abs(reconstructed/c-1))<1e-10
            assert np.asarray(r['mode_samples']).shape==(7,78,3) and np.isfinite(r['mode_samples']).all()
            assert len(r['solves'])==len(r['solve_seconds'])==7
            assert all(t>0 and math.isfinite(t) for t in r['solve_seconds'])
            assert sum(r['solve_seconds'])<=r['seconds']*1.000001
            for solve in r['solves']:
                assert 0<=solve['original_rhs_residual']<=1e-6 and solve['rhs_kernel_fraction']<1e-8
                if r['route']!='original':
                    assert solve['condensation_enabled'] and solve['mass_perturbation']==0
                    assert solve['gradient_defect']<1e-10
                    assert solve['internal']=={2:0,3:3,4:11}[cohort['order']]*cohort['ne']
                if r['route']=='compatible-sweep':
                    recovery(solve)
                    assert 0<=solve['recovered_original_residual']<=1e-6
        comparisons=cohort['comparisons']
        assert len(comparisons)==8
        assert {(c['route'],c['repeat']) for c in comparisons}=={(p,k) for p in ('condensed','compatible-sweep') for k in range(4)}
        for c in comparisons:
            a=next(r for r in runs if r['route']==c['route'] and r['repeat']==c['repeat'])
            b=next(r for r in runs if r['route']=='original' and r['repeat']==c['repeat'])
            relative=np.asarray(a['coefficients'])/b['coefficients']-1
            assert np.allclose(relative,c['relative_coefficients'],rtol=1e-8,atol=1e-14)
            assert max(abs(relative))<1e-6
            errors=np.asarray(c['volume_relative_field_errors'])
            assert errors.shape==(7,) and np.all((0<=errors)&(errors<1e-6))
            assert a['sample_points_m']==b['sample_points_m'] and len(a['sample_points_m'])==78
            delta=np.asarray(a['mode_samples'])-b['mode_samples']
            assert max(np.linalg.norm(delta,axis=(1,2))/np.linalg.norm(b['mode_samples'],axis=(1,2)))<1e-6


if __name__=='__main__':
    data=json.loads((ROOT/'docs/data/compatible_recovery_study.json').read_text())
    diagnosis=json.loads((ROOT/'docs/data/curl_sweep_diagnosis.json').read_text())
    verify(data,diagnosis)
    for control in ('coefficient','field','residual','coverage','convergence','boundary','global_recovery','curvature_control'):
        bad,diag=copy.deepcopy(data),copy.deepcopy(diagnosis)
        row=next(r for r in bad['cohorts'][0]['repeats'] if r['route']=='compatible-sweep')
        if control=='coefficient':row['coefficients'][0]*=1.01
        elif control=='field':row['mode_samples'][0][0][2]+=1
        elif control=='residual':row['solves'][0]['recovered_original_residual']=1e-3
        elif control=='coverage':bad['cohorts'].pop()
        elif control=='convergence':row['solves'][0]['converged']=False
        elif control=='boundary':row['solves'][1]['boundary_dof_max']=1
        elif control=='global_recovery':row['solves'][0]['global_matrix_assembled']=True
        else:diag['cases'].pop()
        try:verify(bad,diag)
        except AssertionError:pass
        else:raise AssertionError(f'Corruption accepted: {control}')
    print('PASS compatible element recovery, full CLN parity, causal controls and eight corruptions')
