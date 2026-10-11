"""Portable numerical/coverage gate for full CLN local-recovery evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def verify(data):
    assert data['schema']==1 and data['complete'] is True
    assert set(data['source_sha256_lf'])=={
        'validation/element_cln_study.py','validation/cln3d/element_recovery.py',
        'validation/cln3d/CLN_AT.py','validation/cln3d/_iccg.py',
        'validation/cln3d/wire_pencils.py','validation/formulation_ranking.py'}
    for name,digest in data['source_sha256_lf'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name
    assert data['runtime']['threads']==4 and data['runtime']['host']
    assert len(data['cohorts'])==4
    assert {(c['maxh'],c['order']) for c in data['cohorts']}=={(.003,2),(.003,3),(.002,3),(.003,4)}
    for cohort in data['cohorts']:
        assert cohort['stages']==3 and cohort['ne']>0 and len(cohort['mesh_sha256'])==64
        runs=cohort['repeats']
        assert len(runs)==16
        assert {(r['route'],r['repeat']) for r in runs}=={(p,k) for p in ('original','full-cholesky','condensed','condensed+sweep') for k in range(4)}
        for r in runs:
            assert math.isfinite(r['seconds']) and r['seconds']>0
            assert 0<=r['setup_seconds']<=r['seconds']
            if r['status']=='rejected':
                assert r['route']=='condensed+sweep'
                assert r['failed_stage'] and 0<=r['completed_solves']<7
                f=r['failure_diagnostics']
                assert f and all(math.isfinite(x) for x in f.values())
                assert max(f['curl_fit_residual'],f['relative_curl_field_error'])>1e-8
                assert 'coefficients' not in r
                continue
            assert r['status']=='pass'
            c=np.asarray(r['coefficients']);norm2=np.asarray(r['mode_norm2'])
            samples=np.asarray(r['mode_samples'])
            assert c.shape==norm2.shape==(7,) and np.all(c>0) and np.all(norm2>0)
            assert samples.shape==(7,78,3) and np.isfinite(samples).all()
            assert np.asarray(r['sample_points_m']).shape==(78,3)
            reconstructed=np.where(np.arange(7)%2==0,1/(1e6*norm2),norm2/(4e-7*np.pi))
            assert np.max(abs(reconstructed/c-1))<1e-10
            assert len(r['solves'])==len(r['solve_seconds'])==7
            assert all(math.isfinite(t) and t>0 for t in r['solve_seconds'])
            assert sum(r['solve_seconds'])<=r['seconds']*1.000001
            for solve in r['solves']:
                assert math.isfinite(solve['original_rhs_residual']) and 0<=solve['original_rhs_residual']<1e-6
                assert 0<=solve['rhs_kernel_fraction']<1e-8
                if r['route']!='original':
                    assert solve['mass_perturbation']==0 and solve['gradient_defect']<1e-10
                    assert solve['full_free']==solve['trace_free']+solve['internal']
                    internal=0 if r['route']=='full-cholesky' else {2:0,3:3,4:11}[cohort['order']]*cohort['ne']
                    assert solve['internal']==internal
                    assert solve['condensation_enabled']==(r['route']!='full-cholesky')
                    assert solve['local_seconds']>=0 and solve['trace_seconds']>=0
        expected={(r['route'],r['repeat']) for r in runs if r['route']!='original' and r['status']=='pass'}
        assert {(r['route'],r['repeat']) for r in cohort['comparisons']}==expected
        assert len(cohort['comparisons'])==len(expected)
        for comp in cohort['comparisons']:
            ref=next(r for r in runs if r['route']=='original' and r['repeat']==comp['repeat'])
            alt=next(r for r in runs if r['route']==comp['route'] and r['repeat']==comp['repeat'])
            error=np.asarray(alt['coefficients'])/ref['coefficients']-1
            assert np.allclose(error,comp['relative_coefficients'],atol=1e-14,rtol=1e-8)
            assert np.max(abs(error))<1e-5
            volume=np.asarray(comp['volume_relative_field_errors'])
            assert volume.shape==(7,) and np.isfinite(volume).all() and np.all((volume>=0)&(volume<1e-5))
            assert alt['sample_points_m']==ref['sample_points_m']
            a,b=np.asarray(alt['mode_samples']),np.asarray(ref['mode_samples'])
            sampled=np.linalg.norm(a-b,axis=(1,2))/np.linalg.norm(b,axis=(1,2))
            assert np.max(sampled)<1e-5


if __name__=='__main__':
    data=json.loads((ROOT/'docs/data/element_cln_study.json').read_text(encoding='utf8'))
    verify(data)
    for control in ('coefficient','field','residual','coverage','silent_fallback'):
        bad=copy.deepcopy(data)
        row=bad['cohorts'][0]['repeats'][1]
        if control=='coefficient':row['coefficients'][2]*=1.01
        elif control=='field':
            values=np.array(row['mode_samples']);index=np.unravel_index(np.argmax(abs(values)),values.shape)
            values[index]*=1.1;row['mode_samples']=values.tolist()
        elif control=='residual':row['solves'][0]['original_rhs_residual']=1e-3
        elif control=='coverage':bad['cohorts'].pop()
        else:
            rejected=next(r for r in bad['cohorts'][0]['repeats'] if r['status']=='rejected')
            rejected['route']='condensed'
        try:verify(bad)
        except AssertionError:pass
        else:raise AssertionError('Corruption accepted: '+control)
    print('PASS full CLN parity, physical fields, equation residuals, timings and explicit sweep failures; five corruptions rejected')
