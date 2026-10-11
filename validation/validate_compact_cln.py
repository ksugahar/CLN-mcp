"""Portable full-recursion gate for cached load/validation operators."""
import copy
import json
import math
from pathlib import Path
from validate_scalar_cln import sources,verify

ROOT=Path(__file__).resolve().parents[1]
SCALARS=('scalar-compact','scalar-compact-mexcuda','scalar-cached','scalar-cached-mexcuda')


def check(data,host,controls):
    assert data['complete'] is host['complete'] is controls['complete'] is True
    sources(data,26);sources(host,2);sources(controls,27)
    assert host['returncode']==0 and 0<=host['preflight_cpu_busy_fraction']<=.05
    assert host['preflight_gpu']['utilization_percent']==0
    assert not any(host['foreign_compute_job_samples']) and not host.get('monitor_errors')
    assert host['observed_owned_processes']
    assert host['preflight_gpu']['name']==data['matlab_engine']['device']
    binary=json.loads((ROOT/'docs/data/mexcuda_build.json').read_text())['binary_sha256']
    assert data['matlab_engine']['binary_sha256']==host['artifact_sha256']['cln_projection_mex.mexw64']==binary
    prior=json.loads((ROOT/'docs/data/mexcuda_timing_host.json').read_text())
    assert host['artifact_sha256']==prior['artifact_sha256']
    old=json.loads((ROOT/'docs/data/scalar_cln_study.json').read_text())
    nodal=json.loads((ROOT/'docs/data/scalar_nodal_control.json').read_text())
    for route in SCALARS:
        view=copy.deepcopy(data)
        view['source_sha256_lf']={n:data['source_sha256_lf'][n] for n in old['source_sha256_lf']}
        for c in view['cohorts']:
            c['repeats']=[r for r in c['repeats'] if r['route'] in ('original','condensed',route)]
            c['comparisons']=[r for r in c['comparisons'] if r['route'] in ('condensed',route)]
            for r in c['repeats']+c['comparisons']:
                if r['route']==route:r['route']='scalar-VPhi'
        verify(view,nodal,same_host=False)
    for cohort in data['cohorts']:
        assert len(cohort['repeats'])==24 and len(cohort['comparisons'])==20
        for row in cohort['repeats']:
            if row['route'] not in SCALARS:continue
            cached=True
            if cached:
                assert row['cache_scope']=='one full recursion; no reuse across repeats'
                assert set(row['cached_operator_csr_bytes_by_role'])=={'A','T'}
                assert all(isinstance(n,int) and n>0 for n in row['cached_operator_csr_bytes_by_role'].values())
            if 'compact' in row['route']:
                storage=row['unique_operator_csr_bytes']
                assert len(storage)==6 and all(isinstance(v,int) and v>0 for v in storage.values())
                assert set(storage)==set(row['operator_assembly_seconds'])
                assert sum(row['cached_operator_csr_bytes_by_role'].values())==2*sum(storage.values())
                assert sum(row['operator_assembly_seconds'].values())<=row['setup_seconds']
                assert all(math.isfinite(v) and v>0 for v in row['operator_assembly_seconds'].values())
                reference=next(r for r in cohort['repeats'] if r['route']=='scalar-cached' and r['repeat']==row['repeat'])
                assert sum(storage.values())<.5*sum(reference['cached_operator_csr_bytes_by_role'].values())
            for s in row['solves']:
                assert s['physical_integral_order']==13 and s['unpacked_operators_released']
                if cached:
                    assert s['validation_global_vector_operator'] is True
                    assert s['load_assembly']=='NGSolve cached rectangular curl-load operator'
                    assert s['physical_check_assembly']=='NGSolve cached rectangular validation operators'
                    parts=[s[k] for k in ('scalar_load_seconds','scalar_linear_seconds','scalar_check_seconds')]
                    assert all(math.isfinite(t) and t>=0 for t in parts)
                    assert abs(sum(parts)-s['scalar_seconds'])<1e-4
                if row['route'].endswith('mexcuda'):
                    assert s['backend']=='mexcuda-local/cpu-fe' and s['gpu_dtype']=='float64'
                    assert s['projection_backend']=='mexcuda FP64 fused element kernels'
                    parts=[s[k] for k in ('device_upload_seconds','mex_internal_projection_seconds','curl_check_seconds','device_download_seconds','matlab_bridge_seconds')]
                    assert all(math.isfinite(t) and t>=0 for t in parts)
                    assert abs(sum(parts)-s['mex_call_seconds'])<1e-6
    assert controls['host']==data['runtime']['host'] and len(controls['cases'])==12
    assert {(r['curve'],r['order'],r['magnetic']) for r in controls['cases']}=={(c,p,b) for c in (1,3) for p in (2,3,4) for b in (False,True)}
    for r in controls['cases']:
        for key in ('curl_load_relative_error','scalar_operator_relative_error'):
            assert math.isfinite(r[key]) and 0<=r[key]<1e-12
        for key in ('physical_vector_relative_errors','scalar_load_relative_errors'):
            assert len(r[key])==3 and all(math.isfinite(v) and 0<=v<1e-12 for v in r[key])


if __name__=='__main__':
    data=json.loads((ROOT/'docs/data/compact_cln_timing.json').read_text())
    host=json.loads((ROOT/'docs/data/compact_cln_host.json').read_text())
    controls=json.loads((ROOT/'docs/data/compact_cln_controls.json').read_text())
    check(data,host,controls)
    for name in ('residual','field','memory','operator','control','overlap'):
        bad,h,c=copy.deepcopy(data),copy.deepcopy(host),copy.deepcopy(controls)
        row=next(r for r in bad['cohorts'][0]['repeats'] if r['route']=='scalar-compact-mexcuda')
        if name=='residual':row['solves'][0]['original_scalar_residual']=.01
        elif name=='field':row['mode_samples'][0][0][0]+=1
        elif name=='memory':row['unique_operator_csr_bytes']['scalar-load']=0
        elif name=='operator':row['solves'][0]['validation_global_vector_operator']=False
        elif name=='control':c['cases'][0]['physical_vector_relative_errors'][1]=1e-3
        else:h['foreign_compute_job_samples'].append(['foreign job'])
        try:check(bad,h,c)
        except AssertionError:pass
        else:raise AssertionError('Corruption accepted: '+name)
    memory=json.loads((ROOT/'docs/data/compact_cln_memory.json').read_text())
    assert memory['complete'] is False and memory['status']=='interrupted_by_user_shutdown'
    assert memory['host']==data['runtime']['host']
    sources({'source_sha256_lf':memory['planned_source_sha256_lf']},27)
    assert len(memory['observed_console_peaks'])==6
    assert all(math.isfinite(r['peak_working_set_mib']) and r['peak_working_set_mib']>0 for r in memory['observed_console_peaks'])
    assert memory['pending'] and 'not recovered' in memory['provenance']
    print('PASS compact CLN: full physics, native operator equivalence, storage, idle host and six corruptions')
